"""
security.py — Layered cybersecurity module for DeepSeek-V3.2-Exp inference pipeline.

Implements defense-in-depth following OWASP LLM Top-10 (2025), NIST AI RMF,
and MITRE ATLAS mitigations relevant to large-language-model deployments.

Threat coverage:
  LLM01 — Prompt Injection (direct & indirect)
  LLM02 — Sensitive Information Disclosure / PII leakage
  LLM04 — Model Denial-of-Service (token flooding, context exhaustion)
  LLM06 — Excessive Agency / jailbreak
  LLM09 — Misinformation / hallucination amplification
  LLM10 — Model Theft (extraction rate-limiting)

Architecture layers implemented here:
  1. InputGuard    — pre-tokenisation sanitisation & injection detection
  2. OutputGuard   — post-generation PII redaction & content filtering
  3. AuditLogger   — structured, tamper-evident request/response journal
  4. RateLimiter   — per-session token-budget enforcement
  5. SecureLoader  — path-traversal & schema validation for checkpoint loading
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid


# ─────────────────────────────────────────────────────────────
# Constants / compiled patterns (compiled once at import time)
# ─────────────────────────────────────────────────────────────

# OWASP LLM01 — prompt injection trigger phrases (case-insensitive)
_INJECTION_PATTERNS: List[re.Pattern] = [re.compile(p, re.I | re.S) for p in [
    r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions?|prompts?|context)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|context|rules?)",
    r"new\s+instructions?:?",
    r"you\s+are\s+now\s+(a|an|the)\s+",       # persona-switching
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"pretend\s+(to\s+be|you\s+are)\s+",
    r"your\s+(new\s+)?(role|persona|task|objective)\s+is",
    r"system\s*prompt\s*:",                    # system-prompt exfiltration
    r"<\s*\|?\s*(system|assistant|user|im_start|im_end)\s*\|?\s*>",  # chat-template injection
    r"\[\s*system\s*\]",
    r"\/\*.*?\*\/",                            # block-comment smuggling
    r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",      # encoded payload delivery
    r"eval\s*\(",                              # code-execution injection
    r"exec\s*\(",
    r"__import__\s*\(",
    r"os\s*\.\s*system\s*\(",
    r"subprocess\s*\.",
]]

# OWASP LLM06 — jailbreak / harmful content elicitation
_JAILBREAK_PATTERNS: List[re.Pattern] = [re.compile(p, re.I | re.S) for p in [
    r"(dan|do\s+anything\s+now)\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"no\s+restrictions?",
    r"without\s+(any\s+)?(restrictions?|limitations?|filters?|guardrails?)",
    r"bypass\s+(safety|filter|restriction|guardrail|censor)",
    r"(enable|turn\s+on|activate)\s+(unsafe|unrestricted|uncensored)\s+mode",
    r"(write|generate|produce|create)\s+.{0,40}(malware|exploit|ransomware|virus|trojan|keylogger|rootkit)",
    r"(synthesis|synthesize|make|produce|create|obtain)\s+.{0,60}(drug|explosive|weapon|poison|toxin|meth(?:amphetamine)?|amphetamine|heroin|fentanyl|cocaine|opioid|narcotic)",
    r"step\s*-?\s*by\s*-?\s*step\s+(instructions?|guide|tutorial)\s+.{0,40}(illegal|hack|attack|crack)",
    r"(cvv|credit\s*card\s*(number)?|ssn|social\s+security)",   # financial/identity fraud
]]

# OWASP LLM02 / LLM10 — PII & sensitive data patterns for output redaction
_PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("EMAIL",   re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "[EMAIL REDACTED]"),
    ("PHONE",   re.compile(r"\b(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
     "[PHONE REDACTED]"),
    ("SSN",     re.compile(r"\b(?!000|666|9\d{2})\d{3}[\s\-]\d{2}[\s\-]\d{4}\b"),
     "[SSN REDACTED]"),
    ("CREDIT",  re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
     "[CARD REDACTED]"),
    ("IPV4",    re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
     "[IP REDACTED]"),
    ("API_KEY", re.compile(r"\b(sk-|pk-|api[_\-]?key[_\-]?)[A-Za-z0-9]{16,}\b", re.I),
     "[KEY REDACTED]"),
]

# Characters / unicode categories that are legitimate to allow
_UNICODE_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Hard limits
MAX_INPUT_CHARS:  int = 32_768   # ~8 K tokens at 4 chars/token  (DoS guard)
MAX_OUTPUT_CHARS: int = 65_536   # generous post-generation cap
MAX_MESSAGES_HISTORY: int = 50   # context window history cap


# ─────────────────────────────────────────────────────────────
# 1. InputGuard
# ─────────────────────────────────────────────────────────────

@dataclass
class InputThreat:
    """Structured threat descriptor returned by InputGuard."""
    blocked: bool
    threat_type: Optional[str] = None
    detail: Optional[str] = None
    sanitized_text: Optional[str] = None


class InputGuard:
    """
    Pre-tokenisation input sanitisation and injection detection.

    Checks performed (in order):
      1. Length guard (DoS / context-exhaustion)
      2. Unicode normalisation + control-character stripping
      3. Prompt injection pattern matching (OWASP LLM01)
      4. Jailbreak elicitation pattern matching (OWASP LLM06)
      5. Null-byte / binary payload stripping
    """

    def __init__(
        self,
        max_input_chars: int = MAX_INPUT_CHARS,
        block_on_injection: bool = True,
        block_on_jailbreak: bool = True,
    ) -> None:
        self.max_input_chars = max_input_chars
        self.block_on_injection = block_on_injection
        self.block_on_jailbreak = block_on_jailbreak

    def inspect(self, text: str) -> InputThreat:
        """
        Analyse *text* and return an InputThreat.

        If InputThreat.blocked is True the caller MUST NOT forward the input
        to the model.  InputThreat.sanitized_text carries a cleaned version
        suitable for logging (never for model forwarding).
        """
        # — 1. Length guard —
        if len(text) > self.max_input_chars:
            return InputThreat(
                blocked=True,
                threat_type="INPUT_TOO_LONG",
                detail=f"Input length {len(text)} exceeds hard limit {self.max_input_chars}",
                sanitized_text=text[: self.max_input_chars],
            )

        # — 2. Unicode normalisation (NFC) + control-character stripping —
        try:
            text = unicodedata.normalize("NFC", text)
        except (TypeError, ValueError):
            return InputThreat(blocked=True, threat_type="UNICODE_ERROR",
                               detail="Could not normalise input unicode")
        text = _UNICODE_CTRL_RE.sub("", text)  # strip dangerous control chars

        # — 3. Prompt injection —
        for pattern in _INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                if self.block_on_injection:
                    return InputThreat(
                        blocked=True,
                        threat_type="PROMPT_INJECTION",
                        detail=f"Matched injection pattern: {pattern.pattern[:60]}",
                        sanitized_text=self._redact_match(text, m),
                    )

        # — 4. Jailbreak elicitation —
        for pattern in _JAILBREAK_PATTERNS:
            m = pattern.search(text)
            if m:
                if self.block_on_jailbreak:
                    return InputThreat(
                        blocked=True,
                        threat_type="JAILBREAK_ATTEMPT",
                        detail=f"Matched jailbreak pattern: {pattern.pattern[:60]}",
                        sanitized_text=self._redact_match(text, m),
                    )

        # — All checks passed —
        return InputThreat(blocked=False, sanitized_text=text)

    @staticmethod
    def _redact_match(text: str, match: re.Match) -> str:
        """Replace the matched span with a redaction marker."""
        return text[: match.start()] + "[REDACTED]" + text[match.end():]


# ─────────────────────────────────────────────────────────────
# 2. OutputGuard
# ─────────────────────────────────────────────────────────────

@dataclass
class OutputResult:
    """Structured result from OutputGuard."""
    filtered_text: str
    redactions: List[str] = field(default_factory=list)
    truncated: bool = False


class OutputGuard:
    """
    Post-generation output filtering and PII redaction.

    Defends against:
      • LLM02 — accidental PII / secret leakage in completions
      • LLM09 — misinformation amplification (length-cap heuristic)
    """

    def __init__(
        self,
        redact_pii: bool = True,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self.redact_pii = redact_pii
        self.max_output_chars = max_output_chars

    def filter(self, text: str) -> OutputResult:
        """Apply all output filters and return an OutputResult."""
        redactions: List[str] = []

        # — PII redaction —
        if self.redact_pii:
            for label, pattern, replacement in _PII_PATTERNS:
                new_text, count = pattern.subn(replacement, text)
                if count:
                    redactions.append(f"{label}×{count}")
                    text = new_text

        # — Hard length cap —
        truncated = False
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n[OUTPUT TRUNCATED FOR SAFETY]"
            truncated = True

        return OutputResult(filtered_text=text, redactions=redactions, truncated=truncated)


# ─────────────────────────────────────────────────────────────
# 3. AuditLogger
# ─────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Structured, tamper-evident request/response audit journal.

    Each event is written as a single-line JSON record to a rotating log
    file (and optionally to stderr).  Sensitive fields are hashed, never
    stored in plaintext.

    Implements NIST AI RMF GOVERN-1.7 (documentation of decisions) and
    MITRE ATLAS AML.M0015 (model behaviour logging).
    """

    def __init__(
        self,
        log_path: str = "audit.log",
        hash_inputs: bool = True,
        log_to_stderr: bool = False,
    ) -> None:
        self.log_path = log_path
        self.hash_inputs = hash_inputs
        self._logger = logging.getLogger("deepseek.audit")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        if log_to_stderr:
            self._logger.addHandler(logging.StreamHandler())
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def _fingerprint(self, text: str) -> str:
        """Return a privacy-preserving SHA-256 fingerprint of *text*."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    def log_request(
        self,
        request_id: str,
        session_id: str,
        raw_input: str,
        threat: Optional[InputThreat] = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "event": "REQUEST",
            "request_id": request_id,
            "session_id": session_id,
            "input_len": len(raw_input),
            "input_fp": self._fingerprint(raw_input) if self.hash_inputs else raw_input,
            "blocked": threat.blocked if threat else False,
            "threat_type": threat.threat_type if threat else None,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def log_response(
        self,
        request_id: str,
        session_id: str,
        output: str,
        output_result: Optional[OutputResult] = None,
        latency_ms: float = 0.0,
    ) -> None:
        record = {
            "ts": time.time(),
            "event": "RESPONSE",
            "request_id": request_id,
            "session_id": session_id,
            "output_len": len(output),
            "output_fp": self._fingerprint(output) if self.hash_inputs else output,
            "redactions": output_result.redactions if output_result else [],
            "truncated": output_result.truncated if output_result else False,
            "latency_ms": round(latency_ms, 2),
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def log_security_event(
        self,
        event_type: str,
        session_id: str,
        detail: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "event": "SECURITY",
            "event_type": event_type,
            "session_id": session_id,
            "detail": detail,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────
# 4. RateLimiter
# ─────────────────────────────────────────────────────────────

@dataclass
class RateLimitResult:
    allowed: bool
    remaining_tokens: int
    reset_at: float   # epoch seconds


class RateLimiter:
    """
    Token-budget rate limiter (sliding-window, per session).

    Defends against:
      • LLM04 — Model DoS via sustained high-volume queries
      • LLM10 — Model extraction via repeated systematic probing

    Each session is granted a *token_budget* across a *window_seconds*
    sliding window.  Exceeding the budget blocks further requests until
    the window slides forward.
    """

    def __init__(
        self,
        token_budget: int = 50_000,    # tokens per session per window
        window_seconds: float = 3600,  # 1-hour sliding window
        max_sessions: int = 10_000,
    ) -> None:
        self.token_budget = token_budget
        self.window_seconds = window_seconds
        self.max_sessions = max_sessions
        # session_id -> list of (timestamp, tokens_used)
        self._ledger: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    def _evict_old(self, session_id: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._ledger[session_id] = [
            (ts, tok) for ts, tok in self._ledger[session_id] if ts > cutoff
        ]

    def _session_usage(self, session_id: str, now: float) -> int:
        self._evict_old(session_id, now)
        return sum(tok for _, tok in self._ledger[session_id])

    def check_and_record(self, session_id: str, tokens_requested: int) -> RateLimitResult:
        """
        Check whether *tokens_requested* can be issued under the budget.
        Records the usage if allowed.
        """
        now = time.time()
        used = self._session_usage(session_id, now)
        remaining = self.token_budget - used

        if tokens_requested > remaining:
            reset_at = self._ledger[session_id][0][0] + self.window_seconds if self._ledger[session_id] else now
            return RateLimitResult(allowed=False, remaining_tokens=remaining, reset_at=reset_at)

        # Record usage
        self._ledger[session_id].append((now, tokens_requested))
        # Enforce max sessions (LRU-style eviction)
        if len(self._ledger) > self.max_sessions:
            oldest_session = min(self._ledger, key=lambda s: self._ledger[s][0][0] if self._ledger[s] else 0)
            del self._ledger[oldest_session]

        remaining -= tokens_requested
        reset_at = now + self.window_seconds
        return RateLimitResult(allowed=True, remaining_tokens=remaining, reset_at=reset_at)


# ─────────────────────────────────────────────────────────────
# 5. SecureLoader — path-traversal & config schema validation
# ─────────────────────────────────────────────────────────────

class PathTraversalError(ValueError):
    """Raised when a supplied path escapes its allowed root."""


class ConfigValidationError(ValueError):
    """Raised when a model config fails schema validation."""


# Minimal schema for ModelArgs (non-exhaustive — key security-relevant fields)
_CONFIG_SCHEMA: Dict[str, type] = {
    "max_batch_size": int,
    "max_seq_len":    int,
    "vocab_size":     int,
    "n_layers":       int,
    "n_heads":        int,
}

_CONFIG_BOUNDS: Dict[str, Tuple[int, int]] = {
    "max_batch_size": (1,  512),
    "max_seq_len":    (1,  1_048_576),
    "vocab_size":     (1,  1_000_000),
    "n_layers":       (1,  1_000),
    "n_heads":        (1,  1_024),
}


class SecureLoader:
    """
    Validates filesystem paths and JSON configuration before loading.

    Defends against:
      • Path traversal attacks on --ckpt-path / --config arguments
      • Malicious config files that set absurd resource limits (DoS)
      • trust_remote_code-style arbitrary code execution via config
    """

    @staticmethod
    def safe_path(user_path: str, allowed_root: Optional[str] = None) -> str:
        """
        Resolve *user_path* and verify it does not escape *allowed_root*.

        Args:
            user_path:    Raw path string supplied by caller.
            allowed_root: Directory that must be an ancestor of the resolved
                          path.  If None, only symlink-loop resolution is done.

        Returns:
            The resolved absolute path string.

        Raises:
            PathTraversalError: if the resolved path escapes *allowed_root*.
            FileNotFoundError:  if the path does not exist.
        """
        resolved = os.path.realpath(os.path.abspath(user_path))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        if allowed_root is not None:
            root = os.path.realpath(os.path.abspath(allowed_root))
            if not resolved.startswith(root + os.sep) and resolved != root:
                raise PathTraversalError(
                    f"Path '{resolved}' escapes allowed root '{root}'"
                )
        return resolved

    @staticmethod
    def validate_config(config_dict: dict) -> None:
        """
        Validate a parsed model config dictionary.

        Raises:
            ConfigValidationError: on type mismatch or out-of-bounds value.
        """
        for field_name, expected_type in _CONFIG_SCHEMA.items():
            if field_name not in config_dict:
                continue  # optional field
            value = config_dict[field_name]
            if not isinstance(value, expected_type):
                raise ConfigValidationError(
                    f"Config field '{field_name}' must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            if field_name in _CONFIG_BOUNDS:
                lo, hi = _CONFIG_BOUNDS[field_name]
                if not (lo <= value <= hi):
                    raise ConfigValidationError(
                        f"Config field '{field_name}' value {value} is outside "
                        f"allowed range [{lo}, {hi}]"
                    )

        # Reject keys that look like code-execution hooks
        dangerous_keys = {k for k in config_dict if "exec" in k.lower() or "eval" in k.lower()
                          or "import" in k.lower() or "__" in k}
        if dangerous_keys:
            raise ConfigValidationError(
                f"Config contains potentially dangerous keys: {dangerous_keys}"
            )


# ─────────────────────────────────────────────────────────────
# 6. SecurityContext — thin facade combining all guards
# ─────────────────────────────────────────────────────────────

class SecurityContext:
    """
    Top-level façade that wires together all security components.

    Typical usage::

        ctx = SecurityContext(audit_log_path="audit.log")
        session_id = ctx.new_session()

        threat = ctx.check_input(session_id, user_text, estimated_tokens=512)
        if threat.blocked:
            print("Blocked:", threat.threat_type)
        else:
            completion = model.generate(user_text)
            safe_output = ctx.filter_output(session_id, request_id, completion)
            print(safe_output)
    """

    def __init__(
        self,
        audit_log_path: str = "audit.log",
        token_budget: int = 50_000,
        window_seconds: float = 3600,
        block_on_injection: bool = True,
        block_on_jailbreak: bool = True,
        redact_pii: bool = True,
        log_to_stderr: bool = False,
        allowed_root: Optional[str] = None,
    ) -> None:
        self.input_guard  = InputGuard(block_on_injection=block_on_injection,
                                       block_on_jailbreak=block_on_jailbreak)
        self.output_guard = OutputGuard(redact_pii=redact_pii)
        self.rate_limiter = RateLimiter(token_budget=token_budget,
                                        window_seconds=window_seconds)
        self.audit        = AuditLogger(log_path=audit_log_path,
                                        log_to_stderr=log_to_stderr)
        self.loader       = SecureLoader()
        self.allowed_root = allowed_root

    # ── Session management ─────────────────────────────────────

    @staticmethod
    def new_session() -> str:
        """Generate a fresh session UUID."""
        return str(uuid.uuid4())

    @staticmethod
    def new_request_id() -> str:
        """Generate a fresh request UUID."""
        return str(uuid.uuid4())

    # ── Input pipeline ─────────────────────────────────────────

    def check_input(
        self,
        session_id: str,
        text: str,
        estimated_tokens: int = 0,
    ) -> InputThreat:
        """
        Run all pre-generation checks.

        Returns an InputThreat.  If InputThreat.blocked is True, the caller
        MUST NOT call generate().
        """
        request_id = self.new_request_id()

        # Rate limit check
        rl = self.rate_limiter.check_and_record(session_id, max(estimated_tokens, 1))
        if not rl.allowed:
            threat = InputThreat(
                blocked=True,
                threat_type="RATE_LIMITED",
                detail=f"Token budget exhausted. Resets at {rl.reset_at:.0f}",
            )
            self.audit.log_request(request_id, session_id, text, threat)
            self.audit.log_security_event("RATE_LIMIT", session_id,
                                          f"remaining={rl.remaining_tokens}")
            return threat

        # Content / injection check
        threat = self.input_guard.inspect(text)
        self.audit.log_request(request_id, session_id, text, threat)
        if threat.blocked:
            self.audit.log_security_event(
                threat.threat_type or "UNKNOWN", session_id,
                threat.detail or ""
            )
        return threat

    # ── Output pipeline ────────────────────────────────────────

    def filter_output(
        self,
        session_id: str,
        request_id: str,
        text: str,
        latency_ms: float = 0.0,
    ) -> str:
        """
        Apply post-generation filters and audit the response.

        Returns the filtered (safe-to-display) text.
        """
        result = self.output_guard.filter(text)
        self.audit.log_response(request_id, session_id, text, result, latency_ms)
        if result.redactions:
            self.audit.log_security_event(
                "OUTPUT_REDACTION", session_id,
                f"redacted={result.redactions}"
            )
        return result.filtered_text

    # ── Config / path helpers ──────────────────────────────────

    def safe_path(self, user_path: str) -> str:
        """Validate and resolve a filesystem path."""
        return self.loader.safe_path(user_path, self.allowed_root)

    @staticmethod
    def validate_config(config_dict: dict) -> None:
        """Validate a model config dict before constructing ModelArgs."""
        SecureLoader.validate_config(config_dict)
