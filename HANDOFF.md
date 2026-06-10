# Session Handoff — DeepSeek-V3.2-Exp Cybersecurity Hardening

**Branch:** `claude/ai-cybersecurity-research-6cf2l`
**Date:** 2026-06-10
**Status:** Complete — pushed, all tests green

---

## What Was Done

### Research

Performed multi-source threat research covering:

- **OWASP LLM Top-10 (2025)** — Prompt injection (LLM01) is rated the #1 threat; jailbreaking, PII leakage, DoS/extraction are LLM02/04/06/10.
- **NIST CAISI evaluation (Sept 2025)** — DeepSeek models are ~12× more likely than U.S. frontier models to follow malicious agent instructions.
- **Cisco/Qualys/HiddenLayer findings** — DeepSeek R1 showed 100% attack success rate against HarmBench; V3 failed jailbreak tests broadly.
- **Infrastructure incidents** — 3.2 Tbps DDoS (Jan 2025), XSS on CDN endpoint, exposed data-processing database, keystroke data routed to Chinese-state-linked servers.

### Code Changes

| File | Change |
|------|--------|
| `inference/security.py` | **New** — 300-line layered security module (see below) |
| `inference/generate.py` | **Modified** — SecurityContext integrated into both interactive and batch modes |
| `inference/tests/test_security.py` | **New** — 55 unit tests, all passing |
| `inference/tests/__init__.py` | **New** — makes tests a package |
| `.github/workflows/python-package-conda.yml` | **Modified** — added `security-scan` CI job |
| `.gitignore` | **New** — excludes `__pycache__`, audit logs, build artefacts |

---

## Architecture of `inference/security.py`

```
SecurityContext  (top-level façade)
├── InputGuard       pre-tokenisation checks
│     • 18 prompt-injection regex patterns (OWASP LLM01)
│     • 11 jailbreak/elicitation patterns  (OWASP LLM06)
│     • Unicode NFC normalisation + control-char stripping
│     • Hard input length cap (32 768 chars default)
├── OutputGuard      post-generation filters
│     • PII redaction: email, phone, SSN, credit card, IPv4, API key
│     • Hard output length cap (65 536 chars default)
├── AuditLogger      structured JSON audit journal
│     • One record per request + one per response
│     • Inputs stored as SHA-256 fingerprints (privacy-preserving)
│     • Security events logged separately
├── RateLimiter      sliding-window token budget per session
│     • Default: 50 000 tokens / 1 hour window  (LLM04 / LLM10)
│     • Per-session, independent ledgers
└── SecureLoader     filesystem + config validation
      • Path-traversal guard for --ckpt-path / --config args
      • JSON config schema + numeric bounds validation
      • Rejects configs containing __dunder__ or exec/eval keys
```

### How to Use

```python
from security import SecurityContext

ctx = SecurityContext(audit_log_path="audit.log", token_budget=50_000)
session_id = ctx.new_session()

threat = ctx.check_input(session_id, user_text, estimated_tokens=len(user_text)//4)
if threat.blocked:
    print(f"Blocked: {threat.threat_type} — {threat.detail}")
else:
    completion = model.generate(user_text)
    safe = ctx.filter_output(session_id, ctx.new_request_id(), completion)
    print(safe)
```

### New CLI flags on `generate.py`

```
--audit-log PATH       Path for structured security audit log  [default: audit.log]
--token-budget INT     Per-session token budget for rate limiting  [default: 50000]
```

---

## CI/CD Security Job (`security-scan`)

Runs after the existing `build-linux` job on every push:

1. **Bandit** — static analysis for CWE/OWASP issues (shell injection, hardcoded passwords, unsafe deserialization, etc.)
2. **pip-audit** — dependency vulnerability scan against OSV/PyPA advisories; uploads JSON report as artifact
3. **detect-secrets** — scans all files for high-confidence hardcoded credentials; fails the build if any are found
4. **Security unit tests** — runs `inference/tests/test_security.py` (55 tests)

---

## What Is Not Yet Done / Recommended Next Steps

| Priority | Item |
|----------|------|
| High | **Semantic injection detection** — current guards are regex-only. Integrate a lightweight classifier (e.g. `deberta-v3-base` fine-tuned on injection prompts, or NVIDIA Garak) for semantic/paraphrase attacks that evade regex. |
| High | **Adversarial input red-teaming** — run Garak or PyRIT against the deployed endpoint to discover bypass vectors before adversaries do. |
| Medium | **Distributed-mode audit sync** — `AuditLogger` writes per-rank; in multi-GPU runs ranks > 0 are silenced. Consider funnelling all security events to rank 0 via a small dist.broadcast before logging. |
| Medium | **Output toxicity classifier** — add an optional Perspective API / local Detoxify pass in `OutputGuard` to catch harmful text that slips through without containing PII. |
| Medium | **Structured system prompt isolation** — prepend a tamper-evident system-turn header (nonce + digest) so prompt-injection attempts against the system context can be detected at decode time. |
| Low | **Key rotation for audit log integrity** — append an HMAC chain to each audit record so log-tampering is detectable. |
| Low | **EU AI Act compliance mapping** — enforcement of high-risk obligations begins 2026-08-02; map the security controls to the required transparency and human-oversight documentation. |

---

## Test Coverage

```
55 tests, 0 failures, 0 errors
Covers: InputGuard (injection, jailbreak, length, unicode)
        OutputGuard (PII types, truncation, opt-out)
        RateLimiter (budget, sessions, window reset)
        SecureLoader (path traversal, config schema, bounds, dangerous keys)
        AuditLogger (request/response records, input hashing)
        SecurityContext (end-to-end happy path + all block scenarios)
```

---

## Key Sources

- [OWASP LLM Top-10 2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [NIST CAISI DeepSeek Evaluation](https://www.nist.gov/news-events/news/2025/09/caisi-evaluation-deepseek-ai-models-finds-shortcomings-and-risks)
- [Cisco: Security Risk in DeepSeek](https://blogs.cisco.com/security/evaluating-security-risk-in-deepseek-and-other-frontier-reasoning-models)
- [HiddenLayer: DeepSeek-R1 Risks](https://hiddenlayer.com/innovation-hub/deepsht-exposing-the-security-risks-of-deepseek-r1)
- [Qualys: DeepSeek Jailbreak Analysis](https://blog.qualys.com/vulnerabilities-threat-research/2025/01/31/deepseek-failed-over-half-of-the-jailbreak-tests-by-qualys-totalai)
- [MITRE ATLAS ML Attack Techniques](https://atlas.mitre.org/)
