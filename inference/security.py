"""
security.py — Hardening utilities for DeepSeek-V3.2-Exp inference.

Provides:
  - validate_path()        safe path resolution + traversal prevention
  - validate_env_int()     safe integer parsing of environment variables
  - sanitize_prompt()      strip null bytes, enforce max length
  - check_file_size()      prevent OOM from oversized input files
  - enforce_https()        configure requests / HuggingFace to use HTTPS + TLS verify
  - validate_positive_int()  guard against zero/negative numeric args
"""

import os
import ssl
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum bytes allowed for an input prompt file (default 10 MB)
MAX_INPUT_FILE_BYTES: int = 10 * 1024 * 1024

# Maximum characters for a single interactive prompt (default 32 KB)
MAX_PROMPT_CHARS: int = 32_768

# Minimum TLS version accepted for outbound HTTPS connections
_MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_path(
    path: str,
    *,
    must_exist: bool = False,
    allowed_base: Optional[str] = None,
    description: str = "path",
) -> Path:
    """Resolve *path* and guard against directory-traversal attacks.

    Args:
        path: The user-supplied or CLI-supplied file/directory path.
        must_exist: If True, raise FileNotFoundError when the resolved path
            does not exist on disk.
        allowed_base: If given, the resolved path must be an ancestor of
            (or equal to) this directory. Raises ValueError otherwise.
        description: Human-readable label used in error messages.

    Returns:
        pathlib.Path — the fully-resolved, safe path.

    Raises:
        ValueError: On traversal attempt or disallowed base.
        FileNotFoundError: When must_exist=True and the path is absent.
    """
    if not path or not path.strip():
        raise ValueError(f"Empty {description} is not allowed")

    resolved = Path(path).resolve()

    if allowed_base is not None:
        base = Path(allowed_base).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(
                f"{description} '{path}' resolves to '{resolved}' which is "
                f"outside the allowed directory '{base}'"
            )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"{description} '{path}' (resolved: '{resolved}') does not exist"
        )

    logger.debug("Validated %s: %s → %s", description, path, resolved)
    return resolved


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------

def validate_env_int(name: str, default: int, *, min_val: int = 0) -> int:
    """Return ``int(os.getenv(name, default))``, raising on bad values.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.
        min_val: Minimum acceptable value (inclusive).

    Returns:
        Parsed integer.

    Raises:
        ValueError: When the variable is set but not a valid integer, or
            the value is below *min_val*.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer"
        )
    if value < min_val:
        raise ValueError(
            f"Environment variable {name}={value} must be >= {min_val}"
        )
    return value


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

def sanitize_prompt(text: str, *, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Strip dangerous characters and enforce a maximum prompt length.

    Removes null bytes and other non-printable control characters that
    could be used for injection or cause downstream parsing issues.

    Args:
        text: Raw user input string.
        max_chars: Hard character limit; input is truncated with a warning
            if exceeded.

    Returns:
        Sanitised string.

    Raises:
        ValueError: When *text* is empty after sanitization.
    """
    # Remove null bytes and C0/C1 control codes except newline/tab
    cleaned = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or (ord(ch) >= 32 and ord(ch) != 127)
    )
    if len(cleaned) > max_chars:
        logger.warning(
            "Prompt truncated from %d to %d characters", len(cleaned), max_chars
        )
        cleaned = cleaned[:max_chars]
    if not cleaned.strip():
        raise ValueError("Prompt is empty after sanitization")
    return cleaned


# ---------------------------------------------------------------------------
# File-size guard
# ---------------------------------------------------------------------------

def check_file_size(path: Path, *, max_bytes: int = MAX_INPUT_FILE_BYTES) -> None:
    """Raise an error if *path* exceeds *max_bytes*.

    Prevents out-of-memory conditions from oversized input files being
    read into memory all at once.

    Args:
        path: Resolved file path.
        max_bytes: Maximum allowed file size in bytes.

    Raises:
        ValueError: When the file is larger than *max_bytes*.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Input file '{path}' is {size:,} bytes, "
            f"which exceeds the {max_bytes:,}-byte limit. "
            "Split the file into smaller batches."
        )


# ---------------------------------------------------------------------------
# HTTPS / TLS enforcement
# ---------------------------------------------------------------------------

def enforce_https() -> None:
    """Configure the process for secure outbound HTTPS connections.

    - Sets REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE so the *requests* library
      and libcurl use the system CA store.
    - Disables HuggingFace Hub telemetry.
    - Sets HF_HUB_DISABLE_IMPLICIT_TOKEN to prevent unintended token use.
    - Verifies that Python's ssl module supports a strong minimum TLS
      version (TLS 1.2+).
    - Raises RuntimeError if the current ssl build cannot satisfy the
      minimum version requirement.

    Should be called once at process start (before any network I/O).
    """
    # Verify TLS support
    ctx = ssl.create_default_context()
    try:
        ctx.minimum_version = _MIN_TLS_VERSION
    except AttributeError:
        raise RuntimeError(
            "Python ssl module does not support setting minimum_version. "
            "Upgrade to Python >= 3.7 with a modern OpenSSL."
        )

    # Point requests / curl at the system CA bundle
    ca_bundle = _find_ca_bundle()
    if ca_bundle:
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
        logger.debug("CA bundle set to %s", ca_bundle)
    else:
        logger.warning(
            "Could not locate system CA bundle; "
            "HTTPS certificate verification may be weakened."
        )

    # Opt out of HuggingFace telemetry and implicit credential leakage
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # Enforce HTTPS-only for HuggingFace Hub
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    logger.debug("HTTPS enforcement configured (TLS >= %s)", _MIN_TLS_VERSION.name)


def _find_ca_bundle() -> Optional[str]:
    """Return the path to the system CA certificate bundle, or None."""
    candidates = [
        "/etc/ssl/certs/ca-certificates.crt",   # Debian / Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",      # RHEL / CentOS
        "/etc/ssl/ca-bundle.pem",                 # OpenSUSE
        "/usr/share/ssl/certs/ca-bundle.crt",     # older RHEL
        "/usr/local/share/certs/ca-root-nss.crt", # FreeBSD
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Numeric parameter guards
# ---------------------------------------------------------------------------

def validate_positive_int(value: int, name: str) -> int:
    """Ensure *value* is a positive integer (> 0).

    Args:
        value: The integer to validate.
        name: Parameter name for the error message.

    Returns:
        The original value if valid.

    Raises:
        ValueError: When *value* is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def validate_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    """Raise ValueError when *value* is not evenly divisible by *divisor*.

    Args:
        value: The dividend.
        divisor: The divisor.
        value_name: Name of the dividend parameter (for error messages).
        divisor_name: Name of the divisor parameter (for error messages).

    Raises:
        ValueError: When value % divisor != 0.
    """
    validate_positive_int(divisor, divisor_name)
    if value % divisor != 0:
        raise ValueError(
            f"{value_name} ({value}) must be divisible by "
            f"{divisor_name} ({divisor})"
        )
