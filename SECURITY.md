# Security Policy

## Supported Versions

This repository is an **experimental inference implementation** of DeepSeek-V3.2.
Security fixes are applied to the `main` branch only.

## Reporting a Vulnerability

Please **do not** file a public GitHub issue for security vulnerabilities.

Report vulnerabilities by emailing the maintainer directly (see the repository
contact). You should receive a response within 72 hours. If the issue is
confirmed, a patch will be prepared and the vulnerability will be disclosed
publicly after a fix is available.

---

## Security Measures in This Codebase

### 1. Path Traversal Prevention (`inference/security.py`)

All CLI-supplied file paths pass through `validate_path()` before any file I/O:

- Paths are resolved to absolute form with `Path.resolve()`.
- When `allowed_base` is specified, the resolved path must be a descendant of
  that directory. Any `../` traversal that escapes the base raises `ValueError`.
- Empty and whitespace-only paths are rejected.

**Protected surfaces:** `--ckpt-path`, `--config`, `--input-file` in
`generate.py`; `--hf-ckpt-path`, `--save-path` in `convert.py`. Every file
path discovered by glob inside `convert.py` is also re-validated against its
source directory.

### 2. Input Sanitization (`inference/security.py`)

Interactive user prompts pass through `sanitize_prompt()`:

- Null bytes (`\x00`) and C0/C1 control characters (except `\n` and `\t`) are
  stripped to prevent injection attacks.
- A hard character limit (`MAX_PROMPT_CHARS`, default 32 768) prevents
  memory-exhaustion via extremely long inputs.

### 3. File Size Limits (`inference/security.py`)

Batch input files are checked with `check_file_size()` before being read into
memory. Files larger than `MAX_INPUT_FILE_BYTES` (default 10 MB) are rejected
to prevent out-of-memory denial-of-service attacks.

### 4. Safe Environment Variable Parsing (`inference/security.py`)

`WORLD_SIZE`, `RANK`, and `LOCAL_RANK` are parsed via `validate_env_int()`,
which raises a descriptive `ValueError` instead of crashing with an unhandled
`ValueError`/`TypeError` when the variable contains a non-integer value.

### 5. HTTPS / TLS Enforcement (`inference/security.py`)

`enforce_https()` is called at process start in `generate.py`:

- A TLS context is created and `minimum_version` is set to TLS 1.2 to ensure
  no legacy protocol downgrades occur.
- `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` are set to the system CA bundle so
  the `requests` library and libcurl verify server certificates.
- `HF_HUB_DISABLE_TELEMETRY=1` opts out of HuggingFace telemetry.
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` prevents unintended credential leakage
  from `~/.cache/huggingface/token`.

### 6. Dependency Version Pinning (`inference/requirements.txt`, `environment.yml`)

All Python dependencies specify minimum **and** maximum version bounds using
compatible-release constraints (e.g. `torch>=2.4.0,<3.0.0`). This eliminates
the supply-chain risk of silently installing a future major version that could
introduce breaking changes or malicious code.

`tilelang` is pinned to the exact version (`==0.1.6`) required for the custom
CUDA kernels in `kernel.py`.

### 7. Proper Exception Handling (replace `assert` with `ValueError`)

Production input validation uses `ValueError` / `TypeError` instead of
`assert` statements. Python assertions are disabled when the interpreter runs
with the `-O` flag, meaning assertion-based guards are silently bypassed in
optimised deployments.

### 8. Safe File Formats

Model weights are stored and loaded as **safetensors** (`.safetensors`), not as
Python pickle files. Safetensors cannot execute arbitrary code during
deserialization, unlike `torch.load()` without `weights_only=True`.

---

## What This Repository Does NOT Protect Against

- **Adversarial prompts / jailbreaks**: The inference engine is a research tool.
  It does not include a content-filtering layer.
- **Untrusted model weights**: Loading weights from an untrusted source is
  outside the scope of this implementation. Always verify checksums when
  distributing model files.
- **Multi-tenant deployments**: The code is designed for single-user or
  internal research use. Do not expose `generate.py` directly to the internet
  without an additional authentication and rate-limiting layer.
