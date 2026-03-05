"""Tests for inference/security.py hardening utilities."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_DIR = REPO_ROOT / "inference"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from security import (  # noqa: E402
    validate_path,
    validate_env_int,
    sanitize_prompt,
    check_file_size,
    validate_positive_int,
    validate_divisible,
    enforce_https,
    MAX_PROMPT_CHARS,
    MAX_INPUT_FILE_BYTES,
)


class TestValidatePath(unittest.TestCase):
    """Tests for validate_path() — directory-traversal prevention."""

    def test_returns_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_path(tmpdir, must_exist=True)
            self.assertIsInstance(result, Path)
            self.assertTrue(result.is_absolute())

    def test_must_exist_raises_on_missing(self):
        with self.assertRaises(FileNotFoundError):
            validate_path("/nonexistent/path/abc123", must_exist=True)

    def test_must_exist_false_allows_missing(self):
        # Should not raise even if path doesn't exist
        result = validate_path("/tmp/does_not_exist_xyz", must_exist=False)
        self.assertIsInstance(result, Path)

    def test_allowed_base_accepts_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            child = os.path.join(tmpdir, "subdir")
            os.makedirs(child)
            result = validate_path(child, must_exist=True, allowed_base=tmpdir)
            self.assertTrue(str(result).startswith(tmpdir))

    def test_allowed_base_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            traversal = os.path.join(tmpdir, "..", "..", "etc", "passwd")
            with self.assertRaises(ValueError):
                validate_path(traversal, allowed_base=tmpdir)

    def test_allowed_base_rejects_absolute_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_path("/etc/passwd", allowed_base=tmpdir)

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            validate_path("")

    def test_whitespace_path_raises(self):
        with self.assertRaises(ValueError):
            validate_path("   ")

    def test_allowed_base_equals_path(self):
        """The path itself (not just its children) should be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_path(tmpdir, must_exist=True, allowed_base=tmpdir)
            self.assertEqual(result, Path(tmpdir).resolve())


class TestValidateEnvInt(unittest.TestCase):
    """Tests for validate_env_int() — safe env-var parsing."""

    def setUp(self):
        # Clean up any env vars we might touch
        for var in ("_TEST_INT_VAR",):
            os.environ.pop(var, None)

    def tearDown(self):
        os.environ.pop("_TEST_INT_VAR", None)

    def test_returns_default_when_unset(self):
        self.assertEqual(validate_env_int("_TEST_INT_VAR", default=5), 5)

    def test_parses_valid_integer(self):
        os.environ["_TEST_INT_VAR"] = "42"
        self.assertEqual(validate_env_int("_TEST_INT_VAR", default=1), 42)

    def test_raises_on_non_integer(self):
        os.environ["_TEST_INT_VAR"] = "not_a_number"
        with self.assertRaises(ValueError):
            validate_env_int("_TEST_INT_VAR", default=1)

    def test_raises_on_float_string(self):
        os.environ["_TEST_INT_VAR"] = "3.14"
        with self.assertRaises(ValueError):
            validate_env_int("_TEST_INT_VAR", default=1)

    def test_raises_below_min_val(self):
        os.environ["_TEST_INT_VAR"] = "0"
        with self.assertRaises(ValueError):
            validate_env_int("_TEST_INT_VAR", default=1, min_val=1)

    def test_accepts_min_val_boundary(self):
        os.environ["_TEST_INT_VAR"] = "1"
        self.assertEqual(validate_env_int("_TEST_INT_VAR", default=0, min_val=1), 1)

    def test_raises_on_empty_string(self):
        os.environ["_TEST_INT_VAR"] = ""
        with self.assertRaises(ValueError):
            validate_env_int("_TEST_INT_VAR", default=1)


class TestSanitizePrompt(unittest.TestCase):
    """Tests for sanitize_prompt() — input sanitization."""

    def test_normal_text_unchanged(self):
        text = "Hello, how are you?"
        self.assertEqual(sanitize_prompt(text), text)

    def test_strips_null_bytes(self):
        result = sanitize_prompt("hello\x00world")
        self.assertNotIn("\x00", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_strips_control_characters(self):
        # BEL (0x07), BS (0x08), ESC (0x1b) should be removed
        result = sanitize_prompt("hello\x07\x08\x1bworld")
        self.assertEqual(result, "helloworld")

    def test_preserves_newlines_and_tabs(self):
        text = "line one\nline two\ttabbed"
        result = sanitize_prompt(text)
        self.assertIn("\n", result)
        self.assertIn("\t", result)

    def test_truncates_at_max_chars(self):
        long_input = "a" * (MAX_PROMPT_CHARS + 100)
        result = sanitize_prompt(long_input)
        self.assertEqual(len(result), MAX_PROMPT_CHARS)

    def test_custom_max_chars(self):
        result = sanitize_prompt("hello world", max_chars=5)
        self.assertEqual(result, "hello")

    def test_empty_after_sanitize_raises(self):
        with self.assertRaises(ValueError):
            sanitize_prompt("\x00\x01\x02\x03")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            sanitize_prompt("   \t\n  ")

    def test_unicode_text_preserved(self):
        text = "こんにちは 안녕하세요 مرحبا"
        result = sanitize_prompt(text)
        self.assertEqual(result, text)


class TestCheckFileSize(unittest.TestCase):
    """Tests for check_file_size() — OOM prevention."""

    def test_small_file_passes(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            fname = f.name
        try:
            check_file_size(Path(fname), max_bytes=MAX_INPUT_FILE_BYTES)
        finally:
            os.unlink(fname)

    def test_oversized_file_raises(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 100)
            fname = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                check_file_size(Path(fname), max_bytes=50)
            self.assertIn("50", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_exact_limit_passes(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 50)
            fname = f.name
        try:
            check_file_size(Path(fname), max_bytes=50)
        finally:
            os.unlink(fname)


class TestValidatePositiveInt(unittest.TestCase):
    """Tests for validate_positive_int()."""

    def test_positive_value_passes(self):
        self.assertEqual(validate_positive_int(1, "x"), 1)
        self.assertEqual(validate_positive_int(256, "n_experts"), 256)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_int(0, "mp")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_int(-1, "mp")

    def test_bool_raises(self):
        with self.assertRaises(TypeError):
            validate_positive_int(True, "mp")

    def test_float_raises(self):
        with self.assertRaises(TypeError):
            validate_positive_int(1.0, "mp")  # type: ignore[arg-type]


class TestValidateDivisible(unittest.TestCase):
    """Tests for validate_divisible()."""

    def test_evenly_divisible_passes(self):
        validate_divisible(256, 8, "n_experts", "mp")  # no exception

    def test_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            validate_divisible(257, 8, "n_experts", "mp")

    def test_zero_divisor_raises(self):
        with self.assertRaises(ValueError):
            validate_divisible(256, 0, "n_experts", "mp")


class TestEnforceHttps(unittest.TestCase):
    """Tests for enforce_https() — TLS / HTTPS configuration."""

    def test_runs_without_error(self):
        """enforce_https() should not raise on a standard Python install."""
        enforce_https()  # no exception

    def test_sets_hf_telemetry_env(self):
        os.environ.pop("HF_HUB_DISABLE_TELEMETRY", None)
        enforce_https()
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "1")

    def test_sets_hf_implicit_token_env(self):
        os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        enforce_https()
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"), "1")

    def test_ca_bundle_env_set_when_bundle_exists(self):
        """If a CA bundle is found, REQUESTS_CA_BUNDLE should be set."""
        enforce_https()
        # If a CA bundle was located, env var must point to a real file
        ca = os.environ.get("REQUESTS_CA_BUNDLE")
        if ca:
            self.assertTrue(os.path.isfile(ca), f"CA bundle path is not a file: {ca}")


if __name__ == "__main__":
    unittest.main()
