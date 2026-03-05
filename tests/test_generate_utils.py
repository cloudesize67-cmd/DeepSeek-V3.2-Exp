"""Tests for utility functions in generate.py."""
import subprocess
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_DIR = REPO_ROOT / "inference"


def _setup_mocks():
    """Mock heavy dependencies so generate.py can be imported without GPU/tilelang."""
    if "kernel" not in sys.modules:
        m = types.ModuleType("kernel")
        m.act_quant = lambda x, s=None: (x, s)
        m.fp8_gemm = lambda x, s, w, ws: x
        m.fp8_index = lambda q, k, idx: q
        sys.modules["kernel"] = m


def _import_generate():
    """Import generate module with dependencies mocked."""
    _setup_mocks()
    if str(INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFERENCE_DIR))
    if "generate" in sys.modules:
        return sys.modules["generate"]
    import generate
    return generate


class TestSampleFunction(unittest.TestCase):
    """Tests for the sample() token-sampling function in generate.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            cls.torch = None

    def setUp(self):
        if self.torch is None:
            self.skipTest("torch not available in this environment")

    def test_sample_output_shape(self):
        gen = _import_generate()
        torch = self.torch
        logits = torch.randn(3, 100)
        result = gen.sample(logits)
        self.assertEqual(result.shape, (3,))

    def test_sample_with_near_zero_temperature(self):
        """Near-zero temperature should produce near-argmax results."""
        gen = _import_generate()
        torch = self.torch
        logits = torch.zeros(1, 10)
        logits[0, 7] = 100.0
        result = gen.sample(logits, temperature=1e-8)
        self.assertEqual(result.item(), 7)

    def test_sample_indices_in_valid_range(self):
        """Sampled indices must be within [0, vocab_size)."""
        gen = _import_generate()
        torch = self.torch
        vocab_size = 200
        logits = torch.randn(4, vocab_size)
        result = gen.sample(logits)
        self.assertTrue((result >= 0).all().item())
        self.assertTrue((result < vocab_size).all().item())

    def test_sample_high_temperature(self):
        """High temperature should still produce valid indices."""
        gen = _import_generate()
        torch = self.torch
        logits = torch.randn(2, 50)
        result = gen.sample(logits, temperature=10.0)
        self.assertEqual(result.shape, (2,))

    def test_sample_default_temperature(self):
        """Default temperature=1.0 should work."""
        gen = _import_generate()
        torch = self.torch
        logits = torch.randn(5, 32000)
        result = gen.sample(logits)
        self.assertEqual(result.shape, (5,))


class TestFlake8CiRules(unittest.TestCase):
    """Verify all Python source files pass flake8 with the rules enforced by CI."""

    def test_flake8_e9_f63_f7_f82(self):
        """E9xx, F63x, F7xx, F82x errors must be zero — this is what CI checks."""
        result = subprocess.run(
            [
                sys.executable, "-m", "flake8",
                str(INFERENCE_DIR),
                "--select=E9,F63,F7,F82",
                "--count",
                "--show-source",
                "--statistics",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"flake8 CI check failed:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
