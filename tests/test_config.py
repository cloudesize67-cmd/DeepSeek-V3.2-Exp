"""Tests for model configuration and ModelArgs dataclass."""
import json
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_DIR = REPO_ROOT / "inference"


try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _import_model_args():
    """Import ModelArgs with kernel dependency mocked out (requires torch)."""
    if "kernel" not in sys.modules:
        m = types.ModuleType("kernel")
        m.act_quant = lambda x, s=None: (x, s)
        m.fp8_gemm = lambda x, s, w, ws: x
        m.fp8_index = lambda q, k, idx: q
        sys.modules["kernel"] = m
    if str(INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFERENCE_DIR))
    if "model" in sys.modules:
        return sys.modules["model"].ModelArgs
    import model
    return model.ModelArgs


class TestConfigJson(unittest.TestCase):
    """Tests for inference/config_671B_v3.2.json."""

    CONFIG_PATH = INFERENCE_DIR / "config_671B_v3.2.json"

    def test_config_file_exists(self):
        self.assertTrue(self.CONFIG_PATH.exists(), "config_671B_v3.2.json not found")

    def test_config_is_valid_json(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_config_required_fields_present(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        required = [
            "vocab_size", "dim", "n_layers", "n_heads",
            "n_routed_experts", "n_activated_experts",
        ]
        for field in required:
            self.assertIn(field, data, f"Missing required field: {field}")

    def test_config_integer_fields_are_positive(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        int_fields = [
            "vocab_size", "dim", "n_layers", "n_heads",
            "n_routed_experts", "n_activated_experts", "n_shared_experts",
        ]
        for field in int_fields:
            self.assertGreater(data[field], 0, f"{field} should be positive")

    def test_config_671b_values(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        self.assertEqual(data["vocab_size"], 129280)
        self.assertEqual(data["dim"], 7168)
        self.assertEqual(data["n_layers"], 61)
        self.assertEqual(data["n_heads"], 128)
        self.assertEqual(data["n_routed_experts"], 256)
        self.assertEqual(data["n_activated_experts"], 8)
        self.assertEqual(data["dtype"], "fp8")
        self.assertEqual(data["scale_fmt"], "ue8m0")

    def test_activated_experts_leq_routed_experts(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        self.assertLessEqual(
            data["n_activated_experts"],
            data["n_routed_experts"],
            "n_activated_experts must not exceed n_routed_experts",
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
class TestModelArgs(unittest.TestCase):
    """Tests for ModelArgs dataclass."""

    def test_default_instantiation(self):
        ModelArgs = _import_model_args()
        args = ModelArgs()
        self.assertEqual(args.vocab_size, 102400)
        self.assertEqual(args.n_layers, 27)
        self.assertEqual(args.dtype, "bf16")
        self.assertIsNone(args.scale_fmt)

    def test_load_from_config_json(self):
        ModelArgs = _import_model_args()
        with open(INFERENCE_DIR / "config_671B_v3.2.json") as f:
            config = json.load(f)
        args = ModelArgs(**config)
        self.assertEqual(args.vocab_size, 129280)
        self.assertEqual(args.dim, 7168)
        self.assertEqual(args.n_layers, 61)
        self.assertEqual(args.n_routed_experts, 256)
        self.assertEqual(args.dtype, "fp8")
        self.assertEqual(args.scale_fmt, "ue8m0")

    def test_n_activated_leq_n_routed(self):
        ModelArgs = _import_model_args()
        with open(INFERENCE_DIR / "config_671B_v3.2.json") as f:
            config = json.load(f)
        args = ModelArgs(**config)
        self.assertLessEqual(
            args.n_activated_experts,
            args.n_routed_experts,
            "Activated experts must not exceed total routed experts",
        )

    def test_n_dense_layers_lt_n_layers(self):
        ModelArgs = _import_model_args()
        args = ModelArgs()
        self.assertLess(args.n_dense_layers, args.n_layers)

    def test_dtype_values(self):
        ModelArgs = _import_model_args()
        valid_dtypes = {"bf16", "fp8"}
        args_bf16 = ModelArgs(dtype="bf16")
        args_fp8 = ModelArgs(dtype="fp8")
        self.assertIn(args_bf16.dtype, valid_dtypes)
        self.assertIn(args_fp8.dtype, valid_dtypes)

    def test_score_func_values(self):
        ModelArgs = _import_model_args()
        valid_funcs = {"softmax", "sigmoid"}
        args = ModelArgs()
        self.assertIn(args.score_func, valid_funcs)

    def test_qk_dimensions_positive(self):
        ModelArgs = _import_model_args()
        args = ModelArgs()
        self.assertGreater(args.qk_nope_head_dim, 0)
        self.assertGreater(args.qk_rope_head_dim, 0)
        self.assertGreater(args.v_head_dim, 0)

    def test_max_seq_len_positive(self):
        ModelArgs = _import_model_args()
        args = ModelArgs()
        self.assertGreater(args.max_seq_len, 0)
        self.assertGreater(args.max_batch_size, 0)


if __name__ == "__main__":
    unittest.main()
