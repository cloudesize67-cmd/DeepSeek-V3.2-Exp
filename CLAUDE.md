# CLAUDE.md — DeepSeek-V3.2-Exp

This file documents the codebase structure, development workflows, and conventions for AI assistants working on this repository.

## Project Overview

DeepSeek-V3.2-Exp is an experimental 671B-parameter Mixture-of-Experts (MoE) transformer language model. Its primary contribution is **DeepSeek Sparse Attention (DSA)** — a fine-grained sparse attention mechanism that substantially reduces long-context training and inference compute while maintaining output quality on par with DeepSeek-V3.1-Terminus.

Key innovations:
- **DSA (DeepSeek Sparse Attention)**: A learned sparse index head selects the top-k most relevant tokens per layer, replacing full attention at decode time.
- **FP8 quantization (E4M3)**: Weights and activations are quantized to 8-bit floating point with block-wise scaling, dramatically reducing memory and compute.
- **Multi-Head Latent Attention (MLA)**: Low-rank projections for Q and KV compress the KV cache.
- **MoE routing**: 256 routed experts + 1 shared expert per MoE layer; 8 experts activated per token.

The `inference/` directory contains a standalone, research-oriented implementation intended to help the community understand the architecture. Production deployments should use SGLang, vLLM, or FlashMLA kernels.

---

## Repository Layout

```
DeepSeek-V3.2-Exp/
├── inference/
│   ├── model.py                 # Full transformer implementation (922 lines)
│   ├── kernel.py                # TileLang CUDA kernels (274 lines)
│   ├── generate.py              # Inference entrypoint: batch + interactive chat (186 lines)
│   ├── convert.py               # HuggingFace → distributed checkpoint converter (100 lines)
│   ├── config_671B_v3.2.json    # Model hyperparameters
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # Quick-start commands
├── .github/
│   └── workflows/
│       └── python-package-conda.yml  # CI: flake8 lint + pytest
├── README.md                    # Overview, benchmarks, deployment options
├── DeepSeek_V3_2.pdf            # Technical paper
├── LICENSE                      # MIT
└── cost.jpg                     # Efficiency comparison figure
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Framework | PyTorch (CUDA required) |
| Quantization | FP8 E4M3 (block-wise, 128-element blocks) |
| Activation precision | BFloat16 |
| Kernel DSL | TileLang `==0.1.6` |
| Distributed | `torch.distributed` / NCCL |
| Checkpoint format | SafeTensors |
| CI | GitHub Actions + pytest + flake8 |

**Python dependencies** (`inference/requirements.txt`):
```
torch
transformers
safetensors
fast_hadamard_transform
tilelang==0.1.6
```

---

## Model Configuration (`config_671B_v3.2.json`)

| Parameter | Value | Meaning |
|---|---|---|
| `vocab_size` | 129280 | Token vocabulary |
| `dim` | 7168 | Model hidden dimension |
| `inter_dim` | 18432 | Dense MLP intermediate dimension |
| `moe_inter_dim` | 2048 | MoE expert intermediate dimension |
| `n_layers` | 61 | Total transformer layers |
| `n_dense_layers` | 3 | First 3 layers use dense MLP (not MoE) |
| `n_heads` | 128 | Attention heads |
| `n_routed_experts` | 256 | Routed MoE experts |
| `n_shared_experts` | 1 | Always-active shared expert |
| `n_activated_experts` | 8 | Experts activated per token |
| `n_expert_groups` | 8 | Expert grouping for load balancing |
| `n_limited_groups` | 4 | Max groups from which experts are selected |
| `q_lora_rank` | 1536 | Low-rank dim for query projection (MLA) |
| `kv_lora_rank` | 512 | Low-rank dim for KV projection (MLA) |
| `qk_nope_head_dim` | 128 | Query/key head dim without positional encoding |
| `qk_rope_head_dim` | 64 | Query/key head dim with RoPE |
| `v_head_dim` | 128 | Value head dimension |
| `dtype` | `fp8` | Weight/activation quantization format |
| `scale_fmt` | `ue8m0` | Scaling format for FP8 |
| `index_n_heads` | 64 | Sparse attention index heads |
| `index_head_dim` | 128 | Index head dimension |
| `index_topk` | 2048 | Tokens selected by sparse attention |

Do not modify this config for experiments — create a new JSON file and pass it via `--config`.

---

## Source File Guide

### `inference/model.py` — Core Architecture

Key classes (in dependency order):

| Class | Purpose |
|---|---|
| `ModelArgs` | Dataclass of all hyperparameters; populated from the JSON config |
| `ParallelEmbedding` | Token embedding sharded across `model_parallel` GPUs |
| `Linear` | Base linear with FP8 quantization; wraps `act_quant` + `fp8_gemm` |
| `ColumnParallelLinear` | Linear with output sharded across GPUs (column-wise) |
| `RowParallelLinear` | Linear with input sharded across GPUs; all-reduces output |
| `RMSNorm` / `LayerNorm` | Normalization layers |
| `MLA` | Multi-Head Latent Attention; handles prefill (full) and decode (sparse) |
| `Indexer` | Separate attention head that produces sparse token indices |
| `MoE` | Mixture-of-Experts block: routes tokens → top-k experts |
| `Gate` | Expert routing logits + top-k selection with load balancing |
| `Block` | Single transformer layer: `MLA` + (`MoE` or dense `MLP`) |
| `Transformer` | Full model: embedding → N blocks → RMSNorm → output projection |

Key free functions:

| Function | Purpose |
|---|---|
| `act_quant(x, block_size)` | Quantizes a BF16 tensor to FP8 with per-block scaling |
| `weight_dequant(w, s, block_size)` | Dequantizes FP8 weights back to BF16 |
| `fp8_gemm(A, As, B, Bs)` | FP8 matrix multiply using TileLang kernel |
| `precompute_freqs_cis(...)` | Precomputes RoPE frequency tensors |
| `apply_rotary_emb(x, freqs_cis)` | Applies RoPE to query/key tensors |

### `inference/kernel.py` — TileLang CUDA Kernels

Three kernels, all defined with TileLang's research-oriented DSL:

| Kernel | Purpose |
|---|---|
| `act_quant_kernel` | Quantizes activations to FP8 E4M3 in 128-element blocks |
| `fp8_gemm_kernel` | Matrix multiply with FP8 inputs and BF16 output |
| `fp8_index_kernel` | Computes sparse attention logits for the indexer module |

Use TileLang for any new CUDA kernels added to this project — do not write raw `.cu` files.

### `inference/generate.py` — Inference Interface

- `sample(logits, temperature)` — temperature sampling with greedy fallback
- `generate(model, prompts, ...)` — prefill + decode loop, returns token sequences
- `main()` — CLI with `--interactive` flag for chat; uses `transformers` tokenizer and chat templates

Distributed: rank 0 orchestrates I/O; all ranks participate in forward passes via `torchrun`.

### `inference/convert.py` — Checkpoint Conversion

Converts HuggingFace SafeTensors shards to the model-parallel format expected by `generate.py`. Shards expert weights across `--model-parallel` GPUs. Run once before inference.

---

## Development Workflows

### 1. Environment Setup

```bash
pip install -r inference/requirements.txt
```

Requires CUDA-capable GPU(s). FP8 hardware acceleration requires Hopper (H100) or newer. Earlier Ampere GPUs can run in BF16 simulation mode but will be slower.

### 2. Convert a HuggingFace Checkpoint

```bash
cd inference
export HF_CKPT_PATH=/path/to/hf_model
export SAVE_PATH=/path/to/output
export MP=8  # set to your GPU count

python convert.py \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --save-path ${SAVE_PATH} \
    --n-experts 256 \
    --model-parallel ${MP}
```

### 3. Run Inference

```bash
cd inference
export MP=8

# Interactive chat
torchrun --nproc-per-node ${MP} generate.py \
    --ckpt-path ${SAVE_PATH} \
    --config config_671B_v3.2.json \
    --interactive

# Batch inference (edit prompts in generate.py main())
torchrun --nproc-per-node ${MP} generate.py \
    --ckpt-path ${SAVE_PATH} \
    --config config_671B_v3.2.json
```

### 4. Linting and Tests

The CI pipeline runs these two commands (see `.github/workflows/python-package-conda.yml`):

```bash
# Syntax and fatal errors only (strict)
flake8 inference/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Style warnings (non-blocking)
flake8 inference/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Tests
pytest
```

Run these before committing. The CI will fail on E9/F63/F7/F82 errors.

### 5. SGLang Deployment

```bash
# H200
docker pull lmsysorg/sglang:dsv32
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2-Exp \
    --tp 8 --dp 8 --enable-dp-attention
```

See README.md for ROCm (MI350) and NPU docker images.

### 6. vLLM Deployment

vLLM provides day-0 support. Refer to the official vLLM recipes documentation for DeepSeek-V3.2-Exp.

---

## Coding Conventions

| Aspect | Convention |
|---|---|
| Class names | PascalCase (`ColumnParallelLinear`, `ModelArgs`) |
| Function/variable names | snake_case (`act_quant`, `n_routed_experts`) |
| Type hints | Required on all function signatures |
| Docstrings | Required for all classes and public methods |
| Default activation dtype | `torch.bfloat16` — always be explicit with `dtype=` arguments |
| Comments | Only for non-obvious WHY (hidden constraints, workarounds); never describe what the code does |

---

## Critical Implementation Notes

### RoPE Layout in Indexer (Bug Fixed 2025-11-17)

**The indexer module (`Indexer`) and the MLA attention module use different RoPE tensor layouts:**

- `MLA`: expects **interleaved** layout for RoPE input
- `Indexer`: expects **non-interleaved** layout for RoPE input

Mixing these up causes silent degradation in sparse attention quality. This was a known bug in earlier versions that has been fixed in the current code. When modifying positional embedding code, preserve this distinction carefully.

### FP8 Quantization Boundaries

FP8 quantization (`act_quant`) is applied at linear layer boundaries inside `Linear.forward()`. Activations flowing between layers remain in BF16. Do not apply `act_quant` elsewhere unless you're adding a new quantizable linear layer.

### Distributed Parallelism

All modules must correctly handle `model_parallel` sharding:
- Use `ColumnParallelLinear` for weight matrices split along the output dimension
- Use `RowParallelLinear` for weight matrices split along the input dimension; this class handles the `all_reduce` automatically
- Never add bare `dist.all_reduce` calls outside of `RowParallelLinear` without strong justification

### Expert Distribution

In `MoE`, routed experts are sharded across GPUs. Each GPU holds `n_routed_experts // model_parallel` experts. The shared expert is replicated on all GPUs.

### TileLang Kernel Versioning

`tilelang==0.1.6` is pinned. Do not upgrade this dependency without validating all three kernels in `kernel.py` against the new API — TileLang's DSL is research software with breaking changes between versions.

---

## External Resources

- **Technical paper**: `DeepSeek_V3_2.pdf` (in repo root)
- **TileLang kernels** (research/readable): https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32
- **High-performance indexer kernels**: https://github.com/deepseek-ai/DeepGEMM/pull/200
- **High-performance sparse attention kernels**: https://github.com/deepseek-ai/FlashMLA/pull/98
- **HuggingFace model**: `deepseek-ai/DeepSeek-V3.2-Exp`

---

## License

MIT License — see `LICENSE`.
