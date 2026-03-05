# CLAUDE.md — DeepSeek-V3.2-Exp

This file provides guidance for AI assistants (Claude and others) working on this repository.

---

## Project Overview

**DeepSeek-V3.2-Exp** is an experimental inference implementation of the DeepSeek-V3.2 large language model — a 671-billion parameter mixture-of-experts (MoE) transformer. The core innovation is **DeepSeek Sparse Attention (DSA)**, a fine-grained sparse attention mechanism that reduces computation in long-context scenarios while maintaining output quality comparable to V3.1-Terminus.

This repository contains:
- A custom distributed inference engine (pure PyTorch + CUDA)
- TileLang CUDA kernels for FP8 quantization and sparse attention indexing
- Tooling to convert HuggingFace checkpoints to the distributed inference format

---

## Repository Structure

```
DeepSeek-V3.2-Exp/
├── inference/
│   ├── model.py              # Core transformer architecture (922 lines)
│   ├── generate.py           # Inference entry point (interactive + batch)
│   ├── convert.py            # HuggingFace → distributed checkpoint converter
│   ├── kernel.py             # Custom TileLang CUDA kernels
│   ├── config_671B_v3.2.json # Model hyperparameter configuration
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Inference quick-start guide
├── .github/
│   └── workflows/
│       └── python-package-conda.yml  # CI: lint (flake8) + test (pytest)
├── README.md                 # Project overview, benchmarks, deployment guides
├── DeepSeek_V3_2.pdf         # Technical paper on DSA
└── cost.jpg                  # Cost comparison chart
```

---

## Key Components

### `inference/model.py` — Core Architecture

The main model implementation. Key classes:

| Class | Role |
|---|---|
| `ModelArgs` | Dataclass of all model hyperparameters (loaded from JSON config) |
| `Transformer` | Full model: embedding → N blocks → RMSNorm → output projection |
| `Block` | Single transformer layer: MLA attention + MoE/MLP feed-forward |
| `MLA` | Multi-head Latent Attention with sparse indexing via `Indexer` |
| `Indexer` | Computes top-k sparse attention indices (the DSA mechanism) |
| `MoE` | Mixture-of-Experts layer with `Gate` routing |
| `Expert` | Individual expert feed-forward network |
| `MLP` | Dense feed-forward (used in first 3 layers) |
| `ParallelEmbedding` | Vocabulary embedding sharded across TP ranks |
| `ColumnParallelLinear` / `RowParallelLinear` | Tensor-parallel linear layers |

Module-level globals control distributed state:
```python
world_size: int   # total number of TP processes
rank: int         # this process's rank
block_size: int   # sparse attention block size
```

### `inference/generate.py` — Inference Entry Point

Handles model loading, KV cache management, and generation loops.

- **Interactive mode:** `--interactive` flag → REPL chat session
- **Batch mode:** reads prompts from stdin / file
- Uses `torchrun` for multi-GPU launch

### `inference/convert.py` — Checkpoint Conversion

Converts HuggingFace safetensors (sharded) to the distributed per-rank format expected by `generate.py`.

### `inference/kernel.py` — Custom CUDA Kernels

TileLang kernels for:
- FP8 (float8_e4m3fn) weight quantization with block-wise scaling
- Sparse attention index computation

---

## Development Workflows

### Environment Setup

```bash
pip install -r inference/requirements.txt
# Key packages: torch, transformers, safetensors,
#               fast_hadamard_transform, tilelang==0.1.6
```

### Model Conversion (HuggingFace → Distributed Format)

```bash
python inference/convert.py \
  --hf-ckpt-path ${HF_CKPT_PATH} \
  --save-path ${SAVE_PATH} \
  --n-experts 256 \
  --model-parallel ${MP}
```

### Running Inference

```bash
# Interactive chat
torchrun --nproc-per-node ${MP} inference/generate.py \
  --ckpt-path ${SAVE_PATH} \
  --config inference/config_671B_v3.2.json \
  --interactive

# Batch generation
torchrun --nproc-per-node ${MP} inference/generate.py \
  --ckpt-path ${SAVE_PATH} \
  --config inference/config_671B_v3.2.json
```

### Alternative Deployment (no conversion needed)

- **SGLang:** See README.md for Docker-based deployment
- **vLLM:** See README.md for vLLM integration

---

## CI/CD

**GitHub Actions** (`.github/workflows/python-package-conda.yml`):
- Triggered on every push
- Python 3.10, Ubuntu latest, Conda environment
- Steps:
  1. Lint with `flake8` — enforces: E9xx, F63x, F7xx, F82x error classes
  2. Test with `pytest`

Flake8 linting rules to follow when writing code:
- No syntax errors (E9)
- No undefined names (F821, F822)
- No undefined `__all__` exports (F823)
- No invalid escape sequences (F63x)
- No issues with `__future__` imports (F7xx)

---

## Code Conventions

### Naming
- **Classes:** `PascalCase` — `Transformer`, `MoE`, `RMSNorm`
- **Functions/methods:** `snake_case` — `apply_rotary_emb`, `precompute_freqs_cis`
- **Private attributes:** leading underscore — `_norm`, `_scale`
- **Config fields:** `snake_case` in `ModelArgs` dataclass

### Imports
Grouped in order: stdlib → torch/transformers → local modules

### Type Hints
Used throughout — `Optional`, `Literal`, `Tuple`, `List` from `typing`

### Docstrings
Class and method docstrings are present; maintain this style when adding new code.

### Distributed Computing Patterns
- Always guard rank-0-only operations: `if rank == 0:`
- Use `dist.all_reduce` / `dist.broadcast` for synchronization
- Tensor parallel layers use `ColumnParallelLinear` (splits output dim) and `RowParallelLinear` (splits input dim)

### FP8 Quantization
- Weights stored as `float8_e4m3fn`
- Block-wise scaling factors stored alongside weights
- Scaling uses `"ue8m0"` format (unsigned exponent 8-bit, mantissa 0-bit)

---

## Model Configuration (`config_671B_v3.2.json`)

Key hyperparameters:

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 129,280 | Vocabulary tokens |
| `dim` | 7,168 | Model hidden dimension |
| `n_layers` | 61 | Total transformer layers |
| `n_dense_layers` | 3 | First 3 layers use dense MLP (not MoE) |
| `n_heads` | 128 | Attention heads |
| `n_routed_experts` | 256 | Total MoE experts |
| `n_activated_experts` | 8 | Experts activated per token |
| `n_shared_experts` | 1 | Always-active shared expert |
| `topk_idx` | 2,048 | Sparse attention top-k positions |

---

## Architecture Notes for AI Assistants

1. **Sparse Attention (DSA):** The `Indexer` class selects the top-k most relevant positions for each query head, reducing attention computation from O(n²) to O(n·k). When modifying attention code, ensure the indexing logic is consistent with the block structure.

2. **MoE Routing:** The `Gate` module routes tokens to experts using a softmax over expert logits. Load balancing is implicit — do not modify the routing without understanding expert utilization.

3. **YARN Positional Embeddings:** Long-context support uses YARN scaling of RoPE frequencies. Parameters `rope_scaling_factor`, `original_seq_len`, `beta_fast`, `beta_slow`, `mscale` in `ModelArgs` control this.

4. **Distributed Execution:** All tensor shapes in comments/docstrings reflect per-rank shapes. Global shapes are `world_size` times larger on the TP-split dimension.

5. **Checkpoint Format:** After `convert.py`, each rank loads its own shard from `${SAVE_PATH}/rank_${rank}/`. Expert weights are stored separately from non-expert weights.

---

## What This Repository Does NOT Include

- Training code (inference-only)
- Tokenizer implementation (uses HuggingFace `transformers` tokenizer)
- Dedicated test suite (CI references pytest, but no tests exist yet)
- Evaluation harnesses

---

## Branch Convention

Development branches follow: `claude/<feature-description>-<session-id>`

---

## External Resources

- [TileLang](https://github.com/tile-ai/tilelang) — CUDA kernel DSL used in `kernel.py`
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — FP8 GEMM kernels (referenced in README)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — Optimized MLA attention (referenced in README)
- Technical paper: `DeepSeek_V3_2.pdf` in this repo
