import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

from security import validate_path, validate_positive_int, validate_divisible


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "wq_b": ("wq_b", None),
    "wk": ("wk", None),
    "k_norm": ("k_norm", None),
    "weights_proj": ("weights_proj", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    # Validate numeric arguments before any I/O
    validate_positive_int(n_experts, "n_experts")
    validate_positive_int(mp, "model_parallel")
    validate_divisible(n_experts, mp, "n_experts", "model_parallel")

    # Validate filesystem paths
    src_dir = validate_path(hf_ckpt_path, must_exist=True, description="--hf-ckpt-path")
    dst_dir = validate_path(save_path, description="--save-path")

    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(str(src_dir), "*.safetensors"))):
        # Validate each discovered file is still inside src_dir
        validate_path(file_path, must_exist=True, allowed_base=str(src_dir), description="safetensors file")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                parts = name.split(".")
                if len(parts) < 2:
                    raise ValueError(
                        f"Unexpected tensor name format (too few components): {name!r}"
                    )
                key = parts[-2]
                if key not in mapping:
                    raise ValueError(f"Key {key!r} not found in mapping for tensor {name!r}")
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        expert_parts = name.split(".")
                        if len(expert_parts) < 3:
                            raise ValueError(
                                f"Cannot parse expert index from tensor name: {name!r}"
                            )
                        try:
                            idx = int(expert_parts[-3])
                        except ValueError:
                            raise ValueError(
                                f"Non-integer expert index in tensor name: {name!r}"
                            )
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        if param.size(dim) % mp != 0:
                            raise ValueError(
                                f"Dimension {dim} of tensor {name!r} "
                                f"(size {param.size(dim)}) must be divisible by "
                                f"model_parallel ({mp})"
                            )
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(str(dst_dir), exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(str(dst_dir), f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(str(src_dir), "*token*")):
        # Ensure tokenizer files are inside src_dir before copying
        validate_path(file_path, must_exist=True, allowed_base=str(src_dir), description="tokenizer file")
        new_file_path = os.path.join(str(dst_dir), os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
