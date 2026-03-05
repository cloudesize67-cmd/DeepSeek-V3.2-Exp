import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from security import (
    enforce_https,
    validate_path,
    validate_env_int,
    sanitize_prompt,
    check_file_size,
    MAX_PROMPT_CHARS,
)


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    if max(prompt_lens) > model.max_seq_len:
        raise ValueError(
            f"Prompt length {max(prompt_lens)} exceeds model maximum sequence "
            f"length (max_seq_len={model.max_seq_len})"
        )
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    # Enforce HTTPS / TLS for all outbound network connections
    enforce_https()

    # Validate all file-system paths before any I/O
    ckpt_dir = validate_path(ckpt_path, must_exist=True, description="--ckpt-path")
    config_file = validate_path(config, must_exist=True, description="--config")
    if input_file:
        input_path = validate_path(input_file, must_exist=True, description="--input-file")
        check_file_size(input_path)

    # Safe environment variable parsing (raises ValueError on bad input)
    world_size = validate_env_int("WORLD_SIZE", default=1, min_val=1)
    rank = validate_env_int("RANK", default=0, min_val=0)
    local_rank = validate_env_int("LOCAL_RANK", default=0, min_val=0)

    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config_file) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    print("load model")
    load_model(model, os.path.join(str(ckpt_dir), f"model{rank}-mp{world_size}.safetensors"))
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                raw_prompt = input(">>> ")
            elif rank == 0:
                raw_prompt = input(">>> ")
                objects = [raw_prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                raw_prompt = objects[0]
            if raw_prompt == "/exit":
                break
            elif raw_prompt == "/clear":
                messages.clear()
                continue
            # Sanitize prompt: strip control characters, enforce length limit
            try:
                prompt = sanitize_prompt(raw_prompt, max_chars=MAX_PROMPT_CHARS)
            except ValueError as exc:
                print(f"[Warning] {exc}")
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_path) as f:
            prompts = f.read().split("\n\n")
        if len(prompts) > args.max_batch_size:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) exceeds maximum batch size "
                f"({args.max_batch_size})"
            )
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        ValueError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    if not args.input_file and not args.interactive:
        raise ValueError("Either --input-file or --interactive must be specified")
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
