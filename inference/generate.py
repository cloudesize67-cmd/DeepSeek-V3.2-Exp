import os
import json
import time
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from security import SecurityContext


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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
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
    audit_log: str = "audit.log",
    token_budget: int = 50_000,
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
        audit_log (str, optional): Path for the structured security audit log. Defaults to "audit.log".
        token_budget (int, optional): Per-session token budget for rate limiting. Defaults to 50000.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    # ── Security: validate paths and config before loading ───────────────────
    sec = SecurityContext(
        audit_log_path=audit_log,
        token_budget=token_budget,
        log_to_stderr=(rank == 0),
    )
    ckpt_path = sec.safe_path(ckpt_path)
    config    = sec.safe_path(config)

    with open(config) as f:
        config_dict = json.load(f)
    sec.validate_config(config_dict)          # raises ConfigValidationError on bad values
    args = ModelArgs(**config_dict)
    # ────────────────────────────────────────────────────────────────────────

    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    print("I'm DeepSeek 👋")

    if interactive:
        session_id = sec.new_session()
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]

            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            # ── Security: input inspection ───────────────────────────────
            estimated_tokens = len(prompt) // 4  # ~4 chars per token heuristic
            threat = sec.check_input(session_id, prompt, estimated_tokens)
            if threat.blocked:
                print(f"[BLOCKED] {threat.threat_type}: {threat.detail}")
                continue
            # ─────────────────────────────────────────────────────────────

            messages.append({"role": "user", "content": prompt})
            # Guard context-window history growth
            if len(messages) > 2 * 50:  # 50 turns = 100 message objects
                messages = messages[-100:]

            request_id = sec.new_request_id()
            t0 = time.monotonic()
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            latency_ms = (time.monotonic() - t0) * 1000

            # ── Security: output filtering ───────────────────────────────
            safe_completion = sec.filter_output(session_id, request_id, completion, latency_ms)
            # ─────────────────────────────────────────────────────────────

            print(safe_completion)
            messages.append({"role": "assistant", "content": safe_completion})
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"

        session_id = sec.new_session()
        safe_prompts: List[str] = []
        for prompt in prompts:
            threat = sec.check_input(session_id, prompt, len(prompt) // 4)
            if threat.blocked:
                print(f"[SKIPPED] {threat.threat_type}: {threat.detail}")
                safe_prompts.append("")  # placeholder so indices align
            else:
                safe_prompts.append(prompt)

        prompt_tokens = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True
            ) for p in safe_prompts
        ]
        t0 = time.monotonic()
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        latency_ms = (time.monotonic() - t0) * 1000
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        for prompt, completion in zip(safe_prompts, completions):
            request_id = sec.new_request_id()
            safe_completion = sec.filter_output(session_id, request_id, completion, latency_ms)
            print("Prompt:", prompt)
            print("Completion:", safe_completion)
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
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--audit-log", type=str, default="audit.log",
                        help="Path for structured security audit log")
    parser.add_argument("--token-budget", type=int, default=50_000,
                        help="Per-session token budget for rate limiting")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive,
         args.max_new_tokens, args.temperature, args.audit_log, args.token_budget)
