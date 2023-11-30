"""A main script for running LocalLLMs"""
from __future__ import annotations

import os
import platform
import time
from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# MODEL = "HuggingFaceH4/zephyr-7b-beta"  # ~2.5 tok/s
# MODEL = "meta-llama/Llama-2-7b-chat-hf"  # ~15.8 tok/s
# MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # ~2.4 tok/s
# MODEL = "TheBloke/zephyr-7B-beta-AWQ"
# MODEL = "TheBloke/Llama-2-7B-Chat-AWQ"

MODEL = "Qwen/Qwen-1_8B-Chat"


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers with concise answers, explaining your reasoning step by step"
PIRATE_SYSTEM_PROMPT = "You are a pirate chatbot who always responds with Arr!"


def is_macos() -> bool:
    """Returns whether the current OS is macOS"""
    return platform.system() == "Darwin"


@contextmanager
def timer(name: str = ""):
    """A context manager for timing code execution"""
    import time

    start = time.perf_counter_ns()
    yield
    elapsed = time.perf_counter_ns() - start

    # Be smart about the units of time
    if elapsed < 1e6:
        unit = "ns"
        actual_elapsed = float(elapsed)
    elif elapsed < 500e6:
        unit = "ms"
        actual_elapsed = elapsed / 1e6
    else:
        unit = "s"
        actual_elapsed = elapsed / 1e9

    if name:
        print(f"[{name}] ", end="")
    print(f"elapsed {actual_elapsed:.3f}{unit}")


def parse_model_with_revision(model_id: str) -> tuple[str, Optional[str]]:
    """Parse a model ID with an optional revision"""
    if ":" in model_id:
        model_id, _, revision = model_id.partition(":")
    else:
        revision = None
    return model_id, revision


def supports_system(model_id: str) -> bool:
    """Returns whether the model supports system prompts"""
    DOES_NOT_SUPPORT_SYSTEM = ["mistralai/Mistral-7B-Instruct-v0.1"]
    return model_id not in DOES_NOT_SUPPORT_SYSTEM


def prompt(
    tokenizer: AutoTokenizer,
    message: str,
    system: Optional[str] = None,
    supports_system: bool = True,
):
    """Generate a prompt for the model"""
    system = system or DEFAULT_SYSTEM_PROMPT
    if supports_system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
    else:
        # The model does not support a system prompt, so we must manually add it to the user prompt
        messages = [{"role": "user", "content": system + "\n\n" + message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model(
    model_id: str,
    revision: Optional[str] = None,
    show_timings: bool = True,
    trust_remote_code: bool = False,
    **model_kwargs,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the model and its tokenizer, optionally printing timings"""
    token = os.environ.get("HUGGINGFACE_TOKEN")
    # Load the tokenizer
    with timer("tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, trust_remote_code=trust_remote_code
        )

    # Load the model
    with timer("model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, trust_remote_code=trust_remote_code, **model_kwargs
        )

    return tokenizer, model


def generate_with_timing(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message: str,
    system: Optional[str] = None,
    supports_system: bool = True,
    return_tokens: bool = False,
    include_prompt: bool = False,
    **gen_kwargs,
):
    """Generate a response to a message, printing the generation"""

    default_gen_kwargs = {
        "do_sample": True,
        "max_new_tokens": 64,
        "penalty_alpha": 0.6,
        "top_k": 4,
    }
    actual_kwargs = default_gen_kwargs | gen_kwargs

    model_prompt = prompt(
        tokenizer, message, system=system, supports_system=supports_system
    )
    tokens = tokenizer(model_prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    output = model.generate(**tokens, **actual_kwargs)
    elapsed_s = time.perf_counter() - start

    # Compute the tok/s
    n_prompt_tokens = tokens.input_ids.shape[1]
    n_new_tokens = output.shape[1] - n_prompt_tokens
    tok_per_s = n_new_tokens / elapsed_s

    print(f"elapsed {elapsed_s:.3f}s @ {tok_per_s:.3f}tok/s ({n_new_tokens} tokens)")

    if not include_prompt:
        # Remove the prompt tokens from the output
        output = output[:, n_prompt_tokens:]

    if return_tokens:
        # Just return the tokens as they are
        return output

    # Decode the output
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main(model_id: str, message: str) -> int:
    model_id, revision = parse_model_with_revision(model_id)
    # Sanity check the model
    if is_macos():
        if "AWQ" in model_id:
            print("error: macOS does not support AWQ quantization")
            return 1

    # torch_dtype = torch.float16 if is_macos() else torch.bfloat16

    # Load the model
    tokenizer, model = load_model(
        model_id,
        revision=revision,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    print(f"Loaded model {model} to device: {model.device}")

    # Generate some text
    streamer = TextStreamer(tokenizer)
    generate_with_timing(
        model, tokenizer, message, streamer=streamer, max_new_tokens=16
    )
    # The text will be printed by the streamer
    # print("------ GENERATED TEXT ------")
    # print(text)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "message",
        type=str,
        help="The message to send to the model",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        help="The model ID to use (default to %(default)s)",
        default=MODEL,
    )
    args = parser.parse_args()
    raise SystemExit(main(model_id=args.model_id, message=args.message))
