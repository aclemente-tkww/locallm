"""A main script for running LocalLLMs"""
from __future__ import annotations

import os
import platform
import time
from contextlib import contextmanager
from typing import Optional

import huggingface_hub
from vllm import LLM, SamplingParams

# MODEL = "HuggingFaceH4/zephyr-7b-beta"  # ~2.5 tok/s
# MODEL = "meta-llama/Llama-2-7b-chat-hf"  # ~15.8 tok/s
# MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # ~2.4 tok/s
# MODEL = "TheBloke/zephyr-7B-beta-AWQ"
MODEL = "TheBloke/Llama-2-7B-Chat-AWQ"


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


def generate_with_timing(llm: LLM, message: str, **comp_params):
    params = {
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "top_k": 40,
        "max_tokens": 1024,
    }
    actual_params = params | comp_params
    sampling_params = SamplingParams(**actual_params)

    start = time.perf_counter()
    output = llm.generate(message, sampling_params)
    elapsed = time.perf_counter() - start
    output = output[0]

    n_prompt_tokens = len(output.prompt_token_ids)
    comp = output.outputs[0]
    n_new_tokens = len(comp.token_ids) - n_prompt_tokens
    tok_per_s = n_new_tokens / elapsed
    print(
        f"elapsed {elapsed:.3f}s @ {tok_per_s:.3f}tok/s ({n_new_tokens} tokens) - (finish_reason={comp.finish_reason})"
    )
    return comp.text


def main(model_id: str, message: str) -> int:
    if is_macos():
        print("error: macOS is not supported")

    model_id, revision = parse_model_with_revision(model_id)

    # Login into the huggingface-hub CLI
    huggingface_hub.login(token=os.environ.get("HUGGINGFACE_TOKEN"))

    # FIXME(alvaro): We need to make sure that the huggingface-hub CLI is logged in
    # NOTE(alvaro): Apparently vLLM does not allow bfloat16 on GPUs with
    # "compute capability" < 8.0, and the Tesla T4 is 7.5

    # torch_dtype = torch.float16 if is_macos() else torch.bfloat16
    llm = LLM(model_id, max_model_len=4096)

    # Generate some text
    generate_with_timing(llm, message, max_tokens=64)

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
