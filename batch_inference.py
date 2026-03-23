"""Batch prompt inference for Z-Image."""

import os
from pathlib import Path
import time
from itertools import cycle, islice

import torch

from model_paths import (
    ATTENTION_BACKEND,
    BATCH_OUTPUT_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PARALLEL_BATCH_SIZE,
    PROMPTS_FILE,
    SEED_MODE,
    SEED_STATE_FILE,
    SEED_VALUE,
    SERIAL_BATCH_COUNT,
    STAGE_OFFLOAD,
)
from utils import AttentionBackend, load_from_fixed_paths, resolve_seed_sequence, set_attention_backend
from zimage import generate


def read_prompts(path: str) -> list[str]:
    """Read prompts from a text file (one per line, empty lines skipped)."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


PROMPTS = read_prompts(os.environ.get("PROMPTS_FILE", str(PROMPTS_FILE)))


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from the prompt."""

    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def build_prompt_schedule(prompts: list[str], total_count: int) -> list[str]:
    """Build a fixed-length prompt schedule and cycle when needed."""
    if total_count <= 0:
        raise ValueError("SERIAL_BATCH_COUNT must be >= 1")
    return list(islice(cycle(prompts), total_count))


def select_device() -> str:
    """Choose the best available device without repeating detection logic."""

    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            print("Chosen device: mps")
            return "mps"
        print("Chosen device: cpu")
        return "cpu"


def main():
    dtype = torch.bfloat16
    compile = False
    stage_offload = STAGE_OFFLOAD
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    num_inference_steps = 8
    guidance_scale = 0.0
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", ATTENTION_BACKEND)
    output_dir = BATCH_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    total_count = SERIAL_BATCH_COUNT
    parallel_batch_size = PARALLEL_BATCH_SIZE
    if parallel_batch_size <= 0:
        raise ValueError("PARALLEL_BATCH_SIZE must be >= 1")

    device = select_device()
    seeds, seed_mode = resolve_seed_sequence(SEED_MODE, SEED_VALUE, total_count, SEED_STATE_FILE)
    preview_count = min(5, len(seeds))
    print(f"Chosen seeds ({seed_mode}): {seeds[:preview_count]}{' ...' if len(seeds) > preview_count else ''}")
    print(f"Serial batch count: {total_count}")
    print(f"Parallel batch size: {parallel_batch_size}")

    load_device = "cpu" if stage_offload and device == "cuda" else device
    components = load_from_fixed_paths(
        device=load_device,
        dtype=dtype,
        compile=compile,
        vae_device="cpu",
        text_encoder_device="cpu",
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    scheduled_prompts = build_prompt_schedule(PROMPTS, total_count)

    for idx, prompt in enumerate(scheduled_prompts, start=1):
        seed = seeds[idx - 1]
        output_path = output_dir / f"prompt-{idx:02d}-seed{seed}-{slugify(prompt)}.png"
        generator = torch.Generator(device).manual_seed(seed)

        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            execution_device=device,
            stage_offload=stage_offload,
        )
        elapsed = time.time() - start_time
        images[0].save(output_path)
        print(f"[{idx}/{total_count}] Saved {output_path} in {elapsed:.2f} seconds")

    print("Done.")


if __name__ == "__main__":
    main()
