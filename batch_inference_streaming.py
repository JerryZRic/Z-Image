"""Batch inference with pre-encoded prompts and streaming decode.

This variant differs from batch_inference.py:

1. Prompt embeddings are prepared up front and cached in RAM.
2. The text encoder is unloaded before denoising starts.
3. The transformer and VAE stay together on GPU during generation.
4. Each image is decoded and saved immediately after denoising.
"""

import os
from itertools import cycle, islice
from pathlib import Path
import time

import torch

from model_paths import (
    ATTENTION_BACKEND,
    BATCH_OUTPUT_DIR,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    FINAL_IMAGE_FORMAT,
    FINAL_OUTPUT_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NEGATIVE_PROMPTS_FILE,
    PARALLEL_BATCH_SIZE,
    PREVIEW_IMAGE_FORMAT,
    PREVIEW_JPEG_QUALITY,
    PROMPTS_FILE,
    SAVE_FINAL_IMAGES,
    SEED_MODE,
    SEED_STATE_FILE,
    SEED_VALUE,
    SERIAL_BATCH_COUNT,
)
from utils import AttentionBackend, debug_memory_snapshot, load_from_fixed_paths, resolve_seed_sequence, set_attention_backend
from zimage.pipeline import clone_module_to_device, cleanup_cuda_stage, decode_latents, encode_prompt_embeddings, sample_latents


def read_prompts(path: str) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def read_first_optional_prompt(path: str) -> str | None:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return None
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                return prompt
    return None


def slugify(text: str, max_len: int = 60) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def build_prompt_schedule(prompts: list[str], total_count: int) -> list[str]:
    if total_count <= 0:
        raise ValueError("SERIAL_BATCH_COUNT must be >= 1")
    return list(islice(cycle(prompts), total_count))


def select_device() -> str:
    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Chosen device: mps")
        return "mps"
    print("Chosen device: cpu")
    return "cpu"


def main():
    memory_debug = os.environ.get("ZIMAGE_DEBUG_MEMORY", "0") == "1"
    dtype = torch.bfloat16
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    num_inference_steps = DEFAULT_INFERENCE_STEPS
    guidance_scale = DEFAULT_GUIDANCE_SCALE
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", ATTENTION_BACKEND)
    preview_format = PREVIEW_IMAGE_FORMAT.lower()
    if preview_format not in {"jpg", "jpeg", "png", "bmp"}:
        raise ValueError("PREVIEW_IMAGE_FORMAT must be one of: jpg, jpeg, png, bmp")
    final_format = FINAL_IMAGE_FORMAT.lower()
    if final_format not in {"png", "bmp", "jpg", "jpeg"}:
        raise ValueError("FINAL_IMAGE_FORMAT must be one of: png, bmp, jpg, jpeg")
    output_dir = BATCH_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = FINAL_OUTPUT_DIR
    if SAVE_FINAL_IMAGES:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    total_count = SERIAL_BATCH_COUNT
    parallel_batch_size = PARALLEL_BATCH_SIZE
    if parallel_batch_size != 1:
        print("PARALLEL_BATCH_SIZE is currently informational for this streaming script. Using serial decode with batch size 1.")

    prompts = read_prompts(os.environ.get("PROMPTS_FILE", str(PROMPTS_FILE)))
    negative_prompt = read_first_optional_prompt(
        os.environ.get("NEGATIVE_PROMPTS_FILE", str(NEGATIVE_PROMPTS_FILE))
    )
    scheduled_prompts = build_prompt_schedule(prompts, total_count)

    device = select_device()
    if device != "cuda":
        raise RuntimeError("This streaming batch script is intended for CUDA execution.")

    seeds, seed_mode = resolve_seed_sequence(SEED_MODE, SEED_VALUE, total_count, SEED_STATE_FILE)
    preview_count = min(5, len(seeds))
    print(f"Chosen seeds ({seed_mode}): {seeds[:preview_count]}{' ...' if len(seeds) > preview_count else ''}")
    print(f"Serial batch count: {total_count}")
    print(f"Parallel batch size: {parallel_batch_size}")

    components = load_from_fixed_paths(
        device="cpu",
        dtype=dtype,
        compile=False,
        vae_device="cpu",
        text_encoder_device="cpu",
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    unique_prompts = list(dict.fromkeys(scheduled_prompts))
    prompt_embed_cache = {}

    print(f"Encoding {len(unique_prompts)} unique prompt(s)...")
    text_encoder_runtime = clone_module_to_device(
        components["text_encoder"],
        device,
        next(components["text_encoder"].parameters()).dtype,
    )
    prompt_embeds_list, negative_prompt_embeds_list = encode_prompt_embeddings(
        text_encoder_runtime,
        components["tokenizer"],
        prompt=unique_prompts,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        target_device="cpu",
    )
    for prompt, embeds in zip(unique_prompts, prompt_embeds_list):
        prompt_embed_cache[prompt] = embeds
    negative_prompt_embed_cache = {}
    if negative_prompt_embeds_list:
        for prompt, embeds in zip(unique_prompts, negative_prompt_embeds_list):
            negative_prompt_embed_cache[prompt] = embeds
    del text_encoder_runtime
    cleanup_cuda_stage(device)
    if memory_debug:
        debug_memory_snapshot(
            "stream_after_text_encode",
            modules={"text_encoder_cpu": components["text_encoder"]},
            tensors={"cached_prompt_embeds_cpu": list(prompt_embed_cache.values())},
        )

    print("Loading transformer and VAE to GPU...")
    transformer_runtime = clone_module_to_device(
        components["transformer"],
        device,
        next(components["transformer"].parameters()).dtype,
    )
    vae_runtime = clone_module_to_device(
        components["vae"],
        device,
        torch.bfloat16,
    )
    if memory_debug:
        debug_memory_snapshot(
            "stream_transformer_vae_on_gpu",
            modules={
                "transformer_cpu": components["transformer"],
                "transformer_gpu": transformer_runtime,
                "vae_cpu": components["vae"],
                "vae_gpu": vae_runtime,
            },
        )

    total_start = time.time()
    final_decode_jobs = []
    for idx, prompt in enumerate(scheduled_prompts, start=1):
        seed = seeds[idx - 1]
        stem = f"stream-{idx:02d}-seed{seed}-{slugify(prompt)}"
        output_path = output_dir / f"{stem}.{preview_format}"
        generator = torch.Generator(device).manual_seed(seed)

        start_time = time.time()
        prompt_embed_gpu = prompt_embed_cache[prompt].to(device)
        negative_prompt_embed_gpu = None
        if guidance_scale > 1.0 and prompt in negative_prompt_embed_cache:
            negative_prompt_embed_gpu = negative_prompt_embed_cache[prompt].to(device)
        latents = sample_latents(
            transformer_runtime,
            components["scheduler"],
            prompt_embeds_list=[prompt_embed_gpu],
            negative_prompt_embeds_list=[negative_prompt_embed_gpu] if negative_prompt_embed_gpu is not None else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if memory_debug:
            debug_memory_snapshot(
                f"stream_before_decode_{idx:02d}",
                modules={"transformer_gpu": transformer_runtime, "vae_gpu": vae_runtime},
                tensors={
                    "prompt_embed_gpu": prompt_embed_gpu,
                    "negative_prompt_embed_gpu": negative_prompt_embed_gpu,
                    "latents_gpu": latents,
                },
            )
        try:
            images = decode_latents(vae_runtime, latents, output_type="pil")
        except Exception:
            if memory_debug:
                debug_memory_snapshot(
                    f"stream_decode_failure_{idx:02d}",
                    modules={"transformer_gpu": transformer_runtime, "vae_gpu": vae_runtime},
                    tensors={
                        "prompt_embed_gpu": prompt_embed_gpu,
                        "negative_prompt_embed_gpu": negative_prompt_embed_gpu,
                        "latents_gpu": latents,
                    },
                )
            raise
        if SAVE_FINAL_IMAGES:
            final_decode_jobs.append((idx, seed, prompt, stem, latents.detach().to("cpu")))
        save_kwargs = {}
        if preview_format in {"jpg", "jpeg"}:
            save_kwargs["quality"] = PREVIEW_JPEG_QUALITY
            save_kwargs["format"] = "JPEG"
        elif preview_format == "png":
            save_kwargs["format"] = "PNG"
        elif preview_format == "bmp":
            save_kwargs["format"] = "BMP"
        images[0].save(output_path, **save_kwargs)
        elapsed = time.time() - start_time
        print(f"[{idx}/{total_count}] Saved {output_path} in {elapsed:.2f} seconds")
        del prompt_embed_gpu
        if negative_prompt_embed_gpu is not None:
            del negative_prompt_embed_gpu
        del latents
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if memory_debug:
            debug_memory_snapshot(
                f"stream_after_cleanup_{idx:02d}",
                modules={"transformer_gpu": transformer_runtime, "vae_gpu": vae_runtime},
            )

    total_elapsed = time.time() - total_start
    print(f"Total streaming batch time: {total_elapsed:.2f} seconds")

    del transformer_runtime
    del vae_runtime
    cleanup_cuda_stage(device)

    if SAVE_FINAL_IMAGES and final_decode_jobs:
        print("Running final float32 VAE decode for saved latents...")
        final_start = time.time()
        final_vae_runtime = clone_module_to_device(
            components["vae"],
            device,
            torch.float32,
        )
        if memory_debug:
            debug_memory_snapshot(
                "stream_final_vae_on_gpu",
                modules={"vae_cpu": components["vae"], "vae_gpu": final_vae_runtime},
            )
        for idx, seed, prompt, stem, final_latents_cpu in final_decode_jobs:
            final_path = final_output_dir / f"{stem}-final.{final_format}"
            final_step_start = time.time()
            if memory_debug:
                debug_memory_snapshot(
                    f"stream_before_final_decode_{idx:02d}",
                    modules={"vae_gpu": final_vae_runtime},
                    tensors={"final_latents_cpu": final_latents_cpu},
                )
            final_images = decode_latents(final_vae_runtime, final_latents_cpu, output_type="pil")
            final_save_kwargs = {}
            if final_format in {"jpg", "jpeg"}:
                final_save_kwargs["format"] = "JPEG"
                final_save_kwargs["quality"] = 100
            elif final_format == "png":
                final_save_kwargs["format"] = "PNG"
            elif final_format == "bmp":
                final_save_kwargs["format"] = "BMP"
            final_images[0].save(final_path, **final_save_kwargs)
            final_elapsed = time.time() - final_step_start
            print(f"[final {idx}/{total_count}] Saved {final_path} in {final_elapsed:.2f} seconds")
            del final_images
            del final_latents_cpu
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        final_total_elapsed = time.time() - final_start
        print(f"Total final decode time: {final_total_elapsed:.2f} seconds")
        del final_vae_runtime
        cleanup_cuda_stage(device)


if __name__ == "__main__":
    main()
