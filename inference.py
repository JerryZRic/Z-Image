"""Z-Image PyTorch Native Inference."""

import os
from pathlib import Path
import time
import warnings

import torch

warnings.filterwarnings("ignore")
from model_paths import (
    ATTENTION_BACKEND,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NEGATIVE_PROMPTS_FILE,
    PROMPTS_FILE,
    SEED_MODE,
    SEED_STATE_FILE,
    SEED_VALUE,
    SINGLE_OUTPUT_DIR,
    STAGE_OFFLOAD,
)
from utils import AttentionBackend, load_from_fixed_paths, resolve_seed, set_attention_backend
from zimage import generate


def read_first_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                return prompt
    raise ValueError(f"No prompts found in {prompt_path}")


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


def main():
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    stage_offload = STAGE_OFFLOAD
    output_dir = SINGLE_OUTPUT_DIR
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    num_inference_steps = DEFAULT_INFERENCE_STEPS
    guidance_scale = DEFAULT_GUIDANCE_SCALE
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", ATTENTION_BACKEND)
    prompt = read_first_prompt(os.environ.get("PROMPTS_FILE", str(PROMPTS_FILE)))
    negative_prompt = read_first_optional_prompt(
        os.environ.get("NEGATIVE_PROMPTS_FILE", str(NEGATIVE_PROMPTS_FILE))
    )

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")
    seed, seed_mode = resolve_seed(SEED_MODE, SEED_VALUE, SEED_STATE_FILE)
    print(f"Chosen seed: {seed} ({seed_mode})")
    # Load models
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

    # Gen an image
    start_time = time.time()
    images = generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
        execution_device=device,
        stage_offload=stage_offload,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"zimage-{timestamp}-seed{seed}.png"
    images[0].save(output_path)
    print(f"Saved image to: {output_path}")

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
