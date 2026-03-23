"""Z-Image PyTorch Native Inference."""

import os
from pathlib import Path
import time
import warnings

import torch

warnings.filterwarnings("ignore")
from model_paths import (
    ATTENTION_BACKEND,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEED_MODE,
    SEED_STATE_FILE,
    SEED_VALUE,
    SINGLE_OUTPUT_DIR,
    STAGE_OFFLOAD,
)
from utils import AttentionBackend, load_from_fixed_paths, resolve_seed, set_attention_backend
from zimage import generate


def main():
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    stage_offload = STAGE_OFFLOAD
    output_dir = SINGLE_OUTPUT_DIR
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    num_inference_steps = 8
    guidance_scale = 0.0
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", ATTENTION_BACKEND)
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
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
