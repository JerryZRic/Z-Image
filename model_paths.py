"""Centralized local paths for all model components.

Edit this file to point the project at your local model files.
All inference entrypoints read model paths from here.
"""

from pathlib import Path


# Model variant switch:
# - "turbo": Tongyi-MAI/Z-Image-Turbo
# - "base": Tongyi-MAI/Z-Image
MODEL_VARIANT = "base"

MODEL_ROOTS = {
    "turbo": Path("/media/zric/Games/LLM/Tongyi-MAI/Z-Image-Turbo"),
    "base": Path("/media/zric/Games/LLM/Tongyi-MAI/Z-Image"),
}

if MODEL_VARIANT not in MODEL_ROOTS:
    raise ValueError(f"MODEL_VARIANT must be one of: {', '.join(MODEL_ROOTS)}")

MODEL_DEFAULTS = {
    # Official Turbo guidance recommends 9 scheduler steps (8 effective DiT forwards) and cfg=0.0.
    "turbo": {
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
    },
    # Official Base guidance recommends 28-50 steps and cfg around 3.0-5.0.
    "base": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    },
}

# Fill in your fixed local paths here.
# You can keep them as absolute paths, or use paths relative to the repo root.
MODEL_ROOT = MODEL_ROOTS[MODEL_VARIANT]
TRANSFORMER_DIR = MODEL_ROOT / "transformer"
VAE_DIR = MODEL_ROOT / "vae"
TEXT_ENCODER_DIR = MODEL_ROOT / "text_encoder"
TOKENIZER_DIR = MODEL_ROOT / "tokenizer"
SCHEDULER_DIR = MODEL_ROOT / "scheduler"


# Optional batch inference input path.
PROMPTS_FILE = Path("prompts/prompt1.txt")
NEGATIVE_PROMPTS_FILE = Path("prompts/negative1.txt")
SINGLE_OUTPUT_DIR = Path("/dev/shm/outputs")
BATCH_OUTPUT_DIR = Path("/dev/shm/outputs")


# Seed behavior:
# - "fixed": always use SEED_VALUE
# - "random": sample a fresh seed each run
# - "increment": increase from the last used seed, starting at SEED_VALUE
# - "decrement": decrease from the last used seed, starting at SEED_VALUE
SEED_MODE = "random"
SEED_VALUE = 42
SEED_STATE_FILE = Path("/dev/shm/outputs/.zimage_seed_state.json")


# Batch behavior:
# - SERIAL_BATCH_COUNT controls how many images batch_inference.py will run in total.
#   Prompts are read from PROMPTS_FILE, empty lines are skipped, and the list cycles if needed.
# - PARALLEL_BATCH_SIZE is the lower-level simultaneous batch size. Keep it at 1 for the current staged-offload setup.
SERIAL_BATCH_COUNT = 4
PARALLEL_BATCH_SIZE = 1


# Inference parameters shared by single and batch generation.
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
DEFAULT_INFERENCE_STEPS = MODEL_DEFAULTS[MODEL_VARIANT]["num_inference_steps"]
DEFAULT_GUIDANCE_SCALE = MODEL_DEFAULTS[MODEL_VARIANT]["guidance_scale"]
ATTENTION_BACKEND = "flash"
STAGE_OFFLOAD = True
PREVIEW_IMAGE_FORMAT = "jpg"
PREVIEW_JPEG_QUALITY = 95
SAVE_FINAL_IMAGES = True
FINAL_OUTPUT_DIR = Path("/dev/shm/outputs/final")
FINAL_IMAGE_FORMAT = "png"
