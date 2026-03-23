"""Centralized local paths for all model components.

Edit this file to point the project at your local model files.
All inference entrypoints read model paths from here.
"""

from pathlib import Path


# Fill in your fixed local paths here.
# You can keep them as absolute paths, or use paths relative to the repo root.
MODEL_ROOT = Path("/media/zric/Games/LLM/Tongyi-MAI/Z-Image-Turbo")
TRANSFORMER_DIR = MODEL_ROOT / "transformer"
VAE_DIR = MODEL_ROOT / "vae"
TEXT_ENCODER_DIR = MODEL_ROOT / "text_encoder"
TOKENIZER_DIR = MODEL_ROOT / "tokenizer"
SCHEDULER_DIR = MODEL_ROOT / "scheduler"


# Optional batch inference input path.
PROMPTS_FILE = Path("prompts/prompt1.txt")
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
ATTENTION_BACKEND = "flash"
STAGE_OFFLOAD = True
