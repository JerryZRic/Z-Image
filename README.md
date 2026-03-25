# Z-Image Python Inference Notes

This repository is a personal fork of the official Z-Image native PyTorch implementation.

The model files used by this fork are expected to be downloaded separately from Hugging Face:

- Z-Image-Turbo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Z-Image: https://huggingface.co/Tongyi-MAI/Z-Image

The goal of this fork is simple:

- learn and verify the full text-to-image inference path in plain Python
- keep the code easy to inspect and modify
- optimize runtime on a 64 GB RAM / 16 GB VRAM machine with practical engineering changes

The original upstream README has been preserved as [README.upstream.md](README.upstream.md).

## What This Fork Changes

This fork keeps the original native inference structure, but adds a workflow that is more practical on smaller GPUs:

- centralized local configuration in [model_paths.py](model_paths.py)
- fixed local model paths instead of auto-downloading with `ensure_model_weights`
- staged model offload inspired by ComfyUI-style memory management
- faster attention backend support via FlashAttention
- configurable seed behavior
- configurable prompt/output/batch behavior

In this fork, the most important optimization is staged offload:

- a CPU-resident copy of `text_encoder`, `transformer`, and `vae` stays in RAM
- only the model needed for the current stage is copied to VRAM
- the temporary GPU copy is deleted after that stage finishes

This is similar in spirit to ComfyUI's offload workflow. ComfyUI's memory manager unloads models back to their configured offload device, which is often CPU RAM in lower-VRAM modes. This fork uses a slightly different tactic: it keeps the CPU copy resident and creates temporary GPU execution copies, so we avoid sending the active GPU instance back to CPU after each stage.

This fork also supports faster attention backends. On the tested environment, the best practical setup so far is:

- `ATTENTION_BACKEND = "flash"`
- `STAGE_OFFLOAD = True`

On the tested machine, this staged-offload strategy was not just a memory workaround. A local benchmark on the Z-Image transformer showed that CPU-side `deepcopy` was significantly faster than moving the same transformer from GPU back to CPU, which strongly supports the "CPU-resident master copy + temporary GPU execution copy" design used in this fork.

## Why This Exists

The official project already works. This fork exists because I wanted a version that is:

- easier to read as a learning project
- easier to tune on a local machine
- easier to compare against GUI tools like ComfyUI
- more explicit about seed control, prompt scheduling, output location, and memory behavior

## Current Workflow

Install the Python dependencies first:

```bash
pip install -e .
```

Then edit [model_paths.py](model_paths.py) so it points at your local model files and preferred runtime settings.

Single image inference:

```bash
python inference.py
```

Batch inference from a prompt text file:

```bash
python batch_inference.py
```

Streaming batch inference with prompt pre-encoding and immediate preview output:

```bash
python batch_inference_streaming.py
```

Prompts are read from `prompts/prompt1.txt` by default, unless you change `PROMPTS_FILE` in [model_paths.py](model_paths.py).

The `prompts/` directory is tracked, but prompt text files are intentionally ignored. Create your own prompt file there, for example `prompts/prompt1.txt`.

## Configuration

Most day-to-day settings now live in [model_paths.py](model_paths.py).

Important ones:

- model variant:
  - `MODEL_VARIANT`
- model directories:
  - `TRANSFORMER_DIR`
  - `VAE_DIR`
  - `TEXT_ENCODER_DIR`
  - `TOKENIZER_DIR`
  - `SCHEDULER_DIR`
- prompt files:
  - `PROMPTS_FILE`
  - `NEGATIVE_PROMPTS_FILE`
- output paths:
  - `SINGLE_OUTPUT_DIR`
  - `BATCH_OUTPUT_DIR`
  - `FINAL_OUTPUT_DIR`
- image size:
  - `IMAGE_WIDTH`
  - `IMAGE_HEIGHT`
- seed behavior:
  - `SEED_MODE`
  - `SEED_VALUE`
  - `SEED_STATE_FILE`
- batch behavior:
  - `SERIAL_BATCH_COUNT`
  - `PARALLEL_BATCH_SIZE`
- runtime behavior:
  - `ATTENTION_BACKEND`
  - `STAGE_OFFLOAD`
  - `CFG_NORMALIZATION`
  - `SAVE_FINAL_IMAGES`
  - `PREVIEW_IMAGE_FORMAT`
  - `FINAL_IMAGE_FORMAT`

### Model Variants

This fork can switch between the official model families with:

- `MODEL_VARIANT = "turbo"`
- `MODEL_VARIANT = "base"`

This changes both the model root directory and the default inference presets:

- `turbo`
  - `DEFAULT_INFERENCE_STEPS = 8`
  - `DEFAULT_GUIDANCE_SCALE = 0.0`
- `base`
  - `DEFAULT_INFERENCE_STEPS = 50`
  - `DEFAULT_GUIDANCE_SCALE = 5.0`

### Prompt Files

- `PROMPTS_FILE` is the main prompt text file.
- `NEGATIVE_PROMPTS_FILE` is an optional negative prompt text file.

`inference.py` uses the first non-empty line from `PROMPTS_FILE`.

`batch_inference.py` and `batch_inference_streaming.py` use all non-empty lines in `PROMPTS_FILE`.

If `NEGATIVE_PROMPTS_FILE` exists, the first non-empty line is used as the negative prompt. If the file does not exist or is empty, the negative prompt is treated as disabled.

### Seed Modes

Supported seed modes:

- `fixed`
- `random`
- `increment`
- `decrement`

Behavior:

- `fixed`: every image uses the same seed
- `random`: every image gets an independent random seed
- `increment`: seeds increase across runs
- `decrement`: seeds decrease across runs

The actual seed is written into the output filename, so every generated image remains traceable.

### CFG Normalization

This fork also exposes `CFG_NORMALIZATION`.

It only matters when classifier-free guidance is actually active, which means:

- `guidance_scale > 1.0`
- and a negative prompt is present

In practice, this mainly matters for `base` experiments. With the default `turbo` preset (`guidance_scale = 0.0`), `CFG_NORMALIZATION` is effectively inert.

## Batch Behavior

This fork distinguishes two different batch concepts:

- `SERIAL_BATCH_COUNT`: how many images to generate in total
- `PARALLEL_BATCH_SIZE`: lower-level simultaneous batch size

Right now, `PARALLEL_BATCH_SIZE` is kept at `1` by default because the staged-offload path is the main optimization target.

For serial batch scheduling:

- prompts are read from the prompt file
- empty lines are skipped
- if `SERIAL_BATCH_COUNT` is smaller than the number of prompts, only the first prompts are used
- if `SERIAL_BATCH_COUNT` is larger than the number of prompts, the prompt list loops from the beginning until the requested count is reached

## Batch Modes

This fork currently provides two batch workflows:

### `batch_inference.py`

This is the simpler baseline batch runner:

- each image runs through the full pipeline independently
- easier to understand
- useful as a reference path

### `batch_inference_streaming.py`

This is the more optimized batch runner:

- prompt embeddings are prepared up front and cached in RAM
- the text encoder is not re-run for every image
- the transformer stays on GPU across the batch
- the VAE runs in `bfloat16` for faster/lower-memory preview decode
- each image is decoded and saved immediately after denoising
- preview images can be written as JPEG for fast local inspection
- after the batch finishes, the stored latents can optionally be decoded again with a `float32` VAE for final-quality outputs

At the moment, this is the recommended batch workflow for the tested hardware configuration.

## FlashAttention

If you want the best speed from this fork, use FlashAttention.

Recommended approach:

1. build FlashAttention yourself for your exact Python / PyTorch / CUDA environment
2. install it into your inference environment

This fork does not commit a prebuilt wheel to the repository. If you already have a compatible FlashAttention wheel for your exact environment, you can install that locally instead of rebuilding.

The environment used during testing was:

- Python 3.12
- PyTorch 2.10.0 + CUDA 12.8

If your environment differs, compiling your own build is the safer option.

## Attention Backends

This fork exposes multiple attention backends through configuration:

- `flash`
- `flash_varlen`
- `_flash_3`
- `_flash_varlen_3`
- `mps_flash`
- `native`
- `_native_flash`
- `_native_math`

The currently best-tested option in this fork is `flash`.

Backends that are not explicitly implemented in the codebase are not automatically supported. For example, installing another attention library such as SageAttention does not make it available by itself; the backend must also be wired into the local attention dispatch code.

## Experimental Overrides

The native implementation uses a dynamic image-size-dependent shift (`mu`) by default.

If you want to compare that behavior against the Hugging Face Space style fixed shift, you can set:

```bash
ZIMAGE_FIXED_SHIFT=3.0
```

When this environment variable is set, the sampling path uses the fixed value instead of the native dynamic `calculate_shift(...)` result.

## Output Locations

Outputs are configured to go to the RAM-backed filesystem by default:

- single image outputs: `/dev/shm/outputs`
- batch outputs: `/dev/shm/outputs`

Single-image filenames include:

- timestamp
- seed

Batch filenames include:

- index
- seed
- prompt slug

## Practical Notes

- This fork is intentionally opinionated toward local experimentation and learning.
- The official upstream documentation is still the best reference for model-family details and baseline usage.
- If you want the original project documentation, see [README.upstream.md](README.upstream.md).

## Suggested Setup

For a machine similar to the one used during testing:

- Intel Core i5-12490F
- DDR4-3200 64 GB RAM
- NVIDIA RTX 4070 Ti Super 16 GB VRAM
- local model files already downloaded

try:

- `ATTENTION_BACKEND = "flash"`
- `STAGE_OFFLOAD = True`
- a local FlashAttention build
- outputs on `/dev/shm`

## Acknowledgment

All model architecture, weights, and original native implementation belong to the original Z-Image project. This fork is an experimental, learning-oriented adaptation focused on local inference workflow and runtime tuning.
