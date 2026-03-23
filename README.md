# Z-Image Python Inference Notes

This repository is a personal fork of the official Z-Image native PyTorch implementation.

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

## Why This Exists

The official project already works. This fork exists because I wanted a version that is:

- easier to read as a learning project
- easier to tune on a local machine
- easier to compare against GUI tools like ComfyUI
- more explicit about seed control, prompt scheduling, output location, and memory behavior

## Current Workflow

Single image inference:

```bash
python inference.py
```

Batch inference from a prompt text file:

```bash
python batch_inference.py
```

Prompts are read from [prompts/prompt1.txt](prompts/prompt1.txt) by default, unless you change `PROMPTS_FILE` in [model_paths.py](model_paths.py).

The `prompts/` directory is tracked, but prompt text files are intentionally ignored. Create your own prompt file there, for example `prompts/prompt1.txt`.

## Configuration

Most day-to-day settings now live in [model_paths.py](model_paths.py).

Important ones:

- model directories:
  - `TRANSFORMER_DIR`
  - `VAE_DIR`
  - `TEXT_ENCODER_DIR`
  - `TOKENIZER_DIR`
  - `SCHEDULER_DIR`
- output paths:
  - `SINGLE_OUTPUT_DIR`
  - `BATCH_OUTPUT_DIR`
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

## FlashAttention

If you want the best speed from this fork, use FlashAttention.

Recommended approach:

1. build FlashAttention yourself for your exact Python / PyTorch / CUDA environment
2. install it into your inference environment

If your environment matches mine closely enough, you can also try the included wheel:

- [wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl](wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl)

This wheel was tested with:

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

- 64 GB system RAM
- 16 GB VRAM
- local model files already downloaded

try:

- `ATTENTION_BACKEND = "flash"`
- `STAGE_OFFLOAD = True`
- a local FlashAttention build
- outputs on `/dev/shm`

## Acknowledgment

All model architecture, weights, and original native implementation belong to the original Z-Image project. This fork is an experimental, learning-oriented adaptation focused on local inference workflow and runtime tuning.
