"""Benchmark transformer deepcopy vs GPU->CPU transfer.

This script measures two costs for the Z-Image transformer:

1. The time to deepcopy a CPU-resident transformer.
2. The time to move a GPU-resident transformer back to CPU.

Each benchmark runs 10 times and prints the average.
"""

import copy
import gc
import time

import torch

from config import (
    DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
    DEFAULT_TRANSFORMER_DIM,
    DEFAULT_TRANSFORMER_F_PATCH_SIZE,
    DEFAULT_TRANSFORMER_IN_CHANNELS,
    DEFAULT_TRANSFORMER_N_HEADS,
    DEFAULT_TRANSFORMER_N_KV_HEADS,
    DEFAULT_TRANSFORMER_N_LAYERS,
    DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
    DEFAULT_TRANSFORMER_NORM_EPS,
    DEFAULT_TRANSFORMER_PATCH_SIZE,
    DEFAULT_TRANSFORMER_QK_NORM,
    DEFAULT_TRANSFORMER_T_SCALE,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)
from model_paths import TRANSFORMER_DIR
from utils.loader import load_config, load_sharded_safetensors
from zimage.transformer import ZImageTransformer2DModel


RUNS = 10
DTYPE = torch.bfloat16


def load_transformer_cpu() -> ZImageTransformer2DModel:
    transformer_dir = TRANSFORMER_DIR
    config = load_config(str(transformer_dir / "config.json"))

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(config.get("all_patch_size", DEFAULT_TRANSFORMER_PATCH_SIZE)),
            all_f_patch_size=tuple(config.get("all_f_patch_size", DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
            in_channels=config.get("in_channels", DEFAULT_TRANSFORMER_IN_CHANNELS),
            dim=config.get("dim", DEFAULT_TRANSFORMER_DIM),
            n_layers=config.get("n_layers", DEFAULT_TRANSFORMER_N_LAYERS),
            n_refiner_layers=config.get("n_refiner_layers", DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
            n_heads=config.get("n_heads", DEFAULT_TRANSFORMER_N_HEADS),
            n_kv_heads=config.get("n_kv_heads", DEFAULT_TRANSFORMER_N_KV_HEADS),
            norm_eps=config.get("norm_eps", DEFAULT_TRANSFORMER_NORM_EPS),
            qk_norm=config.get("qk_norm", DEFAULT_TRANSFORMER_QK_NORM),
            cap_feat_dim=config.get("cap_feat_dim", DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
            rope_theta=config.get("rope_theta", ROPE_THETA),
            t_scale=config.get("t_scale", DEFAULT_TRANSFORMER_T_SCALE),
            axes_dims=config.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=config.get("axes_lens", ROPE_AXES_LENS),
        ).to(DTYPE)

    state_dict = load_sharded_safetensors(transformer_dir, device="cpu", dtype=DTYPE)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    transformer = transformer.to("cpu")
    transformer.eval()
    return transformer


def benchmark_cpu_deepcopy(cpu_transformer: ZImageTransformer2DModel, runs: int = RUNS) -> list[float]:
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        copied = copy.deepcopy(cpu_transformer)
        end = time.perf_counter()
        times.append(end - start)
        del copied
        gc.collect()
    return times


def benchmark_gpu_to_cpu(cpu_transformer: ZImageTransformer2DModel, runs: int = RUNS) -> list[float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU->CPU benchmark.")

    times = []
    for _ in range(runs):
        gc.collect()
        torch.cuda.empty_cache()

        gpu_transformer = copy.deepcopy(cpu_transformer).to("cuda")
        torch.cuda.synchronize()

        start = time.perf_counter()
        moved_back = gpu_transformer.to("cpu")
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

        del gpu_transformer
        del moved_back
        gc.collect()
        torch.cuda.empty_cache()

    return times


def print_stats(name: str, times: list[float]) -> None:
    avg = sum(times) / len(times)
    print(f"{name}:")
    print(f"  runs: {len(times)}")
    print(f"  average: {avg:.4f} s")
    print(f"  min: {min(times):.4f} s")
    print(f"  max: {max(times):.4f} s")


def main() -> None:
    print("Loading transformer to CPU...")
    cpu_transformer = load_transformer_cpu()
    print("Transformer loaded.\n")

    cpu_deepcopy_times = benchmark_cpu_deepcopy(cpu_transformer, runs=RUNS)
    print_stats("CPU deepcopy", cpu_deepcopy_times)
    print()

    if torch.cuda.is_available():
        gpu_to_cpu_times = benchmark_gpu_to_cpu(cpu_transformer, runs=RUNS)
        print_stats("GPU->CPU transfer", gpu_to_cpu_times)
        print()

        deepcopy_avg = sum(cpu_deepcopy_times) / len(cpu_deepcopy_times)
        transfer_avg = sum(gpu_to_cpu_times) / len(gpu_to_cpu_times)
        faster = "CPU deepcopy" if deepcopy_avg < transfer_avg else "GPU->CPU transfer"
        print("Comparison:")
        print(f"  faster on average: {faster}")
        print(f"  average difference: {abs(deepcopy_avg - transfer_avg):.4f} s")
    else:
        print("CUDA not available, skipped GPU->CPU transfer benchmark.")


if __name__ == "__main__":
    main()
