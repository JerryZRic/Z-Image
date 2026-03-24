"""Benchmark image save formats for preview/final output decisions.

This script compares save time and file size for:

- JPEG
- PNG with compression level 0
- BMP

Each format is saved multiple times and the script reports average time and size.
"""

from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image


RUNS = 10
WIDTH = 720
HEIGHT = 1280
OUTPUT_DIR = Path("/dev/shm/zimage_save_benchmark")


def make_test_image(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """Create a deterministic test image with enough structure for compression differences."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]

    image = noise.copy()
    image[..., 0] = ((image[..., 0].astype(np.uint16) + x) // 2).astype(np.uint8)
    image[..., 1] = ((image[..., 1].astype(np.uint16) + y) // 2).astype(np.uint8)
    image[..., 2] = ((image[..., 2].astype(np.uint16) + ((x + y) // 2)) // 2).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def benchmark_format(image: Image.Image, fmt: str, suffix: str, save_kwargs: dict, runs: int = RUNS) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = []
    sizes = []

    for i in range(runs):
        output_path = OUTPUT_DIR / f"benchmark-{fmt.lower()}-{i:02d}.{suffix}"
        if output_path.exists():
            output_path.unlink()

        start = time.perf_counter()
        image.save(output_path, format=fmt, **save_kwargs)
        end = time.perf_counter()

        times.append(end - start)
        sizes.append(output_path.stat().st_size)

    return {
        "format": fmt,
        "runs": runs,
        "avg_time_s": statistics.mean(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "avg_size_bytes": statistics.mean(sizes),
        "min_size_bytes": min(sizes),
        "max_size_bytes": max(sizes),
    }


def format_bytes(size: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"


def print_result(result: dict) -> None:
    print(f"{result['format']}:")
    print(f"  runs: {result['runs']}")
    print(f"  average save time: {result['avg_time_s']:.4f} s")
    print(f"  min save time: {result['min_time_s']:.4f} s")
    print(f"  max save time: {result['max_time_s']:.4f} s")
    print(f"  average file size: {format_bytes(result['avg_size_bytes'])}")
    print(f"  min file size: {format_bytes(result['min_size_bytes'])}")
    print(f"  max file size: {format_bytes(result['max_size_bytes'])}")


def main() -> None:
    image = make_test_image()
    print(f"Benchmark image size: {WIDTH}x{HEIGHT}")
    print(f"Runs per format: {RUNS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    configs = [
        ("JPEG", "jpg", {"quality": 95}),
        ("PNG", "png", {"compress_level": 0}),
        ("BMP", "bmp", {}),
    ]

    results = []
    for fmt, suffix, kwargs in configs:
        result = benchmark_format(image, fmt, suffix, kwargs, runs=RUNS)
        results.append(result)
        print_result(result)
        print()

    fastest = min(results, key=lambda item: item["avg_time_s"])
    smallest = min(results, key=lambda item: item["avg_size_bytes"])

    print("Summary:")
    print(f"  fastest average save: {fastest['format']} ({fastest['avg_time_s']:.4f} s)")
    print(f"  smallest average size: {smallest['format']} ({format_bytes(smallest['avg_size_bytes'])})")


if __name__ == "__main__":
    main()
