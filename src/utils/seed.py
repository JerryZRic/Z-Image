"""Seed utilities for inference entrypoints."""

import json
import random
from pathlib import Path
from typing import List, Tuple


VALID_SEED_MODES = {"fixed", "random", "increment", "decrement"}


def _read_seed_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_seed_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def resolve_seed(seed_mode: str, seed_value: int, state_file: Path) -> Tuple[int, str]:
    """Resolve the runtime seed based on the configured mode."""
    if seed_mode not in VALID_SEED_MODES:
        raise ValueError(f"Invalid seed mode: {seed_mode}. Expected one of {sorted(VALID_SEED_MODES)}")

    state_file = Path(state_file)

    if seed_mode == "fixed":
        seed = int(seed_value)
    elif seed_mode == "random":
        seed = random.randint(0, 2**32 - 1)
    else:
        state = _read_seed_state(state_file)
        last_seed = state.get("last_seed")
        if last_seed is None:
            seed = int(seed_value)
        elif seed_mode == "increment":
            seed = int(last_seed) + 1
        else:
            seed = int(last_seed) - 1

    _write_seed_state(state_file, {"last_seed": int(seed), "mode": seed_mode})
    return int(seed), seed_mode


def resolve_seed_sequence(seed_mode: str, seed_value: int, count: int, state_file: Path) -> Tuple[List[int], str]:
    """Resolve a list of seeds for batch generation."""
    if count <= 0:
        raise ValueError("Seed sequence count must be >= 1")
    if seed_mode not in VALID_SEED_MODES:
        raise ValueError(f"Invalid seed mode: {seed_mode}. Expected one of {sorted(VALID_SEED_MODES)}")

    state_file = Path(state_file)

    if seed_mode == "fixed":
        seeds = [int(seed_value)] * count
    elif seed_mode == "random":
        seeds = [random.randint(0, 2**32 - 1) for _ in range(count)]
    else:
        state = _read_seed_state(state_file)
        last_seed = state.get("last_seed")
        current = int(seed_value) if last_seed is None else int(last_seed)
        step = 1 if seed_mode == "increment" else -1
        seeds = []
        for i in range(count):
            current = current if i == 0 and last_seed is None else current + step
            seeds.append(int(current))

    _write_seed_state(state_file, {"last_seed": int(seeds[-1]), "mode": seed_mode})
    return seeds, seed_mode
