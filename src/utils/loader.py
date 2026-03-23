"""Model loading utilities for Z-Image components."""

import json
import os
from pathlib import Path
import sys
from typing import Optional, Union

from loguru import logger
from safetensors.torch import load_file
import torch
from transformers import AutoModel, AutoTokenizer

from config import (
    DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS,
    DEFAULT_SCHEDULER_SHIFT,
    DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING,
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
    DEFAULT_VAE_IN_CHANNELS,
    DEFAULT_VAE_LATENT_CHANNELS,
    DEFAULT_VAE_NORM_NUM_GROUPS,
    DEFAULT_VAE_OUT_CHANNELS,
    DEFAULT_VAE_SCALING_FACTOR,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)
from zimage.autoencoder import AutoencoderKL as LocalAutoencoderKL
from zimage.scheduler import FlowMatchEulerDiscreteScheduler

DIFFUSERS_AVAILABLE = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_safetensors(weight_dir: Path, device: str = "cuda", dtype: Optional[torch.dtype] = None) -> dict:
    """Load sharded safetensors from a directory."""
    weight_dir = Path(weight_dir)
    index_files = list(weight_dir.glob("*.safetensors.index.json"))

    state_dict = {}
    if index_files:
        with open(index_files[0], "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
        for shard_file in shard_files:
            shard_path = weight_dir / shard_file
            shard_state = load_file(str(shard_path), device=str(device))
            state_dict.update(shard_state)
    else:
        safetensors_files = list(weight_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {weight_dir}")
        state_dict = load_file(str(safetensors_files[0]), device=str(device))

    if dtype is not None:
        state_dict = {k: v.to(dtype) if v.dtype != dtype else v for k, v in state_dict.items()}

    return state_dict


def _normalize_component_dir(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _load_components_from_dirs(
    transformer_dir: Union[str, Path],
    vae_dir: Union[str, Path],
    text_encoder_dir: Union[str, Path],
    tokenizer_dir: Union[str, Path],
    scheduler_dir: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
    compile: bool = False,
    vae_device: Optional[str] = None,
    text_encoder_device: Optional[str] = None,
) -> dict:
    transformer_dir = _normalize_component_dir(transformer_dir)
    vae_dir = _normalize_component_dir(vae_dir)
    text_encoder_dir = _normalize_component_dir(text_encoder_dir)
    tokenizer_dir = _normalize_component_dir(tokenizer_dir)
    scheduler_dir = _normalize_component_dir(scheduler_dir)

    repo_src_dir = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(repo_src_dir))
    from zimage.transformer import ZImageTransformer2DModel

    if verbose:
        logger.info("Loading Z-Image from configured component paths")

    if vae_device is None:
        vae_device = "cpu"
    if text_encoder_device is None:
        text_encoder_device = "cpu"

    if verbose:
        logger.info("Loading DiT...")
    transformer_config = load_config(str(transformer_dir / "config.json"))

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(transformer_config.get("all_patch_size", DEFAULT_TRANSFORMER_PATCH_SIZE)),
            all_f_patch_size=tuple(transformer_config.get("all_f_patch_size", DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
            in_channels=transformer_config.get("in_channels", DEFAULT_TRANSFORMER_IN_CHANNELS),
            dim=transformer_config.get("dim", DEFAULT_TRANSFORMER_DIM),
            n_layers=transformer_config.get("n_layers", DEFAULT_TRANSFORMER_N_LAYERS),
            n_refiner_layers=transformer_config.get("n_refiner_layers", DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
            n_heads=transformer_config.get("n_heads", DEFAULT_TRANSFORMER_N_HEADS),
            n_kv_heads=transformer_config.get("n_kv_heads", DEFAULT_TRANSFORMER_N_KV_HEADS),
            norm_eps=transformer_config.get("norm_eps", DEFAULT_TRANSFORMER_NORM_EPS),
            qk_norm=transformer_config.get("qk_norm", DEFAULT_TRANSFORMER_QK_NORM),
            cap_feat_dim=transformer_config.get("cap_feat_dim", DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
            rope_theta=transformer_config.get("rope_theta", ROPE_THETA),
            t_scale=transformer_config.get("t_scale", DEFAULT_TRANSFORMER_T_SCALE),
            axes_dims=transformer_config.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=transformer_config.get("axes_lens", ROPE_AXES_LENS),
        ).to(dtype)

    state_dict = load_sharded_safetensors(transformer_dir, device="cpu", dtype=dtype)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    if verbose:
        logger.info("Moving DiT to target device...")
    transformer = transformer.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    transformer.eval()

    if verbose:
        logger.info("Loading VAE...")
    vae_config = load_config(str(vae_dir / "config.json"))

    vae = LocalAutoencoderKL(
        in_channels=vae_config.get("in_channels", DEFAULT_VAE_IN_CHANNELS),
        out_channels=vae_config.get("out_channels", DEFAULT_VAE_OUT_CHANNELS),
        down_block_types=tuple(vae_config.get("down_block_types", ("DownEncoderBlock2D",))),
        up_block_types=tuple(vae_config.get("up_block_types", ("UpDecoderBlock2D",))),
        block_out_channels=tuple(vae_config.get("block_out_channels", (64,))),
        layers_per_block=vae_config.get("layers_per_block", 1),
        latent_channels=vae_config.get("latent_channels", DEFAULT_VAE_LATENT_CHANNELS),
        norm_num_groups=vae_config.get("norm_num_groups", DEFAULT_VAE_NORM_NUM_GROUPS),
        scaling_factor=vae_config.get("scaling_factor", DEFAULT_VAE_SCALING_FACTOR),
        shift_factor=vae_config.get("shift_factor", None),
        use_quant_conv=vae_config.get("use_quant_conv", True),
        use_post_quant_conv=vae_config.get("use_post_quant_conv", True),
        mid_block_add_attention=vae_config.get("mid_block_add_attention", True),
    )
    vae._init_kwargs = {
        "in_channels": vae_config.get("in_channels", DEFAULT_VAE_IN_CHANNELS),
        "out_channels": vae_config.get("out_channels", DEFAULT_VAE_OUT_CHANNELS),
        "down_block_types": tuple(vae_config.get("down_block_types", ("DownEncoderBlock2D",))),
        "up_block_types": tuple(vae_config.get("up_block_types", ("UpDecoderBlock2D",))),
        "block_out_channels": tuple(vae_config.get("block_out_channels", (64,))),
        "layers_per_block": vae_config.get("layers_per_block", 1),
        "latent_channels": vae_config.get("latent_channels", DEFAULT_VAE_LATENT_CHANNELS),
        "norm_num_groups": vae_config.get("norm_num_groups", DEFAULT_VAE_NORM_NUM_GROUPS),
        "scaling_factor": vae_config.get("scaling_factor", DEFAULT_VAE_SCALING_FACTOR),
        "shift_factor": vae_config.get("shift_factor", None),
        "use_quant_conv": vae_config.get("use_quant_conv", True),
        "use_post_quant_conv": vae_config.get("use_post_quant_conv", True),
        "mid_block_add_attention": vae_config.get("mid_block_add_attention", True),
    }

    vae_state_dict = load_sharded_safetensors(vae_dir, device="cpu")
    vae.load_state_dict(vae_state_dict, strict=False)
    del vae_state_dict
    vae.to(device=vae_device, dtype=torch.float32)
    vae.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        logger.info("Loading Text Encoder...")
    text_encoder = AutoModel.from_pretrained(
        str(text_encoder_dir),
        dtype=dtype,
        trust_remote_code=True,
    )
    text_encoder.to(text_encoder_device)
    text_encoder.eval()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if verbose:
        logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir) if tokenizer_dir.exists() else str(text_encoder_dir),
        trust_remote_code=True,
    )

    if verbose:
        logger.info("Loading Scheduler...")
    scheduler_config = load_config(str(scheduler_dir / "scheduler_config.json"))
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scheduler_config.get("num_train_timesteps", DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS),
        shift=scheduler_config.get("shift", DEFAULT_SCHEDULER_SHIFT),
        use_dynamic_shifting=scheduler_config.get("use_dynamic_shifting", DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING),
    )

    if compile:
        if verbose:
            logger.info("Compiling DiT and VAE...")
        transformer = torch.compile(transformer)
        vae = torch.compile(vae)

    if verbose:
        logger.success("All components loaded successfully")

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def load_from_local_dir(
    model_dir: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
    compile: bool = False,
    vae_device: Optional[str] = None,
    text_encoder_device: Optional[str] = None,
) -> dict:
    """Load all Z-Image components from a local model root directory."""
    model_dir = Path(model_dir).expanduser().resolve()

    if verbose:
        logger.info(f"Loading Z-Image from: {model_dir}")

    return _load_components_from_dirs(
        transformer_dir=model_dir / "transformer",
        vae_dir=model_dir / "vae",
        text_encoder_dir=model_dir / "text_encoder",
        tokenizer_dir=model_dir / "tokenizer",
        scheduler_dir=model_dir / "scheduler",
        device=device,
        dtype=dtype,
        verbose=verbose,
        compile=compile,
        vae_device=vae_device,
        text_encoder_device=text_encoder_device,
    )


def load_from_fixed_paths(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
    compile: bool = False,
    vae_device: Optional[str] = None,
    text_encoder_device: Optional[str] = None,
) -> dict:
    """Load all Z-Image components from the fixed paths in ``model_paths.py``."""
    from model_paths import SCHEDULER_DIR, TEXT_ENCODER_DIR, TOKENIZER_DIR, TRANSFORMER_DIR, VAE_DIR

    return _load_components_from_dirs(
        transformer_dir=TRANSFORMER_DIR,
        vae_dir=VAE_DIR,
        text_encoder_dir=TEXT_ENCODER_DIR,
        tokenizer_dir=TOKENIZER_DIR,
        scheduler_dir=SCHEDULER_DIR,
        device=device,
        dtype=dtype,
        verbose=verbose,
        compile=compile,
        vae_device=vae_device,
        text_encoder_device=text_encoder_device,
    )
