"""
Training utilities for LoRA/QLoRA configuration, seeding, and parameter profiling.
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any
from transformers import BitsAndBytesConfig
from peft import LoraConfig


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def supports_native_bf16() -> bool:
    """
    Returns True only on CUDA GPUs with native hardware BF16 (Ampere SM 8.0+).
    Avoids PyTorch's is_bf16_supported() returning True via software emulation on Turing/T4 (SM 7.5).
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def resolve_torch_dtype(dtype_str: str = "bfloat16") -> torch.dtype:
    """Resolves requested dtype safely for the current GPU hardware."""
    if not torch.cuda.is_available():
        return torch.float32
    if dtype_str == "bfloat16":
        if supports_native_bf16():
            return torch.bfloat16
        return torch.float16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def get_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """Constructs a BitsAndBytesConfig from dictionary."""
    q_cfg = config.get("quantization", {})
    compute_dtype_str = q_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = resolve_torch_dtype(compute_dtype_str)

    return BitsAndBytesConfig(
        load_in_4bit=q_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Constructs a PEFT LoraConfig from dictionary."""
    l_cfg = config.get("lora", {})
    return LoraConfig(
        r=l_cfg.get("r", 16),
        lora_alpha=l_cfg.get("lora_alpha", 32),
        lora_dropout=l_cfg.get("lora_dropout", 0.05),
        bias=l_cfg.get("bias", "none"),
        target_modules=l_cfg.get("target_modules", "all-linear"),
        task_type=l_cfg.get("task_type", "CAUSAL_LM"),
    )


def log_trainable_parameters(model: Any) -> Dict[str, Any]:
    """Prints and returns trainable parameter count vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    stats = {
        "trainable_params": trainable_params,
        "all_params": all_param,
        "trainable_percent": pct,
    }
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {pct:.4f}%"
    )
    return stats
