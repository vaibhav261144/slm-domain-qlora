"""
Main QLoRA fine-tuning script using Hugging Face Transformers, PEFT, and TRL.
Supports config-driven experiments, 4-bit NF4 quantization, and chat template formatting.
"""

import argparse
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig

from src.data.dataset_loader import load_jsonl_dataset
from src.training.utils import (
    set_seed,
    load_yaml_config,
    get_quantization_config,
    get_lora_config,
    log_trainable_parameters,
)


def train(config_path: str):
    print(f"Loading experiment configuration from: {config_path}")
    cfg = load_yaml_config(config_path)

    # 1. Set seed for reproducibility
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # 2. Check GPU environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    model_id = cfg["model"]["model_id"]
    print(f"Base model ID: {model_id}")

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Configure Quantization (QLoRA NF4)
    bnb_config = None
    if cfg.get("quantization", {}).get("load_in_4bit", True) and device == "cuda":
        bnb_config = get_quantization_config(cfg)
        print("4-bit NF4 Quantization enabled.")
    else:
        print("Running in full precision or without 4-bit bitsandbytes.")

    # 5. Load Base Model
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
        )

    # 6. Apply LoRA Adapter
    lora_config = get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    log_trainable_parameters(model)

    # 7. Load Datasets
    data_cfg = cfg["data"]
    train_dataset = load_jsonl_dataset(data_cfg["train_file"])
    val_dataset = load_jsonl_dataset(data_cfg["val_file"]) if data_cfg.get("val_file") else None

    print(f"Loaded {len(train_dataset)} training examples.")
    if val_dataset:
        print(f"Loaded {len(val_dataset)} validation examples.")

    # 8. Setup SFTConfig
    train_args = cfg["training"]
    output_dir = train_args.get("output_dir", "outputs/run")
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_args.get("num_train_epochs", 3),
        per_device_train_batch_size=train_args.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_args.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_args.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_args.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_args.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_args.get("warmup_ratio", 0.03),
        weight_decay=train_args.get("weight_decay", 0.01),
        gradient_checkpointing=train_args.get("gradient_checkpointing", True),
        logging_steps=train_args.get("logging_steps", 10),
        eval_strategy=train_args.get("eval_strategy", "steps") if val_dataset else "no",
        eval_steps=train_args.get("eval_steps", 100),
        save_strategy=train_args.get("save_strategy", "steps"),
        save_steps=train_args.get("save_steps", 100),
        save_total_limit=train_args.get("save_total_limit", 2),
        seed=seed,
        fp16=(torch_dtype == torch.float16 and device == "cuda"),
        bf16=(torch_dtype == torch.bfloat16 and device == "cuda"),
        max_seq_length=data_cfg.get("max_seq_length", 1024),
        report_to=train_args.get("report_to", "none"),
    )

    # 9. Initialize SFTTrainer
    # Passing processing_class=tokenizer ensures native chat template tokenization
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 10. Train
    print("Beginning model training...")
    train_result = trainer.train()
    print("Training complete! Metrics:", train_result.metrics)

    # 11. Save trained adapter and tokenizer
    adapter_dir = os.path.join(output_dir, "adapter")
    print(f"Saving fine-tuned LoRA adapter to: {adapter_dir}")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("LoRA Adapter and Tokenizer successfully persisted.")
    return adapter_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to experiment YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
