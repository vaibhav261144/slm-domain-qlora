"""
LoRA Adapter Merging Tool.
Merges trained LoRA adapter weights into a 16-bit copy of the base model
using `merge_and_unload()` to produce a standalone model ready for deployment,
GGUF quantization, or vLLM serving.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_standalone(
    base_model_id: str,
    adapter_dir: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    print(f"Loading 16-bit base model: {base_model_id}")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Executing merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")) else base_model_id
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged standalone model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("Merged model and tokenizer successfully saved!")

    if push_to_hub and hub_model_id:
        print(f"Uploading merged model to Hugging Face Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("Upload complete!")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA Adapter with Base Model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model Hugging Face ID")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to trained LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, default="outputs/merged_model", help="Path to save merged weights")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged weights to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Target Hugging Face Hub repo ID")
    args = parser.parse_args()

    merge_lora_to_standalone(
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
