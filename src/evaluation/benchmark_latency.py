"""
Inference latency, throughput, and memory benchmarking tool.
Measures tokens/sec, p50, p95, p99 latency, and GPU VRAM usage.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def benchmark_inference_efficiency(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    test_prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    warmup_runs: int = 3,
    num_runs: int = 20,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if not test_prompts:
        test_prompts = [
            "Explain the difference between parameter-efficient fine-tuning and full model fine-tuning.",
            "What are the key eligibility guidelines for smallholder agricultural subsidy programs?",
            "Extract entities: Customer John Doe purchased 3 units of widget A for 15 dollars each on Oct 12.",
        ]

    # Reset peak memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup runs
    for i in range(warmup_runs):
        p = test_prompts[i % len(test_prompts)]
        inp = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=16, do_sample=False)

    latencies = []
    generated_token_counts = []

    for i in range(num_runs):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        num_gen = out.shape[1] - inputs.input_ids.shape[1]
        latencies.append(t1 - t0)
        generated_token_counts.append(num_gen)

    lat_arr = np.array(latencies)
    tok_arr = np.array(generated_token_counts)
    total_tokens = np.sum(tok_arr)
    total_time = np.sum(lat_arr)

    peak_vram_gb = 0.0
    if device == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "model": adapter_path or base_model_id,
        "device": device,
        "runs": num_runs,
        "total_tokens_generated": int(total_tokens),
        "tokens_per_second": round(float(total_tokens / total_time), 2),
        "latency_sec": {
            "mean": round(float(np.mean(lat_arr)), 4),
            "p50": round(float(np.percentile(lat_arr, 50)), 4),
            "p95": round(float(np.percentile(lat_arr, 95)), 4),
            "p99": round(float(np.percentile(lat_arr, 99)), 4),
        },
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
