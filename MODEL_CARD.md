---
language:
- en
- hi
license: apache-2.0
tags:
- peft
- lora
- qlora
- small-language-models
- text-generation
- slm
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# Model Card: Domain-Adapted Small Language Model (LoRA/QLoRA)

## Model Details
- **Model Description:** A parameter-efficient fine-tuned Small Language Model (SLM) adapted to domain-specific instructions and structured responses using Low-Rank Adaptation (LoRA / QLoRA).
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (or `Llama-3.2-1B-Instruct` / `Llama-3.2-3B-Instruct`)
- **Adapter Type:** PEFT LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit NormalFloat (NF4) with double quantization during training

## Intended Use
- **Primary Use Case:** High-throughput, low-latency domain task execution (domain question answering, structured entity extraction, schema adherence) on edge devices or single-GPU servers.
- **Out-of-Scope:** Open-domain freeform chat outside the adapted domain without grounding.

## Training Hyperparameters
- **Rank ($r$):** 16
- **Scaling Factor ($\alpha$):** 32
- **Dropout:** 0.05
- **Target Modules:** `all-linear` (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
- **Optimizer:** `paged_adamw_8bit`
- **Learning Rate:** `2e-4` with cosine learning rate schedule
- **Precision:** bfloat16 / float16 compute with NF4 4-bit weights
- **Loss Masking:** Completion-only (prompt tokens masked from gradient calculation)

## Evaluation Results
| Model Configuration | Exact Match (%) | Token F1 | JSON Validity (%) | p50 Latency (ms) | Peak VRAM (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base Model (0-shot) | - | - | - | - | - |
| Base Model (Few-shot) | - | - | - | - | - |
| RAG Baseline | - | - | - | - | - |
| **SLM + QLoRA (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

## Limitations and Bias
- The adapter is specialized for the target domain and may exhibit slight degradation on out-of-domain conversational benchmarks (catastrophic forgetting).
- Factuality is constrained to domain training boundaries.
