# Research Paper Manuscript Blueprint (arXiv Ready)

**Title:** *Empirical Trade-offs in Fine-Tuning Small Language Models with LoRA and QLoRA versus Prompting and Retrieval-Augmented Generation*

## 1. Abstract
- Context: Growth of Small Language Models (SLMs, 0.5B–8B) as efficient alternatives to frontier models.
- Problem: Limited empirical comparison across fine-tuning (LoRA/QLoRA), RAG, and prompting under fixed compute budgets.
- Methodology: Comprehensive evaluation across model sizes (1B, 3B, 7B), rank ablations ($r \in \{4, 8, 16, 32, 64\}$), LoRA vs QLoRA, and latency/cost benchmarks.
- Key Results: 1B/3B QLoRA model matches or surpasses few-shot prompted 8B models while cutting inference latency by X% and memory footprint by Y%.

## 2. Introduction
- Background: Why enterprise and edge deployments need small, private, low-latency models.
- Research Gap: Lack of systematic study comparing parameter-efficient fine-tuning with RAG under constrained memory (8–16 GB VRAM).
- Concrete Contributions:
  1. A curated, validated open domain benchmark dataset with complete datasheet.
  2. Rigorous empirical comparison of LoRA vs QLoRA vs RAG vs Prompting.
  3. Ablation analysis on rank scaling, target modules, and base vs. instruct initialization.
  4. Cost, latency, and general capability retention (forgetting) characterization.

## 3. Related Work
- Parameter-Efficient Fine-Tuning (LoRA: Hu et al., 2021; QLoRA: Dettmers et al., 2023; DoRA: Liu et al., 2024).
- Fine-Tuning vs Retrieval-Augmented Generation (Ovadia et al., 2023).
- Small Language Model scaling (Llama 3.2, Qwen 2.5, Gemma 2, SmolLM).

## 4. Task Specification & Dataset Curation
- Task Definition: Input representation, output requirements, and evaluation metrics.
- Data Collection: Provenance, licensing, and ethical filtering.
- Data Processing: Deduplication, chat template normalization, leak-free document-level splitting (80/10/10).

## 5. Experimental Setup & Methodology
- Models evaluated: 1B, 3B, and 7B variants.
- LoRA/QLoRA configuration: Rank $r$, $\alpha=2r$, `all-linear` projection, completion-only loss masking.
- Baselines:
  - Zero-shot prompting (base & instruct)
  - 3-shot / 5-shot in-context learning
  - Dense RAG pipeline (FAISS + sentence-transformers)
  - Large frontier model ceiling reference
- Statistical rigor: 3 random seeds per run, 95% bootstrap confidence intervals, paired significance testing.

## 6. Results and Empirical Analysis
- Primary Task Performance (EM, F1, JSON Accuracy).
- LoRA (16-bit) vs QLoRA (4-bit): Accuracy parity and memory savings.
- Rank Ablations: Diminishing returns of rank $r > 16$.
- Learning Curves: Performance vs dataset size ($N = 250, 500, 1k, 2k, 5k, \text{Full}$).

## 7. Operational Efficiency: Latency, Memory & Cost
- Throughput (tokens/sec) and p50/p95 response latency.
- Peak VRAM footprint during training and inference.
- Cost per 1,000 queries compared to API-based large models.

## 8. Capability Retention & Error Analysis
- Evaluation on general NLP benchmarks (MMLU / GSM8k) to quantify catastrophic forgetting.
- Qualitative taxonomy of failure modes (formatting errors, hallucination, out-of-distribution drift).

## 9. Conclusion & Limitations
- Summary of findings and recommendations for practitioners.
- Limitations: Dataset size constraints, compute hardware boundaries.
