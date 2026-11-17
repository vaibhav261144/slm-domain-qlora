"""Baselines module: Zero-shot, Few-shot prompting, and RAG."""
from .prompt_baseline import evaluate_prompting_baseline
from .rag_baseline import SimpleRAGPipeline

__all__ = ["evaluate_prompting_baseline", "SimpleRAGPipeline"]
