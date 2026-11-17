"""
Data preprocessing, deduplication, and token-length profiling utilities.
"""

import hashlib
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_message_hash(messages: List[Dict[str, str]]) -> str:
    """Computes a SHA-256 fingerprint for a list of messages for deduplication."""
    serialized = "".join(
        f"{m.get('role', '')}:{m.get('content', '').strip().lower()}"
        for m in messages
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def deduplicate_dataset(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicates a list of chat records based on normalized message content.
    Returns (deduplicated_records, number_of_duplicates_removed).
    """
    seen_hashes = set()
    unique_records = []
    duplicates_removed = 0

    for rec in records:
        h = compute_message_hash(rec.get("messages", []))
        if h in seen_hashes:
            duplicates_removed += 1
        else:
            seen_hashes.add(h)
            unique_records.append(rec)

    return unique_records, duplicates_removed


def compute_token_length_stats(
    records: List[Dict[str, Any]],
    tokenizer: Any
) -> Dict[str, float]:
    """
    Calculates token length statistics (mean, std, median, 95th percentile, max)
    using the base model's chat template or tokenizer.
    Helps set max_seq_length correctly (as specified in report step 7/10).
    """
    lengths = []
    for rec in records:
        messages = rec.get("messages", [])
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer.encode(formatted_text)
            except Exception:
                text = " ".join([m.get("content", "") for m in messages])
                tokens = tokenizer.encode(text)
        else:
            text = " ".join([m.get("content", "") for m in messages])
            tokens = tokenizer.encode(text)
        lengths.append(len(tokens))

    arr = np.array(lengths)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
    }
