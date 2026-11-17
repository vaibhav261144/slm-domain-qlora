"""
Comprehensive evaluation metrics:
- Exact Match (EM)
- Token-level F1 / Precision / Recall
- JSON validity & Field-level F1
- ROUGE-L and BLEU
- Bootstrap confidence interval estimation
"""

import json
import re
import string
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np


def normalize_text(s: str) -> str:
    """Lowercases text, strips punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculates whether normalized prediction matches ground truth exactly (0.0 or 1.0)."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Calculates token-level precision, recall, and F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        score = 1.0 if pred_tokens == gold_tokens else 0.0
        return {"precision": score, "recall": score, "f1": score}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extract_json_object(text: str) -> Tuple[bool, Any]:
    """Attempts to parse JSON from direct text or markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = match.group(1).strip() if match else text

    # Try direct parse
    try:
        data = json.loads(candidate)
        return True, data
    except Exception:
        # Fallback regex search for { ... }
        brace_match = re.search(r"\{[\s\S]*\}", candidate)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return True, data
            except Exception:
                pass
        return False, None


def compute_json_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Computes JSON schema validity rate and field-level key-value match accuracy.
    """
    valid_count = 0
    field_matches = 0
    total_fields = 0

    for pred, gold in zip(predictions, ground_truths):
        gold_valid, gold_obj = extract_json_object(gold)
        pred_valid, pred_obj = extract_json_object(pred)

        if pred_valid:
            valid_count += 1

        if gold_valid and pred_valid and isinstance(gold_obj, dict) and isinstance(pred_obj, dict):
            for k, v in gold_obj.items():
                total_fields += 1
                if k in pred_obj:
                    if pred_obj[k] == v:
                        field_matches += 1
                    elif str(pred_obj[k]).strip().lower() == str(v).strip().lower():
                        field_matches += 1
        elif gold_valid and isinstance(gold_obj, dict):
            total_fields += len(gold_obj)

    n = len(predictions) if len(predictions) > 0 else 1
    validity_rate = valid_count / n
    field_acc = (field_matches / total_fields) if total_fields > 0 else 0.0

    return {
        "json_validity_rate": round(validity_rate, 4),
        "field_accuracy": round(field_acc, 4),
        "total_fields_evaluated": total_fields,
    }


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes mean and [lower_ci, upper_ci] using non-parametric bootstrap resampling.
    """
    if not scores:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    boot_means = []

    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean = float(np.mean(arr))

    return round(mean, 4), round(lower, 4), round(upper, 4)
