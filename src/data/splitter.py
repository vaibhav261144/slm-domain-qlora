"""
Dataset splitting utility with leakage prevention.
Supports grouping by document/source ID or random seeded splits.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    group_key: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records into train, validation, and test sets.
    If group_key is provided (e.g., 'document_id', 'source'), all records sharing
    the same group value will be assigned to the same partition to prevent leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    if group_key is not None:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for rec in records:
            g = rec.get(group_key, "unknown")
            groups.setdefault(g, []).append(rec)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n_total = len(records)
        train_target = int(n_total * train_ratio)
        val_target = int(n_total * val_ratio)

        train_records, val_records, test_records = [], [], []
        current_train_count = 0
        current_val_count = 0

        for gk in group_keys:
            items = groups[gk]
            if current_train_count < train_target:
                train_records.extend(items)
                current_train_count += len(items)
            elif current_val_count < val_target:
                val_records.extend(items)
                current_val_count += len(items)
            else:
                test_records.extend(items)
    else:
        shuffled = list(records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_records = shuffled[:n_train]
        val_records = shuffled[n_train : n_train + n_val]
        test_records = shuffled[n_train + n_val :]

    return train_records, val_records, test_records


def save_splits_to_disk(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "data/splits"
) -> Dict[str, Path]:
    """Saves splits to JSONL files in the target directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_path / "train.jsonl",
        "val": out_path / "val.jsonl",
        "test": out_path / "test.jsonl"
    }

    for split_name, file_path in paths.items():
        data = locals()[split_name]
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return paths
