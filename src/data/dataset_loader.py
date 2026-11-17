"""
Dataset loading and schema validation module.
Supports JSONL files with standard conversational schema:
{"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from datasets import Dataset


def validate_chat_schema(entry: Dict[str, Any], entry_idx: int = 0) -> bool:
    """
    Validates whether an entry conforms to the messages format.
    Raises ValueError with descriptive message if invalid.
    """
    if "messages" not in entry:
        raise ValueError(
            f"Entry {entry_idx} is missing the required 'messages' key. Found keys: {list(entry.keys())}"
        )
    messages = entry["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(
            f"Entry {entry_idx}: 'messages' must be a list containing at least a user and assistant message."
        )

    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(
            f"Entry {entry_idx}: 'messages' must contain at least one 'user' and one 'assistant' message."
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Entry {entry_idx}, message {i}: must be a dictionary.")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Entry {entry_idx}, message {i}: missing 'role' or 'content'.")
        if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
            raise ValueError(f"Entry {entry_idx}, message {i}: 'content' must be a non-empty string.")

    return True


def load_jsonl_dataset(
    file_path: Union[str, Path],
    validate: bool = True
) -> Dataset:
    """
    Loads a JSONL dataset into a Hugging Face Dataset object after validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {idx + 1} of {path}: {e}")

            if validate:
                validate_chat_schema(data, entry_idx=idx + 1)
            records.append(data)

    if not records:
        raise ValueError(f"No records found in dataset file: {path}")

    return Dataset.from_list(records)
