"""
Unit tests for data loader, schema validator, and splitter.
"""

import pytest
from src.data.dataset_loader import validate_chat_schema
from src.data.preprocessor import deduplicate_dataset
from src.data.splitter import split_dataset


def test_validate_chat_schema_valid():
    entry = {
        "messages": [
            {"role": "user", "content": "What is LoRA?"},
            {"role": "assistant", "content": "Low-Rank Adaptation is a PEFT technique."}
        ]
    }
    assert validate_chat_schema(entry) is True


def test_validate_chat_schema_missing_assistant():
    entry = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is LoRA?"},
        ]
    }
    with pytest.raises(ValueError, match="at least one 'user' and one 'assistant'"):
        validate_chat_schema(entry)


def test_deduplicate_dataset():
    records = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}, # duplicate
        {"messages": [{"role": "user", "content": "Different"}, {"role": "assistant", "content": "Response."}]}
    ]
    unique, removed = deduplicate_dataset(records)
    assert len(unique) == 2
    assert removed == 1


def test_split_dataset_ratios():
    records = [{"id": i, "messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(100)]
    train, val, test = split_dataset(records, 0.8, 0.1, 0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_dataset_grouped_no_leakage():
    records = [
        {"doc_id": "doc_A", "messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "1"}]},
        {"doc_id": "doc_A", "messages": [{"role": "user", "content": "2"}, {"role": "assistant", "content": "2"}]},
        {"doc_id": "doc_B", "messages": [{"role": "user", "content": "3"}, {"role": "assistant", "content": "3"}]},
        {"doc_id": "doc_C", "messages": [{"role": "user", "content": "4"}, {"role": "assistant", "content": "4"}]},
    ]
    train, val, test = split_dataset(records, 0.5, 0.25, 0.25, group_key="doc_id", seed=42)
    train_docs = {r["doc_id"] for r in train}
    val_docs = {r["doc_id"] for r in val}
    test_docs = {r["doc_id"] for r in test}

    # Ensure zero overlap between splits
    assert len(train_docs.intersection(val_docs)) == 0
    assert len(train_docs.intersection(test_docs)) == 0
    assert len(val_docs.intersection(test_docs)) == 0
