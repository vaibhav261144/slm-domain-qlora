"""
Unit tests for evaluation metrics.
"""

from src.evaluation.metrics import (
    compute_exact_match,
    compute_token_f1,
    extract_json_object,
    compute_json_metrics,
    bootstrap_confidence_interval,
)


def test_exact_match():
    assert compute_exact_match("Hello World!", "hello world") == 1.0
    assert compute_exact_match("Apples", "Oranges") == 0.0


def test_token_f1():
    res = compute_token_f1("the quick brown fox", "quick brown fox jumps")
    assert res["f1"] > 0.5
    assert res["recall"] < 1.0


def test_extract_json():
    # Direct json
    ok, obj = extract_json_object('{"a": 1, "b": 2}')
    assert ok is True
    assert obj == {"a": 1, "b": 2}

    # Markdown json
    markdown = '```json\n{"status": "ok"}\n```'
    ok, obj = extract_json_object(markdown)
    assert ok is True
    assert obj["status"] == "ok"


def test_compute_json_metrics():
    preds = ['{"name": "Alice", "age": 30}']
    golds = ['{"name": "Alice", "age": 30}']
    metrics = compute_json_metrics(preds, golds)
    assert metrics["json_validity_rate"] == 1.0
    assert metrics["field_accuracy"] == 1.0


def test_bootstrap_confidence_interval():
    scores = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    mean, low, high = bootstrap_confidence_interval(scores, n_bootstraps=500)
    assert low <= mean <= high
