"""
Model Evaluator runner.
Evaluates base models or LoRA adapters on frozen test split, computes metrics,
and saves reports to results/metrics/.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_json_metrics,
    bootstrap_confidence_interval,
)


class ModelEvaluator:
    def __init__(
        self,
        base_model_id: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path

        print(f"Loading tokenizer: {base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")) else base_model_id
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model: {base_model_id}")
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if adapter_path:
            print(f"Applying LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = gen_tokens[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate_test_set(
        self,
        test_records: List[Dict[str, Any]],
        task_type: str = "qa",
        max_new_tokens: int = 256,
        output_report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs greedy generation and evaluates against references.
        """
        predictions = []
        ground_truths = []
        em_scores = []
        f1_scores = []
        items_detail = []

        for item in tqdm(test_records, desc="Evaluating test set"):
            msgs = item["messages"]
            prompt_msgs = [m for m in msgs if m["role"] in ("system", "user")]
            ref_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            gold_str = ref_msg["content"].strip() if ref_msg else ""

            pred_str = self.generate(prompt_msgs, max_new_tokens=max_new_tokens)

            predictions.append(pred_str)
            ground_truths.append(gold_str)

            em = compute_exact_match(pred_str, gold_str)
            f1_dict = compute_token_f1(pred_str, gold_str)

            em_scores.append(em)
            f1_scores.append(f1_dict["f1"])

            items_detail.append({
                "id": item.get("id"),
                "prediction": pred_str,
                "ground_truth": gold_str,
                "em": em,
                "token_f1": f1_dict["f1"],
            })

        em_mean, em_low, em_high = bootstrap_confidence_interval(em_scores)
        f1_mean, f1_low, f1_high = bootstrap_confidence_interval(f1_scores)

        summary = {
            "model": self.adapter_path or self.base_model_id,
            "num_test_samples": len(test_records),
            "exact_match": {"mean": em_mean, "ci_95": [em_low, em_high]},
            "token_f1": {"mean": f1_mean, "ci_95": [f1_low, f1_high]},
        }

        if task_type == "extraction" or "{" in ground_truths[0]:
            json_metrics = compute_json_metrics(predictions, ground_truths)
            summary["json_metrics"] = json_metrics

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": items_detail}, f, indent=2, ensure_ascii=False)
            print(f"Saved evaluation report to: {output_report_path}")

        return summary
