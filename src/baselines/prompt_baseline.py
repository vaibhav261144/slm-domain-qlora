"""
Zero-shot and Few-shot prompting baselines runner.
Evaluates base models without fine-tuning across the test set under identical decoding parameters.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class PromptBaselineRunner:
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch_dtype or (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generates response using greedy decoding (temperature=0)."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = gen_tokens[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def run_eval(
        self,
        test_records: List[Dict[str, Any]],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
    ) -> List[Dict[str, Any]]:
        """
        Runs generation across all test records, returning predictions paired with references.
        """
        results = []
        for item in tqdm(test_records, desc="Evaluating Prompting Baseline"):
            msgs = item["messages"]
            user_msg = next((m for m in msgs if m["role"] == "user"), None)
            ref_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            sys_msg = next((m for m in msgs if m["role"] == "system"), None)

            eval_prompt = []
            if sys_msg:
                eval_prompt.append(sys_msg)

            if few_shot_examples:
                eval_prompt.extend(few_shot_examples)

            if user_msg:
                eval_prompt.append(user_msg)

            pred = self.generate_response(eval_prompt, max_new_tokens=max_new_tokens)
            results.append({
                "id": item.get("id"),
                "prompt": user_msg["content"] if user_msg else "",
                "prediction": pred,
                "ground_truth": ref_msg["content"] if ref_msg else "",
            })
        return results


def evaluate_prompting_baseline(
    model_id: str,
    test_records: List[Dict[str, Any]],
    few_shot: bool = False,
    few_shot_k: int = 3,
) -> List[Dict[str, Any]]:
    runner = PromptBaselineRunner(model_id)
    few_shot_examples = None
    if few_shot and len(test_records) > few_shot_k:
        # Note: in rigorous evaluation, few-shot examples should come strictly from the validation split
        few_shot_examples = []
        for ex in test_records[:few_shot_k]:
            u = next((m for m in ex["messages"] if m["role"] == "user"), None)
            a = next((m for m in ex["messages"] if m["role"] == "assistant"), None)
            if u and a:
                few_shot_examples.extend([u, a])
    return runner.run_eval(test_records, few_shot_examples=few_shot_examples)
