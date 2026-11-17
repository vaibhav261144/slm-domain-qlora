"""
Production FastAPI Serving Endpoint for the fine-tuned model.
Includes request validation, health checks, generation latency logging, and error handling.
"""

import os
import time
import torch
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(
    title="SLM LoRA/QLoRA Serving API",
    description="Inference API for domain-adapted Small Language Model",
    version="1.0.0",
)

# Global model state
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text")


class GenerationRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversational messages list")
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    response: str
    tokens_generated: int
    latency_ms: float
    model_id: str


@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER
    base_model_id = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = os.getenv("ADAPTER_PATH", None)

    print(f"Initializing model server on {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained(adapter_path or base_model_id)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    MODEL = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}")
        MODEL = PeftModel.from_pretrained(MODEL, adapter_path)

    MODEL.eval()
    print("Model successfully loaded and ready for inference.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL is not None,
    }


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    t0 = time.perf_counter()
    raw_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        prompt_text = TOKENIZER.apply_chat_template(
            raw_msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template: {e}")

    inputs = TOKENIZER(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=(req.temperature > 0.0),
            temperature=req.temperature if req.temperature > 0.0 else None,
            top_p=req.top_p if req.temperature > 0.0 else None,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )

    t1 = time.perf_counter()
    new_tokens = out[0][inputs.input_ids.shape[1] :]
    answer = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

    return GenerationResponse(
        response=answer,
        tokens_generated=int(new_tokens.shape[0]),
        latency_ms=round((t1 - t0) * 1000.0, 2),
        model_id=os.getenv("ADAPTER_PATH", os.getenv("BASE_MODEL_ID", "local")),
    )
