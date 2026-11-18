"""
Interactive Gradio Web Application for Hugging Face Spaces or local demo.
Compares base zero-shot model against the LoRA fine-tuned adapter side-by-side.
"""

import os
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "outputs/run1/adapter")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer: {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=dtype if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
base_model.eval()

tuned_model = None
if os.path.exists(ADAPTER_PATH):
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tuned_model.eval()
else:
    print(f"Adapter path '{ADAPTER_PATH}' not found. Demo will run base model only until adapter is trained.")


def generate_single(model, prompt_text: str, system_msg: str, max_tokens: int, temp: float):
    messages = []
    if system_msg.strip():
        messages.append({"role": "system", "content": system_msg.strip()})
    messages.append({"role": "user", "content": prompt_text.strip()})

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temp > 0.0),
            temperature=temp if temp > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - t0
    new_tokens = out[0][inputs.input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, f"{latency:.2f}s ({len(new_tokens)} tokens)"


def compare_models(user_prompt: str, system_prompt: str, max_new_tokens: int, temperature: float):
    if not user_prompt.strip():
        return "Please enter a prompt.", "", "Please enter a prompt.", ""

    # Generate base model response
    base_out, base_meta = generate_single(base_model, user_prompt, system_prompt, max_new_tokens, temperature)

    # Generate tuned model response
    if tuned_model is not None:
        tuned_out, tuned_meta = generate_single(tuned_model, user_prompt, system_prompt, max_new_tokens, temperature)
    else:
        tuned_out = "[Adapter not yet mounted/trained. Train a LoRA adapter first to view fine-tuned response.]"
        tuned_meta = "N/A"

    return base_out, base_meta, tuned_out, tuned_meta


demo_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(theme=demo_theme, title="Small Language Model LoRA/QLoRA Demo") as demo:
    gr.Markdown("# 🔬 Fine-Tuning Small Language Models with LoRA / QLoRA")
    gr.Markdown(
        "Evaluate the performance, response quality, and latency trade-offs between "
        "an out-of-the-box base SLM and a domain-adapted LoRA/QLoRA model."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="User Prompt / Task Input",
                placeholder="Enter query, clinical text, or invoice to extract...",
                lines=4,
            )
            system_input = gr.Textbox(
                label="System Prompt (Optional)",
                placeholder="You are an expert domain specialist...",
                lines=2,
            )
            with gr.Row():
                tokens_slider = gr.Slider(minimum=32, maximum=1024, value=256, step=32, label="Max New Tokens")
                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature (0=Greedy)")
            submit_btn = gr.Button("Compare Models", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏛️ Base Model (Zero-Shot / Prompted)")
            base_output = gr.Textbox(label="Output", lines=8)
            base_stat = gr.Label(label="Latency & Token Count")

        with gr.Column():
            gr.Markdown("### ⚡ Fine-Tuned Model (LoRA / QLoRA)")
            tuned_output = gr.Textbox(label="Output", lines=8)
            tuned_stat = gr.Label(label="Latency & Token Count")

    submit_btn.click(
        fn=compare_models,
        inputs=[prompt_input, system_input, tokens_slider, temp_slider],
        outputs=[base_output, base_stat, tuned_output, tuned_stat],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
