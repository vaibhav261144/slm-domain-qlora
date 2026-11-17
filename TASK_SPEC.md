# Task Specification: Bilingual Domain QA (English + Hindi)

## 1. Overview & Research Objectives
- **Task Domain:** Public Service & Agriculture Advisory (Government Schemes, Crop Management, Soil Health).
- **Languages:** English (`en`) and Hindi (`hi`).
- **Target Model Family:** Small Language Models (SLMs, ~1.5B–3B parameters), e.g., `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`.
- **Primary Research Questions:**
  1. **RQ1:** Can a QLoRA fine-tuned 1.5B SLM match or outperform an 8B zero-shot prompted model and a Dense RAG pipeline on factual domain accuracy and bilingual fluency?
  2. **RQ2:** How does token economy and tokenizer efficiency differ between English and Indic scripts (Devanagari), and does QLoRA suffer degraded Indic generation without vocabulary adaptation?
  3. **RQ3:** What is the latency and VRAM trade-off: LoRA/QLoRA (parametric memory) vs. RAG (retrieval + long prompt processing)?

## 2. Input / Output Format
Standard chat template format:
```json
{
  "id": "agri_hi_0104",
  "domain": "agriculture",
  "language": "hi",
  "document_id": "doc_wheat_cultivation_2024",
  "messages": [
    {
      "role": "system",
      "content": "आप एक विशेषज्ञ कृषि सलाहकार हैं जो किसानों को फसल प्रबंधन और कीट नियंत्रण पर सटीक और व्यावहारिक समाधान प्रदान करते हैं।"
    },
    {
      "role": "user",
      "content": "गेहूं की फसल में दीमक (termite) नियंत्रण के लिए कौन सी दवा और कितनी मात्रा का प्रयोग करना चाहिए?"
    },
    {
      "role": "assistant",
      "content": "गेहूं में दीमक के प्रकोप से बचाव हेतु बुवाई से पहले बीज उपचार करें। इसके लिए क्लोरपायरीफॉस 20% ई.सी. की 4-5 मिलीलीटर प्रति किलोग्राम बीज की दर से उपचार करें। खड़ी फसल में प्रकोप होने पर सिंचाई जल के साथ क्लोरपायरीफॉस 20% ई.सी. की 1 से 1.5 लीटर मात्रा प्रति एकड़ की दर से प्रयोग करें।"
    }
  ]
}
```

## 3. Evaluation Metrics & Success Thresholds
1. **Primary Factual Metrics:**
   - **Token F1:** $\ge 0.70$ on English; $\ge 0.65$ on Hindi.
   - **Exact Match (EM):** Tracked for entity and numeric matches (dosages, scheme amounts).
   - **ROUGE-L & sacreBLEU:** Measured across both language subsets.
2. **Operational Constraints:**
   - **Peak VRAM:** $< 12 \text{ GB}$ during training (compatible with free Google Colab T4).
   - **p50 Latency:** $\le 800 \text{ ms}$ for 128-token generation.
   - **Cost Target:** $> 10\times$ cheaper per query than proprietary API baselines.
3. **Statistical Significance:**
   - 3 training seeds ($42, 100, 2024$).
   - 95% bootstrap confidence intervals reported on all final metrics.

## 4. Leakage Prevention Strategy
Splitting is strictly enforced by **`document_id`** (`group_key="document_id"` in `src/data/splitter.py`). No question-answer pairs derived from the same agricultural handbook or policy document may appear across both train and test partitions.
