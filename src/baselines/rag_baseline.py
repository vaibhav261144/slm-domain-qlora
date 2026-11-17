"""
Retrieval-Augmented Generation (RAG) baseline implementation.
Uses sentence-transformers for embedding, FAISS for indexing, and local LLM for generation.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss


class SimpleRAGPipeline:
    def __init__(
        self,
        generator_model_id: str,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k

        # 1. Load Embedding Model
        print(f"Loading embedding model: {embedding_model_id}")
        self.embedder = SentenceTransformer(embedding_model_id, device=self.device)

        # 2. Load Generator LLM
        print(f"Loading generator model: {generator_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.generator = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            torch_dtype=dtype if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.generator.eval()

        self.documents: List[str] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def build_index(self, corpus: List[str]):
        """Embeds corpus documents and creates FAISS cosine similarity index."""
        self.documents = corpus
        embeddings = self.embedder.encode(
            corpus,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"FAISS index built successfully with {len(corpus)} documents (dim={dim}).")

    def retrieve(self, query: str) -> List[str]:
        """Retrieves top-k relevant documents for a query."""
        if self.index is None or not self.documents:
            return []
        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        k = min(self.top_k, len(self.documents))
        _, indices = self.index.search(query_emb.astype(np.float32), k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

    def query_with_rag(
        self,
        user_query: str,
        system_instruction: str = "Answer the user question factually using ONLY the provided context.",
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Runs end-to-end RAG retrieval + generation and profiles timing."""
        t0 = time.perf_counter()
        retrieved_docs = self.retrieve(user_query)
        retrieval_time = time.perf_counter() - t0

        context_str = "\n\n".join([f"Context [{i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])
        augmented_prompt = f"Reference Information:\n{context_str}\n\nQuestion: {user_query}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": augmented_prompt},
        ]

        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

        t1 = time.perf_counter()
        with torch.no_grad():
            gen_tokens = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generation_time = time.perf_counter() - t1
        total_time = time.perf_counter() - t0

        new_tokens = gen_tokens[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "query": user_query,
            "response": response,
            "retrieved_contexts": retrieved_docs,
            "latency": {
                "retrieval_sec": round(retrieval_time, 4),
                "generation_sec": round(generation_time, 4),
                "total_sec": round(total_time, 4),
            },
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
        }
