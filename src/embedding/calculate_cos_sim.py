import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from typing import List
from src.embedding.embedder import average_embedder  # 👈 import từ file embedder.py


def chunk_by_tokens(tokenizer, text: str, max_tokens: int = 512, stride: int = None) -> List[str]:
    """Tách văn bản thành nhiều chunk có độ dài ≤ max_tokens"""
    encoding = tokenizer(text, return_attention_mask=False, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    total = len(input_ids)
    if stride is None:
        stride = max_tokens
    chunks = []
    i = 0
    while i < total:
        sub = input_ids[i: i + max_tokens]
        chunk_text = tokenizer.decode(sub, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        i += stride
    return chunks


def compute_similarity(query: str, document: str, model_name: str = "intfloat/multilingual-e5-large", agg: str = "mean") -> float:
    """
    Tính cosine similarity giữa query và document dài.
    - Dùng chiến lược chunking với max_token=512
    - average_embedder() dùng để sinh embedding (import từ embedder.py)
    - agg = 'mean' hoặc 'max' để tổng hợp điểm
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Embed query ---
    query_emb = average_embedder(query)
    query_emb = torch.tensor(query_emb, dtype=torch.float32).to(device)

    # --- Chunk document ---
    chunks = chunk_by_tokens(tokenizer, document, max_tokens=512)
    sims = []

    for chunk in chunks:
        doc_emb = average_embedder(chunk)
        doc_emb = torch.tensor(doc_emb, dtype=torch.float32).to(device)

        # tính cosine similarity
        sim = F.cosine_similarity(query_emb, doc_emb, dim=0).item()
        sims.append(sim)

    if not sims:
        return 0.0

    if agg == "mean":
        return float(np.mean(sims))
    elif agg == "max":
        return float(np.max(sims))
    else:
        raise ValueError("agg must be 'mean' or 'max'")


# ====== Ví dụ sử dụng ======
if __name__ == "__main__":
    query = "What is artificial intelligence?"
    document = """Artificial intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks 
    that normally require human intelligence. These tasks include reasoning, learning, perception, and language understanding."""

    similarity = compute_similarity(query, document, agg="mean")
    print(f"Cosine similarity (mean over chunks): {similarity:.4f}")