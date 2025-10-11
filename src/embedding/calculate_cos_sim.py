import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from typing import List
from src.embedding.embedder import average_embedder  # 👈 import từ file embedder.py
import json
from tqdm import tqdm
import os
import glob


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

    # Đường dẫn thư mục chứa queries và documents
    base_dir = "/Users/hieunguyen/SANNER_2025/data/pool"
    query_dir = os.path.join(base_dir, "queries_pool")
    doc_path = os.path.join(base_dir, "document_pool.json")
    output_path = os.path.join(base_dir, "first_phase_score.json")

    # Đọc document pool
    with open(doc_path, "r", encoding="utf-8") as f:
        document_data = json.load(f)
    documents = document_data.get("document", [])
    print(f"Loaded {len(documents)} documents ✅")

    # Lấy danh sách file query
    query_files = sorted(glob.glob(os.path.join(query_dir, "queries_*.json")))
    print(f"Found {len(query_files)} query files ✅")

    all_results = []

    # Xử lý từng file query
    for qf in query_files:
        print(f"\n🔹 Processing file: {qf}")
        with open(qf, "r", encoding="utf-8") as f:
            query_data = json.load(f)
        queries = query_data.get("query", [])

        # Với mỗi query trong file
        for q in tqdm(queries, desc=f"Computing similarities for {os.path.basename(qf)}"):
            q_id = q["id"]
            q_text = q["text"]

            sim_scores = []
            # So sánh query với từng document
            for d in documents:
                d_id = d["id"]
                d_text = d["text"]

                score = compute_similarity(q_text, d_text, agg="mean")  # hoặc "max" nếu muốn lấy điểm cao nhất
                sim_scores.append({
                    "doc_id": d_id,
                    "similarity": score
                })

            # Lấy top 5 document có similarity cao nhất
            top5 = sorted(sim_scores, key=lambda x: x["similarity"], reverse=True)[:5]

            all_results.append({
                "query_id": q_id,
                "top_5_docs": top5
            })

    # Lưu kết quả ra file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Saved top-5 similarities for all queries to {output_path}")