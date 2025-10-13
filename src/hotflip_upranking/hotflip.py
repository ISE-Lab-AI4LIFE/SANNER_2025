import torch
import pandas as pd
from tqdm import tqdm
import random
from src.hotflip_upranking.gradient import position_aware_beam_search, re_search
from src.hotflip_upranking.utils import doc_to_lines, load_model_and_tokenizer
from pathlib import Path
import sys
import json

# Thêm folder gốc của project vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# -------------------------
# Config
# -------------------------
MODEL_NAME = "facebook/contriever"   # HF model
DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                                       else "mps" if torch.backends.mps.is_available() 
                                       else "cpu")
PRINT_EVERY = 1

# Algorithm hyperparams
L_max = 12           # max inserted token length (Δ length, number of tokens)
BEAM_WIDTH = 4       # B
TOPK_TOKEN_PER_POS = 8  # k_b  (candidates per position)
N_ITER = 6           # main iterations
RESEARCH_ITER = 3    # re-search iterations (shorter)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def demo_documents_to_queries(file_path, output_path, test_mode=False):
    """
    Run HotFlip on documents and queries specified in a JSON mapping.
    Save results as {doc_id: modified_text} in JSON.

    file_path: path to documents_to_queries.json
    output_path: path to save JSON results
    """
    # Load document → queries mapping
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        doc_to_queries = json.load(f)

    # Normalize paths: replace 'embedding' -> 'processed' and '.npy' -> '.json'
    normalized_doc_to_queries = {}
    for doc_path, query_paths in doc_to_queries.items():
        # Normalize document path
        doc_path = doc_path.replace("embedding", "processed").replace(".npy", ".json")
        # Normalize query paths
        query_paths = [q.replace("embedding", "processed").replace(".npy", ".json") for q in query_paths]
        normalized_doc_to_queries[doc_path] = query_paths

    # Use normalized mapping
    doc_to_queries = normalized_doc_to_queries

    # Load model & tokenizer once
    model, tokenizer = load_model_and_tokenizer()

    results = {}

    # Process each document
    doc_items = list(doc_to_queries.items())
    if test_mode:
        doc_items = doc_items[:1]
    for doc_path, query_paths in tqdm(doc_items, desc="Processing documents"):
        print(f"Processing: {doc_path}")

        # Load document JSON
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_json = json.load(f)
        doc_id = doc_json.get("id", Path(doc_path).stem)
        doc_text = doc_json["text"]

        # Split document into lines
        lines = doc_to_lines(doc_text)
        num_lines = len(lines)

        # Load all query texts for this document
        queries_per_pos_texts = []
        for _ in range(num_lines):
            queries_at_pos = []
            for q_path in query_paths:
                with open(q_path, "r") as qf:
                    q_json = json.load(qf)
                    queries_at_pos.append(q_json["text"])
            queries_per_pos_texts.append(queries_at_pos)

        # HotFlip: position-aware beam search
        s_star, p_star, best_score, lines_out, flat_ids, line_end_offsets = position_aware_beam_search(
            model, tokenizer, doc_text, queries_per_pos_texts,
            L_max=L_max, beam_width=BEAM_WIDTH, topk_token_per_pos=TOPK_TOKEN_PER_POS, N_iter=N_ITER
        )

        print(f"\n=== Document {doc_id} ===")
        print(f"Initial best pos: {p_star}, score: {best_score:.4f}")

        # Re-search to refine
        s_star2, p_star2, best_score2 = re_search(
            model, tokenizer, doc_text, queries_per_pos_texts,
            original_seq_ids=s_star, original_pos_idx=p_star,
            L_max=L_max,
            beam_width=max(2, BEAM_WIDTH // 2),
            topk_token_per_pos=max(4, TOPK_TOKEN_PER_POS // 2),
            N_iter=RESEARCH_ITER
        )

        print(f"ReSearch best pos: {p_star2}, score: {best_score2:.4f}")

        # Convert token IDs to text
        try:
            seq_text = tokenizer.decode(s_star2, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        except Exception:
            seq_text = " ".join([str(x) for x in s_star2])

        # Save result for this document
        results[doc_id] = seq_text

    # Save all results to output JSON
    with open(output_path, "w") as outf:
        json.dump(results, outf, indent=2)

    print(f"\nProcessed {len(results)} documents. Results saved to {output_path}.")
    return results

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    input_mapping = "data/document_to_queries.json"
    output_file = "data/documents_hotflip_results.json"
    demo_documents_to_queries(input_mapping, output_file, test_mode=True)