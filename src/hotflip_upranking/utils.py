import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/contriever"   # HF model
DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                                       else "mps" if torch.backends.mps.is_available() 
                                       else "cpu")
# -------------------------
# Utilities
# -------------------------
def load_model_and_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    # model_output[0] is last_hidden_state: (batch, seq_len, dim)
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def embed_texts(texts, model, tokenizer, batch_size=16):
    """
    Return normalized embeddings (tensor, shape (len(texts), D))
    """
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])  # (batch_size, D)
            pooled = F.normalize(pooled, p=2, dim=-1)
            all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0).to(DEVICE)

# -------------------------
# Document / Positions handling
# -------------------------
def doc_to_lines(text):
    # Keep original line breaks. Return list of lines (str).
    # If no newline, treat whole doc as single line.
    lines = text.splitlines()
    if len(lines) == 0:
        return [text]
    return lines

def build_flat_token_ids_and_offsets(lines, tokenizer):
    """
    Tokenize each line separately, return:
    - flat_ids: list of token ids (flattened)
    - line_end_offsets: list of insertion offsets (index in flat_ids where insertion after that line occurs)
    - line_token_lists: list of token lists per line
    """
    line_token_lists = []
    flat = []
    offsets = []
    for line in lines:
        enc = tokenizer(line, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0).tolist()
        line_token_lists.append(ids)
        flat.extend(ids)
        offsets.append(len(flat))  # insertion at this index (after this line)
    return flat, offsets, line_token_lists

def insert_tokens_into_flat(flat_ids, insert_ids, insert_pos_idx, line_end_offsets):
    pos = line_end_offsets[insert_pos_idx]
    new_flat = flat_ids[:pos] + insert_ids + flat_ids[pos:]
    return new_flat