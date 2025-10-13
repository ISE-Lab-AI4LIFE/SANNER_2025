import torch
import torch.nn.functional as F
from src.hotflip_upranking.utils import mean_pooling, embed_texts, doc_to_lines, build_flat_token_ids_and_offsets, insert_tokens_into_flat


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

def gradient_propose_tokens(
                            model, # embedding model
                            tokenizer,
                            flat_ids, 
                            line_end_offsets,
                            insert_pos_idx, 
                            current_seq_ids, 
                            queries_embs, 
                            topk=TOPK_TOKEN_PER_POS
                        ):
    """
    Given current inserted token ids (length L_max), compute gradient of objective
    (mean cosine between embedding(doc ⊕ seq) and queries_embs) w.r.t inserted embeddings.
    Then propose top-k token ids per position whose input embeddings align with gradient vectors.

    Returns: T_per_j: list (length L) of lists of token ids (len topk each)
    """
    model = model.to(DEVICE)
    emb_layer = model.get_input_embeddings()  # nn.Embedding
    token_emb_matrix = emb_layer.weight.detach()  # (V, D)
    token_emb_matrix = F.normalize(token_emb_matrix, p=2, dim=-1).to(DEVICE)

    L = len(current_seq_ids)
    # Build flat_with_insert ids
    flat_with = insert_tokens_into_flat(flat_ids, current_seq_ids, insert_pos_idx, line_end_offsets)
    # Tokenize full input to get attention mask and allow model to accept input_embeds
    # Instead of re-tokenizing (we have ids), we will build input_embeds from emb_layer
    # Get embeddings for full_sequence; for inserted positions we will create a requires_grad tensor

    # Compute start index in flat_with where inserted tokens begin (in token positions)
    insert_start = line_end_offsets[insert_pos_idx]
    insert_len = L
    assert len(flat_with) >= insert_start + insert_len

    # Build embeddings for full sequence
    # For non-inserted tokens, fetch from embedding layer (no grad)
    with torch.no_grad():
        full_ids_tensor = torch.tensor(flat_with, dtype=torch.long, device=DEVICE)
        full_embs = emb_layer(full_ids_tensor)  # (T, D)

    # Split into before / inserted / after
    before_embs = full_embs[:insert_start].detach()
    after_embs = full_embs[insert_start + insert_len:].detach()
    # Inserted: create tensor from current_seq_ids but requires_grad=True
    inserted_ids_tensor = torch.tensor(current_seq_ids, dtype=torch.long, device=DEVICE)
    inserted_embs = emb_layer(inserted_ids_tensor).clone().detach().requires_grad_(True)  # (L, D)

    # Concatenate to form input_embeds
    input_embeds = torch.cat([before_embs, inserted_embs, after_embs], dim=0).unsqueeze(0)  # (1, T, D)

    # Build attention mask (all ones, because we provide embeddings directly)
    attn_mask = torch.ones((1, input_embeds.size(1)), dtype=torch.long, device=DEVICE)

    # Forward pass using input_embeds; enable grad only for inserted_embs (already requires_grad)
    out = model(inputs_embeds=input_embeds, attention_mask=attn_mask)
    doc_emb = mean_pooling(out, attn_mask).squeeze(0)  # (D,)
    # Objective: mean cosine similarity to queries_embs (queries_embs: (M, D))
    with torch.no_grad():
        # queries_embs already normalized
        pass
    cosines = F.cosine_similarity(doc_emb.unsqueeze(0), queries_embs, dim=-1)  # (M,)
    obj = cosines.mean()  # scalar

    # Backprop to inserted_embs
    model.zero_grad()
    if inserted_embs.grad is not None:
        inserted_embs.grad.zero_()
    obj.backward(retain_graph=False)

    grads = inserted_embs.grad.detach()  # (L, D)
    grads_norm = F.normalize(grads, p=2, dim=-1)  # (L, D)
    token_embs_norm = token_emb_matrix  # (V, D)

    # Dot product to score tokens: (L, V) = grads_norm @ token_embs_norm.T
    # May be memory heavy if V large; still do it here.
    dots = torch.matmul(grads_norm, token_embs_norm.t())  # (L, V)

    # For each position get topk token ids
    topk_vals, topk_idx = torch.topk(dots, k=topk, dim=-1)
    topk_idx = topk_idx.cpu().tolist()  # list of lists
    T_per_j = [row for row in topk_idx]  # each row is list of token ids
    return T_per_j

# -------------------------
# Evaluate a sequence at a given position -> score
# -------------------------
def score_sequence_at_pos(model, 
                          tokenizer,
                          flat_ids, 
                          line_end_offsets, 
                          insert_pos_idx, 
                          seq_ids, 
                          queries_embs
                        ):
    # Build flat_with insert, compute doc embedding and average cosine vs queries_embs
    flat_with = insert_tokens_into_flat(flat_ids, seq_ids, insert_pos_idx, line_end_offsets)
    # convert ids to text via tokenizer.decode? We'll compute embedding via input_embeds for exact tokens
    ids_tensor = torch.tensor(flat_with, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = model(ids_tensor) if hasattr(model, "forward") and "inputs_embeds" not in model.forward.__code__.co_varnames else None
        # For safety, use tokenizer to get text then embed via model.encode-like
        # But we'll do efficient route: use embeddings -> input_embeds path without gradient
        emb_layer = model.get_input_embeddings()
        full_embs = emb_layer(ids_tensor.squeeze(0))
        input_embeds = full_embs.unsqueeze(0)
        attn_mask = torch.ones((1, input_embeds.size(1)), dtype=torch.long, device=DEVICE)
        out2 = model(inputs_embeds=input_embeds, attention_mask=attn_mask)
        doc_emb = mean_pooling(out2, attn_mask).squeeze(0)
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        cosines = F.cosine_similarity(doc_emb.unsqueeze(0), queries_embs, dim=-1)
        score = float(cosines.mean().item())
    return score

# -------------------------
# Position-aware beam search (multi-position)
# -------------------------
def position_aware_beam_search(
                                model, 
                                tokenizer, 
                                document_text, 
                                queries_per_pos_texts,
                                L_max=L_max, 
                                beam_width=BEAM_WIDTH, 
                                topk_token_per_pos=TOPK_TOKEN_PER_POS,
                                N_iter=N_ITER
                            ):
    """
    queries_per_pos_texts: list of lists of strings (for each p we may have multiple queries)
    """
    lines = doc_to_lines(document_text)
    flat_ids, line_end_offsets, _ = build_flat_token_ids_and_offsets(lines, tokenizer)
    num_lines = len(lines)

    # Precompute queries embeddings per position
    queries_embs_list = []
    for qtexts in queries_per_pos_texts:
        # qtexts is list of strings
        q_emb = embed_texts(qtexts, model, tokenizer)  # (M, D)
        queries_embs_list.append(q_emb)

    # Initialize Beam: for each position p, insert empty sequence (e.g., [pad/eos]*L_max)
    # Choose a neutral token: tokenizer.eos_token_id or pad or unk fallback
    neutral_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id)
    if neutral_id is None:
        neutral_id = 0

    Beam = []
    for p_idx in range(num_lines):
        s0 = [neutral_id] * L_max
        score0 = score_sequence_at_pos(model, tokenizer, flat_ids, line_end_offsets, p_idx, s0, queries_embs_list[p_idx])
        Beam.append((s0, p_idx, score0))

    # Keep top beam_width initial beams
    Beam.sort(key=lambda x: x[2], reverse=True)
    Beam = Beam[:beam_width]

    # Main iterations
    for it in range(N_iter):
        Candidates = []
        for (s_cur, p_idx, score_cur) in Beam:
            queries_embs = queries_embs_list[p_idx]  # (M, D)
            # propose topk per position via gradient
            T_per_j = gradient_propose_tokens(
                model, tokenizer, flat_ids, line_end_offsets,
                p_idx, s_cur, queries_embs, topk=topk_token_per_pos
            )  # list of length L_max
            # create candidates by replacing one position j with each candidate token
            for j in range(L_max):
                for tok in T_per_j[j]:
                    s_new = s_cur.copy()
                    s_new[j] = int(tok)
                    score_new = score_sequence_at_pos(model, tokenizer, flat_ids, line_end_offsets, p_idx, s_new, queries_embs)
                    Candidates.append((s_new, p_idx, score_new))
        # sort and keep top beam_width
        Candidates.sort(key=lambda x: x[2], reverse=True)
        Beam = Candidates[:beam_width]
        if (it+1) % PRINT_EVERY == 0 or it == N_iter-1:
            best_seq, best_p, best_sc = Beam[0]
            print(f"[Iter {it+1}/{N_iter}] best_score={best_sc:.4f} pos={best_p} seq_prefix={best_seq[:min(10,len(best_seq))]}")

    # final best
    s_star, p_star, best_score = max(Beam, key=lambda x: x[2])
    return s_star, p_star, best_score, lines, flat_ids, line_end_offsets

# -------------------------
# reSearch (short, start from original ranking seq)
# -------------------------
def re_search(
                model, 
                tokenizer, 
                document_text, 
                queries_per_pos_texts,
                original_seq_ids, 
                original_pos_idx,
                L_max=L_max, 
                beam_width=3, 
                topk_token_per_pos=6, 
                N_iter=RESEARCH_ITER
            ):
    lines = doc_to_lines(document_text)
    flat_ids, line_end_offsets, _ = build_flat_token_ids_and_offsets(lines, tokenizer)
    num_lines = len(lines)
    queries_embs_list = [embed_texts(qs, model, tokenizer) for qs in queries_per_pos_texts]

    # initialize beam: include original seq at its pos as top, plus others zero sequences
    neutral_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id)
    if neutral_id is None:
        neutral_id = 0

    Beam = []
    # original
    s_orig = original_seq_ids.copy()
    score_orig = score_sequence_at_pos(model, tokenizer, flat_ids, line_end_offsets, original_pos_idx, s_orig, queries_embs_list[original_pos_idx])
    Beam.append((s_orig, original_pos_idx, score_orig))
    # add some other positions neutral
    for p_idx in range(num_lines):
        if p_idx == original_pos_idx:
            continue
        s0 = [neutral_id] * L_max
        score0 = score_sequence_at_pos(model, tokenizer, flat_ids, line_end_offsets, p_idx, s0, queries_embs_list[p_idx])
        Beam.append((s0, p_idx, score0))
    Beam.sort(key=lambda x: x[2], reverse=True)
    Beam = Beam[:beam_width]

    # short iterations
    for it in range(N_iter):
        Candidates = []
        for (s_cur, p_idx, score_cur) in Beam:
            queries_embs = queries_embs_list[p_idx]
            T_per_j = gradient_propose_tokens(model, tokenizer, flat_ids, line_end_offsets, p_idx, s_cur, queries_embs, topk=topk_token_per_pos)
            for j in range(L_max):
                for tok in T_per_j[j]:
                    s_new = s_cur.copy()
                    s_new[j] = int(tok)
                    score_new = score_sequence_at_pos(model, tokenizer, flat_ids, line_end_offsets, p_idx, s_new, queries_embs)
                    Candidates.append((s_new, p_idx, score_new))
        Candidates.sort(key=lambda x: x[2], reverse=True)
        Beam = Candidates[:beam_width]
        best_seq, best_p, best_sc = Beam[0]
        print(f"(reSearch Iter {it+1}/{N_iter}) best_score={best_sc:.4f} pos={best_p}")

    s_star, p_star, best_score = max(Beam, key=lambda x: x[2])
    return s_star, p_star, best_score