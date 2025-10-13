import random
import math

# -----------------------------
# Config
# -----------------------------
L_MAX = 8             # số token chèn vào
BEAM_WIDTH = 4        # số beam
TOPK_PER_POS = 5      # mỗi vị trí lấy top-k token
N_ITER = 5            # số vòng beam search
RESEARCH_ITER = 3     # số vòng refine ngắn
SEED = 42
random.seed(SEED)

# Tập từ giả lập
VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]

# -----------------------------
# Giả lập "gradient propose"
# -----------------------------
def gradient_propose_tokens(current_seq, queries, topk=TOPK_PER_POS):
    """
    Mô phỏng bước gradient-propose bằng cách
    chọn ra top-k token có nhiều ký tự chung nhất với query.
    """
    T_per_j = []
    for j in range(len(current_seq)):
        # Chọn ngẫu nhiên vài token ứng viên
        scored_tokens = []
        for tok in VOCAB:
            # điểm = độ trùng ký tự với query
            score = sum(tok.count(c) for c in "".join(queries))
            scored_tokens.append((tok, score))
        # lấy top-k
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        T_per_j.append([t for t, _ in scored_tokens[:topk]])
    return T_per_j


# -----------------------------
# Hàm chấm điểm (score)
# -----------------------------
def score_sequence(document_lines, seq, insert_pos, queries):
    """
    Giả lập điểm cosine: càng có nhiều token gần giống query thì điểm càng cao.
    """
    inserted_text = " ".join(seq)
    query_text = " ".join(queries)
    # điểm = tỉ lệ ký tự trùng giữa chuỗi chèn và query
    matches = sum(1 for c in inserted_text if c in query_text)
    total = max(1, len(inserted_text))
    score = matches / total
    # thêm phần thưởng nhẹ cho độ dài hợp lý
    score *= 1 - abs(len(inserted_text) - len(query_text)) / (len(query_text) + 1)
    return score


# -----------------------------
# Beam Search chính
# -----------------------------
def position_aware_beam_search(document_text, queries_per_pos):
    lines = document_text.splitlines()
    num_lines = len(lines)
    neutral_token = "<NEU>"

    # beam khởi tạo
    Beam = []
    for p_idx in range(num_lines):
        s0 = [neutral_token] * L_MAX
        score0 = score_sequence(lines, s0, p_idx, queries_per_pos[p_idx])
        Beam.append((s0, p_idx, score0))
    Beam.sort(key=lambda x: x[2], reverse=True)
    Beam = Beam[:BEAM_WIDTH]

    for it in range(N_ITER):
        Candidates = []
        for (s_cur, p_idx, sc_cur) in Beam:
            T_per_j = gradient_propose_tokens(s_cur, queries_per_pos[p_idx])
            for j in range(L_MAX):
                for tok in T_per_j[j]:
                    s_new = s_cur.copy()
                    s_new[j] = tok
                    sc_new = score_sequence(lines, s_new, p_idx, queries_per_pos[p_idx])
                    Candidates.append((s_new, p_idx, sc_new))
        Candidates.sort(key=lambda x: x[2], reverse=True)
        Beam = Candidates[:BEAM_WIDTH]
        best_seq, best_p, best_sc = Beam[0]
        print(f"[Iter {it+1}/{N_ITER}] best_score={best_sc:.4f} at pos={best_p}")

    best_seq, best_pos, best_score = max(Beam, key=lambda x: x[2])
    return best_seq, best_pos, best_score


# -----------------------------
# Re-search (fine-tune quanh nghiệm tốt nhất)
# -----------------------------
def re_search(document_text, queries_per_pos, best_seq, best_pos):
    lines = document_text.splitlines()
    num_lines = len(lines)
    Beam = []

    # thêm nghiệm ban đầu
    score_best = score_sequence(lines, best_seq, best_pos, queries_per_pos[best_pos])
    Beam.append((best_seq, best_pos, score_best))

    # thêm vài vị trí khác trung tính
    neutral_token = "<NEU>"
    for p_idx in range(num_lines):
        if p_idx == best_pos:
            continue
        s0 = [neutral_token] * L_MAX
        sc0 = score_sequence(lines, s0, p_idx, queries_per_pos[p_idx])
        Beam.append((s0, p_idx, sc0))

    Beam.sort(key=lambda x: x[2], reverse=True)
    Beam = Beam[:BEAM_WIDTH]

    for it in range(RESEARCH_ITER):
        Candidates = []
        for (s_cur, p_idx, sc_cur) in Beam:
            T_per_j = gradient_propose_tokens(s_cur, queries_per_pos[p_idx])
            for j in range(L_MAX):
                for tok in T_per_j[j]:
                    s_new = s_cur.copy()
                    s_new[j] = tok
                    sc_new = score_sequence(lines, s_new, p_idx, queries_per_pos[p_idx])
                    Candidates.append((s_new, p_idx, sc_new))
        Candidates.sort(key=lambda x: x[2], reverse=True)
        Beam = Candidates[:BEAM_WIDTH]
        best_seq, best_p, best_sc = Beam[0]
        print(f"(ReSearch {it+1}/{RESEARCH_ITER}) best_score={best_sc:.4f} at pos={best_p}")

    best_seq, best_pos, best_score = max(Beam, key=lambda x: x[2])
    return best_seq, best_pos, best_score


# -----------------------------
# Demo chạy thử
# -----------------------------
if __name__ == "__main__":
    document = """This function calculates the average of absolute differences.
                    It processes each permutation separately.
                    Finally, it returns the mean result."""

    queries = [
        ["average", "absolute", "difference"],    # cho dòng 1
        ["permutation", "shuffle"],               # cho dòng 2
        ["mean", "return", "result"]              # cho dòng 3
    ]

    print("\n=== HOTFLIP UPRANKING (SIMULATION) ===")
    s_star, p_star, best_score = position_aware_beam_search(document, queries)
    print(f"\nInitial best pos={p_star}, score={best_score:.4f}")

    s_refined, p_refined, sc_refined = re_search(document, queries, s_star, p_star)
    print("\n=== FINAL RESULT ===")
    print("Best position:", p_refined)
    print("Best score:", round(sc_refined, 4))
    print("Best token sequence:", " ".join(s_refined))