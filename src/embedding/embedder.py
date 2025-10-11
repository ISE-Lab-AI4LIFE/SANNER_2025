# This code is the embedder used for the dual-chain retrieval phase

from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np

def average_embedder(document: str):
    # chọn thiết bị tối ưu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BGE-M3 (tự nhận GPU qua tham số device)
    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device.type)

    # E5 model
    e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(device)

    # mGTE model (large)
    mGTE_tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    mGTE_model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True).to(device)

    # ====== BGE-M3 embedding ======
    bge_m3_embedding_vector = bge_m3_model.encode(document, batch_size=12, max_length=8192)['dense_vecs']
    bge_m3_embedding_vector = np.array(bge_m3_embedding_vector, dtype=np.float32).squeeze()

    # ====== E5 embedding ======
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    e5_inputs = e5_tokenizer([document], max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        e5_outputs = e5_model(**e5_inputs)
    e5_embedding_vector = average_pool(e5_outputs.last_hidden_state, e5_inputs["attention_mask"])
    e5_embedding_vector = F.normalize(e5_embedding_vector, p=2, dim=1)
    e5_embedding_vector = e5_embedding_vector.squeeze(0).detach().cpu().numpy()

    # ====== mGTE embedding ======
    mGTE_inputs = mGTE_tokenizer([document], max_length=8192, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        mGTE_outputs = mGTE_model(**mGTE_inputs)
    mGTE_embedding_vector = mGTE_outputs.last_hidden_state[:, 0, :]
    mGTE_embedding_vector = F.normalize(mGTE_embedding_vector, p=2, dim=1)
    mGTE_embedding_vector = mGTE_embedding_vector.squeeze(0).detach().cpu().numpy()

    # ====== average fusion ======
    final_embedding = (bge_m3_embedding_vector + e5_embedding_vector + mGTE_embedding_vector) / 3.0
    return final_embedding.astype(np.float32)