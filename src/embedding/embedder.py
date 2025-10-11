from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np

class AverageEmbedder:
    def __init__(self, device: str = None):
        # chọn thiết bị tối ưu
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() 
                                       else "mps" if torch.backends.mps.is_available() 
                                       else "cpu")
        else:
            self.device = torch.device(device)

        # ===== Load model 1 lần =====
        self.bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=self.device.type)

        self.e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)

        self.mGTE_tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.mGTE_model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True).to(self.device)

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, document: str) -> np.ndarray:
        # ===== BGE-M3 =====
        bge_m3_embedding_vector = self.bge_m3_model.encode(document, batch_size=12, max_length=8192)['dense_vecs']
        bge_m3_embedding_vector = np.array(bge_m3_embedding_vector, dtype=np.float32).squeeze()

        # ===== E5 =====
        e5_inputs = self.e5_tokenizer([document], max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            e5_outputs = self.e5_model(**e5_inputs)
        e5_embedding_vector = self.average_pool(e5_outputs.last_hidden_state, e5_inputs["attention_mask"])
        e5_embedding_vector = F.normalize(e5_embedding_vector, p=2, dim=1)
        e5_embedding_vector = e5_embedding_vector.squeeze(0).detach().cpu().numpy()

        # ===== mGTE =====
        mGTE_inputs = self.mGTE_tokenizer([document], max_length=8192, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            mGTE_outputs = self.mGTE_model(**mGTE_inputs)
        mGTE_embedding_vector = mGTE_outputs.last_hidden_state[:, 0, :]
        mGTE_embedding_vector = F.normalize(mGTE_embedding_vector, p=2, dim=1)
        mGTE_embedding_vector = mGTE_embedding_vector.squeeze(0).detach().cpu().numpy()

        # ===== Average fusion =====
        final_embedding = (bge_m3_embedding_vector + e5_embedding_vector + mGTE_embedding_vector) / 3.0
        return final_embedding.astype(np.float32)