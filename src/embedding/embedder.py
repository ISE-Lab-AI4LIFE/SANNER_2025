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

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, document: str) -> np.ndarray:
        # ===== BGE-M3 =====
        bge_m3_embedding_vector = self.bge_m3_model.encode(document, batch_size=12, max_length=8192)['dense_vecs']
        bge_m3_embedding_vector = np.array(bge_m3_embedding_vector, dtype=np.float32).squeeze()
        return bge_m3_embedding_vector.astype(np.float32)