# embedder.py
from FlagEmbedding import BGEM3FlagModel
import torch
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

    def embed(self, documents: str | list[str]) -> np.ndarray:
        if isinstance(documents, str):
            documents = [documents]
            single = True
        else:
            single = False

        embedding_vectors = self.bge_m3_model.encode(documents, batch_size=12, max_length=8192)['dense_vecs']
        embedding_vectors = np.array(embedding_vectors, dtype=np.float32)

        if single:
            return embedding_vectors[0]
        else:
            return embedding_vectors