from typing import List
import numpy as np

# 1. Tạo Adapter để LangChain hiểu được SentenceTransformer
class HelperEmbeddingsAdapter:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Chuyển đổi output của SentenceTransformer (numpy) sang list (LangChain yêu cầu)
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # Tương tự cho câu query đơn lẻ
        embedding = self.model.encode(text)
        return embedding.tolist()

    # Hàm __call__ để dự phòng nếu LangChain gọi trực tiếp object
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)