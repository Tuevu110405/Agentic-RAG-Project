import torch
import faiss
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List
from langchain_core.documents import Document
from src.rag.utils import tokenize_bm25, normalize_score_bm25

class VectorDB:
    # build database with multiple collections
    def __init__(
            self,
            # documents: List[str],
            embed_model, # Pass vectordatabase instance
    ):
        self.semantic_index = {}
        self.bm25_index = {}
        self.bm25_id = {}
        self.embedding_model = embed_model
        self.documents = {}

    def _build_rv(self, documents):
        # Initialize BM25 index
        tokenized_docs = [tokenize_bm25(doc) for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs)
        bm25_id = list(range(len(tokenized_docs)))
        return bm25_index, bm25_id

    def _build_db(self, documents):
        # 1. Gọi API để lấy embedding (trả về numpy array)
        # Class wrapper đã xử lý việc loop qua từng document
        embeddings = self.embedding_model.encode(documents)

        # 2. Chuẩn hóa vector (L2 Normalization) bằng Numpy
        # Để dùng Cosine Similarity với Faiss IndexFlatIP, vector phải được chuẩn hóa
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Tránh lỗi chia cho 0
        embeddings = embeddings / np.maximum(norms, 1e-10)

        # 3. Chuyển sang float32 cho Faiss
        embeddings = embeddings.astype("float32")

        # 4. Tạo Index Faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f"Initialize semantic index successfully. Size: {embeddings.shape}")
        return index

    def add_collection(self,
                       collection_name: str,
                       documents: List[str],
                       ):
        # add collection to database
        if collection_name not in self.semantic_index.keys():
            self.semantic_index[collection_name] = self._build_db(documents)
            self.bm25_index[collection_name], self.bm25_id[collection_name] = self._build_rv(documents)
            self.documents[collection_name] = documents
            print(f"Collection {collection_name} added successfully")
        else:
            print(f"Collection {collection_name} already exists")

    def semantic_search(self, query: str, top_k: int, collection_name):
        # 1. Encode query
        query_embedding = self.embedding_model.encode(query) # Trả về (1, dim) hoặc (dim,)

        # 2. Reshape nếu cần (đảm bảo là 2D array)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 3. Chuẩn hóa query (L2 Norm)
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / np.maximum(norm, 1e-10)

        # 4. Search
        distance, index = self.semantic_index[collection_name].search(query_embedding.astype("float32"), top_k)

        return distance[0], index[0]

    def text_search(self, query: str, top_k: int, collection_name: str):
        tokenized_query = tokenize_bm25(query)
        scores = self.bm25_index[collection_name].get_scores(tokenized_query)
        normalized_scores = normalize_score_bm25(scores)

        # Lấy top k index
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        results = [(self.bm25_id[collection_name][idx], normalized_scores[idx]) for idx in top_indices]
        return results

    def hybrid_search(self, query_dense: str, query_sparse : str, top_k: int,collection_name: str, weights: list = [0.5, 0.5]):
        """
        Args:
            query: Câu truy vấn (String)
            top_k: Số lượng kết quả
            weights: [Semantic weight, Keyword weight]
        """
        # Cả semantic và keyword đều dùng chung string query đầu vào
        sem_distance, sem_index = self.semantic_search(query_dense, top_k, collection_name)
        text_results = self.text_search(query_sparse, top_k, collection_name)

        combined_scores = {}

        # Cộng điểm Semantic (weights[0])
        for idx, score in zip(sem_index, sem_distance):
            if idx != -1: # Faiss trả về -1 nếu không tìm thấy
                combined_scores[idx] = combined_scores.get(idx, 0.0) + weights[0] * score

        # Cộng điểm Keyword/BM25 (weights[1])
        for idx, score in text_results:
            combined_scores[idx] = combined_scores.get(idx, 0.0) + weights[1] * score

        # Sắp xếp kết quả cuối cùng
        sorted_results = sorted(list(combined_scores.items()), key=lambda x: x[1], reverse=True)[:top_k]

        final_docs = []
        for idx, score in sorted_results:
            doc = Document(
                page_content=self.documents[collection_name][idx],
                metadata={"id": int(idx), "score": float(score)}
            )
            final_docs.append(doc)

        return final_docs