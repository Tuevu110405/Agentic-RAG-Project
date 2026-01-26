from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Literal, List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize
import re

class TextSplitter:
    def __init__(
            self,
            separators: List[str] = ['\n\n','\n',' ',''],
            chunk_size = 300,
            chunk_overlap = 0,
            embed_model = None
    ):
        '''
            Args:
                This project mainly use semantic chunking, remember to pass HuggingFace embedding model
        '''
        self.splitter = RecursiveCharacterTextSplitter(
            separators= separators,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        self.embed_model = embed_model

    def recursive_splitter(self, documents):
        return self.splitter.split_documents(documents)


    def semantic_splitting_v2(self, sentences, threshold=0.6):
        '''
            Split sentences into chunks
            Args:
                sentences: list of sentences
                embed_model: embedding model
                threshold: cosine similarity threshold

        '''
        # 1. Handling input exception
        if not sentences:
            return []

        # 2. Encode all sentences once (Batch processing) -> increasing speed
        # convert_to_numpy=True assure that these embeddings are numpy
        embeddings = self.embed_model.encode(sentences, convert_to_tensor=False)
        print("Encode for chunking sucessfully")

        # 3. Initialize the chunks list
        chunks = [[sentences[0]]] # Bắt đầu chunk đầu tiên với câu đầu

        # 4. traversing the next sentences
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            current_embedding = embeddings[i].reshape(1, -1)
            prev_embedding = embeddings[i-1].reshape(1, -1)

            # 5. Cosine Similarity Calculation
            # Returning matrix [[score]], get [0][0] to get real score
            sim_score = cosine_similarity(prev_embedding, current_embedding)[0][0]

            # 6. linking related sentences
            if sim_score >= threshold:
                # if semantic score is high, combining into current chunk
                chunks[-1].append(current_sentence)
            else:
                # Nếu khác nhau (score thấp), tạo chunk mới
                # if semantic score is low, creating new chunk
                chunks.append([current_sentence])

        # 7. Combining each chunk into a paragraph
        final_chunks = [' '.join(chunk) for chunk in chunks]

        return final_chunks

class Loader:
    def __init__(
            self,
            embed_model,
            split_kwargs = {
                     "chunk_size" : 300,
                     "chunk_overlap" : 20
                 }
    ):

        self.embed_model = embed_model

        self.doc_splitter = TextSplitter(embed_model=self.embed_model, **split_kwargs)

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # 1
        text = re.sub(r'<[^>]+>','', text)
        # 2
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        # 3
        vietnamese_chars = "a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
        pattern = f"[^{vietnamese_chars}\d\s.,?!:;]"
        text = re.sub(pattern, '', text)
        # 4
        text = re.sub(r'\s([.,?!:;])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def semantic_load(self, txt_file_path, semantic_threshold):
        with open(txt_file_path, "r", encoding = 'utf-8') as f:
            sentences = f.read()
        sentences = self._clean_text(sentences)
        sentences = sent_tokenize(sentences)

        chunks = self.doc_splitter.semantic_splitting_v2(sentences=sentences, threshold=semantic_threshold)
        return chunks