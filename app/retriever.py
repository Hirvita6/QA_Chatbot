import os
from typing import List, Dict
import json
from sentence_transformers import SentenceTransformer
import numpy as np


try:
    import faiss
except Exception:
    faiss = None


class Retriever:
    """Lightweight FAISS-backed retriever using SentenceTransformers.

    Responsibilities:
    - load or create a FAISS index and accompanying metadata file
    - add text chunks (with source metadata) to the index
    - search the index for nearest neighbors to a query
    """

    def __init__(self, index_path: str, meta_path: str, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(embed_model_name)
        # dimension of the sentence embeddings
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadatas: List[Dict] = []
        self._load()

    def _load(self):
        # load faiss index if present
        if os.path.exists(self.index_path) and faiss is not None:
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception:
                self.index = None

        # load metadata json if present
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadatas = json.load(f)
            except Exception:
                self.metadatas = []

        # if index not loaded, create a new one
        if self.index is None:
            if faiss is None:
                raise RuntimeError("faiss is required. Please install faiss-cpu or faiss-gpu.")
            # simple flat L2 index
            self.index = faiss.IndexFlatL2(self.dim)

    def save(self):
        # persist index and metadata
        if faiss is not None and self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
            except Exception:
                # ignore write failures here; caller can handle
                pass

        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def reset(self, remove_files: bool = False):
        """Reset the in-memory index and metadata to empty. Optionally remove persisted files.

        Call this when you want to start fresh (for example, after a new upload that should
        replace the previous index). After adding documents call `save()` to persist.
        """
        if faiss is None:
            raise RuntimeError("faiss is required to create or reset the index")

        # create a fresh index
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadatas = []

        if remove_files:
            try:
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
            except Exception:
                pass
            try:
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
            except Exception:
                pass

    def add_documents(self, texts: List[str], sources: List[str]):
        """Add text chunks and their source metadata to the index.

        texts: list of text chunks
        sources: list of source file paths (same length)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # ensure embeddings have batch dimension
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, 0)

        # FAISS expects float32
        self.index.add(embeddings.astype('float32'))
        for t, s in zip(texts, sources):
            self.metadatas.append({"text": t, "source": s})

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        if q_emb.dtype != "float32":
            q_emb = q_emb.astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results: List[Dict] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append({"score": float(score), "text": meta.get("text"), "source": meta.get("source")})
        return results