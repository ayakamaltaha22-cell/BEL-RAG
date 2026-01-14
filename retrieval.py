from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import DocChunk


@dataclass
class RetrievalResult:
    chunk: DocChunk
    score: float


class TfidfRetriever:
    """Simple vector retrieval backend (TF-IDF cosine). Replace with dense embeddings in production."""
    def __init__(self, chunks: Sequence[DocChunk], ngram_range=(1, 2), max_features: int = 50000):
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words="english")
        self._fit()

    def _fit(self) -> None:
        texts = [c.text for c in self.chunks]
        self.mat = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.mat).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(sims[i])) for i in idx]
