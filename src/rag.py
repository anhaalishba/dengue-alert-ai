"""Simple RAG: chunk docs/*.txt, TF-IDF retrieval (no extra downloads).
Upgrade idea: swap TF-IDF for sentence-transformers embeddings + FAISS/Chroma."""
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


def load_chunks():
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        for para in [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]:
            chunks.append({"source": os.path.basename(path), "text": para})
    return chunks


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        idx = sims.argsort()[::-1][:k]
        return [dict(self.chunks[i], score=float(sims[i])) for i in idx if sims[i] > 0]
