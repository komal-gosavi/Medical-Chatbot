"""
pinecone_embeddings.py
-----------------------
A tiny LangChain-compatible "Embeddings" class that calls Pinecone's
own HOSTED embedding model (multilingual-e5-large) over the network,
instead of running a model locally with sentence-transformers/torch.

WHY: torch + sentence-transformers need 1-2GB RAM just to import.
Render's free tier only has 512MB RAM, so the app would crash with
a silent Out-Of-Memory kill (no port ever opens, no error printed).

This class needs almost zero RAM -- it just sends text over HTTPS
to Pinecone and gets vectors back. Free to use with your existing
Pinecone account.

Model: multilingual-e5-large -> 1024 dimensions.
(This means your Pinecone index dimension must be 1024, not 384.)
"""

from typing import List
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone


class PineconeHostedEmbeddings(Embeddings):
    """LangChain-compatible embeddings using Pinecone's hosted inference API."""

    def __init__(self, api_key: str, model: str = "multilingual-e5-large"):
        self._pc = Pinecone(api_key=api_key)
        self._model = model

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Call Pinecone's inference API in batches (max 96 texts per call)."""
        all_vectors: List[List[float]] = []
        batch_size = 96

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._pc.inference.embed(
                model=self._model,
                inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"},
            )
            all_vectors.extend([item["values"] for item in result])

        return all_vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (used when storing chunks in Pinecone)."""
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query (used when a user asks a question)."""
        return self._embed([text], input_type="query")[0]
