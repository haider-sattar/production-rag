import re
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag.retrieval.retriever import RetrievedChunk

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    """
    Normalize text for BM25 while preserving useful terms,
    acronyms, years, and values such as 5G, LMICs, and 2025.
    """

    tokens = re.findall(
        pattern=r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b",
        string=text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


class BM25Retriever:
    """
    Qdrant mein stored chunk payloads par lexical BM25 search.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError(
                f"No chunks found in Qdrant collection: "
                f"{self.collection_name}"
            )

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_chunks(self) -> list[RetrievedChunk]:
        """
        Qdrant se saare chunk payloads pagination ke saath load karta hai.
        Vectors load nahi karta because BM25 ko sirf text chahiye.
        """

        chunks: list[RetrievedChunk] = []
        offset: Any = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                text = str(payload.get("text", ""))

                if not text.strip():
                    continue

                chunks.append(
                    RetrievedChunk(
                        text=text,
                        document_id=str(
                            payload.get("document_id", "")
                        ),
                        filename=str(
                            payload.get("filename", "")
                        ),
                        page_number=int(
                            payload.get("page_number", -1)
                        ),
                        chunk_index=int(
                            payload.get("chunk_index", -1)
                        ),
                        score=0.0,
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Query ke exact lexical matches ke basis par top chunks return karta hai.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(scores[index]),
                )
            )

        return results