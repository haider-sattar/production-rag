import re
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag.retrieval.retriever import RetrievedChunk
from rag.retrieval.vector_store import VectorStore

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
    Perform lexical BM25 retrieval over chunks stored in Qdrant.

    BM25 indexes are built per document so lexical statistics and
    search results never mix chunks from different uploaded PDFs.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

        # Cache one BM25 index per document. This avoids rebuilding the
        # lexical index on every query while still keeping documents
        # strictly isolated from one another.
        self._chunks_by_document: dict[
            str,
            list[RetrievedChunk],
        ] = {}
        self._bm25_by_document: dict[
            str,
            BM25Okapi,
        ] = {}

    def _load_chunks(
        self,
        document_id: str,
    ) -> list[RetrievedChunk]:
        """
        Load only one document's chunk payloads from Qdrant.

        Vectors are not loaded because BM25 only needs text.
        """

        chunks: list[RetrievedChunk] = []
        offset: Any = None

        document_filter = (
            VectorStore.build_document_filter(
                document_id=document_id,
            )
        )

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=document_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                text = str(
                    payload.get(
                        "text",
                        "",
                    )
                )

                if not text.strip():
                    continue

                chunks.append(
                    RetrievedChunk(
                        text=text,
                        document_id=str(
                            payload.get(
                                "document_id",
                                "",
                            )
                        ),
                        filename=str(
                            payload.get(
                                "filename",
                                "",
                            )
                        ),
                        page_number=int(
                            payload.get(
                                "page_number",
                                -1,
                            )
                        ),
                        chunk_index=int(
                            payload.get(
                                "chunk_index",
                                -1,
                            )
                        ),
                        score=0.0,
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return chunks

    def _get_document_index(
        self,
        document_id: str,
    ) -> tuple[
        list[RetrievedChunk],
        BM25Okapi | None,
    ]:
        """
        Return the cached BM25 index for one document.

        The index is created lazily the first time that document is
        queried. This also allows the API to start when Qdrant contains
        no uploaded documents yet.
        """

        if document_id in self._chunks_by_document:
            return (
                self._chunks_by_document[
                    document_id
                ],
                self._bm25_by_document.get(
                    document_id
                ),
            )

        chunks = self._load_chunks(
            document_id=document_id,
        )

        self._chunks_by_document[
            document_id
        ] = chunks

        if not chunks:
            return chunks, None

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        self._bm25_by_document[
            document_id
        ] = bm25

        return chunks, bm25

    def invalidate_document(
        self,
        document_id: str,
    ) -> None:
        """
        Remove one document from the local BM25 cache.

        Call this after that document is uploaded again, replaced,
        or deleted so its next query rebuilds the index from Qdrant.
        """

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        self._chunks_by_document.pop(
            document_id,
            None,
        )

        self._bm25_by_document.pop(
            document_id,
            None,
        )

    def search(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Return top lexical matches from one document only.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        chunks, bm25 = (
            self._get_document_index(
                document_id=document_id,
            )
        )

        if bm25 is None:
            return []

        scores = bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []

        for index in ranked_indices:
            chunk = chunks[index]

            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    document_id=(
                        chunk.document_id
                    ),
                    filename=chunk.filename,
                    page_number=(
                        chunk.page_number
                    ),
                    chunk_index=(
                        chunk.chunk_index
                    ),
                    score=float(
                        scores[index]
                    ),
                )
            )

        return results
