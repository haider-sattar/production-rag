from dataclasses import replace

from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.embeddings import EmbeddingService
from rag.retrieval.retriever import RetrievedChunk, Retriever


class HybridRetriever:
    """
    Dense aur BM25 results ko Reciprocal Rank Fusion
    ke through combine karta hai.
    """

    def __init__(
        self,
        dense_retriever: Retriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        candidate_k: int = 20,
        rrf_constant: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if rrf_constant < 0:
            raise ValueError(
                "rrf_constant cannot be negative"
            )

        if dense_retriever is None:
            embedding_service = EmbeddingService()

            dense_retriever = Retriever(
                embedding_service=embedding_service,
            )

        if bm25_retriever is None:
            bm25_retriever = BM25Retriever()

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    @staticmethod
    def _chunk_key(
        chunk: RetrievedChunk,
    ) -> tuple[str, int]:
        """
        Ek document chunk ki stable identity.

        Same chunk dense aur BM25 dono mein aaye to
        RRF scores isi key ke through combine honge.
        """

        return (
            chunk.document_id,
            chunk.chunk_index,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Dense aur BM25 candidates retrieve karke
        unko RRF se rank karta hai.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        candidate_limit = max(
            self.candidate_k,
            top_k,
        )

        dense_results = self.dense_retriever.search(
            query=query,
            top_k=candidate_limit,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_limit,
        )

        rrf_scores: dict[
            tuple[str, int],
            float,
        ] = {}

        chunks_by_key: dict[
            tuple[str, int],
            RetrievedChunk,
        ] = {}

        result_lists = [
            dense_results,
            bm25_results,
        ]

        for results in result_lists:
            for rank, chunk in enumerate(
                results,
                start=1,
            ):
                key = self._chunk_key(chunk)

                chunks_by_key[key] = chunk

                rank_score = 1.0 / (
                    self.rrf_constant + rank
                )

                rrf_scores[key] = (
                    rrf_scores.get(key, 0.0)
                    + rank_score
                )

        ranked_keys = sorted(
            rrf_scores,
            key=lambda key: rrf_scores[key],
            reverse=True,
        )

        final_results: list[RetrievedChunk] = []
        seen_pages: set[tuple[str, int]] = set()

        for key in ranked_keys:
            original_chunk = chunks_by_key[key]

            page_key = (
                original_chunk.document_id,
                original_chunk.page_number,
            )

            if page_key in seen_pages:
                continue

            seen_pages.add(page_key)

            final_results.append(
                replace(
                    original_chunk,
                    score=rrf_scores[key],
                )
            )

            if len(final_results) >= top_k:
                break

        return final_results