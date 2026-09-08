from dataclasses import replace

from sentence_transformers import CrossEncoder

from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.retriever import RetrievedChunk


class RerankingRetriever:
    """
    Retrieve hybrid candidates from one document and rerank them
    with a cross-encoder for final relevance ordering.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        rerank_candidates: int = 30,
        batch_size: int = 16,
    ) -> None:
        if rerank_candidates <= 0:
            raise ValueError(
                "rerank_candidates must be greater than 0"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        if hybrid_retriever is None:
            # The hybrid stage must generate at least as many
            # candidates as the cross-encoder is expected to rerank.
            hybrid_retriever = HybridRetriever(
                candidate_k=rerank_candidates,
                rrf_constant=60,
            )

        self.hybrid_retriever = (
            hybrid_retriever
        )
        self.rerank_candidates = (
            rerank_candidates
        )
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name,
        )

    def invalidate_document(
        self,
        document_id: str,
    ) -> None:
        """
        Invalidate cached retrieval state for one document.

        The hybrid retriever forwards this to BM25, which keeps
        the per-document lexical cache.
        """

        self.hybrid_retriever.invalidate_document(
            document_id=document_id,
        )

    def search(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve candidates from one document, score them against
        the query with the cross-encoder, and return the best top-k.
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

        candidate_limit = max(
            self.rerank_candidates,
            top_k,
        )

        candidates = (
            self.hybrid_retriever.search(
                query=query,
                document_id=document_id,
                top_k=candidate_limit,
            )
        )

        if not candidates:
            return []

        query_chunk_pairs = [
            (
                query,
                candidate.text,
            )
            for candidate in candidates
        ]

        reranker_scores = self.model.predict(
            query_chunk_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked_results = [
            replace(
                candidate,
                score=float(score),
            )
            for candidate, score in zip(
                candidates,
                reranker_scores,
                strict=True,
            )
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]
