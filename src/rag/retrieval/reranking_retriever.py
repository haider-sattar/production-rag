from dataclasses import replace

from sentence_transformers import CrossEncoder

from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.retriever import RetrievedChunk


class RerankingRetriever:
    """
    Hybrid retrieval se candidates leta hai aur cross-encoder
    ke through unki final relevance ranking improve karta hai.
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

        self.hybrid_retriever = hybrid_retriever
        self.rerank_candidates = rerank_candidates
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Hybrid candidates ko query ke against cross-encoder
        se score karta hai aur best top-k results return karta hai.
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
            self.rerank_candidates,
            top_k,
        )

        candidates = self.hybrid_retriever.search(
            query=query,
            top_k=candidate_limit,
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