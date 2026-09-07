import argparse
import json
from pathlib import Path
from typing import Any

from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.embeddings import EmbeddingService
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranking_retriever import RerankingRetriever
from rag.retrieval.retriever import Retriever


def hit_at_k(
    retrieved_pages: list[int],
    relevant_pages: set[int],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant page appears
    within the first k retrieved results.

    Note:
        This is Hit@K / Success@K rather than strict Recall@K.
    """

    top_k_pages = retrieved_pages[:k]

    return float(
        any(
            page in relevant_pages
            for page in top_k_pages
        )
    )


def reciprocal_rank(
    retrieved_pages: list[int],
    relevant_pages: set[int],
) -> float:
    """
    Calculate reciprocal rank of the first relevant result.

    Examples:
        relevant result at rank 1 -> 1.0
        relevant result at rank 2 -> 0.5
        relevant result at rank 4 -> 0.25
        no relevant result        -> 0.0
    """

    for rank, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


def build_retriever(
    retriever_name: str,
    rerank_candidates: int,
):
    """
    Build the retrieval system selected through the CLI.

    For reranking, the hybrid retriever first generates
    a candidate pool. The cross-encoder then reorders
    those candidates by query-passage relevance.
    """

    if retriever_name == "dense":
        embedding_service = EmbeddingService()

        return Retriever(
            embedding_service=embedding_service,
        )

    if retriever_name == "bm25":
        return BM25Retriever()

    if retriever_name == "hybrid":
        return HybridRetriever(
            candidate_k=20,
            rrf_constant=60,
        )

    if retriever_name == "reranked":
        # Ensure the hybrid stage can generate enough
        # candidates for experiments such as 10, 20, or 30.
        hybrid_candidate_k = max(
            20,
            rerank_candidates,
        )

        hybrid_retriever = HybridRetriever(
            candidate_k=hybrid_candidate_k,
            rrf_constant=60,
        )

        return RerankingRetriever(
            hybrid_retriever=hybrid_retriever,
            rerank_candidates=rerank_candidates,
            batch_size=16,
        )

    raise ValueError(
        f"Unsupported retriever: {retriever_name}"
    )


def main() -> None:
    """
    Evaluate a retrieval configuration against the
    audited page-level RAG evaluation dataset.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dense, BM25, hybrid, or reranked "
            "retrieval against the gold-standard dataset."
        )
    )

    parser.add_argument(
        "--retriever",
        choices=[
            "dense",
            "bm25",
            "hybrid",
            "reranked",
        ],
        default="dense",
        help=(
            "Retrieval system to evaluate. "
            "Default: dense"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of final retrieval results to evaluate. "
            "Must be at least 5. Default: 5"
        ),
    )

    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help=(
            "Number of hybrid candidates passed to the "
            "cross-encoder reranker. Only used when "
            "--retriever reranked. Default: 20"
        ),
    )

    args = parser.parse_args()

    if args.top_k < 5:
        raise ValueError(
            "top-k must be at least 5 because the evaluator "
            "calculates Hit@1, Hit@3, and Hit@5"
        )

    if args.rerank_candidates <= 0:
        raise ValueError(
            "rerank-candidates must be greater than 0"
        )

    if (
        args.retriever == "reranked"
        and args.rerank_candidates < args.top_k
    ):
        raise ValueError(
            "rerank-candidates must be greater than or "
            "equal to top-k"
        )

    retriever_name = args.retriever
    top_k = args.top_k

    dataset_path = Path(
        "evaluation/questions.json"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not questions:
        raise ValueError(
            "Evaluation dataset is empty"
        )

    retriever = build_retriever(
        retriever_name=retriever_name,
        rerank_candidates=args.rerank_candidates,
    )

    total_hit_1 = 0.0
    total_hit_3 = 0.0
    total_hit_5 = 0.0
    total_reciprocal_rank = 0.0

    detailed_results: list[dict[str, Any]] = []

    for item in questions:
        query = item["query"]

        relevant_pages = set(
            item["relevant_pages"]
        )

        results = retriever.search(
            query=query,
            top_k=top_k,
        )

        retrieved_pages = [
            result.page_number
            for result in results
        ]

        hit_1 = hit_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=1,
        )

        hit_3 = hit_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=3,
        )

        hit_5 = hit_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
        )

        total_hit_1 += hit_1
        total_hit_3 += hit_3
        total_hit_5 += hit_5
        total_reciprocal_rank += rr

        detailed_results.append(
            {
                "id": item["id"],
                "query": query,
                "query_type": item.get(
                    "query_type"
                ),
                "difficulty": item.get(
                    "difficulty"
                ),
                "relevant_pages": sorted(
                    relevant_pages
                ),
                "retrieved_pages": (
                    retrieved_pages
                ),
                "hit@1": hit_1,
                "hit@3": hit_3,
                "hit@5": hit_5,
                "reciprocal_rank": rr,
            }
        )

    num_questions = len(questions)

    metrics: dict[str, Any] = {
        "retriever": retriever_name,
        "num_questions": num_questions,
        "top_k": top_k,
        "hit@1": (
            total_hit_1 / num_questions
        ),
        "hit@3": (
            total_hit_3 / num_questions
        ),
        "hit@5": (
            total_hit_5 / num_questions
        ),
        "mrr": (
            total_reciprocal_rank
            / num_questions
        ),
    }

    if retriever_name == "reranked":
        metrics["rerank_candidates"] = (
            args.rerank_candidates
        )

    output = {
        "metrics": metrics,
        "results": detailed_results,
    }

    results_directory = Path(
        "evaluation/results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if retriever_name == "reranked":
        output_filename = (
            f"reranked_k"
            f"{args.rerank_candidates}"
            "_results.json"
        )
    else:
        output_filename = (
            f"{retriever_name}_results.json"
        )

    output_path = (
        results_directory
        / output_filename
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print(
        f"{retriever_name.upper()} "
        "Retrieval Evaluation"
    )
    print("=" * 60)

    print(
        f"Questions: {num_questions}"
    )

    if retriever_name == "reranked":
        print(
            "Rerank candidates: "
            f"{args.rerank_candidates}"
        )

    print(
        f"Final top-k: {top_k}"
    )

    print(
        f"Hit@1: {metrics['hit@1']:.4f}"
    )
    print(
        f"Hit@3: {metrics['hit@3']:.4f}"
    )
    print(
        f"Hit@5: {metrics['hit@5']:.4f}"
    )
    print(
        f"MRR:   {metrics['mrr']:.4f}"
    )

    print()
    print(
        "Saved detailed results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()