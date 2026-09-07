import argparse

from rag.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search document chunks using BM25."
    )

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results",
    )

    args = parser.parse_args()

    retriever = BM25Retriever()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print()
    print(f"Query: {args.query}")
    print(f"Retrieved chunks: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print()
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"BM25 score: {result.score:.4f}")
        print(f"File: {result.filename}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_index}")
        print("-" * 70)
        print(result.text)


if __name__ == "__main__":
    main()