from dotenv import load_dotenv

from rag.generation.clients.gemini_client import (
    GeminiClient,
)
from rag.generation.generator import Generator
from rag.retrieval.reranking_retriever import (
    RerankingRetriever,
)


def main() -> None:
    load_dotenv()

    query = input(
        "Question: "
    ).strip()

    if not query:
        raise ValueError(
            "Question cannot be empty"
        )

    retriever = RerankingRetriever(
        rerank_candidates=30,
    )

    chunks = retriever.search(
        query=query,
        top_k=5,
    )

    generator = Generator(
        llm_client=GeminiClient(),
    )

    result = generator.generate(
        query=query,
        chunks=chunks,
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.answer)

    print()
    print("=" * 60)
    print("CITED SOURCES")
    print("=" * 60)

    if not result.citations:
        print(
            "No sources were cited."
        )

    for citation in result.citations:
        print(
            f"[Source {citation.source_id}] "
            f"{citation.filename}, "
            f"page {citation.page_number}, "
            f"chunk {citation.chunk_index}"
        )

    print()
    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    for index, chunk in enumerate(
        result.sources,
        start=1,
    ):
        print(
            f"[Source {index}] "
            f"{chunk.filename}, "
            f"page {chunk.page_number}, "
            f"score={chunk.score:.4f}"
        )


if __name__ == "__main__":
    main()