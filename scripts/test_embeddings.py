from rag.retrieval.embeddings import EmbeddingService


def dot_product(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculate the dot product between two vectors.

    Because our embeddings are normalized to length 1,
    their dot product is equivalent to cosine similarity.
    """

    return sum(
        a * b
        for a, b in zip(vector_a, vector_b, strict=True)
    )


def main() -> None:
    embedding_service = EmbeddingService()

    sentences = [
        "Machine learning predicts equipment failures.",
        "AI can detect machines that are likely to break down.",
        "Paris is the capital of France.",
    ]

    # Create one vector for every sentence.
    embeddings = embedding_service.embed_batch(sentences)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    similarity_ab = dot_product(
        embeddings[0],
        embeddings[1],
    )

    similarity_ac = dot_product(
        embeddings[0],
        embeddings[2],
    )

    print()
    print(f"A: {sentences[0]}")
    print(f"B: {sentences[1]}")
    print(f"C: {sentences[2]}")

    print()
    print(f"Similarity A <-> B: {similarity_ab:.4f}")
    print(f"Similarity A <-> C: {similarity_ac:.4f}")


if __name__ == "__main__":
    main()