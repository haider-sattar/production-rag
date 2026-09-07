import argparse



from rag.ingestion.chunker import chunk_pages
from rag.ingestion.parser import parse_pdf
from rag.retrieval.embeddings import EmbeddingService

from rag.retrieval.vector_store import VectorStore

def main() -> None:
    """
    Development CLI for testing the PDF ingestion and chunking pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the Production RAG system."
    )

    parser.add_argument(
        "file",
        help="Path to the PDF file.",
    )

    args = parser.parse_args()

    # Load the embedding model/tokenizer.
    embedding_service = EmbeddingService()

    # Stage 1: PDF -> structured pages.
    pages = parse_pdf(args.file)

    print(f"Parsed pages: {len(pages)}")

    # Stage 2: pages -> token-aware chunks.
    chunks = chunk_pages(
        pages=pages,
        # These are initial development values.
        # Later we will evaluate chunk sizes experimentally.
        max_tokens=220,
        overlap_tokens=40,
        # Pass the embedding model's tokenizer functionality
        # into our independent chunker.
        token_counter=embedding_service.count_tokens,
        token_splitter=embedding_service.split_by_tokens,
    )
    # Convert chunk text into embedding vectors.
    texts = [chunk.text for chunk in chunks]

    embeddings = embedding_service.embed_batch(texts)

    print(f"Created embeddings: {len(embeddings)}")

    # Connect to Qdrant.
    vector_store = VectorStore()

    # Ensure the collection exists before storing points.
    vector_store.create_collection()

    # Store vectors together with chunk metadata.
    vector_store.store_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"Stored {len(chunks)} chunks in Qdrant")

    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        token_count = embedding_service.count_tokens(chunk.text)

        print()
        print("=" * 70)
        print(f"Chunk: {chunk.chunk_index}")
        print(f"Page: {chunk.page_number}")
        print(f"Tokens: {token_count}")
        print("-" * 70)
        print(chunk.text)


if __name__ == "__main__":
    main()
