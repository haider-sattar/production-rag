import argparse

from rag.ingestion.service import IngestionService


def main() -> None:
    """
    Development CLI for ingesting a PDF into the Production RAG system.
    """

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the Production RAG system."
    )

    parser.add_argument(
        "file",
        help="Path to the PDF file.",
    )

    args = parser.parse_args()

    ingestion_service = IngestionService()

    result = ingestion_service.ingest_pdf(
        file_path=args.file,
    )

    print(
        f"Ingested document: {result.filename}"
    )
    print(
        f"Document ID: {result.document_id}"
    )
    print(
        f"Parsed pages: {result.page_count}"
    )
    print(
        f"Stored chunks: {result.chunk_count}"
    )


if __name__ == "__main__":
    main()
