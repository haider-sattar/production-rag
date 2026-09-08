from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rag.ingestion.chunker import chunk_pages
from rag.ingestion.parser import parse_pdf
from rag.retrieval.embeddings import EmbeddingService
from rag.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class IngestionResult:
    """
    Summary returned after a document has been indexed successfully.
    """

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class IngestionService:
    """
    Orchestrates the complete PDF ingestion pipeline.

    Pipeline:
        PDF
        -> parse pages
        -> create token-aware chunks
        -> generate embeddings
        -> store chunks and vectors in Qdrant
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        max_tokens: int = 220,
        overlap_tokens: int = 40,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0"
            )

        if overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens cannot be negative"
            )

        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_tokens"
            )

        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def ingest_pdf(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> IngestionResult:
        """
        Parse, chunk, embed, and store one PDF.

        Args:
            file_path:
                Local path to the PDF.

            document_id:
                Optional externally generated identifier. If omitted,
                a fresh UUID is generated for this ingestion.

        Returns:
            Metadata describing the indexed document.

        Raises:
            ValueError:
                If the PDF contains no extractable text/chunks.
        """

        path = Path(file_path)

        if document_id is None:
            document_id = str(uuid4())
        elif not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        pages = parse_pdf(
            file_path=path,
            document_id=document_id,
        )

        chunks = chunk_pages(
            pages=pages,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            token_counter=(
                self.embedding_service.count_tokens
            ),
            token_splitter=(
                self.embedding_service.split_by_tokens
            ),
        )

        if not chunks:
            raise ValueError(
                "PDF contains no extractable text"
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_service.embed_batch(
                texts
            )
        )

        self.vector_store.create_collection()

        self.vector_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return IngestionResult(
            document_id=document_id,
            filename=path.name,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
