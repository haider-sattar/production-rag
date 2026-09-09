import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from rag.api.schemas import (
    CitationResponse,
    DocumentDeleteResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from rag.generation.clients.gemini_client import GeminiClient
from rag.generation.generator import Generator
from rag.ingestion.service import IngestionService
from rag.observability.langfuse import get_langfuse_client
from rag.observability.logging import configure_logging
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.embeddings import EmbeddingService
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranking_retriever import RerankingRetriever
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore

load_dotenv()

configure_logging()

logger = logging.getLogger(
    "rag.api"
)


MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024


def create_retriever(
    embedding_service: EmbeddingService | None = None,
    qdrant_url: str = "http://localhost:6333",
) -> RerankingRetriever:
    """
    Create the production retrieval pipeline.
    """

    if embedding_service is None:
        embedding_service = EmbeddingService()

    dense_retriever = Retriever(
        embedding_service=embedding_service,
        url=qdrant_url,
    )

    bm25_retriever = BM25Retriever(
        url=qdrant_url,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        candidate_k=30,
        rrf_constant=60,
    )

    return RerankingRetriever(
        hybrid_retriever=hybrid_retriever,
        rerank_candidates=30,
    )


def create_generator() -> Generator:
    """
    Create the production generation pipeline.
    """

    return Generator(
        llm_client=GeminiClient(),
    )


def create_app(
    initialize_rag: bool = True,
) -> FastAPI:
    """
    Application factory.

    initialize_rag=True:
        Production/development application.
        Heavy RAG components are loaded during startup.

    initialize_rag=False:
        Lightweight application used for unit tests.
    """

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:

        if initialize_rag:
            embedding_service = EmbeddingService()
            qdrant_url = os.getenv(
                "QDRANT_URL",
                "http://localhost:6333",
            )

            vector_store = VectorStore(
                url=qdrant_url,
            )

            app.state.retriever = (
                create_retriever(
                    embedding_service=embedding_service,
                    qdrant_url=qdrant_url,
                )
            )

            app.state.generator = (
                create_generator()
            )

            app.state.ingestion_service = (
                IngestionService(
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                )
            )

        yield

    application = FastAPI(
        title="Production RAG API",
        version="0.1.0",
        description=(
            "Production-style Retrieval-Augmented Generation API "
            "with hybrid retrieval, reranking, generation, "
            "validated citations, and observability."
        ),
        lifespan=lifespan,
    )

    def get_retriever(
        request: Request,
    ) -> RerankingRetriever:
        """
        Return the application-wide retriever.
        """

        return request.app.state.retriever

    def get_generator(
        request: Request,
    ) -> Generator:
        """
        Return the application-wide generator.
        """

        return request.app.state.generator

    def get_ingestion_service(
        request: Request,
    ) -> IngestionService:
        """
        Return the application-wide ingestion service.
        """

        return request.app.state.ingestion_service

    @application.get("/health")
    def health_check() -> dict[str, str]:
        """
        Verify that the API process is running.
        """

        return {
            "status": "ok",
        }

    @application.post(
        "/documents",
        response_model=DocumentUploadResponse,
        status_code=201,
    )
    async def upload_document(
        file: Annotated[
            UploadFile,
            File(
                description="PDF document to index.",
            ),
        ],
        ingestion_service: Annotated[
            IngestionService,
            Depends(get_ingestion_service),
        ],
        retriever: Annotated[
            RerankingRetriever,
            Depends(get_retriever),
        ],
    ) -> DocumentUploadResponse:
        """
        Upload and index one PDF.

        The endpoint validates the upload, assigns a fresh document_id,
        runs the existing ingestion pipeline, and returns metadata needed
        for subsequent /query requests.
        """

        filename = Path(
            file.filename or "document.pdf"
        ).name

        if Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported",
            )

        contents = await file.read(
            MAX_PDF_SIZE_BYTES + 1
        )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty",
            )

        if len(contents) > MAX_PDF_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 20 MB upload limit",
            )

        if not contents.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=415,
                detail="Uploaded file is not a valid PDF",
            )

        document_id = str(
            uuid4()
        )

        logger.info(
            (
                "document_upload_started "
                "document_id=%s "
                "filename=%r "
                "size_bytes=%d"
            ),
            document_id,
            filename,
            len(contents),
        )

        try:
            with TemporaryDirectory() as temp_dir:
                temp_path = (
                    Path(temp_dir) / filename
                )

                temp_path.write_bytes(
                    contents
                )

                result = await run_in_threadpool(
                    ingestion_service.ingest_pdf,
                    temp_path,
                    document_id,
                )

            # The dense retriever reads Qdrant directly. BM25 keeps
            # a per-document cache, so invalidate it after indexing.
            retriever.invalidate_document(
                document_id=document_id,
            )

            logger.info(
                (
                    "document_upload_completed "
                    "document_id=%s "
                    "filename=%r "
                    "pages=%d "
                    "chunks=%d"
                ),
                result.document_id,
                result.filename,
                result.page_count,
                result.chunk_count,
            )

            return DocumentUploadResponse(
                document_id=result.document_id,
                filename=result.filename,
                page_count=result.page_count,
                chunk_count=result.chunk_count,
            )

        except HTTPException:
            raise

        except ValueError as exc:
            logger.warning(
                (
                    "document_upload_failed "
                    "document_id=%s "
                    "error=%s"
                ),
                document_id,
                exc,
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            logger.exception(
                (
                    "document_upload_failed "
                    "document_id=%s "
                    "error_type=internal"
                ),
                document_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Document ingestion failed",
            ) from exc

    @application.delete(
        "/documents/{document_id}",
        response_model=DocumentDeleteResponse,
    )
    def delete_document(
        document_id: str,
        ingestion_service: Annotated[
            IngestionService,
            Depends(get_ingestion_service),
        ],
        retriever: Annotated[
            RerankingRetriever,
            Depends(get_retriever),
        ],
    ) -> DocumentDeleteResponse:
        """
        Delete a document's vectors and invalidate its retrieval cache.
        """

        logger.info(
            (
                "document_delete_started "
                "document_id=%s"
            ),
            document_id,
        )

        try:
            ingestion_service.vector_store.delete_document(
                document_id=document_id,
            )

            retriever.invalidate_document(
                document_id=document_id,
            )

            logger.info(
                (
                    "document_delete_completed "
                    "document_id=%s"
                ),
                document_id,
            )

            return DocumentDeleteResponse(
                document_id=document_id,
                deleted=True,
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                (
                    "document_delete_failed "
                    "document_id=%s "
                    "error_type=internal"
                ),
                document_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Document deletion failed",
            ) from exc

    @application.post(
        "/query",
        response_model=QueryResponse,
    )
    def query_rag(
        request: QueryRequest,
        retriever: Annotated[
            RerankingRetriever,
            Depends(get_retriever),
        ],
        generator: Annotated[
            Generator,
            Depends(get_generator),
        ],
    ) -> QueryResponse:
        """
        Execute the complete RAG pipeline.

        The request is traced in Langfuse with separate
        retrieval and generation observations.
        """

        request_id = str(
            uuid4()
        )

        total_start = time.perf_counter()

        langfuse = get_langfuse_client()

        logger.info(
            (
                "request_started "
                "request_id=%s "
                "document_id=%s "
                "question=%r"
            ),
            request_id,
            request.document_id,
            request.question,
        )

        try:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="rag-query",
                input={
                    "document_id": request.document_id,
                    "question": request.question,
                },
                metadata={
                    "request_id": request_id,
                    "document_id": request.document_id,
                },
            ) as rag_span:

                retrieval_start = (
                    time.perf_counter()
                )

                with langfuse.start_as_current_observation(
                    as_type="retriever",
                    name="hybrid-retrieval-reranking",
                    input={
                        "query": request.question,
                        "document_id": request.document_id,
                        "top_k": 5,
                        "rerank_candidates": 30,
                    },
                ) as retrieval_span:

                    chunks = retriever.search(
                        query=request.question,
                        document_id=request.document_id,
                        top_k=5,
                    )

                    if not chunks:
                        raise HTTPException(
                            status_code=404,
                            detail=(
                                "No indexed chunks found for "
                                "the requested document_id"
                            ),
                        )

                    retrieval_ms = (
                        time.perf_counter()
                        - retrieval_start
                    ) * 1000

                    retrieval_span.update(
                        output=[
                            {
                                "rank": rank,
                                "filename": (
                                    chunk.filename
                                ),
                                "page_number": (
                                    chunk.page_number
                                ),
                                "chunk_index": (
                                    chunk.chunk_index
                                ),
                                "score": (
                                    float(chunk.score)
                                ),
                            }
                            for rank, chunk in enumerate(
                                chunks,
                                start=1,
                            )
                        ],
                        metadata={
                            "retrieval_ms": (
                                retrieval_ms
                            ),
                            "retrieved_chunks": (
                                len(chunks)
                            ),
                        },
                    )

                generation_start = (
                    time.perf_counter()
                )

                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name="gemini-generation",
                    model="gemini-2.5-flash",
                    input={
                        "document_id": request.document_id,
                        "question": request.question,
                        "sources": [
                            {
                                "source_id": index,
                                "filename": (
                                    chunk.filename
                                ),
                                "page_number": (
                                    chunk.page_number
                                ),
                                "chunk_index": (
                                    chunk.chunk_index
                                ),
                            }
                            for index, chunk in enumerate(
                                chunks,
                                start=1,
                            )
                        ],
                    },
                ) as generation_span:

                    result = generator.generate(
                        query=request.question,
                        chunks=chunks,
                    )

                    generation_ms = (
                        time.perf_counter()
                        - generation_start
                    ) * 1000

                    generation_span.update(
                        output={
                            "answer": (
                                result.answer
                            ),
                            "citations": [
                                citation.source_id
                                for citation
                                in result.citations
                            ],
                        },
                        metadata={
                            "generation_ms": (
                                generation_ms
                            ),
                            "citation_count": (
                                len(
                                    result.citations
                                )
                            ),
                        },
                    )

                citations = [
                    CitationResponse(
                        source_id=(
                            citation.source_id
                        ),
                        filename=(
                            citation.filename
                        ),
                        page_number=(
                            citation.page_number
                        ),
                        chunk_index=(
                            citation.chunk_index
                        ),
                    )
                    for citation in result.citations
                ]

                total_ms = (
                    time.perf_counter()
                    - total_start
                ) * 1000

                rag_span.update(
                    output={
                        "answer": (
                            result.answer
                        ),
                        "citations": [
                            citation.source_id
                            for citation
                            in result.citations
                        ],
                    },
                    metadata={
                        "request_id": (
                            request_id
                        ),
                        "document_id": (
                            request.document_id
                        ),
                        "retrieval_ms": (
                            retrieval_ms
                        ),
                        "generation_ms": (
                            generation_ms
                        ),
                        "total_ms": (
                            total_ms
                        ),
                        "retrieved_chunks": (
                            len(chunks)
                        ),
                        "citations": (
                            len(citations)
                        ),
                    },
                )

            logger.info(
                (
                    "request_completed "
                    "request_id=%s "
                    "document_id=%s "
                    "retrieval_ms=%.2f "
                    "generation_ms=%.2f "
                    "total_ms=%.2f "
                    "retrieved_chunks=%d "
                    "citations=%d"
                ),
                request_id,
                request.document_id,
                retrieval_ms,
                generation_ms,
                total_ms,
                len(chunks),
                len(citations),
            )

            return QueryResponse(
                answer=result.answer,
                citations=citations,
            )

        except HTTPException:
            raise

        except ValueError as exc:
            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            logger.warning(
                (
                    "request_failed "
                    "request_id=%s "
                    "error_type=value_error "
                    "total_ms=%.2f "
                    "error=%s"
                ),
                request_id,
                total_ms,
                exc,
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            logger.exception(
                (
                    "request_failed "
                    "request_id=%s "
                    "error_type=internal "
                    "total_ms=%.2f"
                ),
                request_id,
                total_ms,
            )

            raise HTTPException(
                status_code=500,
                detail="Internal RAG pipeline error",
            ) from exc

    return application


app = create_app()
