import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request

from rag.api.schemas import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
)
from rag.generation.clients.gemini_client import GeminiClient
from rag.generation.generator import Generator
from rag.observability.langfuse import get_langfuse_client
from rag.observability.logging import configure_logging
from rag.retrieval.reranking_retriever import RerankingRetriever


load_dotenv()

configure_logging()

logger = logging.getLogger(
    "rag.api"
)


def create_retriever() -> RerankingRetriever:
    """
    Create the production retrieval pipeline.
    """

    return RerankingRetriever(
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
            app.state.retriever = (
                create_retriever()
            )

            app.state.generator = (
                create_generator()
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

    @application.get("/health")
    def health_check() -> dict[str, str]:
        """
        Verify that the API process is running.
        """

        return {
            "status": "ok",
        }

    @application.post(
        "/query",
        response_model=QueryResponse,
    )
    def query_rag(
        request: QueryRequest,
        retriever: RerankingRetriever = Depends(
            get_retriever
        ),
        generator: Generator = Depends(
            get_generator
        ),
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
            "request_started request_id=%s question=%r",
            request_id,
            request.question,
        )

        try:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="rag-query",
                input={
                    "question": request.question,
                },
                metadata={
                    "request_id": request_id,
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
                        "top_k": 5,
                        "rerank_candidates": 30,
                    },
                ) as retrieval_span:

                    chunks = retriever.search(
                        query=request.question,
                        top_k=5,
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
                    "retrieval_ms=%.2f "
                    "generation_ms=%.2f "
                    "total_ms=%.2f "
                    "retrieved_chunks=%d "
                    "citations=%d"
                ),
                request_id,
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