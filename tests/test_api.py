from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.generator import GeneratedAnswer
from rag.ingestion.service import IngestionResult
from rag.retrieval.retriever import RetrievedChunk

TEST_DOCUMENT_ID = "test-document-id"


class FakeRetriever:
    """
    Deterministic retriever used in API unit tests.

    It avoids loading embedding models, BM25, Qdrant,
    or the cross-encoder.
    """

    def search(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="AI is discussed in this test chunk.",
                document_id=document_id,
                filename="test.pdf",
                page_number=1,
                chunk_index=0,
                score=0.9,
            )
        ]

    def invalidate_document(
        self,
        document_id: str,
    ) -> None:
        """
        Match the production retriever interface used after uploads.
        """


class EmptyRetriever(FakeRetriever):
    """
    Simulates a document_id with no indexed chunks.
    """

    def search(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return []


class FakeGenerator:
    """
    Fake generator that avoids calling Gemini.
    """

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Test answer",
            citations=[],
            sources=[],
        )


class FailingGenerator:
    """
    Simulates an unexpected generation failure.
    """

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        raise RuntimeError(
            "Simulated provider failure"
        )


class FakeIngestionService:
    """
    Simulates successful document ingestion without parsing,
    embedding, or talking to Qdrant.
    """

    def ingest_pdf(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> IngestionResult:
        if document_id is None:
            raise ValueError(
                "Test ingestion requires document_id"
            )

        path = Path(file_path)

        return IngestionResult(
            document_id=document_id,
            filename=path.name,
            page_count=2,
            chunk_count=4,
        )


def create_test_client(
    retriever: FakeRetriever | None = None,
    generator: FakeGenerator | FailingGenerator | None = None,
) -> TestClient:
    """
    Create a lightweight API instance without loading
    real RAG models or external services.
    """

    app = create_app(
        initialize_rag=False,
    )

    app.state.retriever = (
        retriever
        if retriever is not None
        else FakeRetriever()
    )

    app.state.generator = (
        generator
        if generator is not None
        else FakeGenerator()
    )

    app.state.ingestion_service = (
        FakeIngestionService()
    )

    return TestClient(app)


def test_health_endpoint() -> None:
    with create_test_client() as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_query_endpoint() -> None:
    with create_test_client() as client:
        response = client.post(
            "/query",
            json={
                "document_id": TEST_DOCUMENT_ID,
                "question": "What is AI?",
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "Test answer",
        "citations": [],
    }


def test_query_requires_document_id() -> None:
    with create_test_client() as client:
        response = client.post(
            "/query",
            json={
                "question": "What is AI?",
            },
        )

    assert response.status_code == 422


def test_query_rejects_empty_question() -> None:
    with create_test_client() as client:
        response = client.post(
            "/query",
            json={
                "document_id": TEST_DOCUMENT_ID,
                "question": "",
            },
        )

    assert response.status_code == 422


def test_query_returns_404_for_unknown_document() -> None:
    with create_test_client(
        retriever=EmptyRetriever(),
    ) as client:
        response = client.post(
            "/query",
            json={
                "document_id": "missing-document",
                "question": "What is AI?",
            },
        )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "No indexed chunks found for "
            "the requested document_id"
        ),
    }


def test_query_handles_pipeline_failure() -> None:
    with create_test_client(
        generator=FailingGenerator(),
    ) as client:
        response = client.post(
            "/query",
            json={
                "document_id": TEST_DOCUMENT_ID,
                "question": "What is AI?",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal RAG pipeline error",
    }


def test_upload_document() -> None:
    pdf_bytes = (
        b"%PDF-1.7\n"
        b"fake test PDF bytes"
    )

    with create_test_client() as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "test.pdf",
                    pdf_bytes,
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 201

    payload = response.json()

    UUID(payload["document_id"])

    assert payload["filename"] == "test.pdf"
    assert payload["page_count"] == 2
    assert payload["chunk_count"] == 4


def test_upload_rejects_non_pdf_extension() -> None:
    with create_test_client() as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "notes.txt",
                    b"%PDF-1.7\nfake",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415

    assert response.json() == {
        "detail": "Only PDF files are supported",
    }


def test_upload_rejects_invalid_pdf_content() -> None:
    with create_test_client() as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "test.pdf",
                    b"this is not a PDF",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 415

    assert response.json() == {
        "detail": "Uploaded file is not a valid PDF",
    }
