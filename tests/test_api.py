from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.generator import GeneratedAnswer


class FakeRetriever:
    """
    Deterministic retriever used in API unit tests.

    It avoids loading embedding models, BM25, Qdrant,
    or the cross-encoder.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list:
        return []


class FakeGenerator:
    """
    Fake generator that avoids calling Gemini.
    """

    def generate(
        self,
        query: str,
        chunks: list,
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
        chunks: list,
    ) -> GeneratedAnswer:
        raise RuntimeError(
            "Simulated provider failure"
        )


def create_test_client() -> TestClient:
    """
    Create a lightweight API instance without loading
    real RAG models.
    """

    app = create_app(
        initialize_rag=False,
    )

    app.state.retriever = FakeRetriever()
    app.state.generator = FakeGenerator()

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
                "question": "What is AI?",
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "Test answer",
        "citations": [],
    }


def test_query_rejects_empty_question() -> None:
    with create_test_client() as client:
        response = client.post(
            "/query",
            json={
                "question": "",
            },
        )

    assert response.status_code == 422


def test_query_handles_pipeline_failure() -> None:
    app = create_app(
        initialize_rag=False,
    )

    app.state.retriever = FakeRetriever()
    app.state.generator = FailingGenerator()

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What is AI?",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal RAG pipeline error",
    }