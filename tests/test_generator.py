import pytest

from rag.generation.generator import Generator
from rag.retrieval.retriever import RetrievedChunk


class FakeLLMClient:
    """
    Fake LLM returning the structured format expected
    by Generator.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return """
        {
            "answer": "AI readiness depends on four foundations.",
            "citations": [1]
        }
        """


def make_chunk(
    page_number: int = 42,
    chunk_index: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=(
            "The four foundations of AI readiness are "
            "connectivity, compute, context, and competency."
        ),
        document_id="doc-1",
        filename="test.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        score=0.95,
    )


def test_generator_returns_answer() -> None:
    generator = Generator(
        llm_client=FakeLLMClient(),
    )

    chunk = make_chunk()

    result = generator.generate(
        query=(
            "What are the four foundations "
            "of AI readiness?"
        ),
        chunks=[chunk],
    )

    assert result.answer == (
        "AI readiness depends on four foundations."
    )

    assert result.sources == [chunk]


def test_generator_builds_trusted_citation() -> None:
    generator = Generator(
        llm_client=FakeLLMClient(),
    )

    chunk = make_chunk(
        page_number=42,
        chunk_index=143,
    )

    result = generator.generate(
        query="What are the four foundations?",
        chunks=[chunk],
    )

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.source_id == 1
    assert citation.filename == "test.pdf"
    assert citation.page_number == 42
    assert citation.chunk_index == 143


def test_generator_rejects_empty_query() -> None:
    generator = Generator(
        llm_client=FakeLLMClient(),
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        generator.generate(
            query="   ",
            chunks=[],
        )


class EmptyFakeLLMClient:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return "   "


def test_generator_rejects_empty_llm_response() -> None:
    generator = Generator(
        llm_client=EmptyFakeLLMClient(),
    )

    with pytest.raises(
        ValueError,
        match="LLM returned an empty response",
    ):
        generator.generate(
            query="What is AI?",
            chunks=[make_chunk()],
        )


class InvalidJSONFakeLLMClient:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return "This is not JSON."


def test_generator_rejects_invalid_json() -> None:
    generator = Generator(
        llm_client=InvalidJSONFakeLLMClient(),
    )

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        generator.generate(
            query="What is AI?",
            chunks=[make_chunk()],
        )


class InvalidCitationFakeLLMClient:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return """
        {
            "answer": "Some answer.",
            "citations": [99]
        }
        """


def test_generator_rejects_unknown_source_id() -> None:
    generator = Generator(
        llm_client=InvalidCitationFakeLLMClient(),
    )

    with pytest.raises(
        ValueError,
        match="does not exist in retrieved context",
    ):
        generator.generate(
            query="What is AI?",
            chunks=[make_chunk()],
        )


class DuplicateCitationFakeLLMClient:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return """
        {
            "answer": "Some answer.",
            "citations": [1, 1, 1]
        }
        """


def test_generator_deduplicates_citations() -> None:
    generator = Generator(
        llm_client=DuplicateCitationFakeLLMClient(),
    )

    result = generator.generate(
        query="What is AI?",
        chunks=[make_chunk()],
    )

    assert len(result.citations) == 1
    assert result.citations[0].source_id == 1