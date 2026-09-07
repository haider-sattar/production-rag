from rag.generation.prompt_builder import (
    build_context,
    build_user_prompt,
)
from rag.retrieval.retriever import RetrievedChunk


def make_chunk(
    text: str,
    page_number: int,
    chunk_index: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        document_id="doc-1",
        filename="test.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        score=0.9,
    )


def test_build_context_formats_sources() -> None:
    chunks = [
        make_chunk(
            text="First chunk content.",
            page_number=10,
            chunk_index=1,
        ),
        make_chunk(
            text="Second chunk content.",
            page_number=11,
            chunk_index=2,
        ),
    ]

    context = build_context(
        chunks=chunks,
    )

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "First chunk content." in context
    assert "Second chunk content." in context

    # Citation metadata is intentionally not exposed to the LLM.
    assert "Page:" not in context
    assert "test.pdf" not in context


def test_build_user_prompt_contains_query() -> None:
    chunks = [
        make_chunk(
            text="Some evidence.",
            page_number=5,
            chunk_index=1,
        )
    ]

    prompt = build_user_prompt(
        query="What is AI?",
        chunks=chunks,
    )

    assert "What is AI?" in prompt
    assert "Some evidence." in prompt
    assert "Question:" in prompt


def test_build_user_prompt_rejects_empty_query() -> None:
    chunks = []

    try:
        build_user_prompt(
            query="   ",
            chunks=chunks,
        )
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"
    else:
        raise AssertionError(
            "Expected ValueError for empty query"
        )