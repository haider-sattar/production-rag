import pytest

from rag.ingestion.chunker import (
    chunk_pages,
    merge_pieces,
    recursive_split,
    split_preserving_separator,
)
from rag.ingestion.parser import DocumentPage


def fake_token_counter(text: str) -> int:
    """
    Simple deterministic token counter used only for unit tests.

    We treat each whitespace-separated word as one token.

    This is NOT intended to reproduce the MiniLM tokenizer.
    It lets us test chunking behavior without loading an ML model.
    """

    return len(text.split())


def fake_token_splitter(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    Simple token-based fallback used only in unit tests.

    Words act as tokens and are grouped according to max_tokens.
    """

    words = text.split()

    return [
        " ".join(words[start : start + max_tokens])
        for start in range(0, len(words), max_tokens)
    ]


def make_page(
    text: str,
    document_id: str = "test-document",
    filename: str = "test.pdf",
    page_number: int = 1,
) -> DocumentPage:
    """
    Create a fake parsed PDF page.

    This keeps chunker tests independent from the PDF parser itself.
    """

    return DocumentPage(
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        text=text,
    )


def test_short_text_creates_one_chunk() -> None:
    """
    Text already below the token limit should remain as one chunk.
    """

    page = make_page("Machine learning predicts failures.")

    chunks = chunk_pages(
        pages=[page],
        max_tokens=10,
        overlap_tokens=2,
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Machine learning predicts failures."


def test_empty_page_creates_no_chunks() -> None:
    """
    Pages containing no useful text should be ignored.
    """

    page = make_page("   ")

    chunks = chunk_pages(
        pages=[page],
        max_tokens=10,
        overlap_tokens=2,
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    assert chunks == []


def test_metadata_is_preserved() -> None:
    """
    Document metadata must survive the chunking process because
    it will later be stored in Qdrant and used for citations.
    """

    page = make_page(
        text="Machine learning predicts equipment failure.",
        document_id="abc123",
        filename="report.pdf",
        page_number=8,
    )

    chunks = chunk_pages(
        pages=[page],
        max_tokens=10,
        overlap_tokens=2,
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    assert chunks[0].document_id == "abc123"
    assert chunks[0].filename == "report.pdf"
    assert chunks[0].page_number == 8
    assert chunks[0].chunk_index == 0


def test_chunk_indices_increase() -> None:
    """
    Every chunk should receive a unique sequential chunk index.
    """

    page = make_page("one two three four five six seven eight nine ten")

    chunks = chunk_pages(
        pages=[page],
        max_tokens=4,
        overlap_tokens=1,
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    indices = [chunk.chunk_index for chunk in chunks]

    assert indices == list(range(len(chunks)))


def test_all_chunks_respect_max_token_limit() -> None:
    """
    No final chunk should exceed the configured model token budget.
    """

    page = make_page(
        "one two three four five six seven eight "
        "nine ten eleven twelve thirteen fourteen"
    )

    chunks = chunk_pages(
        pages=[page],
        max_tokens=5,
        overlap_tokens=1,
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    for chunk in chunks:
        assert fake_token_counter(chunk.text) <= 5


def test_split_preserves_separator() -> None:
    """
    Structural splitting should not silently remove sentence punctuation.
    """

    text = "Machine learning works. Predictive maintenance helps."

    pieces = split_preserving_separator(
        text=text,
        separator=". ",
    )

    reconstructed = "".join(pieces)

    assert reconstructed == text


def test_recursive_split_prefers_natural_boundaries() -> None:
    """
    Large text should be split using available structural boundaries
    before falling back to hard token slicing.
    """

    text = "Machine learning predicts failures.\n\nSensors measure machine vibration."

    pieces = recursive_split(
        text=text,
        max_tokens=4,
        separators=["\n\n", "\n", ". ", " "],
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    assert len(pieces) > 1

    for piece in pieces:
        assert fake_token_counter(piece) <= 4


def test_hard_token_fallback() -> None:
    """
    If no useful structural separator remains, the supplied token
    splitter should guarantee pieces remain under max_tokens.
    """

    text = "one two three four five six seven eight"

    pieces = recursive_split(
        text=text,
        max_tokens=3,
        # Empty separators force the final token-based fallback.
        separators=[],
        token_counter=fake_token_counter,
        token_splitter=fake_token_splitter,
    )

    assert len(pieces) == 3

    for piece in pieces:
        assert fake_token_counter(piece) <= 3


def test_merge_pieces_creates_overlap() -> None:
    """
    Content from the end of one chunk should be carried into the
    following chunk when overlap is enabled.
    """

    pieces = [
        "one two ",
        "three four ",
        "five six ",
        "seven eight",
    ]

    chunks = merge_pieces(
        pieces=pieces,
        max_tokens=4,
        overlap_tokens=2,
        token_counter=fake_token_counter,
    )

    assert len(chunks) > 1

    # The ending piece of chunk 1 should also appear in chunk 2.
    assert "three four" in chunks[0]
    assert "three four" in chunks[1]


def test_zero_overlap_does_not_repeat_content() -> None:
    """
    overlap_tokens=0 must completely disable overlap.
    """

    pieces = [
        "one two ",
        "three four ",
        "five six ",
    ]

    chunks = merge_pieces(
        pieces=pieces,
        max_tokens=4,
        overlap_tokens=0,
        token_counter=fake_token_counter,
    )

    assert len(chunks) == 2

    assert chunks[0] == "one two three four"
    assert chunks[1] == "five six"


def test_invalid_max_tokens_raises_error() -> None:
    page = make_page("hello world")

    with pytest.raises(ValueError):
        chunk_pages(
            pages=[page],
            max_tokens=0,
            overlap_tokens=0,
            token_counter=fake_token_counter,
            token_splitter=fake_token_splitter,
        )


def test_negative_overlap_raises_error() -> None:
    page = make_page("hello world")

    with pytest.raises(ValueError):
        chunk_pages(
            pages=[page],
            max_tokens=10,
            overlap_tokens=-1,
            token_counter=fake_token_counter,
            token_splitter=fake_token_splitter,
        )


def test_overlap_must_be_smaller_than_max_tokens() -> None:
    page = make_page("hello world")

    with pytest.raises(ValueError):
        chunk_pages(
            pages=[page],
            max_tokens=10,
            overlap_tokens=10,
            token_counter=fake_token_counter,
            token_splitter=fake_token_splitter,
        )
