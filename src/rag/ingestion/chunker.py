from collections.abc import Callable
from dataclasses import dataclass
import re

from rag.ingestion.parser import DocumentPage


@dataclass
class DocumentChunk:
    """
    Represents one final piece of a document that will later be embedded.

    Source metadata stays attached to the chunk so that retrieved results
    can be traced back to the original PDF and page.
    """

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


# Separators are ordered from strongest document structure
# to weakest structure.
SEPARATORS = [
    "\n\n",  # Paragraph boundary
    "\n",  # Line boundary
    ". ",  # Approximate sentence boundary
    " ",  # Word boundary
]


def split_preserving_separator(text: str, separator: str) -> list[str]:
    """
    Split text while keeping separators in the resulting pieces.

    Normal str.split() removes the separator. For document processing,
    we generally do not want chunking to silently remove punctuation
    or whitespace from the original text.
    """

    # Parentheses create a capturing group, which makes re.split()
    # include the separator in its result.
    parts = re.split(f"({re.escape(separator)})", text)

    pieces: list[str] = []

    # Reattach each separator to the text that appeared before it.
    for index in range(0, len(parts), 2):
        piece = parts[index]

        if index + 1 < len(parts):
            piece += parts[index + 1]

        if piece:
            pieces.append(piece)

    return pieces


def recursive_split(
    text: str,
    max_tokens: int,
    separators: list[str],
    token_counter: Callable[[str], int],
    token_splitter: Callable[[str, int], list[str]],
) -> list[str]:
    """
    Recursively split text while preserving the strongest available
    document boundaries.

    Chunk size is measured using tokens from the embedding model rather
    than raw character count.

    Args:
        text:
            Text to split.

        max_tokens:
            Maximum number of model tokens allowed in one piece.

        separators:
            Boundaries to try, ordered from strongest to weakest.

        token_counter:
            Function that returns the number of embedding-model tokens
            contained in a string.

        token_splitter:
            Last-resort function capable of splitting text directly
            according to the embedding model's tokenizer.

    Returns:
        Pieces where each piece is at most max_tokens.
    """

    if not text.strip():
        return []

    # If the complete text already fits into our token budget,
    # preserve it instead of splitting unnecessarily.
    if token_counter(text) <= max_tokens:
        return [text]

    # If paragraph/line/sentence/word boundaries were all exhausted,
    # perform a hard token-based split as the final fallback.
    if not separators:
        return token_splitter(text, max_tokens)

    separator = separators[0]

    # If this particular boundary does not exist in the text,
    # try the next weaker boundary.
    if separator not in text:
        return recursive_split(
            text=text,
            max_tokens=max_tokens,
            separators=separators[1:],
            token_counter=token_counter,
            token_splitter=token_splitter,
        )

    # Split without silently deleting punctuation or whitespace.
    pieces = split_preserving_separator(text, separator)

    final_pieces: list[str] = []

    for piece in pieces:
        if not piece.strip():
            continue

        # Keep pieces that already fit.
        if token_counter(piece) <= max_tokens:
            final_pieces.append(piece)

        else:
            # The piece is still too large, so recursively try
            # weaker structural boundaries.
            smaller_pieces = recursive_split(
                text=piece,
                max_tokens=max_tokens,
                separators=separators[1:],
                token_counter=token_counter,
                token_splitter=token_splitter,
            )

            final_pieces.extend(smaller_pieces)

    return final_pieces


def merge_pieces(
    pieces: list[str],
    max_tokens: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    """
    Pack small structure-aware pieces into final chunks.

    Final chunks stay under max_tokens, while some complete pieces
    from the previous chunk are reused to provide contextual overlap.
    """

    if not pieces:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []

    for piece in pieces:
        # Check what the chunk would look like if this piece were added.
        candidate = "".join(current_parts + [piece])

        # If adding this piece exceeds the token budget,
        # finalize the current chunk first.
        if current_parts and token_counter(candidate) > max_tokens:
            chunk_text = "".join(current_parts).strip()

            if chunk_text:
                chunks.append(chunk_text)

            # ---------------------------------------------------------
            # Build overlap from the END of the completed chunk.
            # ---------------------------------------------------------

            # If overlap is disabled, start the next chunk fresh.
            if overlap_tokens == 0:
                overlap_parts: list[str] = []

            else:
                overlap_parts = []

                # Work backwards through the previous chunk because
                # overlap should come from its ending context.
                for previous_part in reversed(current_parts):
                    candidate_overlap = previous_part + "".join(overlap_parts)

                    # If we already have overlap and another complete
                    # piece would exceed the overlap target, stop.
                    if (
                        overlap_parts
                        and token_counter(candidate_overlap) > overlap_tokens
                    ):
                        break

                    overlap_parts.insert(0, previous_part)

                    # Stop once we have approximately reached
                    # the requested overlap.
                    if token_counter("".join(overlap_parts)) >= overlap_tokens:
                        break

            # The overlap plus the new piece must still fit
            # inside max_tokens.
            #
            # If it does not, remove the oldest overlapping
            # pieces until it fits.
            while overlap_parts:
                candidate_with_piece = "".join(overlap_parts + [piece])

                if token_counter(candidate_with_piece) <= max_tokens:
                    break

                overlap_parts.pop(0)

            # Start the next chunk using the retained overlap.
            current_parts = overlap_parts

        # Add the new piece to either:
        # 1. the existing chunk, or
        # 2. the newly started overlapping chunk.
        current_parts.append(piece)

    # Save the final unfinished chunk after the loop.
    if current_parts:
        final_chunk = "".join(current_parts).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def chunk_pages(
    pages: list[DocumentPage],
    max_tokens: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
    token_splitter: Callable[[str, int], list[str]],
) -> list[DocumentChunk]:
    """
    Convert parsed PDF pages into token-aware, overlapping chunks.

    Every PDF page is processed independently so page-level source
    metadata remains reliable for citations.

    Args:
        pages:
            Parsed pages returned by parse_pdf().

        max_tokens:
            Maximum number of embedding-model tokens in one chunk.

        overlap_tokens:
            Approximate number of tokens shared between neighboring chunks.

        token_counter:
            Function for counting tokens using the embedding model tokenizer.

        token_splitter:
            Function used for hard token splitting if natural boundaries
            cannot produce a sufficiently small piece.

    Returns:
        DocumentChunk objects ready for embedding.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        # Empty PDF pages contain nothing useful for retrieval.
        if not page.text.strip():
            continue

        # Stage 1:
        # Break the page into natural pieces while respecting
        # the embedding model's token limit.
        pieces = recursive_split(
            text=page.text,
            max_tokens=max_tokens,
            separators=SEPARATORS,
            token_counter=token_counter,
            token_splitter=token_splitter,
        )

        # Stage 2:
        # Pack those pieces into reasonably sized chunks and
        # preserve contextual overlap between neighboring chunks.
        page_chunks = merge_pieces(
            pieces=pieces,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            token_counter=token_counter,
        )

        # Stage 3:
        # Attach source metadata to every final chunk.
        for text in page_chunks:
            chunks.append(
                DocumentChunk(
                    document_id=page.document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )

            chunk_index += 1

    return chunks
