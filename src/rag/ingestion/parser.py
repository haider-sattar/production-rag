from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import fitz


@dataclass
class DocumentPage:
    """
    Represents one extracted page from a source document.

    Source metadata stays attached to every page so it can later
    propagate to chunks, retrieval results, and citations.
    """

    document_id: str
    filename: str
    page_number: int
    text: str


def parse_pdf(
    file_path: str | Path,
    document_id: str | None = None,
) -> list[DocumentPage]:
    """
    Read a PDF and convert each page into a structured DocumentPage object.

    Args:
        file_path:
            Path to the PDF file.

        document_id:
            Optional identifier assigned by the ingestion layer.
            If omitted, a new UUID is generated for backward compatibility
            with CLI/development usage.

    Returns:
        One DocumentPage object per PDF page.

    Raises:
        FileNotFoundError:
            If the supplied file does not exist.

        ValueError:
            If the supplied file is not a PDF or document_id is empty.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {path.suffix}"
        )

    if document_id is None:
        document_id = str(uuid4())
    elif not document_id.strip():
        raise ValueError(
            "document_id cannot be empty"
        )

    pages: list[DocumentPage] = []

    with fitz.open(path) as document:
        for index, page in enumerate(document):
            text = page.get_text(
                "text"
            ).strip()

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    filename=path.name,
                    page_number=index + 1,
                    text=text,
                )
            )

    return pages
