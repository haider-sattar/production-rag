from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import fitz


@dataclass
class DocumentPage:
    """
    Represents one extracted page from a source document.

    We keep metadata together with the page text because later
    this information will travel with chunks into the vector database
    and allow us to generate citations.
    """

    document_id: str
    filename: str
    page_number: int
    text: str


def parse_pdf(file_path: str | Path) -> list[DocumentPage]:
    """
    Read a PDF and convert each page into a structured DocumentPage object.

    Args:
        file_path:
            Path to the PDF file. It may be provided either as a string
            or as a pathlib.Path object.

    Returns:
        A list containing one DocumentPage object per PDF page.

    Raises:
        FileNotFoundError:
            If the supplied file does not exist.

        ValueError:
            If the supplied file is not a PDF.
    """

    # Convert the input into a Path object so we can work with paths
    # using methods such as .exists(), .name, and .suffix.
    path = Path(file_path)

    # Fail early with a clear error instead of allowing PyMuPDF
    # to fail later with a less understandable exception.
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # For this parser we currently support only PDF files.
    # .lower() also allows extensions such as ".PDF".
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # Give every ingested document a unique identifier.
    # We do not use the filename because multiple documents can have
    # the same filename.
    document_id = str(uuid4())

    # This list will contain the structured representation
    # of every page extracted from the PDF.
    pages: list[DocumentPage] = []

    # Using a context manager ensures the PDF is closed automatically,
    # even if an exception occurs while processing one of the pages.
    with fitz.open(path) as document:
        # enumerate() gives us both:
        #   index -> 0, 1, 2, ...
        #   page  -> the actual PyMuPDF page object
        for index, page in enumerate(document):
            # Extract plain text from the current PDF page.
            # strip() removes unnecessary whitespace at the beginning/end.
            text = page.get_text("text").strip()

            # Store the extracted text together with metadata.
            # page_number is index + 1 because Python uses zero-based
            # indexing while humans normally refer to PDF pages from 1.
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    filename=path.name,
                    page_number=index + 1,
                    text=text,
                )
            )

    # Return all parsed pages to the next stage of the pipeline.
    return pages
