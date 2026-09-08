from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for the RAG query endpoint.

    document_id identifies the uploaded PDF that retrieval must
    be restricted to.
    """

    document_id: str = Field(
        min_length=1,
        description="Identifier of the uploaded document to query.",
    )
    question: str = Field(
        min_length=1,
        description="User question to answer from the selected document.",
    )


class DocumentUploadResponse(BaseModel):
    """
    Metadata returned after a PDF has been parsed, chunked,
    embedded, and stored successfully.
    """

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class CitationResponse(BaseModel):
    """
    Validated citation metadata returned by the API.
    """

    source_id: int
    filename: str
    page_number: int
    chunk_index: int


class QueryResponse(BaseModel):
    """
    Final API response containing the grounded answer
    and validated citations.
    """

    answer: str
    citations: list[CitationResponse]
