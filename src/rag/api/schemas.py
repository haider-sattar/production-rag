from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for the RAG query endpoint.
    """

    question: str = Field(
        min_length=1,
        description="User question to answer from the indexed documents.",
    )


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