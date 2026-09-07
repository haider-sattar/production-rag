from dataclasses import dataclass

from qdrant_client import QdrantClient

from rag.retrieval.embeddings import EmbeddingService


@dataclass
class RetrievedChunk:
    """
    Represents one chunk returned by the retrieval system.

    We use our own application-level object instead of exposing
    Qdrant's internal result objects to the rest of the project.
    """

    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float


class Retriever:
    """
    Retrieves semantically relevant document chunks from Qdrant.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ) -> None:
        # Query embeddings must be created with the same model
        # used for the stored document embeddings.
        self.embedding_service = embedding_service

        # Qdrant connection.
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Embed a query and retrieve the top-k most similar chunks.

        Args:
            query:
                Natural-language search query.

            top_k:
                Maximum number of chunks to return.

        Returns:
            Retrieved chunks ordered from most to least similar.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Turn the user's query into a dense vector.
        query_vector = self.embedding_service.embed_text(query)

        # Ask Qdrant for the closest stored vectors.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        retrieved_chunks: list[RetrievedChunk] = []

        for result in response.points:
            payload = result.payload or {}

            retrieved_chunks.append(
                RetrievedChunk(
                    text=str(payload.get("text", "")),
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    page_number=int(payload.get("page_number", -1)),
                    chunk_index=int(payload.get("chunk_index", -1)),
                    score=float(result.score),
                )
            )

        return retrieved_chunks