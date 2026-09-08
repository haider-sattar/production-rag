from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.ingestion.chunker import DocumentChunk


class VectorStore:
    """
    Handles communication with Qdrant.

    Responsibilities:
    - create the vector collection
    - create payload indexes used for filtering
    - store chunk embeddings
    - delete all chunks belonging to one document
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
        vector_size: int = 384,
    ) -> None:
        """
        Connect to Qdrant.

        Args:
            url:
                Address where Qdrant is running.

            collection_name:
                Name of the Qdrant collection used for document chunks.

            vector_size:
                Number of dimensions produced by the embedding model.
                MiniLM-L6-v2 produces 384-dimensional embeddings.
        """

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        """
        Ensure the Qdrant collection and required indexes exist.

        Stored data is never deleted when this method is called.
        """

        if not self.client.collection_exists(
            self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """
        Create indexes for payload fields used in filtering.

        document_id is indexed because queries will frequently restrict
        retrieval to a single uploaded document.
        """

        collection_info = self.client.get_collection(
            collection_name=self.collection_name,
        )

        if (
            "document_id"
            not in collection_info.payload_schema
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    @staticmethod
    def build_document_filter(
        document_id: str,
    ) -> Filter:
        """
        Build a Qdrant filter restricting results to one document.
        """

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    ),
                )
            ]
        )

    @staticmethod
    def build_point_id(
        chunk: DocumentChunk,
    ) -> str:
        """
        Build a deterministic globally unique Qdrant point ID.

        chunk_index alone is not sufficient because every document
        starts chunk numbering from zero.

        UUID5 gives us a deterministic ID based on:
            document_id + chunk_index

        Re-ingesting the same document therefore updates the same
        points instead of creating duplicates.
        """

        return str(
            uuid5(
                NAMESPACE_URL,
                (
                    f"{chunk.document_id}:"
                    f"{chunk.chunk_index}"
                ),
            )
        )

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks and embeddings in Qdrant.

        Each chunk corresponds to exactly one vector and one
        deterministic Qdrant point.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return

        points: list[PointStruct] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            points.append(
                PointStruct(
                    id=self.build_point_id(
                        chunk
                    ),
                    vector=embedding,
                    payload={
                        "document_id": (
                            chunk.document_id
                        ),
                        "filename": chunk.filename,
                        "page_number": (
                            chunk.page_number
                        ),
                        "chunk_index": (
                            chunk.chunk_index
                        ),
                        "text": chunk.text,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every stored chunk belonging to one document.
        """

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.build_document_filter(
                document_id=document_id,
            ),
            wait=True,
        )