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

from rag.config import (
    QDRANT_API_KEY,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_URL,
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
        url: str = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        collection_name: str = "documents",
        vector_size: int = 384,
    ) -> None:
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )

        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        """
        Ensure the Qdrant collection and required indexes exist.

        Existing stored data is not deleted.
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

        document_id is indexed because queries frequently restrict
        retrieval to one uploaded document.
        """

        collection_info = (
            self.client.get_collection(
                collection_name=(
                    self.collection_name
                ),
            )
        )

        if (
            "document_id"
            not in collection_info.payload_schema
        ):
            self.client.create_payload_index(
                collection_name=(
                    self.collection_name
                ),
                field_name="document_id",
                field_schema=(
                    PayloadSchemaType.KEYWORD
                ),
                wait=True,
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

        UUID5 produces the same point ID for the same:
            document_id + chunk_index

        Re-ingesting the same document therefore updates those
        points instead of producing duplicate point IDs.
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
        batch_size: int = 32,
    ) -> None:
        """
        Store document chunks and embeddings in Qdrant.

        Points are uploaded in batches so large documents do not create
        one oversized remote request.

        Args:
            chunks:
                Document chunks to store.

            embeddings:
                Dense vector corresponding to each chunk.

            batch_size:
                Maximum number of Qdrant points uploaded per request.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
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
                        "filename": (
                            chunk.filename
                        ),
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

        for start in range(
            0,
            len(points),
            batch_size,
        ):
            batch = points[
                start : start + batch_size
            ]

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=batch,
                wait=True,
            )

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every stored chunk belonging to one document.
        """

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=(
                self.build_document_filter(
                    document_id=document_id,
                )
            ),
            wait=True,
        )