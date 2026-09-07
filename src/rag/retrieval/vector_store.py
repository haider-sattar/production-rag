from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.ingestion.chunker import DocumentChunk


class VectorStore:
    """
    Handles communication with Qdrant.

    Responsibilities:
    - create the vector collection
    - store chunk embeddings
    - later: search for similar vectors
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
        Create the Qdrant collection if it does not already exist.
        """

        # We do not want to delete/recreate the collection every time
        # the application starts because that would destroy stored data.
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks and their embedding vectors in Qdrant.

        Each chunk corresponds to exactly one vector and one Qdrant point.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        points: list[PointStruct] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            # For now we build a deterministic numeric point ID using
            # the chunk index. Later we'll improve this for multiple
            # documents/users so IDs cannot collide.
            point_id = chunk.chunk_index

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
            )

        # Upsert means:
        # - insert a point if the ID does not exist
        # - update it if the ID already exists
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )