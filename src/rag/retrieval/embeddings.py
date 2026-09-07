from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Owns the embedding model and its tokenizer.

    The tokenizer is used both for token-aware chunking and by the
    embedding model itself when creating semantic vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        # Loading the model can take time and memory, so we load it once
        # when the service starts rather than once per chunk.
        self.model = SentenceTransformer(model_name)

        # Use exactly the tokenizer associated with our embedding model.
        self.tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokenizer tokens contained in the text.
        """

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        return len(encoded["input_ids"])

    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        """
        Hard-split text according to tokenizer tokens.

        This should rarely be used. It exists only as the final fallback
        when structural boundaries cannot create sufficiently small pieces.
        """

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        token_ids = encoded["input_ids"]

        pieces: list[str] = []

        for start in range(0, len(token_ids), max_tokens):
            token_slice = token_ids[start : start + max_tokens]

            # Convert the token IDs back into readable text.
            piece = self.tokenizer.decode(
                token_slice,
                skip_special_tokens=True,
            ).strip()

            if piece:
                pieces.append(piece)

        return pieces

    def embed_text(self, text: str) -> list[float]:
        """
        Convert one text string into a dense embedding vector.

        Returns:
            A list of floating-point values representing the semantic
            meaning of the input text.
        """

        if not text.strip():
            raise ValueError("Cannot embed empty text")

        # SentenceTransformer internally:
        # text -> tokenizer -> token IDs -> transformer -> pooled embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        # The model returns a NumPy array.
        # Convert it into a normal Python list for easier storage/serialization.
        return embedding.tolist()


    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple texts into embeddings efficiently.

        Batch processing is much faster than calling embed_text()
        repeatedly for hundreds or thousands of document chunks.
        """

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()