from rag.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """
You are a retrieval-augmented generation assistant.

Answer the user's question using only the provided retrieved context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the retrieved context does not contain enough information,
  clearly say that the available context is insufficient.
- Only cite sources that directly support the answer.
- Citations must refer only to the source numbers provided in
  the retrieved context.
- Do not invent source numbers.
- Do not invent page numbers or filenames.
- Do not place citation markers inside the answer text.
- Return citations only through the "citations" JSON field.
- Return only valid JSON.
- Do not wrap the JSON in Markdown code fences.

Your response must have exactly this structure:

{
  "answer": "Your grounded answer here.",
  "citations": [1, 2]
}

The "citations" field must contain only the integer source numbers
that directly support the answer.

If the context is insufficient, return:

{
  "answer": "The retrieved context does not contain enough information to answer this question.",
  "citations": []
}
""".strip()


def build_context(
    chunks: list[RetrievedChunk],
) -> str:
    """
    Convert retrieved chunks into numbered context blocks.

    Source numbers are temporary identifiers used by the LLM.
    Real citation metadata such as filename and page number is
    attached later by the application after validation.
    """

    if not chunks:
        return "No retrieved context is available."

    context_parts: list[str] = []

    for source_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        context_parts.append(
            f"[Source {source_number}]\n"
            f"Content:\n{chunk.text}"
        )

    return "\n\n".join(context_parts)


def build_user_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build the grounded user prompt sent to the LLM.

    The model receives numbered context passages and must return
    structured JSON containing an answer and cited source IDs.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty"
        )

    context = build_context(
        chunks=chunks,
    )

    return (
        "Retrieved context:\n\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Return the answer using the required JSON format."
    )