from langfuse import get_client


def get_langfuse_client():
    """
    Return the shared Langfuse client.

    Langfuse automatically reads these environment variables:

    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_BASE_URL

    The client is used to create traces and observations
    for the RAG pipeline.
    """

    return get_client()