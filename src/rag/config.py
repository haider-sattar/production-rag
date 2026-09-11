import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
)

QDRANT_TIMEOUT_SECONDS = float(
    os.getenv(
        "QDRANT_TIMEOUT_SECONDS",
        "120",
    )
)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]