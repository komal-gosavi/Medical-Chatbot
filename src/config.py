"""
config.py
---------
Loads all settings from your .env file.
We removed OPENAI_API_KEY and added GROQ_API_KEY (100% free).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file automatically


def load_config() -> dict:
    """
    Read all environment variables and return them as a dictionary.
    Raises ValueError if required keys are missing.
    """

    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    groq_key     = os.environ.get("GROQ_API_KEY", "").strip()

    # ── Validation ──────────────────────────────────────────────────────────
    missing = []
    if not pinecone_key:
        missing.append("PINECONE_API_KEY")
    if not groq_key:
        missing.append("GROQ_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please add them to your .env file."
        )
    # ────────────────────────────────────────────────────────────────────────

    return {
        # API keys
        "PINECONE_API_KEY" : pinecone_key,
        "GROQ_API_KEY"     : groq_key,

        # Pinecone settings
        "PINECONE_INDEX"   : os.environ.get("PINECONE_INDEX", "medical-chatbot"),
        "PINECONE_CLOUD"   : os.environ.get("PINECONE_CLOUD", "aws"),
        "PINECONE_REGION"  : os.environ.get("PINECONE_REGION", "us-east-1"),

        # Embedding model -- now hosted by Pinecone (no local torch/RAM needed)
        "EMBEDDING_MODEL"  : os.environ.get(
            "EMBEDDING_MODEL",
            "multilingual-e5-large"   # 1024-dim, runs on Pinecone's servers
        ),
        "EMBEDDING_DIMENSION": int(os.environ.get("EMBEDDING_DIMENSION", 1024)),

        # Groq / LLM settings
        "GROQ_MODEL"       : os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),

        # Server settings (used when running locally)
        "HOST"             : os.environ.get("HOST", "127.0.0.1"),
        "PORT"             : int(os.environ.get("PORT", 8000)),
        "DEBUG"            : os.environ.get("DEBUG", "False").lower() == "true",
    }
