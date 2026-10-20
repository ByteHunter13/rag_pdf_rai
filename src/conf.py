"""
Configuración centralizada

Todo lo que necesiten los otros archivos
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT_DIR / "pdfs"
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
CHUNKS_DIR = ROOT_DIR / "data" / "chunks"

PDF_METADATA: dict[str, dict[str, str]] = {
    "AN 1702 FBI in Food Product Manufacturing with HH-XRF.pdf": {
        "topic": "Fast and Efficient Foreign Body Identification (FBI) in Food Product Manufacturing with HH-XRF",
        "doc_type": "application_note",
        "language": "en",
    },
}

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Base de datos (Postgres local con pgvector)
DATABASE_URL = os.getenv("DATABASE_URL")
VECTOR_TABLE = "chunks"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimensión por defecto de text-embedding-3-small; debe coincidir con vector(N) en la tabla
EMBEDDING_BATCH_SIZE = 100  # textos por request a la API

# Generación de respuestas
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Recuperación
TOP_K = 5  # chunks que se pasan como contexto al LLM

def ensure_dirs() -> None:
    """Crear los directorios de datos intermedios si no existen todavía"""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)