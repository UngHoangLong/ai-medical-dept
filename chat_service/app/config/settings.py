from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # =======================
    # LLM / Models
    # =======================
    CHAT_MODEL: str = "deepseek-v4-flash"
    LLM_TEMPERATURE_CHAT: float = 0.7

    CORS_ORIGINS: str = "*"
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 3

    # =======================
    # API Keys
    # =======================
    DEEPSEEK_API_KEY: str

    # =======================
    # Streaming
    # =======================
    ENABLE_STREAMING: bool = True

    # =======================
    # Memory / Vector Store
    # =======================
    VECTOR_STORE_TYPE: Literal["faiss", "chroma", "pgvector"] = "faiss"
    VECTOR_STORE_PATH: str = "./data/vector_store"

    # =======================
    # Observability
    # =======================
    ENABLE_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None

    # =======================
    # MCP_SERVER
    # =======================
    MCP_SERVER_URL: str = "http://localhost:8000/mcp"
    
    
    # =======================
    # Database PostgresSQL
    # =======================
    PGUSER: str = "postgres"
    PGPORT: int = 5432
    PGDATABASE: str = "postgres"
    PGPASSWORD: str = "postgres"
    PGHOST: str = "127.0.0.1"
    DB_URI: str = None

    class Config:
        env_file = ".env"

settings = Settings()


