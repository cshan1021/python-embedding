from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 경로 설정
    BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
    @property
    def STATIC_PATH(self) -> Path: return self.BASE_PATH / "app" / "static"
    @property
    def TEMPLATES_PATH(self) -> Path: return self.BASE_PATH / "app" / "templates"

    # OLLAMA API URL
    OLLAMA_ENDPOINT: str = "http://localhost:11434/api/embeddings"
    MODEL_NAME: str = "nicolasfer45/Octen-Embedding-4B-GGUF:latest"

    # Qdrant
    QDRANT_PATH: str = "./data/qdrant"
    QDRANT_COLLECTION_BLOCK_INGREDIENTS: str = "block_ingredients"

    # 환경 변수
    PROJECT_NAME: str = "Python LLM Serving"
    DB_URL: str = "sqlite:///./test.db"
    
    # env 파일 설정 (자동으로 파일 읽기)
    model_config = SettingsConfigDict(env_file=".env")

# 전역에서 사용할 객체 생성
settings = Settings()