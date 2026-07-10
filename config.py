"""환경설정.

.env에서 OPENAI_API_KEY와 모델명을 읽는다. 키 값은 로그로 남기지 않는다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
MEMORY_DB = BASE_DIR / "chat_memory.db"
KNOWLEDGE_DIR = BASE_DIR / "output"


@dataclass(frozen=True)
class Settings:
    api_key: str
    chat_model: str
    embedding_model: str
    tavily_enabled: bool


class ConfigError(RuntimeError):
    pass


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )
    # 라이브러리 INFO 로그가 너무 시끄러워서 낮춘다.
    for noisy in ("httpx", "urllib3", "chromadb", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_settings() -> Settings:
    """.env를 읽어 설정을 만든다. OPENAI_API_KEY가 없으면 에러."""
    load_dotenv(BASE_DIR / ".env")

    api_key = _clean(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise ConfigError("OPENAI_API_KEY가 없습니다. .env를 확인하세요. (.env.example 참고)")

    return Settings(
        api_key=api_key,
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        tavily_enabled=bool(_clean(os.getenv("TAVILY_API_KEY"))),
    )


def _clean(value: str | None) -> str:
    """빈 값이나 .env.example의 자리표시자를 걸러낸다."""
    if not value:
        return ""
    value = value.strip()
    if value.endswith("...") or value in {"sk-...", "tvly-..."}:
        return ""
    return value
