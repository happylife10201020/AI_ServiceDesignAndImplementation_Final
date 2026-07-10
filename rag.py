"""RAG: 강의자료 PDF를 텍스트로 뽑아 청크로 나누고 Chroma에 넣는다.

지식원은 과정에서 받은 강의자료 PDF(day10 RAG, day11 LangGraph, day13 Agent)와
과제 공지 PDF. 텍스트 추출은 pdf_extractor.extract_raw를 그대로 쓴다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_DIR, CHROMA_DIR, Settings
from pdf_extractor import extract_raw
from providers import get_embeddings

logger = logging.getLogger("rag")

COLLECTION_NAME = "course_docs"

# 인덱싱 대상. 없는 파일은 건너뛴다.
SOURCE_PDFS = [
    BASE_DIR.parent / "day10" / "RAG.pdf",
    BASE_DIR.parent / "day11" / "LangGraph.pdf",
    BASE_DIR.parent / "day13" / "Agent.pdf",
    BASE_DIR / "Agent 평가 과제 공지.pdf",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _load_documents() -> list[Document]:
    docs: list[Document] = []
    for pdf_path in SOURCE_PDFS:
        if not pdf_path.exists():
            logger.warning("강의자료 없음, 건너뜀: %s", pdf_path)
            continue

        raw = extract_raw(pdf_path)
        if not raw or not raw.strip():
            logger.warning("텍스트 없음, 건너뜀: %s", pdf_path.name)
            continue

        docs.append(
            Document(page_content=raw, metadata={"source": pdf_path.name, "path": str(pdf_path)})
        )
        logger.info("문서 로드: %s (%d자)", pdf_path.name, len(raw))

    return docs


def build_vectorstore(settings: Settings, *, rebuild: bool = False) -> Chroma:
    """Chroma 벡터스토어를 만들거나 기존 것을 불러온다. rebuild=True면 다시 만든다."""
    embeddings = get_embeddings(settings)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    existing = vectorstore._collection.count()
    if existing > 0 and not rebuild:
        logger.info("기존 벡터스토어 사용 (청크 %d개)", existing)
        return vectorstore

    if rebuild and existing > 0:
        logger.info("기존 인덱스 삭제 후 재구성 (청크 %d개)", existing)
        vectorstore.reset_collection()

    documents = _load_documents()
    if not documents:
        raise RuntimeError("인덱싱할 강의자료를 찾지 못했다. SOURCE_PDFS 경로를 확인해라.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("청크 %d개 임베딩 시작", len(chunks))

    vectorstore.add_documents(chunks)
    logger.info("인덱싱 완료: 청크 %d개 -> %s", len(chunks), CHROMA_DIR)
    return vectorstore


def get_retriever(settings: Settings, *, k: int = 4):
    vectorstore = build_vectorstore(settings, rebuild=False)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def index_is_ready() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
