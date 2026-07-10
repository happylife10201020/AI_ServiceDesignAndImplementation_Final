"""RAG: 강의자료 PDF를 텍스트로 뽑아 청크로 나누고 Chroma에 넣는다.

지식원은 repo에 포함된 knowledge/ 폴더의 PDF다(RAG, LangGraph, Agent, 과제 공지).
다른 PC에서 clone해도 바로 인덱싱되도록 문서를 repo 안에 둔다.
텍스트 추출은 pdf_extractor.extract_raw를 그대로 쓴다.
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

# 인덱싱 대상 문서 폴더 (repo에 포함). 여기에 PDF를 넣으면 인덱싱된다.
DOCS_DIR = BASE_DIR / "knowledge"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _load_documents() -> list[Document]:
    docs: list[Document] = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
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
        raise RuntimeError(f"인덱싱할 PDF가 없다. {DOCS_DIR}에 PDF를 넣어라.")

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
