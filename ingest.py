"""강의자료 벡터 인덱스를 만드는 스크립트.

main.py가 알아서 인덱스를 만들긴 하지만, 미리 만들거나 강의자료가 바뀌어
다시 만들 때 쓴다.

    python ingest.py            인덱스 없으면 생성
    python ingest.py --rebuild  기존 인덱스 지우고 재생성
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import ConfigError, configure_logging, load_settings
from rag import build_vectorstore

logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="강의자료 RAG 인덱스 구축")
    parser.add_argument("--rebuild", action="store_true", help="기존 인덱스 지우고 재생성")
    args = parser.parse_args()

    configure_logging()
    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("임베딩 모델: %s", settings.embedding_model)
    try:
        vs = build_vectorstore(settings, rebuild=args.rebuild)
    except Exception as exc:  # noqa: BLE001
        logger.error("인덱싱 실패: %s", exc)
        return 1

    logger.info("인덱스 준비 완료. 청크 수: %d", vs._collection.count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
