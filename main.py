"""AI 서비스 설계 학습 튜터.

강의자료(RAG/LangGraph/Agent PDF)를 근거로 학생 질문에 답하는 멀티턴 에이전트다.
RAG 검색, 도구 호출, SqliteSaver 메모리, LangGraph 분기/반복, 가드레일,
Pydantic 구조화 출력을 한데 묶었다.

사용법:
    python main.py                 대화형 실행
    python main.py --rebuild-index 강의자료 인덱스 재구성 후 실행
    python main.py --diagram       워크플로우 다이어그램만 출력
(의존성은 .venv에 있으니 venv를 활성화하거나 ./.venv/bin/python으로 실행한다.)
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys

from langchain_core.messages import HumanMessage

from config import MEMORY_DB, ConfigError, configure_logging, load_settings
from graph import build_graph
from rag import build_vectorstore

logger = logging.getLogger("main")

WELCOME = (
    "\nAI 서비스 설계 학습 튜터\n"
    "강의자료(LangChain/LangGraph/RAG/Agent)를 근거로 답합니다.\n"
    "예: 'RAG가 뭐야?', 'LangGraph의 조건부 엣지 설명해줘', 'checkpointer 정의'\n"
    "종료하려면 q 또는 quit 입력.\n"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 서비스 설계 학습 튜터")
    parser.add_argument("--rebuild-index", action="store_true", help="강의자료 벡터 인덱스 재구성")
    parser.add_argument("--diagram", action="store_true", help="워크플로우 다이어그램만 출력")
    parser.add_argument("--thread", default="tutor-session-1", help="대화 세션 thread_id")
    return parser.parse_args()


def _print_diagram(graph) -> None:
    print(graph.get_graph().draw_mermaid())


def _chat_loop(graph, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(WELCOME)
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return

        if question.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            return
        if not question:
            continue

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]}, config=config
            )
            print("\n답변>\n" + result["messages"][-1].content)
        except Exception as exc:  # noqa: BLE001 - 개별 오류로 루프가 죽지 않게
            logger.exception("대화 처리 중 오류: %s", exc)
            print(f"\n오류가 발생했습니다: {exc}")


def main() -> int:
    args = _parse_args()
    configure_logging()

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("모델: %s", settings.chat_model)

    # 인덱스가 없으면 만들고, --rebuild-index면 다시 만든다.
    try:
        vectorstore = build_vectorstore(settings, rebuild=args.rebuild_index)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as exc:  # noqa: BLE001
        logger.error("벡터스토어 준비 실패: %s", exc)
        return 1

    conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver(conn)
        graph = build_graph(settings, retriever, checkpointer)

        if args.diagram:
            _print_diagram(graph)
            return 0

        _chat_loop(graph, args.thread)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
