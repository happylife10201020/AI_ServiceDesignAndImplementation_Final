"""에이전트가 쓰는 도구.

rag_search는 강의자료 검색, course_glossary는 용어 사전, web_search는 웹 검색이다.
Tavily 키가 없으면 web_search는 빠지고 나머지 둘로 최소 2개는 채운다.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from config import Settings

logger = logging.getLogger("tools")

# 외부 호출 없이 답하는 용어 사전
_GLOSSARY = {
    "rag": "Retrieval-Augmented Generation. 외부 지식(문서/DB)을 검색해 그 내용을 근거로 "
    "LLM 답변 품질을 높이는 기법.",
    "langgraph": "상태(State)와 노드/엣지로 LLM 워크플로우를 그래프로 설계하는 프레임워크. "
    "분기(conditional edge)와 반복(loop)으로 에이전트 흐름을 구성한다.",
    "stategraph": "LangGraph에서 상태를 공유하며 노드 간 전이를 정의하는 그래프 객체.",
    "checkpointer": "대화/실행 상태를 thread_id 단위로 저장·복원해 멀티턴 메모리를 만드는 컴포넌트 "
    "(InMemorySaver, SqliteSaver 등).",
    "tool": "에이전트가 자연어 판단으로 호출하는 함수. @tool 데코레이터로 정의한다.",
    "outputparser": "LLM 출력을 JSON/Pydantic 같은 구조화된 형태로 바꾸는 컴포넌트.",
    "conditional edge": "노드 결과(상태)에 따라 다음 노드를 정하는 분기 엣지.",
    "middleware": "로깅·가드레일·예외처리 등 운영 안정성을 담당하는 계층.",
    "embedding": "텍스트를 의미 벡터로 바꾼 것. 벡터 유사도로 관련 문서를 찾는다.",
    "runnable": "LangChain의 실행 단위 인터페이스. invoke/stream/batch로 파이프라인을 조합한다.",
}


def build_tools(settings: Settings, retriever) -> list:
    """설정과 리트리버를 묶어 도구 리스트를 만든다."""

    @tool
    def rag_search(query: str) -> str:
        """강의자료(RAG/LangGraph/Agent PDF)에서 질문과 관련된 내용을 찾는다.
        LangChain·LangGraph·RAG·Agent 개념 질문이면 이 도구를 쓴다.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "검색 결과가 없습니다."
        blocks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            blocks.append(f"[{i}] (출처: {source})\n{doc.page_content.strip()}")
        logger.info("rag_search: '%s' -> 문서 %d건", query[:40], len(docs))
        return "\n\n".join(blocks)

    @tool
    def course_glossary(term: str) -> str:
        """LangChain/LangGraph 용어의 정의를 사전에서 찾는다.
        'RAG가 뭐야', 'checkpointer 정의'처럼 용어 뜻을 물을 때 쓴다.
        """
        key = term.strip().lower()
        if key in _GLOSSARY:
            return f"{term}: {_GLOSSARY[key]}"
        for k, v in _GLOSSARY.items():
            if k in key or key in k:
                return f"{k}: {v}"
        available = ", ".join(sorted(_GLOSSARY))
        return f"'{term}' 용어는 사전에 없습니다. 등록된 용어: {available}"

    tools = [rag_search, course_glossary]

    if settings.tavily_enabled:
        from langchain_tavily import TavilySearch

        _tavily = TavilySearch(max_results=3)

        @tool
        def web_search(query: str) -> str:
            """강의자료에 없는 최신/외부 정보가 필요할 때 웹을 검색한다."""
            try:
                result = _tavily.invoke({"query": query})
            except Exception as exc:  # noqa: BLE001
                logger.warning("web_search 실패: %s", exc)
                return f"웹 검색에 실패했습니다: {exc}"
            logger.info("web_search: '%s'", query[:40])
            return str(result)

        tools.append(web_search)
    else:
        logger.info("TAVILY_API_KEY 없음, web_search 비활성화")

    return tools
