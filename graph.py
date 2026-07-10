"""학습 튜터 에이전트의 LangGraph StateGraph.

흐름: guardrail -> agent -> tools -> grade -> (rewrite로 재검색하거나) -> finalize.
- guardrail에서 입력을 거르고,
- agent가 도구를 쓸지 스스로 정하고,
- tools 실행 뒤 grade가 검색 결과의 관련성을 보고,
- 관련이 없으면 rewrite로 질의를 다시 짜서 재검색(반복),
- finalize에서 Pydantic 구조화 출력으로 답을 정리한다.
재검색은 MAX_RETRIEVALS로 상한을 둬서 무한 루프를 막는다.
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import Settings
from guardrail import check_input, log_node, safe_node
from providers import get_chat_model
from schemas import GradeDocuments, StudyAnswer

logger = logging.getLogger("graph")

MAX_RETRIEVALS = 2  # rag 재검색 상한

AGENT_SYSTEM = (
    "당신은 'AI 서비스 설계' 교육과정의 학습 튜터입니다. "
    "학생의 질문에 답하기 위해 필요하면 도구를 사용하세요.\n"
    "- 강의 개념(LangChain/LangGraph/RAG/Agent 등) 질문: rag_search로 강의자료를 검색해 근거로 삼으세요.\n"
    "- 용어의 짧은 정의만 필요하면: course_glossary를 사용하세요.\n"
    "- 강의자료로 답할 수 없는 최신/외부 정보: web_search(가능한 경우)를 사용하세요.\n"
    "검색한 내용을 근거로 정확히 답하고, 근거가 없으면 모른다고 하세요."
)

FINALIZE_SYSTEM = (
    "아래 대화와 도구 검색 결과를 바탕으로 학생에게 줄 최종 답변을 StudyAnswer 형식으로 작성하세요. "
    "answer는 한국어로 간결하게, sources에는 참고한 강의자료 파일명이나 URL을 넣으세요. "
    "근거가 충분하면 confidence를 high, 부족하면 low로 설정하세요."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieval_count: int
    guardrail_passed: bool
    docs_relevant: bool


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _trailing_tool_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    """가장 최근 도구 실행 배치의 ToolMessage들을 모은다."""
    trailing: list[ToolMessage] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            trailing.append(m)
        else:
            break
    return trailing


def build_graph(settings: Settings, retriever, checkpointer):
    """튜터 에이전트 그래프를 컴파일해서 돌려준다."""
    llm = get_chat_model(settings, temperature=0)
    tools = _import_tools(settings, retriever)
    llm_with_tools = llm.bind_tools(tools)
    grader = llm.with_structured_output(GradeDocuments)
    finalizer = llm.with_structured_output(StudyAnswer)

    @log_node("guardrail")
    def guardrail_node(state: AgentState) -> dict:
        # 매 턴 상태를 초기화한 뒤 입력을 검증한다.
        text = _last_human_text(state["messages"])
        ok, reason = check_input(text)
        if not ok:
            return {
                "guardrail_passed": False,
                "retrieval_count": 0,
                "docs_relevant": False,
                "messages": [AIMessage(content=reason)],
            }
        return {"guardrail_passed": True, "retrieval_count": 0, "docs_relevant": False}

    @log_node("agent")
    @safe_node("죄송합니다. 답변 생성 중 오류가 발생했습니다.")
    def agent_node(state: AgentState) -> dict:
        # 도구를 쓸지 말지 모델이 스스로 판단한다.
        response = llm_with_tools.invoke([SystemMessage(AGENT_SYSTEM), *state["messages"]])
        return {"messages": [response]}

    @log_node("grade")
    def grade_node(state: AgentState) -> dict:
        # 직전 도구가 rag_search면 결과 관련성을 평가한다. 아니면 그냥 통과.
        trailing = _trailing_tool_messages(state["messages"])
        last_rag = next((m for m in trailing if m.name == "rag_search"), None)
        count = state.get("retrieval_count", 0)

        if last_rag is None:
            return {"docs_relevant": True, "retrieval_count": count}

        count += 1
        question = _last_human_text(state["messages"])
        try:
            grade: GradeDocuments = grader.invoke(
                [
                    SystemMessage("검색된 문서가 질문에 답하기에 관련 있는지 평가하세요."),
                    HumanMessage(
                        content=f"[질문]\n{question}\n\n[검색된 문서]\n{last_rag.content[:3000]}"
                    ),
                ]
            )
            relevant = grade.relevant
            logger.info("관련성 평가: %s (%s)", relevant, grade.reason)
        except Exception as exc:  # noqa: BLE001 - 평가 실패 시 통과 처리
            logger.warning("관련성 평가 실패, 통과 처리: %s", exc)
            relevant = True

        return {"docs_relevant": relevant, "retrieval_count": count}

    @log_node("rewrite")
    @safe_node("검색 질의 재작성에 실패했습니다.")
    def rewrite_node(state: AgentState) -> dict:
        # 검색이 부족하면 질의를 다시 짜서 재검색을 유도한다.
        question = _last_human_text(state["messages"])
        refined = llm.invoke(
            [
                SystemMessage(
                    "다음 질문을 강의자료 검색에 더 맞게 핵심 키워드 중심으로 한 문장으로 "
                    "다시 쓰세요. 다시 쓴 질의만 출력하세요."
                ),
                HumanMessage(content=question),
            ]
        ).content
        logger.info("질의 재작성: '%s' -> '%s'", question[:30], str(refined)[:30])
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "이전 검색 결과가 부족했습니다. 아래 관점으로 rag_search 또는 "
                        f"web_search를 다시 호출하세요: {refined}"
                    )
                )
            ]
        }

    @log_node("finalize")
    @safe_node("답변 구조화에 실패했습니다.")
    def finalize_node(state: AgentState) -> dict:
        # 대화와 검색 결과를 StudyAnswer로 정리한다.
        answer: StudyAnswer = finalizer.invoke(
            [SystemMessage(FINALIZE_SYSTEM), *state["messages"]]
        )
        return {"messages": [AIMessage(content=_format_answer(answer))]}

    def route_after_guardrail(state: AgentState) -> str:
        return "agent" if state["guardrail_passed"] else END

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    def route_after_grade(state: AgentState) -> str:
        if state["docs_relevant"]:
            return "agent"
        if state.get("retrieval_count", 0) >= MAX_RETRIEVALS:
            logger.info("재검색 상한 도달, 확보된 근거로 답변")
            return "agent"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"agent": "agent", END: END})
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"agent": "agent", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def _import_tools(settings: Settings, retriever) -> list:
    # tools <-> graph 순환 임포트를 피하려고 여기서 임포트한다.
    from tools import build_tools

    return build_tools(settings, retriever)


def _format_answer(ans: StudyAnswer) -> str:
    lines = [ans.answer.strip()]
    if ans.key_points:
        lines.append("\n핵심 요점")
        lines.extend(f"- {p}" for p in ans.key_points)
    if ans.sources:
        lines.append("\n출처")
        lines.extend(f"- {s}" for s in ans.sources)
    lines.append(f"\n신뢰도: {ans.confidence}")
    if ans.follow_up_question:
        lines.append(f"다음 질문: {ans.follow_up_question}")
    return "\n".join(lines)
