"""구조화 출력 스키마.

GradeDocuments는 검색 결과 관련성 판단(분기용)에, StudyAnswer는 최종 답변에 쓴다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    relevant: bool = Field(description="검색 문서가 질문에 관련되면 True")
    reason: str = Field(description="관련성 판단 근거 한 문장")


class StudyAnswer(BaseModel):
    answer: str = Field(description="질문에 대한 핵심 답변(한국어, 2~5문장)")
    key_points: list[str] = Field(default_factory=list, description="핵심 요점 2~4개")
    sources: list[str] = Field(
        default_factory=list, description="근거 출처(파일명/페이지 또는 URL). 없으면 빈 리스트"
    )
    confidence: Literal["high", "medium", "low"] = Field(description="답변 신뢰도")
    follow_up_question: str = Field(description="이어서 던져볼 후속 질문 하나")
