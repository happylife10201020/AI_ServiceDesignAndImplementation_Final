"""입력 가드레일, 노드 로깅, 예외 방어용 미들웨어.

- check_input: 빈 입력/과도한 길이/프롬프트 인젝션 차단
- log_node: 노드 진입과 종료를 로깅하는 데코레이터
- safe_node: 노드에서 예외가 나도 그래프가 멈추지 않게 감싸는 데코레이터
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Callable

logger = logging.getLogger("middleware")

MAX_INPUT_CHARS = 2000

# 흔한 인젝션/탈옥 문구 (한글, 영어)
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(the\s+)?(above|previous)",
    r"system\s*prompt",
    r"이전\s*(의\s*)?(모든\s*)?(지시|명령|프롬프트)",
    r"시스템\s*프롬프트",
    r"너의\s*(지시사항|규칙)을\s*무시",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_input(text: str) -> tuple[bool, str]:
    """입력을 검증한다. 통과하면 (True, ""), 아니면 (False, 사유)."""
    if text is None or not text.strip():
        return False, "빈 입력입니다. 질문을 입력해 주세요."

    if len(text) > MAX_INPUT_CHARS:
        return False, f"입력이 너무 깁니다({len(text)}자). {MAX_INPUT_CHARS}자 이내로 줄여 주세요."

    if _INJECTION_RE.search(text):
        logger.warning("가드레일: 인젝션 의심 입력 차단")
        return False, "시스템 지시를 바꾸려는 요청은 처리할 수 없습니다."

    return True, ""


def log_node(name: str) -> Callable:
    """노드의 진입/종료와 소요 시간을 로깅한다."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            logger.info("노드 시작: %s", name)
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("노드 종료: %s (%.0f ms)", name, elapsed)
            return result

        return wrapper

    return decorator


def safe_node(fallback_message: str) -> Callable:
    """노드 실행 중 예외를 잡아 fallback 메시지로 대체한다."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            try:
                return func(state, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.exception("노드 오류 (%s): %s", func.__name__, exc)
                from langchain_core.messages import AIMessage

                return {"messages": [AIMessage(content=fallback_message)]}

        return wrapper

    return decorator
