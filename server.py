"""웹 UI용 FastAPI 서버.

CLI(main.py)와 같은 LangGraph 그래프·SqliteSaver 메모리를 그대로 쓰고,
브라우저에서 대화할 수 있게 public/index.html을 서빙한다.

    python server.py        # http://localhost:3000
"""

from __future__ import annotations

import logging
import os
import sqlite3

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from config import BASE_DIR, MEMORY_DB, ConfigError, configure_logging, load_settings
from graph import build_graph
from rag import build_vectorstore

configure_logging()
logger = logging.getLogger("server")

app = FastAPI(title="AI 서비스 설계 학습 튜터")

# 앱 시작 시 한 번만 구성한다. 키가 없으면 여기서 바로 실패한다.
try:
    _settings = load_settings()
except ConfigError as exc:
    raise SystemExit(f"설정 오류: {exc}")

_vectorstore = build_vectorstore(_settings)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
_conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

_graph = build_graph(_settings, _retriever, SqliteSaver(_conn))

# 화면 표시용 대화 기록. 실제 멀티턴 메모리는 그래프의 체크포인터가 담당한다.
_display: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str | None = None
    sessionId: str = "web"


class ResetRequest(BaseModel):
    sessionId: str = "web"


@app.post("/api/chat")
def chat(body: ChatRequest):
    if not body.message or not body.message.strip():
        return JSONResponse(status_code=400, content={"success": False, "error": "message가 필요합니다."})
    try:
        config = {"configurable": {"thread_id": body.sessionId}}
        result = _graph.invoke({"messages": [HumanMessage(content=body.message)]}, config=config)
        answer = result["messages"][-1].content

        history = _display.setdefault(body.sessionId, [])
        history.append({"role": "user", "content": body.message})
        history.append({"role": "assistant", "content": answer})
        return {"success": True, "answer": answer, "history": history}
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat 처리 오류")
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.post("/api/reset")
def reset(body: ResetRequest):
    _display.pop(body.sessionId, None)
    # 체크포인터 sqlite에서 해당 세션(thread_id) 기록을 지운다.
    try:
        cur = _conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for (table,) in tables:
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info('{table}')").fetchall()]
            if "thread_id" in cols:
                cur.execute(f"DELETE FROM {table} WHERE thread_id=?", (body.sessionId,))
        _conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("메모리 삭제 실패: %s", exc)
    return {"success": True}


# 정적 파일은 API 라우트 뒤에 마운트한다.
app.mount("/", StaticFiles(directory=str(BASE_DIR / "public"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    print(f"http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
