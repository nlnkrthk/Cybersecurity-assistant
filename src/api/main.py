"""
FastAPI app of the incident-response assistant.

    uvicorn api.main:app            (with src/ on the path; or scripts/api/run_api.py)
    Swagger UI: http://localhost:8000/docs

Startup loads the ML models and builds the assistant graph; the search indexes load
in a background thread so the server is usable at once (GET /health shows progress).
The assistant needs an LLM API key in .env (see agents.llm); without it /chat returns 503 and
the direct ML / search endpoints still work.
"""
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.deps import AppState  # noqa: E402
from api.routes import chat, feedback, judge, ml, search, system  # noqa: E402

logger = logging.getLogger("api")

DEFAULT_CORS_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"  # Streamlit


def _build_assistant(state: AppState) -> None:
    try:
        from agents.graph import Assistant
        from agents.llm import get_chat_model, llm_settings

        llm = get_chat_model()
        state.assistant = Assistant(llm=llm)
        settings = llm_settings()
        state.llm_model = f"{settings['provider']}:{settings['model']}"
    except Exception as exc:
        state.assistant_error = str(exc)
        logger.warning(f"Assistant unavailable: {exc}")


def create_app(assistant: Optional[Any] = None, load_search_index: bool = True) -> FastAPI:
    """Build the app.

    Args:
        assistant: an agents.graph.Assistant to use instead of the default Claude-backed one
            (tests pass one with a scripted LLM).
        load_search_index: start loading the search indexes in the background at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from services.ml_service import get_ml_service

        state = AppState()
        started = time.perf_counter()
        state.ml_service = get_ml_service()
        if assistant is not None:
            state.assistant, state.llm_model = assistant, "custom"
        else:
            _build_assistant(state)
        if load_search_index:
            threading.Thread(target=state.load_search_index, name="search-index-loader", daemon=True).start()
        app.state.services = state
        logger.info(f"API ready in {time.perf_counter() - started:.1f}s (assistant: {state.assistant is not None})")
        yield

    app = FastAPI(
        title="Cybersecurity Incident Response Assistant API",
        version="1.0.0",
        description=(
            "Multi-agent incident-response assistant on CIC-IDS2017 network flows.\n\n"
            "- **/chat**: ask questions in plain language (supervisor + 4 agents, guardrails, agent trace)\n"
            "- **/ml**: attack type, intrusion risk, anomaly, SHAP explanation and priority for flows\n"
            "- **/search**: hybrid retrieval of similar historical incidents\n\n"
            "Flows are given as any subset of the 24 metrics (numbers or strings with units such as "
            "'2 s', '1.5 KB') or as a historical `incident_id`."
        ),
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.getenv("API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Thread-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(f"{request_id} {request.method} {request.url.path} -> {response.status_code} ({elapsed:.0f} ms)")
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception(f"{request_id} unhandled error on {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id or ""},
        )

    app.include_router(system.router)
    app.include_router(chat.router)
    app.include_router(judge.router)
    app.include_router(feedback.router)
    app.include_router(ml.router)
    app.include_router(search.router)
    return app


app = create_app()
