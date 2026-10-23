"""HTTP API. Run with ``jolt serve`` or ``uvicorn jolt.api:app``."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from jolt import __version__
from jolt.config import get_settings
from jolt.ingest import SUPPORTED_SUFFIXES
from jolt.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from jolt.service import JoltService

logger = logging.getLogger("jolt.api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(service: JoltService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or JoltService.from_settings(get_settings())
        yield

    app = FastAPI(
        title="Jolt",
        version=__version__,
        description="Agentic RAG: ask questions over your documents, with citations.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %d in %.0f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        return response

    def svc(request: Request) -> JoltService:
        service: JoltService = request.app.state.service
        return service

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            llm_backend=s.settings.llm_backend,
            indexed_chunks=len(s.store),
        )

    @app.post("/v1/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        # The pipeline is blocking (model inference), so keep it off the event loop.
        return await run_in_threadpool(svc(request).ask, body.question, body.use_web)

    @app.post("/v1/documents", response_model=IngestResponse)
    async def upload(request: Request, files: list[UploadFile] = File(...)) -> IngestResponse:
        s = svc(request)
        upload_dir: Path = s.settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for file in files:
            name = Path(file.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise HTTPException(415, f"Unsupported file type: {name or '(unnamed)'}")
            if file.size is not None and file.size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} is larger than 25 MB")
            target = upload_dir / name
            with target.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            saved.append(target)
        return await run_in_threadpool(s.ingest, saved)

    @app.get("/v1/graph", response_class=Response)
    def graph(request: Request) -> Response:
        return Response(svc(request).graph_mermaid(), media_type="text/plain")

    return app


def __getattr__(name: str) -> FastAPI:
    # Lets `uvicorn jolt.api:app` build the app lazily, so importing this module stays cheap.
    if name == "app":
        return create_app()
    raise AttributeError(name)
