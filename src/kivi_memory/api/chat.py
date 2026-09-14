"""Thin chat API over the existing Kivi read-response pipeline."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from kivi_memory.common.config import DEFAULT_LOCALE
from kivi_memory.document_rag.repository import DocumentRagRepository
from kivi_memory.document_rag.upload import ingest_uploaded_pdf
from kivi_memory.working_memory import ThreadEpisodeStore
from kivi_memory.working_memory.store import RedisWorkingMemoryError


class CreateThreadResponse(BaseModel):
    thread_id: str


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    current_datetime: datetime
    timezone: str = Field(min_length=1)
    locale: str = DEFAULT_LOCALE
    force_document_search: bool = False
    document_ids: list[str] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    thread_id: str
    answer: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessage]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    duplicate: bool
    chunks_indexed: int
    parse_status: str | None = None
    embedding_time: float = 0.0
    db_write_time: float = 0.0
    total_time: float = 0.0
    error: str | None = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    page_count: int


def create_app(
    *,
    read_service: Any | None = None,
    episode_store: ThreadEpisodeStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Kivi Chat API")
    service = read_service
    store = episode_store or (getattr(service, "episode_store", None) if service is not None else ThreadEpisodeStore())
    sweeper_task: asyncio.Task | None = None

    def get_read_service() -> Any:
        nonlocal service
        if service is None:
            from kivi_memory.read_response import ReadResponseService

            service = ReadResponseService(episode_store=store)
        return service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.post("/api/chat/threads", response_model=CreateThreadResponse)
    def create_thread() -> CreateThreadResponse:
        thread_id = str(uuid4())
        try:
            if getattr(store.config, "writeback_on_turn_enabled", False):
                from kivi_memory.writeback.trigger import trigger_all_idle_writeback_checks

                trigger_all_idle_writeback_checks()
        except RedisWorkingMemoryError:
            raise
        except Exception:
            pass
        return CreateThreadResponse(thread_id=thread_id)

    @app.post("/api/chat/threads/{thread_id}/messages", response_model=SendMessageResponse)
    async def send_message(thread_id: str, request: SendMessageRequest) -> SendMessageResponse:
        try:
            response = await get_read_service().handle_user_query(
                thread_id=thread_id,
                user_query=request.message,
                current_datetime=request.current_datetime,
                timezone=request.timezone,
                locale=request.locale,
                force_document_search=request.force_document_search,
                document_ids=request.document_ids,
            )
        except (ValueError, RedisWorkingMemoryError) as exc:
            raise HTTPException(status_code=400, detail=_public_error(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=_public_error(exc)) from exc
        return SendMessageResponse(thread_id=response.thread_id, answer=response.text)

    @app.post("/api/chat/threads/{thread_id}/messages/stream")
    async def stream_message(thread_id: str, request: SendMessageRequest) -> StreamingResponse:
        async def events():
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            answer_was_streamed = False

            def emit(event: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            task = asyncio.create_task(
                get_read_service().handle_user_query(
                    thread_id=thread_id,
                    user_query=request.message,
                    current_datetime=request.current_datetime,
                    timezone=request.timezone,
                    locale=request.locale,
                    force_document_search=request.force_document_search,
                    document_ids=request.document_ids,
                    event_callback=emit,
                )
            )

            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if event.get("type") == "assistant_response":
                    answer_was_streamed = True
                yield _sse(event)

            try:
                response = await task
            except (ValueError, RedisWorkingMemoryError) as exc:
                yield _sse({"type": "error", "message": _public_error(exc)})
                return
            except Exception as exc:
                yield _sse({"type": "error", "message": _public_error(exc)})
                return

            if not answer_was_streamed:
                yield _sse({"type": "assistant_response", "text": response.text})
            yield _sse({"type": "done", "thread_id": response.thread_id})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/chat/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
    def get_thread_messages(thread_id: str) -> ThreadMessagesResponse:
        try:
            episodes = store.get_recent_thread_episodes(thread_id, max_episodes=500)
        except (ValueError, RedisWorkingMemoryError) as exc:
            raise HTTPException(status_code=400, detail=_public_error(exc)) from exc
        messages: list[ChatMessage] = []
        for episode in episodes:
            if str(getattr(episode, "episode_id", "")).startswith("seed-context-"):
                continue
            for message in episode.messages:
                messages.append(ChatMessage(role=message.role.lower(), content=message.text))
        return ThreadMessagesResponse(thread_id=thread_id, messages=messages)

    @app.post("/api/documents", response_model=DocumentUploadResponse)
    async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        try:
            result = await asyncio.to_thread(
                ingest_uploaded_pdf,
                filename=file.filename or "document.pdf",
                stream=file.file,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_public_error(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=_public_error(exc)) from exc
        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            sha256=result.sha256,
            status=result.status,
            duplicate=result.duplicate,
            chunks_indexed=result.chunks_indexed,
            parse_status=result.parse_status,
            embedding_time=result.embedding_time,
            db_write_time=result.db_write_time,
            total_time=result.total_time,
            error=result.error,
        )

    @app.get("/api/documents/{document_id}", response_model=DocumentStatusResponse)
    def get_document_status(document_id: str) -> DocumentStatusResponse:
        row = DocumentRagRepository().get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return DocumentStatusResponse(
            document_id=row["document_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            status=row["status"],
            page_count=row["page_count"],
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.on_event("startup")
    async def start_idle_writeback_sweeper() -> None:
        nonlocal sweeper_task
        if sweeper_task is None:
            sweeper_task = asyncio.create_task(_idle_writeback_loop())

    @app.on_event("shutdown")
    async def stop_idle_writeback_sweeper() -> None:
        if sweeper_task is not None:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except asyncio.CancelledError:
                pass

    return app


def _public_error(exc: Exception) -> str:
    return " ".join(str(exc).split()) or "Kivi chat request failed"


def _sse(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "message")
    return f"event: {event_type}\ndata: {json.dumps(event, default=str)}\n\n"


async def _idle_writeback_loop() -> None:
    from kivi_memory.writeback.trigger import trigger_all_idle_writeback_checks

    while True:
        trigger_all_idle_writeback_checks()
        await asyncio.sleep(60)


app = create_app()
