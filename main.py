"""
FastAPI service for the SHL assessment recommendation agent.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health -> {"status": "ok"}
    POST /chat   -> see schemas.ChatRequest / ChatResponse
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from retrieval import CatalogIndex
from agent import SHLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_service")

CATALOG_PATH = os.environ.get("SHL_CATALOG_PATH", "catalog.json")
REQUEST_TIMEOUT_SECONDS = 30

_index: CatalogIndex | None = None
_agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _agent
    _index = CatalogIndex(CATALOG_PATH)
    _agent = SHLAgent(_index)
    logger.info("Loaded catalog with %d entries from %s", len(_index.entries), CATALOG_PATH)
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    if _index is None or _agent is None:
        raise HTTPException(status_code=503, detail="catalog not loaded")
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent not initialized")

    messages = [{"role": m.role.value, "content": m.content} for m in req.messages]

    loop = asyncio.get_running_loop()
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _agent.handle_turn, messages),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Request timed out after %.1fs", time.monotonic() - start)
        raise HTTPException(
            status_code=504,
            detail="Request took too long to process. Please try again with a more specific query.",
        )
    except Exception:
        logger.exception("Unhandled error in agent.handle_turn")
        raise HTTPException(status_code=500, detail="Internal error while generating a response.")

    return ChatResponse(
        reply=result.reply,
        recommendations=[Recommendation(**r) for r in result.recommendations],
        end_of_conversation=result.end_of_conversation,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
