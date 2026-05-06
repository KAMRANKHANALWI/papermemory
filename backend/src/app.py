from contextlib import asynccontextmanager
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path

from src.routers import collections, metadata, chat, memory, search, selection
from src.services.shared import chat_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG backend starting up...")
    yield
    logger.info("RAG backend shutting down")


app = FastAPI(
    title="Enhanced Document Chat Backend",
    description="Complete RAG system with query classification, memory, and advanced features",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Global exception handlers ─────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "status_code": exc.status_code, "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "message": "Validation error",
            "detail": exc.errors(),
        },
    )


# ── Request timing middleware ──────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} [{duration}ms]"
    )
    return response


# ── Static routes ──────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "GutMiScholar API", "version": "2.0.0"}


@app.get("/api/model-info", tags=["health"])
async def get_model_info():
    try:
        return {"status": "success", "model_info": chat_service.get_model_info()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Routers ────────────────────────────────────────────
app.include_router(collections.router)
app.include_router(metadata.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(search.router)
app.include_router(selection.router)
