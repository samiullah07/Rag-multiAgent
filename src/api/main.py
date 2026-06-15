"""
FastAPI application entry point.
Exposes REST endpoints for the RAG system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from src.api.routers import chat, documents, sessions, eval, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multi-Agent RAG API",
    description="Self-correcting RAG with contradiction detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")

# Mount static frontend
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except Exception as e:
    logger.warning(f"Frontend not mounted: {e}")