"""
Forensic Council API Server
===========================

FastAPI application with WebSocket support for real-time updates.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import hitl, investigation, sessions
from core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for startup and shutdown events.
    
    Initializes infrastructure on startup and closes on shutdown.
    """
    # Startup
    logger.info("Starting Forensic Council API server...")
    
    # Initialize infrastructure connections
    # Note: The pipeline will initialize its own connections on first use
    
    yield
    
    # Shutdown
    logger.info("Shutting down Forensic Council API server...")
    
    # Clean up WebSocket connections
    investigation._websocket_connections.clear()
    investigation._active_pipelines.clear()


# Create FastAPI app
app = FastAPI(
    title="Forensic Council API",
    description="Multi-Agent Forensic Evidence Analysis System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Include routers
app.include_router(investigation.router)
app.include_router(hitl.router)
app.include_router(sessions.router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to ensure no error is ignored.
    Logs the error fully to the console/logging system and returns a standardized 500 response.
    """
    logger.error(f"Global Exception Caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "message": str(exc)},
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Forensic Council API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
