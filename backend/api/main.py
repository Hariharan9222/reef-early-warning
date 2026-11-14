"""
backend/api/main.py
FastAPI application entrypoint for CoralGuard Backend.
Provides CORS middleware, API routing, database initialization, and APScheduler lifecycle management.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database import engine, Base
from backend.api.routes import router as api_router
from backend.ingestion.scheduler import init_scheduler, shutdown_scheduler

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("coralguard.app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and start scheduler
    logger.info("Initializing CoralGuard database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

    init_scheduler()
    yield
    # Shutdown
    shutdown_scheduler()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Early Warning API for Coral Reef Bleaching using Underwater Acoustics, Satellite SST, and Citizen Diver Reports.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes under both /api/v1 and root for maximum client compatibility
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["CoralGuard API v1"])
app.include_router(api_router, prefix="", tags=["Direct Endpoints"])

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": "connected",
        "scheduler_enabled": settings.SCHEDULER_ENABLED
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)

