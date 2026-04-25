"""
Main application entry point.
Features: Auto-logging, environment-based docs, static frontend serving and STT job manager.
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.logger import logger
from app.services.stt_jobs import STTJobManager


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend_dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.PREPARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    settings.STT_JOBS_DIR.mkdir(parents=True, exist_ok=True)

    app.state.stt_jobs = STTJobManager()
    app.state.stt_jobs.start()

    try:
        yield
    finally:
        app.state.stt_jobs.stop()
        logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=settings.OPENAPI_PATH if settings.is_dev else None,
    docs_url=settings.DOCS_PATH if settings.is_dev else None,
    redoc_url=settings.REDOC_PATH if settings.is_dev else None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.2f}ms"
    )

    return response


try:
    from app.api.routers import stt

    app.include_router(stt.router, prefix=settings.BASE_PATH, tags=["STT"])
except ImportError as e:
    logger.warning(f"Could not import one or more routers: {e}")


@app.get(settings.HEALTH_PATH, tags=["System"])
async def health_check():
    return {
        "status": "online",
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION,
        "name": settings.PROJECT_NAME,
        "runtime": {
            "default_model": settings.STT_DEFAULT_MODEL,
            "device": settings.STT_DEVICE,
            "compute_type": settings.STT_DEFAULT_COMPUTE_TYPE,
            "workers": settings.STT_WORKERS,
            "cpu_threads": settings.STT_DEFAULT_CPU_THREADS,
            "num_workers": settings.STT_DEFAULT_NUM_WORKERS,
        },
    }


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=settings.UI_PATH)


@app.get(settings.CONFIG_PATH, include_in_schema=False, tags=["System"])
async def plugin_config():
    return {
        "name": settings.PROJECT_SLUG,
        "display_name": settings.DISPLAY_NAME,
        "ui_entry_point": settings.UI_PATH,
        "configuration_entry_point": f"{settings.CONFIG_PATH}/",
        "documentation_entry_point": settings.DOCS_PATH,
        "healthcheck_entry_point": settings.HEALTH_PATH,
        "version": settings.VERSION,
        "logo": settings.LOGO,
        "description": settings.DESCRIPTION,
        "author": settings.AUTHOR,
        "license": settings.LICENSE,
    }


if FRONTEND_DIST_DIR.exists():
    app.mount(
        settings.UI_PATH,
        StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True),
        name="ui",
    )
    logger.info(f"Frontend mounted at {settings.UI_PATH} from: {FRONTEND_DIST_DIR}")
else:
    logger.warning(f"Frontend dist directory not found: {FRONTEND_DIST_DIR}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_dev,
    )