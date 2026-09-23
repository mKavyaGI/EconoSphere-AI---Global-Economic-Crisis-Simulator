from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog
from app.core.config import settings
from app.core.logging import setup_logging
from app.database.session import engine
from app.neo4j.session import neo4j_conn
from app.middleware.correlation import CorrelationIdMiddleware
from app.api.v1.endpoints import health

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await logger.ainfo("application_startup_attempt", message="Starting EconoSphere AI API...")
    
    if settings.env == "production" or settings.production_safety_mode:
        from app.services.production_health import run_readiness_check
        readiness = run_readiness_check()
        if not readiness.get("ready"):
            await logger.aerror("production_preflight_failure", message="Production preflight integrity check failed. Failing closed.", checks=readiness.get("checks"))
            raise RuntimeError("Production preflight integrity check failed.")
        await logger.ainfo("production_preflight_success", message="Production preflight integrity check passed.")

    await neo4j_conn.connect()
    await logger.ainfo("application_ready")
    yield
    await logger.ainfo("application_shutdown", message="Shutting down EconoSphere AI API...")
    await neo4j_conn.close()
    await engine.dispose()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(HTTPException)
async def standardized_http_exception_handler(request: Request, exc: HTTPException):
    """Returns a unified error JSON schema for all HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def standardized_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Returns a unified error JSON schema for request payload validation failures."""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "code": 422,
            "detail": "Request validation failure",
            "errors": exc.errors(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions (e.g. Neo4j connection issues) to prevent CORS Network Errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": 500,
            "detail": "Internal Server Error: Service unavailable or database connection refused.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.v1.router import api_router

app.include_router(api_router, prefix="/api/v1")
