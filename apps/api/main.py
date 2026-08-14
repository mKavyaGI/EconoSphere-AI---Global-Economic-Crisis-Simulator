"""
EconoSphere AI — FastAPI Application Entry Point (Root Stub)

IMPORTANT: The actual FastAPI application is in app/main.py.

To start the development server, run from the apps/api/ directory:

    uvicorn app.main:app --reload

To start with a specific host/port:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

To run in production (no reload):

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

This file re-exports the FastAPI app so that uvicorn app.main:app
and uvicorn main:app (if running from apps/api/) both resolve correctly.
"""
# Re-export the real app object so both invocation paths work:
#   uvicorn app.main:app   (recommended — explicit)
#   uvicorn main:app       (fallback — from apps/api/ directory)
from app.main import app  # noqa: F401

__all__ = ["app"]
