"""
API module for Vendor Quote Extractor.

Provides FastAPI application and routes for vendor quote extraction.

Usage:
    uvicorn src.api.routes:app --reload --port 8000
"""

from src.api.models import (
    CorrectionRequest,
    ErrorResponse,
    ExtractionResponse,
    HealthResponse,
)
from src.api.routes import app

__all__ = [
    "app",
    "CorrectionRequest",
    "ErrorResponse",
    "ExtractionResponse",
    "HealthResponse",
]
