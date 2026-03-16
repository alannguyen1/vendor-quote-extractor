"""
FastAPI routes for the Vendor Quote Extractor API.

Provides:
- POST /api/v1/extract - Upload and extract data from vendor quote PDFs
- GET /api/v1/extractions/{id} - Retrieve previous extraction by ID
- PUT /api/v1/extractions/{id}/correct - Submit field corrections
- GET /api/v1/health - Health check endpoint
"""

import hashlib
import json
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import configure_logging, get_settings
from src.api.models import (
    CorrectionRecord,
    CorrectionRequest,
    ErrorResponse,
    ExtractionListItem,
    ExtractionListResponse,
    ExtractionResponse,
    HealthResponse,
)
from src.db import (
    Correction,
    Extraction,
    add_correction_example,
    close_db,
    get_session,
    init_db,
)
from src.extraction import ExtractionPipeline
from src.schemas import VendorQuote

# Configure logging
logger = configure_logging()

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown tasks."""
    # Startup: Initialize database
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown: Close database connections
    logger.info("Closing database connections...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Vendor Quote Extractor",
    description="AI-powered extraction of structured data from vendor quotes",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# Configure CORS for Streamlit UI (runs on port 8501)
# Origins are configurable via settings.cors_origins
_settings = get_settings()
# Include localhost 127.0.0.1 variant for local development
_cors_origins = _settings.cors_origins + ["http://127.0.0.1:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Extraction-ID"],  # Allow UI to read extraction ID header
)

# Initialize pipeline lazily so the API can import without provider keys.
_pipeline: ExtractionPipeline | None = None

# PDF magic bytes for validation
PDF_MAGIC_BYTES = b"%PDF"


def get_pipeline() -> ExtractionPipeline:
    """Return a cached pipeline instance, creating it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ExtractionPipeline()
    return _pipeline


def compute_file_hash(pdf_bytes: bytes) -> str:
    """Compute SHA-256 hash of PDF bytes for deduplication."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def apply_field_correction(data: dict, field_path: str, value: str) -> tuple[dict, Any]:
    """
    Apply a correction to nested data structure using field path.

    Supports array indexing (e.g., "line_items[3].item_type").
    Returns (updated_data, original_value).

    Raises:
        ValueError: If field path is invalid or doesn't exist.
    """
    # Parse the field path into parts
    parts = []
    for part in field_path.split("."):
        # Check for array access like "line_items[3]"
        match = re.match(r"(\w+)\[(\d+)\]$", part)
        if match:
            parts.append(match.group(1))
            parts.append(int(match.group(2)))
        else:
            parts.append(part)

    # Navigate to the parent of the target field
    current = data
    for i, part in enumerate(parts[:-1]):
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                raise ValueError(f"Invalid array index at path: {field_path}")
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Invalid field path: {field_path}")
            current = current[part]

    # Get the final key and apply correction
    final_key = parts[-1]
    if isinstance(final_key, int):
        if not isinstance(current, list) or final_key >= len(current):
            raise ValueError(f"Invalid array index at path: {field_path}")
        original_value = current[final_key]
        current[final_key] = value
    else:
        if not isinstance(current, dict):
            raise ValueError(f"Invalid field path: {field_path}")
        original_value = current.get(final_key)
        current[final_key] = value

    return data, original_value


def _create_extraction_record(
    filename: str,
    file_hash: str,
    quote: VendorQuote,
) -> Extraction:
    """
    Create an Extraction record from extracted quote data.

    Helper to eliminate code duplication (Gap 2.4 fix).

    Args:
        filename: Original uploaded filename
        file_hash: SHA-256 hash of PDF bytes
        quote: Extracted VendorQuote

    Returns:
        New Extraction ORM instance (not yet added to session)
    """
    return Extraction(
        filename=filename,
        file_hash=file_hash,
        extracted_json=quote.model_dump_json(),
        overall_confidence=(
            quote.extraction_metadata.overall_confidence
            if quote.extraction_metadata
            else None
        ),
        requires_review=(
            quote.extraction_metadata.requires_review
            if quote.extraction_metadata
            else True
        ),
    )


def _extract_correction_context(quote_data: dict, field_path: str) -> str | None:
    """
    Extract document context for a correction to use in few-shot learning.

    For line item corrections, extracts the item's description as context.
    This helps the LLM understand what kind of content the correction applies to.

    Args:
        quote_data: The full extracted quote data dictionary.
        field_path: The field path being corrected (e.g., "line_items[3].item_type").

    Returns:
        Context string if available, None otherwise.
    """
    # Check if this is a line item correction
    match = re.match(r"line_items\[(\d+)\]\.(\w+)", field_path)
    if match:
        index = int(match.group(1))
        line_items = quote_data.get("line_items", [])
        if index < len(line_items):
            item = line_items[index]
            # Use description as context (most descriptive field)
            description = item.get("description", "")
            if description:
                # Truncate if too long
                return description[:200] if len(description) > 200 else description

    # For vendor corrections, use vendor name
    if field_path.startswith("vendor."):
        vendor = quote_data.get("vendor", {})
        return vendor.get("name")

    # For customer corrections, use customer name
    if field_path.startswith("customer."):
        customer = quote_data.get("customer", {})
        if customer:
            return customer.get("name")

    return None


@app.post(
    "/api/v1/extract",
    response_model=VendorQuote,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Processing failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Extract data from vendor quote PDF",
    description="Upload a PDF document and extract structured vendor quote data.",
)
async def extract_quote(
    response: Response,
    file: UploadFile = File(..., description="PDF file to process (max 10MB)"),
    skip_cache: bool = False,
    session: AsyncSession = Depends(get_session),
) -> VendorQuote:
    """
    Extract structured data from a vendor quote PDF.

    Accepts multipart/form-data with a PDF file upload.
    Returns extracted VendorQuote data with validation status.
    Stores extraction in database for future retrieval.

    Args:
        skip_cache: If True, bypass cache and force fresh extraction.
    """
    # Validate content type
    content_type = file.content_type or ""
    if not content_type.startswith("application/pdf"):
        logger.warning(f"Invalid content type: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Expected PDF, got: {content_type}",
        )

    # Read file content
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to read uploaded file",
        )

    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
    if len(pdf_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Validate PDF magic bytes
    if not pdf_bytes.startswith(PDF_MAGIC_BYTES):
        logger.warning("File does not have PDF magic bytes")
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file. File does not appear to be a valid PDF.",
        )

    # Check for existing extraction by file hash (unless skip_cache is True)
    file_hash = compute_file_hash(pdf_bytes)
    if not skip_cache:
        existing = await session.execute(
            select(Extraction).where(Extraction.file_hash == file_hash)
        )
        existing_extraction = existing.scalar_one_or_none()

        if existing_extraction:
            logger.info(f"Returning cached extraction for hash: {file_hash[:8]}...")
            response.headers["X-Extraction-ID"] = str(existing_extraction.id)
            return VendorQuote.model_validate_json(existing_extraction.extracted_json)
    else:
        logger.info(f"Skipping cache for hash: {file_hash[:8]}...")

    # Process the PDF
    filename = file.filename or "unknown.pdf"
    logger.info(f"Processing file: {filename}, size: {len(pdf_bytes)} bytes")

    try:
        result = await get_pipeline().process(pdf_bytes, filename)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}",
        )

    # Check for pipeline errors
    if result.error:
        logger.error(f"Pipeline returned error: {result.error}")
        raise HTTPException(
            status_code=422,
            detail=result.error,
        )

    if result.quote is None:
        raise HTTPException(
            status_code=422,
            detail="Extraction produced no results",
        )

    # Check for currency rejection
    if result.quote.currency.upper() != "USD":
        raise HTTPException(
            status_code=422,
            detail=f"Currency '{result.quote.currency}' not supported. "
            "Only USD quotes are accepted.",
        )

    # Store extraction in database (update if skip_cache and exists)
    # Uses helper function to avoid code duplication (Gap 2.4 fix)
    # Handles race condition with IntegrityError (Gap 2.2 fix)
    if skip_cache:
        # Check if extraction with this hash already exists
        existing = await session.execute(
            select(Extraction).where(Extraction.file_hash == file_hash)
        )
        existing_extraction = existing.scalar_one_or_none()

        if existing_extraction:
            # Update existing record
            existing_extraction.filename = filename
            existing_extraction.extracted_json = result.quote.model_dump_json()
            existing_extraction.overall_confidence = (
                result.quote.extraction_metadata.overall_confidence
                if result.quote.extraction_metadata
                else None
            )
            existing_extraction.requires_review = (
                result.quote.extraction_metadata.requires_review
                if result.quote.extraction_metadata
                else True
            )
            extraction = existing_extraction
            logger.info(f"Updated existing extraction with ID: {extraction.id}")
        else:
            # Create new record using helper
            extraction = _create_extraction_record(filename, file_hash, result.quote)
            session.add(extraction)
            try:
                await session.flush()
                logger.info(f"Stored new extraction with ID: {extraction.id}")
            except IntegrityError:
                # Race condition: another request inserted the same hash
                await session.rollback()
                existing = await session.execute(
                    select(Extraction).where(Extraction.file_hash == file_hash)
                )
                extraction = existing.scalar_one()
                logger.info(
                    f"Race condition resolved - returning existing extraction: "
                    f"{extraction.id}"
                )
    else:
        # Create new record using helper
        extraction = _create_extraction_record(filename, file_hash, result.quote)
        session.add(extraction)
        try:
            await session.flush()
            logger.info(f"Stored extraction with ID: {extraction.id}")
        except IntegrityError:
            # Race condition: another request inserted the same hash
            await session.rollback()
            existing = await session.execute(
                select(Extraction).where(Extraction.file_hash == file_hash)
            )
            extraction = existing.scalar_one()
            logger.info(
                f"Race condition resolved - returning existing extraction: "
                f"{extraction.id}"
            )

    # Return extraction ID in header for UI correction workflow
    response.headers["X-Extraction-ID"] = str(extraction.id)

    return result.quote


@app.get(
    "/api/v1/extractions",
    response_model=ExtractionListResponse,
    summary="List recent extractions",
    description="Get a list of recent extractions with summary information.",
)
async def list_extractions(
    limit: int = 10,
    session: AsyncSession = Depends(get_session),
) -> ExtractionListResponse:
    """
    List recent extractions for the extraction history sidebar.

    Returns lightweight extraction summaries ordered by creation time (newest first).
    Includes total count for potential pagination.
    """
    # Get total count
    count_result = await session.execute(select(func.count(Extraction.id)))
    total = count_result.scalar() or 0

    # Get recent extractions (capped at max_recent_extractions for performance)
    settings = get_settings()
    result = await session.execute(
        select(Extraction)
        .order_by(Extraction.created_at.desc())
        .limit(min(limit, settings.max_recent_extractions))
    )
    extractions = result.scalars().all()

    items = [
        ExtractionListItem(
            id=e.id,
            filename=e.filename,
            overall_confidence=e.overall_confidence,
            requires_review=e.requires_review,
            created_at=e.created_at.isoformat(),
        )
        for e in extractions
    ]

    return ExtractionListResponse(extractions=items, total=total)


@app.get(
    "/api/v1/extractions/{extraction_id}",
    response_model=ExtractionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Extraction not found"},
    },
    summary="Retrieve previous extraction",
    description="Get a previously stored extraction by its database ID.",
)
async def get_extraction(
    extraction_id: int,
    session: AsyncSession = Depends(get_session),
) -> ExtractionResponse:
    """
    Retrieve a stored extraction by ID.

    Returns the extraction data including the VendorQuote JSON
    and any corrections that have been applied.
    """
    # Fetch extraction from database with corrections eager loaded
    result = await session.execute(
        select(Extraction)
        .where(Extraction.id == extraction_id)
        .options(selectinload(Extraction.corrections))
    )
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=404,
            detail=f"Extraction with ID {extraction_id} not found",
        )

    # Convert corrections to response model
    correction_records = [
        CorrectionRecord(
            id=c.id,
            field_path=c.field_path,
            original_value=c.original_value,
            corrected_value=c.corrected_value,
            created_at=c.created_at.isoformat(),
        )
        for c in extraction.corrections
    ]

    return ExtractionResponse(
        id=extraction.id,
        filename=extraction.filename,
        quote=json.loads(extraction.extracted_json),
        corrections=correction_records,
    )


@app.put(
    "/api/v1/extractions/{extraction_id}/correct",
    response_model=ExtractionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Extraction not found"},
        400: {"model": ErrorResponse, "description": "Invalid field path"},
    },
    summary="Submit field correction",
    description="Correct a field in an extraction. Stores correction for audit trail.",
)
async def correct_extraction(
    extraction_id: int,
    correction: CorrectionRequest,
    session: AsyncSession = Depends(get_session),
) -> ExtractionResponse:
    """
    Apply a correction to a stored extraction.

    Updates the stored JSON and records the correction for auditing.
    Uses JSON path notation for the field (e.g., "line_items[3].item_type").
    Returns the updated extraction with full corrections history.
    """
    # Fetch extraction from database with corrections eager loaded
    result = await session.execute(
        select(Extraction)
        .where(Extraction.id == extraction_id)
        .options(selectinload(Extraction.corrections))
    )
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=404,
            detail=f"Extraction with ID {extraction_id} not found",
        )

    # Parse the current JSON
    quote_data = json.loads(extraction.extracted_json)

    # Apply the correction
    try:
        updated_data, original_value = apply_field_correction(
            quote_data, correction.field_path, correction.corrected_value
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    # Store the correction record
    new_correction = Correction(
        extraction_id=extraction.id,
        field_path=correction.field_path,
        original_value=str(original_value) if original_value is not None else None,
        corrected_value=correction.corrected_value,
    )
    session.add(new_correction)

    # Extract document context for few-shot learning
    # For line item corrections, use the item's description as context
    document_context = _extract_correction_context(quote_data, correction.field_path)

    # Store correction example for few-shot learning
    await add_correction_example(
        session=session,
        field_path=correction.field_path,
        original_value=str(original_value) if original_value is not None else None,
        corrected_value=correction.corrected_value,
        document_context=document_context,
    )

    # Update the extraction JSON
    extraction.extracted_json = json.dumps(updated_data)

    # Flush to get the new correction's ID
    await session.flush()

    logger.info(
        f"Applied correction to extraction {extraction_id}: "
        f"{correction.field_path} = {correction.corrected_value}"
    )

    # Reload corrections to include the new one
    await session.refresh(extraction, ["corrections"])

    # Convert all corrections to response model
    correction_records = [
        CorrectionRecord(
            id=c.id,
            field_path=c.field_path,
            original_value=c.original_value,
            corrected_value=c.corrected_value,
            created_at=c.created_at.isoformat(),
        )
        for c in extraction.corrections
    ]

    return ExtractionResponse(
        id=extraction.id,
        filename=extraction.filename,
        quote=updated_data,
        corrections=correction_records,
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Returns service status and version information.
    No authentication required.
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler to format errors consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
            errors=None,
        ).model_dump(),
    )
