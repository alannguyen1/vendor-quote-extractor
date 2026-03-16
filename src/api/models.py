"""
API-specific request and response models.

These models extend the core schemas for API-specific concerns
like error responses and health checks.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standard error response format for all API errors.

    Used for 4xx and 5xx responses to provide consistent error structure.
    """

    detail: str = Field(..., description="Human-readable error message")
    error_code: str = Field(..., description="Machine-readable error code")
    errors: list[str] | None = Field(
        default=None, description="Additional error details"
    )


class HealthResponse(BaseModel):
    """
    Health check response for monitoring.

    Returns basic status and version information.
    """

    status: str = Field(..., description="Service status (healthy/unhealthy)")
    version: str = Field(..., description="API version")


class CorrectionRequest(BaseModel):
    """
    Request model for submitting field corrections.

    Used with PUT /api/v1/extractions/{id}/correct endpoint.
    Field path uses JSON path notation (e.g., "line_items[3].item_type").
    """

    field_path: str = Field(
        ...,
        description="JSON path to the field being corrected",
        examples=["line_items[3].item_type", "vendor.name", "amounts.grand_total"],
    )
    corrected_value: str = Field(
        ...,
        description="New value for the field (as string)",
    )


class CorrectionRecord(BaseModel):
    """
    Response model for a correction record in audit trail.

    Returned in ExtractionResponse.corrections to show edit history.
    """

    id: int = Field(..., description="Correction record ID")
    field_path: str = Field(..., description="JSON path to the corrected field")
    original_value: str | None = Field(None, description="Value before correction")
    corrected_value: str | None = Field(None, description="Value after correction")
    created_at: str = Field(..., description="ISO timestamp of correction")


class ExtractionResponse(BaseModel):
    """
    Response model wrapping VendorQuote with extraction ID and corrections.

    Returned by GET /api/v1/extractions/{id} to include database ID
    and any corrections that have been applied (audit trail).
    """

    id: int = Field(..., description="Database extraction ID")
    filename: str = Field(..., description="Original uploaded filename")
    quote: dict = Field(..., description="Extracted VendorQuote data")
    corrections: list[CorrectionRecord] = Field(
        default_factory=list,
        description="List of corrections applied to this extraction",
    )


class ExtractionListItem(BaseModel):
    """
    Lightweight extraction summary for list views.

    Used in the recent extractions list to show status without full quote data.
    """

    id: int = Field(..., description="Database extraction ID")
    filename: str = Field(..., description="Original uploaded filename")
    overall_confidence: float | None = Field(
        None, description="Extraction confidence score (0-1)"
    )
    requires_review: bool | None = Field(
        None, description="Whether human review is needed"
    )
    created_at: str = Field(..., description="ISO timestamp of extraction")


class ExtractionListResponse(BaseModel):
    """
    Response model for listing recent extractions.

    Returned by GET /api/v1/extractions endpoint.
    """

    extractions: list[ExtractionListItem] = Field(
        ..., description="List of recent extractions"
    )
    total: int = Field(..., description="Total count of extractions in database")
