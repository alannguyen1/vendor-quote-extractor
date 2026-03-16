"""
Pydantic models for vendor quote data structures.

These models define the schema for extracted vendor quote data,
including validation rules and field constraints.
"""

import re
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.enums import ItemType, TermType


def coerce_date(value):
    """
    Coerce various date formats to Python date object.

    Handles:
    - ISO format: YYYY-MM-DD (passthrough)
    - US format: MM/DD/YYYY, M/D/YYYY
    - Written: "February 12, 2026"
    - Already a date object

    Returns None for None input.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        return value

    value = value.strip()

    # Try ISO format first (YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return datetime.strptime(value, "%Y-%m-%d").date()

    # US format: MM/DD/YYYY or M/D/YYYY
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", value):
        return datetime.strptime(value, "%m/%d/%Y").date()

    # Written format: "February 12, 2026" or "Feb 12, 2026"
    written_patterns = [
        ("%B %d, %Y", r"^[A-Z][a-z]+ \d{1,2}, \d{4}$"),  # February 12, 2026
        ("%b %d, %Y", r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$"),  # Feb 12, 2026
    ]
    for fmt, pattern in written_patterns:
        if re.match(pattern, value):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue

    # Return as-is if we can't parse (let Pydantic handle validation error)
    return value


class SourceRegion(BaseModel):
    """
    Represents a region in the source PDF document.

    Used to link extracted data back to its visual location in the PDF
    for source highlighting in the UI. Coordinates are in PDF points
    (72 points per inch) relative to page origin (bottom-left).

    Why this exists: When users hover over extracted fields in the UI,
    we need to highlight where that data came from in the PDF. This
    enables visual verification and builds trust in the extraction.
    """

    region_id: str = Field(
        ..., description="Unique identifier for this region (e.g., 'table_1_p1')"
    )
    page: int = Field(..., ge=1, description="1-indexed page number in the PDF")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x0, y0, x1, y1] in PDF coordinates. "
        "x0,y0 = bottom-left corner; x1,y1 = top-right corner",
    )
    region_type: Literal["table", "text", "header", "footer", "amount"] = Field(
        ..., description="Type of content in this region"
    )
    content_summary: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Brief summary of region content for debugging",
    )


class Entity(BaseModel):
    """
    Represents a party (vendor or customer) in a transaction.

    The vendor is required for all quotes; customer may be omitted
    for some quote formats.
    """

    name: str = Field(
        ..., min_length=2, description="Entity name (required, min 2 chars)"
    )
    address: Optional[str] = Field(default=None, description="Full address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    email: Optional[str] = Field(default=None, description="Contact email address")


class LineItem(BaseModel):
    """
    Individual line item from a vendor quote.

    Represents a single product, service, or adjustment (credit/discount).
    Extended prices can be negative for credits/discounts.
    """

    line_number: int = Field(..., ge=1, description="Sequential line number (>= 1)")
    description: str = Field(..., min_length=1, description="Item description")
    sku: Optional[str] = Field(default=None, description="Product SKU/part number")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer name")
    quantity: int = Field(..., ge=1, description="Quantity ordered (>= 1)")
    unit_price: float = Field(..., ge=0, description="Price per unit (>= 0)")
    extended_price: float = Field(
        ..., description="Total price (can be negative for credits)"
    )

    # Classification - Optional because LLM may not classify all items
    item_type: Optional[ItemType] = Field(
        default=None,
        description="Item category (hardware, software, services, etc.)",
    )

    # Term information for subscriptions/services
    term_type: Optional[TermType] = Field(
        default=None, description="Billing period type"
    )
    term_length_months: Optional[int] = Field(
        default=None, ge=1, description="Duration in months (>= 1)"
    )

    # Extraction quality indicator
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence (0-1)"
    )

    # Source tracking for UI highlighting
    source_region_id: Optional[str] = Field(
        default=None,
        description="ID of the SourceRegion this item was extracted from",
    )

    @field_validator("extended_price")
    @classmethod
    def validate_extended_price_sign(cls, v: float, info) -> float:
        """
        Warn if extended_price sign doesn't match expected for item type.

        Credits/discounts should be negative; other items should be positive.
        This is a soft validation - we don't reject, just log for review.
        """
        # Note: We don't reject here; validation layer checks consistency
        return v


class Amounts(BaseModel):
    """
    Financial totals from a vendor quote.

    Tracks subtotals, adjustments, and final totals.
    When multiple totals exist (MSRP vs discounted), both are captured
    with selected_finance_total indicating which to use.
    """

    subtotal: float = Field(..., ge=0, description="Sum of line item extended prices")
    discounts: float = Field(default=0.0, description="Total discount amount")
    shipping: float = Field(default=0.0, description="Shipping charges")
    tax: float = Field(default=0.0, description="Tax amount")
    grand_total: float = Field(..., description="Final total after all adjustments")

    # Optional totals for quotes with multiple pricing views
    list_price_total: Optional[float] = Field(
        default=None, description="MSRP/list price total"
    )
    net_price_total: Optional[float] = Field(
        default=None, description="Discounted/net price total"
    )

    # User override for which total to use
    # Validated to only accept valid total field names
    selected_finance_total: Optional[Literal["grand_total", "net_price_total"]] = Field(
        default=None,
        description="Which total to use: 'grand_total' or 'net_price_total'",
    )

    @field_validator("discounts", "shipping", "tax", mode="before")
    @classmethod
    def null_to_zero(cls, v):
        """Coerce null/None values to 0.0 for optional amount fields."""
        return 0.0 if v is None else v


class CommercialTerms(BaseModel):
    """
    Payment and subscription terms from a vendor quote.

    Captures billing terms, subscription duration, and coverage periods.
    For date range terms (e.g., "11/22/2024 - 11/21/2029"), calculate
    subscription_term_months from the date range.
    """

    payment_terms: Optional[str] = Field(
        default=None, description="Payment terms text (e.g., 'Net 30')"
    )
    subscription_term_months: Optional[int] = Field(
        default=None, ge=1, description="Total subscription duration in months"
    )
    auto_renew: Optional[bool] = Field(
        default=None, description="Whether subscription auto-renews"
    )
    coverage_start: Optional[date] = Field(
        default=None, description="Start date of coverage period"
    )
    coverage_end: Optional[date] = Field(
        default=None, description="End date of coverage period"
    )

    @field_validator("coverage_start", "coverage_end", mode="before")
    @classmethod
    def coerce_date_format(cls, v):
        """Coerce various date formats to ISO format."""
        return coerce_date(v)


class ParseBreakdown(BaseModel):
    """Timing breakdown for PDF parsing sub-steps."""

    text_extract_ms: int = Field(..., ge=0, description="PyMuPDF text extraction time")
    table_extract_ms: int = Field(
        ..., ge=0, description="pdfplumber table extraction time"
    )
    image_render_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Page-to-image rendering time (scanned PDFs only)",
    )


class ExtractBreakdown(BaseModel):
    """Timing breakdown for LLM extraction sub-steps."""

    prompt_prep_ms: int = Field(
        ..., ge=0, description="Prompt formatting and preparation time"
    )
    llm_api_ms: int = Field(
        ..., ge=0, description="LLM API call time (network + inference)"
    )
    post_process_ms: int = Field(
        ..., ge=0, description="Confidence calculation and post-processing time"
    )


class ExtractionMetadata(BaseModel):
    """
    Metadata about the extraction process.

    Populated by the pipeline AFTER LLM extraction completes.
    The LLM does not return this data - it's computed from pipeline metrics.
    """

    processing_time_ms: int = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )
    model_used: str = Field(..., description="LLM model used for extraction")
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall extraction confidence (0-1)"
    )
    validation_passed: bool = Field(
        ..., description="Whether all validation rules passed"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation error messages"
    )
    requires_review: bool = Field(
        default=False,
        description="True if confidence < 0.90 or validation failed",
    )

    # Stage timing breakdown
    parse_time_ms: Optional[int] = Field(
        default=None, ge=0, description="PDF parsing stage time in milliseconds"
    )
    extract_time_ms: Optional[int] = Field(
        default=None, ge=0, description="LLM extraction stage time in milliseconds"
    )
    validate_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Validation stage time in milliseconds"
    )

    # Detailed sub-step breakdowns
    parse_breakdown: Optional[ParseBreakdown] = Field(
        default=None, description="Timing breakdown for PDF parsing sub-steps"
    )
    extract_breakdown: Optional[ExtractBreakdown] = Field(
        default=None, description="Timing breakdown for LLM extraction sub-steps"
    )


class VendorQuoteHeader(BaseModel):
    """
    Header-only extraction for the first pass of chunked extraction.

    Contains all quote-level fields EXCEPT line_items. Used when
    extracting headers separately from line items to avoid the
    VendorQuote.line_items min_length=1 constraint forcing hallucination.
    """

    # Quote identification
    quote_id: Optional[str] = Field(
        default=None, description="Quote identifier (may be generated placeholder)"
    )
    quote_date: Optional[date] = Field(
        default=None, description="Date quote was issued"
    )
    valid_until: Optional[date] = Field(
        default=None, description="Quote expiration date"
    )
    currency: str = Field(default="USD", description="Currency code (USD only for MVP)")

    # Parties
    vendor: Entity = Field(..., description="Vendor/seller information")
    customer: Optional[Entity] = Field(
        default=None, description="Customer/buyer information"
    )

    # Financial totals
    amounts: Amounts = Field(..., description="Quote totals and adjustments")

    # Terms
    commercial_terms: Optional[CommercialTerms] = Field(
        default=None, description="Payment and subscription terms"
    )

    @field_validator("quote_date", "valid_until", mode="before")
    @classmethod
    def coerce_date_format(cls, v):
        """Coerce various date formats to ISO format."""
        return coerce_date(v)


class LineItemBatch(BaseModel):
    """
    Batch of line items extracted from a table chunk.

    Used in chunked extraction: each chunk of ~15 table rows
    produces one LineItemBatch. Batches are merged after extraction.
    """

    line_items: list[LineItem] = Field(
        ..., min_length=1, description="Extracted line items from this chunk"
    )


class VendorQuote(BaseModel):
    """
    Root model representing a complete vendor quote extraction.

    This is the primary output of the extraction pipeline.
    All required fields must be present for a valid extraction.

    Note: extraction_metadata is Optional because the LLM extraction
    returns the quote data, and the pipeline populates metadata afterwards.
    """

    # Quote identification
    quote_id: Optional[str] = Field(
        default=None, description="Quote identifier (may be generated placeholder)"
    )
    quote_date: Optional[date] = Field(
        default=None, description="Date quote was issued"
    )
    valid_until: Optional[date] = Field(
        default=None, description="Quote expiration date"
    )
    currency: str = Field(default="USD", description="Currency code (USD only for MVP)")

    # Parties
    vendor: Entity = Field(..., description="Vendor/seller information")
    customer: Optional[Entity] = Field(
        default=None, description="Customer/buyer information"
    )

    # Line items - minimum 1 required
    line_items: list[LineItem] = Field(
        ..., min_length=1, description="List of quote line items"
    )

    # Financial totals
    amounts: Amounts = Field(..., description="Quote totals and adjustments")

    # Terms
    commercial_terms: Optional[CommercialTerms] = Field(
        default=None, description="Payment and subscription terms"
    )

    # Extraction metadata - populated by pipeline, not LLM
    extraction_metadata: Optional[ExtractionMetadata] = Field(
        default=None,
        description="Processing metadata (populated post-extraction by pipeline)",
    )

    # Source regions for UI highlighting - populated by pipeline from ParseResult
    source_regions: list[SourceRegion] = Field(
        default_factory=list,
        description="PDF source regions for UI highlighting (populated by pipeline)",
    )

    @field_validator("quote_date", "valid_until", mode="before")
    @classmethod
    def coerce_date_format(cls, v):
        """Coerce various date formats to ISO format."""
        return coerce_date(v)
