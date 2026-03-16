"""SQLAlchemy ORM models for database persistence.

Defines the database schema for storing extractions and corrections.
Uses SQLAlchemy 2.0+ declarative style with type hints.

Tables:
    - extractions: Stores extracted vendor quotes with file hash for dedup
    - corrections: Tracks field-level edits with audit trail
    - correction_examples: Accumulates correction patterns for few-shot learning
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Extraction(Base):
    """Stores extracted vendor quotes.

    Each extraction is uniquely identified by file hash to prevent
    duplicate processing of the same PDF.

    Attributes:
        id: Auto-incrementing primary key.
        filename: Original uploaded filename.
        file_hash: SHA-256 hash of PDF bytes (unique).
        extracted_json: Full VendorQuote as JSON string.
        overall_confidence: Extraction confidence score (0-1).
        requires_review: Whether human review is needed.
        created_at: Timestamp of extraction.
        corrections: Related correction records.
    """

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    extracted_json: Mapped[str] = mapped_column(Text, nullable=False)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    requires_review: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship to corrections
    corrections: Mapped[list["Correction"]] = relationship(
        "Correction", back_populates="extraction", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_extractions_hash", "file_hash"),)


class Correction(Base):
    """Tracks field-level corrections to extractions.

    Provides audit trail for human edits to extracted data.
    Stores original and corrected values with JSON path notation.

    Attributes:
        id: Auto-incrementing primary key.
        extraction_id: Foreign key to extractions table.
        field_path: JSON path to corrected field (e.g., "line_items[3].item_type").
        original_value: Value before correction (as string).
        corrected_value: Value after correction (as string).
        created_at: Timestamp of correction.
        extraction: Related extraction record.
    """

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    extraction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("extractions.id"), nullable=False
    )
    field_path: Mapped[str] = mapped_column(String(255), nullable=False)
    original_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship to extraction
    extraction: Mapped["Extraction"] = relationship(
        "Extraction", back_populates="corrections"
    )

    __table_args__ = (Index("idx_corrections_extraction", "extraction_id"),)


class CorrectionExample(Base):
    """Stores correction patterns for few-shot learning.

    Accumulates examples across all extractions to improve LLM accuracy.
    When corrections are made, patterns are extracted and stored here
    for inclusion in future extraction prompts as few-shot examples.

    The used_count tracks how often an example has been included in prompts,
    while success_rate tracks whether subsequent extractions of similar
    content were correct (for future optimization).

    Attributes:
        id: Auto-incrementing primary key.
        field_path: Normalized field path (e.g., "line_items.item_type").
        original_value: The incorrectly extracted value.
        corrected_value: The user-provided correct value.
        document_context: Surrounding text from the source document.
        used_count: Number of times included in few-shot prompts.
        success_rate: Success rate of extractions using this example.
        created_at: Timestamp when example was created.
    """

    __tablename__ = "correction_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_path: Mapped[str] = mapped_column(String(255), nullable=False)
    original_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    document_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    used_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (Index("idx_examples_field", "field_path"),)
