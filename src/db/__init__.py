"""Database module for persistence layer.

Provides SQLite storage for extractions and corrections using
SQLAlchemy 2.0+ with async support via aiosqlite.

Exports:
    - Extraction: ORM model for stored extractions
    - Correction: ORM model for field corrections
    - CorrectionExample: ORM model for few-shot learning examples
    - get_session: Async session dependency for FastAPI
    - init_db: Initialize database tables
    - close_db: Cleanup database connections
    - Few-shot learning functions for correction examples
"""

from src.db.database import close_db, get_session, init_db
from src.db.fewshot_service import (
    add_correction_example,
    format_few_shot_prompt,
    get_all_examples,
    get_example_count,
    get_examples_for_field,
    increment_example_usage,
    normalize_field_path,
    update_example_success,
)
from src.db.models import Base, Correction, CorrectionExample, Extraction

__all__ = [
    "Base",
    "Correction",
    "CorrectionExample",
    "Extraction",
    "add_correction_example",
    "close_db",
    "format_few_shot_prompt",
    "get_all_examples",
    "get_example_count",
    "get_examples_for_field",
    "get_session",
    "increment_example_usage",
    "init_db",
    "normalize_field_path",
    "update_example_success",
]
