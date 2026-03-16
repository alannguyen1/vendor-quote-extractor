"""
Extraction module for Vendor Quote Extractor.

Provides PDF parsing, LLM extraction, validation, and pipeline orchestration.

Components:
- PDFParser: Extracts text and tables from PDF documents
- LLMExtractor: Uses LLM for structured data extraction
- Validator: Validates extracted data against business rules
- ExtractionPipeline: Orchestrates the full extraction process
"""

from src.extraction.llm_extractor import ExtractionResult, LLMExtractor
from src.extraction.pdf_parser import ParseResult, PDFParser, TableData
from src.extraction.pipeline import (
    MAX_CONCURRENT_EXTRACTIONS,
    ExtractionPipeline,
    PipelineResult,
    StageResult,
    get_extraction_semaphore,
    reset_extraction_semaphore,
)
from src.extraction.validator import ValidationResult, Validator

__all__ = [
    # PDF Parser
    "PDFParser",
    "ParseResult",
    "TableData",
    # LLM Extractor
    "LLMExtractor",
    "ExtractionResult",
    # Validator
    "Validator",
    "ValidationResult",
    # Pipeline
    "ExtractionPipeline",
    "PipelineResult",
    "StageResult",
    # Concurrency control
    "MAX_CONCURRENT_EXTRACTIONS",
    "get_extraction_semaphore",
    "reset_extraction_semaphore",
]
