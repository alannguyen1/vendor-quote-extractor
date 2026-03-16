"""
Pipeline orchestrator for vendor quote extraction.

Coordinates the extraction process through three stages:
1. PDF Parsing - Extract text and tables from PDF
2. LLM Extraction - Use LLM to extract structured data
3. Validation - Verify extracted data against business rules

Handles timeouts, scanned PDF routing, metadata population, and concurrency control.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from config import get_settings
from src.extraction.llm_extractor import ExtractionResult, LLMExtractor
from src.extraction.pdf_parser import PDFParser
from src.extraction.router import SmartRouter
from src.extraction.validator import Validator
from src.schemas import (
    ExtractBreakdown,
    ExtractionMetadata,
    ParseBreakdown,
    VendorQuote,
)

logger = logging.getLogger("vendor_quote_extractor.pipeline")

# Module-level semaphore to limit concurrent LLM extractions
# Per spec: "Concurrent extractions: 5 minimum" - we support exactly 5
# This prevents rate limiting from LLM providers
MAX_CONCURRENT_EXTRACTIONS = 5
_extraction_semaphore: asyncio.Semaphore | None = None


def get_extraction_semaphore() -> asyncio.Semaphore:
    """
    Get or create the extraction semaphore.

    Uses lazy initialization to ensure semaphore is created in the event loop context.
    Module-level to be shared across all pipeline instances.
    """
    global _extraction_semaphore
    if _extraction_semaphore is None:
        _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)
    return _extraction_semaphore


def reset_extraction_semaphore() -> None:
    """Reset the semaphore (for testing purposes)."""
    global _extraction_semaphore
    _extraction_semaphore = None


@dataclass
class StageResult:
    """
    Result of a single pipeline stage.

    Attributes:
        name: Stage name (parse, extract, validate)
        duration_ms: Time taken in milliseconds
        success: Whether stage completed successfully
        error: Error message if stage failed
    """

    name: str
    duration_ms: int
    success: bool
    error: str | None = None


@dataclass
class PipelineResult:
    """
    Result of the full extraction pipeline.

    Attributes:
        quote: Extracted VendorQuote (None if extraction failed)
        error: Error message if pipeline failed
        stages: List of stage results with timing info
    """

    quote: VendorQuote | None
    error: str | None
    stages: list[StageResult] = field(default_factory=list)


class UnavailableLLMExtractor:
    """Fallback extractor used when no LLM providers are configured."""

    def __init__(self, error: str) -> None:
        self.error = error

    async def extract(
        self,
        text: str,
        tables: list,
        page_count: int,
        pdf_bytes: bytes | None = None,
    ) -> ExtractionResult:
        return ExtractionResult(
            quote=None,
            confidence=0.0,
            error=self.error,
            fallback_used=False,
        )

    async def extract_from_images(
        self, images: list[bytes], pdf_bytes: bytes | None = None
    ) -> ExtractionResult:
        return ExtractionResult(
            quote=None,
            confidence=0.0,
            error=self.error,
            fallback_used=False,
        )


class ExtractionPipeline:
    """
    Orchestrates the vendor quote extraction pipeline.

    Stages:
    1. PDF Parsing - PDFParser extracts text and tables
    2. LLM Extraction - LLMExtractor converts to structured data
    3. Validation - Validator checks business rules

    Features:
    - Automatic routing of scanned PDFs to vision API
    - Per-stage timing and error tracking
    - Timeout handling with partial results
    - Metadata population with confidence and review flags
    """

    def __init__(self) -> None:
        """Initialize pipeline components."""
        self.settings = get_settings()
        self.pdf_parser = PDFParser()
        self.router = SmartRouter(self.settings)
        try:
            self.llm_extractor: LLMExtractor | UnavailableLLMExtractor = LLMExtractor()
        except RuntimeError as exc:
            logger.warning(
                "Initializing pipeline without configured LLM providers: %s", exc
            )
            self.llm_extractor = UnavailableLLMExtractor(str(exc))
        self.validator = Validator()

    async def process(self, pdf_bytes: bytes, filename: str) -> PipelineResult:
        """
        Process a PDF document through the extraction pipeline.

        Args:
            pdf_bytes: Raw PDF file content
            filename: Original filename (for logging)

        Returns:
            PipelineResult with extracted quote and stage info
        """
        stages: list[StageResult] = []
        start_time = time.time()

        logger.info(f"Starting extraction pipeline for: {filename}")

        # Apply timeout to entire pipeline
        timeout_seconds = self.settings.processing_timeout_seconds
        try:
            result = await asyncio.wait_for(
                self._run_pipeline(pdf_bytes, filename, stages, start_time),
                timeout=timeout_seconds,
            )
            return result

        except asyncio.TimeoutError:
            logger.error(f"Pipeline timeout after {timeout_seconds}s for: {filename}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            stages.append(
                StageResult(
                    name="timeout",
                    duration_ms=elapsed_ms,
                    success=False,
                    error=f"Processing timeout after {timeout_seconds} seconds",
                )
            )
            return PipelineResult(
                quote=None,
                error=f"Processing timeout after {timeout_seconds} seconds",
                stages=stages,
            )

    async def _run_pipeline(
        self,
        pdf_bytes: bytes,
        filename: str,
        stages: list[StageResult],
        pipeline_start: float,
    ) -> PipelineResult:
        """
        Run the extraction pipeline stages.

        Internal method that executes the actual pipeline logic.
        """
        # Stage 1: PDF Parsing
        parse_start = time.time()
        parse_result = await asyncio.to_thread(self.pdf_parser.extract, pdf_bytes)
        parse_duration = int((time.time() - parse_start) * 1000)

        stages.append(
            StageResult(
                name="parse",
                duration_ms=parse_duration,
                success=parse_result.error is None,
                error=parse_result.error,
            )
        )

        if parse_result.error:
            logger.error(f"PDF parsing failed: {parse_result.error}")
            return PipelineResult(
                quote=None,
                error=f"PDF parsing failed: {parse_result.error}",
                stages=stages,
            )

        logger.info(
            f"Parsed {parse_result.page_count} pages, "
            f"{len(parse_result.tables)} tables, "
            f"scanned={parse_result.is_scanned} in {parse_duration}ms"
        )

        # Stage 2: LLM Extraction (with concurrency control)
        extract_start = time.time()

        # Determine which extractor to use
        if self.settings.routing_enabled:
            # Use Smart Router to select optimal provider
            routing_decision = self.router.select_provider(parse_result)
            routed_providers = self.router.get_providers_for_decision(routing_decision)

            # Create extractor with routed providers (or use default if no providers)
            if routed_providers:
                extractor = LLMExtractor(providers=routed_providers)
                logger.info(
                    f"Router selected: {routing_decision.provider_name} "
                    f"(reason: {routing_decision.reason}, "
                    f"optimal: {routing_decision.is_optimal}, "
                    f"est_latency: {routing_decision.expected_latency_ms}ms)"
                )
            else:
                extractor = self.llm_extractor
                logger.warning("No routed providers available, using default extractor")
        else:
            # Routing disabled - use default extractor
            extractor = self.llm_extractor

        # Acquire semaphore to limit concurrent LLM calls
        # This prevents rate limiting from providers
        semaphore = get_extraction_semaphore()
        async with semaphore:
            if parse_result.is_scanned and parse_result.images:
                # Route scanned PDFs to vision API
                logger.info("Using vision API for scanned PDF extraction")
                extraction_result = await extractor.extract_from_images(
                    parse_result.images, pdf_bytes
                )
            else:
                # Standard text-based extraction
                extraction_result = await extractor.extract(
                    parse_result.text,
                    parse_result.tables,
                    parse_result.page_count,
                    pdf_bytes,
                )

        extract_duration = int((time.time() - extract_start) * 1000)

        stages.append(
            StageResult(
                name="extract",
                duration_ms=extract_duration,
                success=extraction_result.quote is not None,
                error=extraction_result.error,
            )
        )

        if extraction_result.error or extraction_result.quote is None:
            logger.error(f"LLM extraction failed: {extraction_result.error}")
            return PipelineResult(
                quote=None,
                error=f"Extraction failed: {extraction_result.error}",
                stages=stages,
            )

        logger.info(
            f"Extracted {len(extraction_result.quote.line_items)} line items, "
            f"confidence={extraction_result.confidence:.2f}, "
            f"fallback={'yes' if extraction_result.fallback_used else 'no'} "
            f"in {extract_duration}ms"
        )

        # Stage 3: Validation
        # H6: Pass parse_result for completeness validation
        validate_start = time.time()
        validation_result = self.validator.validate(
            extraction_result.quote,
            parse_result=parse_result,  # H6: enables truncation detection
        )
        validate_duration = int((time.time() - validate_start) * 1000)

        stages.append(
            StageResult(
                name="validate",
                duration_ms=validate_duration,
                success=validation_result.passed,
                error=None if validation_result.passed else "Validation failed",
            )
        )

        logger.info(
            f"Validation {'passed' if validation_result.passed else 'failed'}, "
            f"{len(validation_result.errors)} errors, "
            f"{len(validation_result.warnings)} warnings "
            f"in {validate_duration}ms"
        )

        # Calculate total processing time
        total_time_ms = int((time.time() - pipeline_start) * 1000)

        # Determine model used based on provider that succeeded
        provider_name = extraction_result.provider_used
        if provider_name == "openai":
            model_used = self.settings.llm_model
        elif provider_name == "anthropic":
            model_used = self.settings.anthropic_model
        elif provider_name == "groq":
            model_used = self.settings.groq_model
        elif provider_name == "fireworks":
            model_used = self.settings.fireworks_model
        elif provider_name == "gemini":
            model_used = self.settings.gemini_model
        elif extraction_result.fallback_used:
            model_used = self.settings.anthropic_model
        else:
            model_used = self.settings.llm_model

        # Determine if review is required
        confidence_threshold = self.settings.confidence_threshold
        requires_review = (
            extraction_result.confidence < confidence_threshold
            or not validation_result.passed
        )

        # Extract stage timings from stages list
        stage_timings = {s.name: s.duration_ms for s in stages}

        # Build parse breakdown if available
        parse_breakdown = None
        if parse_result.timing:
            parse_breakdown = ParseBreakdown(
                text_extract_ms=parse_result.timing.text_extract_ms,
                table_extract_ms=parse_result.timing.table_extract_ms,
                image_render_ms=parse_result.timing.image_render_ms,
            )

        # Build extract breakdown if available
        extract_breakdown = None
        if extraction_result.timing:
            extract_breakdown = ExtractBreakdown(
                prompt_prep_ms=extraction_result.timing.prompt_prep_ms,
                llm_api_ms=extraction_result.timing.llm_api_ms,
                post_process_ms=extraction_result.timing.post_process_ms,
            )

        # Populate extraction metadata
        extraction_metadata = ExtractionMetadata(
            processing_time_ms=total_time_ms,
            model_used=model_used,
            overall_confidence=extraction_result.confidence,
            validation_passed=validation_result.passed,
            validation_errors=validation_result.errors,
            requires_review=requires_review,
            parse_time_ms=stage_timings.get("parse"),
            extract_time_ms=stage_timings.get("extract"),
            validate_time_ms=stage_timings.get("validate"),
            parse_breakdown=parse_breakdown,
            extract_breakdown=extract_breakdown,
        )

        # Create final quote with metadata and source regions
        # Source regions enable UI highlighting by linking extracted data
        # back to their visual location in the PDF
        final_quote = extraction_result.quote.model_copy(
            update={
                "extraction_metadata": extraction_metadata,
                "source_regions": parse_result.source_regions,
            }
        )

        # Log timing breakdown for performance analysis
        parse_detail = f"parse={parse_duration}ms"
        if parse_breakdown:
            parse_detail += (
                f" (text={parse_breakdown.text_extract_ms}ms, "
                f"tables={parse_breakdown.table_extract_ms}ms"
            )
            if parse_breakdown.image_render_ms is not None:
                parse_detail += f", render={parse_breakdown.image_render_ms}ms"
            parse_detail += ")"

        extract_detail = f"extract={extract_duration}ms"
        if extract_breakdown:
            extract_detail += (
                f" (prep={extract_breakdown.prompt_prep_ms}ms, "
                f"llm={extract_breakdown.llm_api_ms}ms, "
                f"post={extract_breakdown.post_process_ms}ms)"
            )

        logger.info(
            f"Pipeline complete for {filename}: "
            f"total={total_time_ms}ms [{parse_detail}, {extract_detail}, "
            f"validate={validate_duration}ms], requires_review={requires_review}"
        )

        return PipelineResult(
            quote=final_quote,
            error=None,
            stages=stages,
        )
