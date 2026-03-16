"""
LLM extraction module for structured data extraction from vendor quotes.

Uses the provider abstraction layer for multi-provider support with
automatic failover. Primary provider (OpenAI by default) with Anthropic
failover, extensible to Groq/Fireworks/Gemini.

Integrates few-shot learning from user corrections to improve extraction
accuracy over time (H4 implementation).
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import cast

from config import get_settings
from src.db.database import get_session_factory
from src.db.fewshot_service import format_few_shot_prompt, get_examples_for_field
from src.extraction.pdf_parser import TableData
from src.extraction.providers import BaseLLMProvider, get_providers
from src.schemas import LineItemBatch, VendorQuote, VendorQuoteHeader
from src.schemas.models import LineItem

logger = logging.getLogger("vendor_quote_extractor.llm_extractor")


# Extraction prompt template following PRD specification
EXTRACTION_PROMPT = """
You are a document extraction specialist. Extract structured data from vendor quotes.

RULES:
1. Only extract data that is explicitly present in the document
2. Return null for fields that cannot be found - DO NOT guess
3. For item_type classification:
   - hardware: physical equipment, devices, appliances (look for -HW suffix)
   - software: licenses, subscriptions, SaaS (look for LIC-, license, subscription)
   - services: implementation, support, labour, professional services
   - discount/credit: negative amounts, credits, rebates
   - fee: fees, charges not fitting other categories
   - tax: tax line items
   - shipping: shipping, freight, delivery charges
4. AMOUNT EXTRACTION RULES:
   - grand_total = sum of all line items + tax + shipping (what customer actually pays)
   - net_price_total = grand_total MINUS credits/trade-ins (if any)
   - For line items: Use DISCOUNTED unit prices when available, not MSRP/list prices
   - grand_total MUST be consistent with line items: if items use discounted
     prices, grand_total should be the discounted total, NOT the MSRP total
   - Credits/trade-ins: Extract as NEGATIVE line items (negative extended_price)

   CRITICAL: TCV vs Net distinction (for multi-year contracts):
   - TCV (Total Contract Value) = sum of all items BEFORE credits/trade-ins
   - Net Amount Due = TCV MINUS credits/trade-ins
   - Use TCV as grand_total, Net as net_price_total

   CRITICAL: MSRP vs Discounted distinction (for quotes with discounts):
   - MSRP = list price BEFORE discount
   - Discounted = actual price AFTER discount (what customer pays)
   - Use DISCOUNTED prices for line items AND grand_total (they must match)

5. For subscription terms: extract months if stated, or calculate from date ranges
6. Line numbers should be sequential starting from 1
7. Extended prices can be negative for credits/discounts
8. LINE ITEM COMPLETENESS - CRITICAL:
   - Each data row is tagged [R01], [R02], etc. — every tag is a SEPARATE item
   - Do NOT merge or skip rows even if content looks identical
   - Items with the same product name/SKU represent different locations or groups
   - Tables contain ~{row_count} data rows - your output should be close
   - Do not invent items to reach the count; only extract rows actually present
9. line_number MUST equal the [RNN] tag number (e.g., [R07] → line_number=7)
   - This enables gap detection: missing line_numbers mean dropped rows
   - If you see tags [R01] through [R51], output MUST have line_numbers 1 through 51

CRITICAL FORMATTING RULES:
- ALL dates MUST be ISO 8601 format: YYYY-MM-DD (e.g., "2024-10-03", NOT "10/3/2024")
- For credit/discount line items: unit_price MUST be >= 0, use NEGATIVE extended_price
  Example: credit of $7900 → unit_price: 7900.00, extended_price: -7900.00
- If shipping, tax, or discounts are not found in document, use 0.0 (not null)

DOCUMENT TEXT:
{text}

EXTRACTED TABLES:
{tables}

Extract the vendor quote data according to the schema.
"""

# Vision-specific prompt with instructions for scanned documents
VISION_EXTRACTION_PROMPT = """
You are a document extraction specialist. Extract data from scanned vendor quotes.

RULES:
1. Only extract data that is explicitly visible in the images
2. Return null for fields that cannot be found - DO NOT guess
3. For item_type classification:
   - hardware: physical equipment, devices, appliances (look for -HW suffix)
   - software: licenses, subscriptions, SaaS (look for LIC-, license, subscription)
   - services: implementation, support, labour, professional services
   - discount/credit: negative amounts, credits, rebates
   - fee: fees, charges not fitting other categories
   - tax: tax line items
   - shipping: shipping, freight, delivery charges
4. AMOUNT EXTRACTION RULES:
   - grand_total = sum of all line items + tax + shipping (what customer actually pays)
   - net_price_total = grand_total MINUS credits/trade-ins (if any)
   - For line items: Use DISCOUNTED unit prices when available, not MSRP/list prices
   - grand_total MUST be consistent with line items: if items use discounted
     prices, grand_total should be the discounted total, NOT the MSRP total
   - Credits/trade-ins: Extract as NEGATIVE line items (negative extended_price)

   CRITICAL: TCV vs Net distinction (for multi-year contracts):
   - TCV (Total Contract Value) = sum of all items BEFORE credits/trade-ins
   - Net Amount Due = TCV MINUS credits/trade-ins
   - Use TCV as grand_total, Net as net_price_total

   CRITICAL: MSRP vs Discounted distinction (for quotes with discounts):
   - MSRP = list price BEFORE discount
   - Discounted = actual price AFTER discount (what customer pays)
   - Use DISCOUNTED prices for line items AND grand_total (they must match)

5. For subscription terms: extract months if stated, or calculate from date ranges
6. Line numbers should be sequential starting from 1
7. Extended prices can be negative for credits/discounts
8. LINE ITEM COMPLETENESS - CRITICAL:
   - Extract every distinct line item shown; do not merge or deduplicate similar rows
   - Items with the same product name/SKU often represent different locations or groups
   - Each priced row is a separate line item (exclude subtotals and headers)
   - Do not invent items; only extract rows actually visible in the images

CRITICAL FORMATTING RULES:
- ALL dates MUST be ISO 8601 format: YYYY-MM-DD (e.g., "2024-10-03", NOT "10/3/2024")
- For credit/discount line items: unit_price MUST be >= 0, use NEGATIVE extended_price
  Example: credit of $7900 → unit_price: 7900.00, extended_price: -7900.00
- If shipping, tax, or discounts are not found in document, use 0.0 (not null)

Extract the vendor quote data according to the schema.
"""


@dataclass
class ExtractTimingBreakdown:
    """Timing breakdown for LLM extraction sub-steps."""

    prompt_prep_ms: int
    llm_api_ms: int
    post_process_ms: int


@dataclass
class ExtractionResult:
    """
    Result of LLM extraction.

    Attributes:
        quote: The extracted VendorQuote (None if extraction failed)
        confidence: Extraction confidence score (0-1)
        error: Error message if extraction failed
        fallback_used: True if fallback provider was used
        timing: Timing breakdown for extraction sub-steps
        provider_used: Name of the provider that succeeded
    """

    quote: VendorQuote | None
    confidence: float
    error: str | None
    fallback_used: bool
    timing: ExtractTimingBreakdown | None = None
    provider_used: str | None = None


class LLMExtractor:
    """
    Extracts structured vendor quote data using LLM.

    Features:
    - Provider abstraction with automatic failover
    - Schema-enforced extraction via instructor library
    - Automatic confidence calculation
    - Placeholder quote_id generation for missing IDs

    The extractor uses the provider abstraction layer to support multiple
    LLM providers (OpenAI, Anthropic) with automatic failover when the
    primary provider fails.
    """

    def __init__(self, providers: list[BaseLLMProvider] | None = None) -> None:
        """
        Initialize the extractor with configured providers.

        Args:
            providers: Optional list of providers in failover order.
                      If not provided, uses get_providers() from settings.
        """
        self.settings = get_settings()

        # Use injected providers or get from settings
        if providers is not None:
            self._providers = providers
        else:
            self._providers = get_providers(self.settings)

        if not self._providers:
            msg = "No LLM providers available. Check API key configuration."
            raise RuntimeError(msg)

        # Log provider chain
        provider_names = [p.name for p in self._providers]
        logger.info(f"LLM provider chain: {' -> '.join(provider_names)}")

    @property
    def primary_provider(self) -> BaseLLMProvider:
        """Get the primary (first) provider."""
        return self._providers[0]

    @property
    def has_fallback(self) -> bool:
        """Check if fallback providers are available."""
        return len(self._providers) > 1

    async def extract(
        self,
        text: str,
        tables: list[TableData],
        page_count: int,
        pdf_bytes: bytes | None = None,
    ) -> ExtractionResult:
        """
        Extract structured data from document text and tables.

        Uses chunked extraction for large documents (>chunking_min_rows)
        to prevent LLM deduplication of identical-looking rows.

        Args:
            text: Extracted text from PDF
            tables: Extracted table data
            page_count: Number of pages in document
            pdf_bytes: Original PDF bytes (for placeholder ID generation)

        Returns:
            ExtractionResult with quote data, confidence, and error info
        """
        # Track prompt preparation time
        prep_start = time.time()

        primary_name = self.primary_provider.name

        # Pre-process tables: remove repeated header rows (pdfplumber artifact
        # at page breaks) and strip embedded per-item tax disclosure lines that
        # contain dollar amounts the LLM misinterprets as separate line items.
        tables = self._preprocess_tables(tables)

        # Filter out summary tables (quote headers, grand totals, terms)
        # BEFORE they receive [RNN] tags. Summary tables are small tables
        # whose content matches document-level keywords. They stay available
        # in the raw text for header extraction.
        line_item_tables, _summary_tables = self._filter_line_item_tables(tables)
        total_rows = sum(len(t.rows) for t in line_item_tables)

        # H3: Diagnostic logging — row counts at pipeline entry
        logger.info(
            f"Extraction starting: rows_expected={total_rows}, "
            f"line_item_tables={len(line_item_tables)}, "
            f"summary_tables_filtered={len(_summary_tables)}, "
            f"pages={page_count}, provider={primary_name}"
        )

        # Decision: chunked vs single-pass extraction
        settings = self.settings
        use_chunking = (
            settings.chunking_enabled and total_rows > settings.chunking_min_rows
        )

        # Strategy 2: Provider-aware single-pass bypass for large-context models
        if use_chunking:
            max_single_pass = settings.provider_single_pass_max_rows.get(
                primary_name, 0
            )
            if total_rows <= max_single_pass:
                logger.info(
                    f"Provider bypass: {primary_name} supports "
                    f"{max_single_pass} rows single-pass, "
                    f"skipping chunking for {total_rows} rows"
                )
                use_chunking = False

        if use_chunking:
            logger.info(
                f"Chunked extraction activated: {total_rows} rows > "
                f"{settings.chunking_min_rows} threshold"
            )
            return await self._extract_chunked(
                text, line_item_tables, total_rows, page_count, pdf_bytes, prep_start
            )

        # --- Standard single-pass extraction ---
        return await self._extract_single_pass(
            text, line_item_tables, total_rows, page_count, pdf_bytes, prep_start
        )

    async def _extract_single_pass(
        self,
        text: str,
        tables: list[TableData],
        total_rows: int,
        page_count: int,
        pdf_bytes: bytes | None,
        prep_start: float,
    ) -> ExtractionResult:
        """
        Standard single-pass extraction for documents under the chunking threshold.
        """
        primary_name = self.primary_provider.name
        max_text = self._get_provider_text_limit(primary_name)
        max_tables = self._get_provider_table_limit(primary_name)

        # Pipeline: format -> compact if needed -> prune if over limit -> truncate rows
        compact = total_rows > 30
        formatted_tables = self._format_tables(tables, compact=compact)

        if not compact and len(formatted_tables) > max_tables:
            compact = True
            formatted_tables = self._format_tables(tables, compact=True)

        if len(formatted_tables) > max_tables and tables:
            pruned = self._prune_non_financial_columns(tables)
            formatted_tables = self._format_tables(pruned, compact=compact)

        # H0c: Truncate non-table text for table-heavy documents
        settings = self.settings
        if total_rows > settings.complexity_routing_threshold:
            effective_text_limit = min(max_text, settings.large_doc_text_budget)
        else:
            effective_text_limit = max_text

        text_len = len(text)
        tables_len = len(formatted_tables)
        truncated_text = text[:effective_text_limit]
        if tables_len > max_tables:
            truncated_tables, rows_kept, rows_total = self._truncate_tables_row_aware(
                formatted_tables, max_tables
            )
            prompt_row_count = rows_kept
            trunc_len = len(truncated_tables)
            logger.warning(
                f"Tables truncated from {tables_len:,} to "
                f"{trunc_len:,} characters "
                f"({rows_kept}/{rows_total} rows kept) "
                f"[provider: {primary_name}]"
            )
        else:
            truncated_tables = formatted_tables
            prompt_row_count = total_rows

        if text_len > effective_text_limit:
            logger.warning(
                f"Text truncated from {text_len:,} to "
                f"{effective_text_limit:,} characters "
                f"({text_len - effective_text_limit:,} chars lost) "
                f"[provider: {primary_name}]"
            )

        row_count_str = str(prompt_row_count) if prompt_row_count > 0 else "unknown"
        base_prompt = EXTRACTION_PROMPT.format(
            text=truncated_text,
            tables=truncated_tables,
            row_count=row_count_str,
        )

        few_shot_context = await self._get_few_shot_context()
        if few_shot_context:
            prompt = base_prompt + "\n\n" + few_shot_context
        else:
            prompt = base_prompt

        prompt_prep_ms = int((time.time() - prep_start) * 1000)

        # Try each provider in order until one succeeds
        errors: list[tuple[str, Exception]] = []

        for i, provider in enumerate(self._providers):
            is_fallback = i > 0
            try:
                api_start = time.time()
                quote = cast(VendorQuote, await provider.extract_text(prompt))
                llm_api_ms = int((time.time() - api_start) * 1000)

                # H3: Diagnostic logging — extraction result
                logger.info(
                    f"Single-pass extraction: rows_expected={total_rows}, "
                    f"extracted={len(quote.line_items)}, "
                    f"prompt_rows={prompt_row_count}, provider={provider.name}"
                )

                if is_fallback:
                    logger.info(f"Extraction succeeded with fallback: {provider.name}")

                # H2: Reconciliation — re-extract missing rows BEFORE post-processing
                # so that newly added items get confidence enrichment + source linking
                quote = await self._reconcile_missing_rows(
                    quote, tables, total_rows, provider
                )

                # Auto-correct quantities where math doesn't match
                if self.settings.quantity_auto_correction:
                    quote = quote.model_copy(
                        update={
                            "line_items": self._auto_correct_quantities(
                                quote.line_items
                            )
                        }
                    )

                post_start = time.time()
                quote = self._enrich_line_items_confidence(quote)
                quote = self._link_line_items_to_source_regions(quote, tables)
                confidence = self._calculate_confidence(quote)

                if not quote.quote_id and pdf_bytes:
                    quote = self._add_placeholder_quote_id(quote, pdf_bytes)
                post_process_ms = int((time.time() - post_start) * 1000)

                timing = ExtractTimingBreakdown(
                    prompt_prep_ms=prompt_prep_ms,
                    llm_api_ms=llm_api_ms,
                    post_process_ms=post_process_ms,
                )

                return ExtractionResult(
                    quote=quote,
                    confidence=confidence,
                    error=None,
                    fallback_used=is_fallback,
                    timing=timing,
                    provider_used=provider.name,
                )

            except Exception as e:
                logger.warning(f"{provider.name} extraction failed: {e}")
                errors.append((provider.name, e))

                # If primary failed and fallback needs chunking,
                # switch strategy instead of retrying single-pass.
                if i == 0 and len(self._providers) > 1:
                    fb = self._providers[1]
                    fb_max = settings.provider_single_pass_max_rows.get(fb.name, 0)
                    if total_rows > fb_max and total_rows > settings.chunking_min_rows:
                        logger.info(
                            f"Fallback {fb.name} needs chunking "
                            f"({total_rows} > {fb_max}), switching"
                        )
                        return await self._extract_chunked(
                            text,
                            tables,
                            total_rows,
                            page_count,
                            pdf_bytes,
                            time.time(),
                            providers_override=self._providers[1:],
                        )
                continue

        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        if len(errors) > 1:
            error_msg = f"Both providers failed. {error_details}"
        else:
            error_msg = f"Extraction failed: {error_details}"

        return ExtractionResult(
            quote=None,
            confidence=0.0,
            error=error_msg,
            fallback_used=len(errors) > 1,
        )

    # ------------------------------------------------------------------ #
    # Chunked extraction (Phase 3: H1)
    # ------------------------------------------------------------------ #

    async def _extract_chunked(
        self,
        text: str,
        tables: list[TableData],
        total_rows: int,
        page_count: int,
        pdf_bytes: bytes | None,
        prep_start: float,
        providers_override: list[BaseLLMProvider] | None = None,
    ) -> ExtractionResult:
        """
        Two-phase chunked extraction for large documents.

        Phase A: Extract header (vendor, customer, amounts, terms) in one call.
        Phase B: Extract line items in chunks of ~chunk_rows_per_call rows.
        Merge: Combine header + all line-item batches into a VendorQuote.

        Header and chunk extraction run concurrently via asyncio.gather()
        with a configurable semaphore to control LLM API concurrency.
        """
        settings = self.settings
        providers = providers_override or self._providers
        primary = providers[0]

        # --- Prepare all prompts upfront ---
        max_text = self._get_provider_text_limit(primary.name)
        truncated_text = text[:max_text]
        table_summary = self._build_table_summary(tables, total_rows)
        header_prompt = self._build_header_prompt(truncated_text, table_summary)

        chunks = self._chunk_table_rows(tables, settings.chunk_rows_per_call)
        chunk_prompts = [self._build_line_item_chunk_prompt(ct) for ct in chunks]

        logger.info(
            f"Parallel chunked extraction: {len(chunks)} chunks, "
            f"~{settings.chunk_rows_per_call} rows each, "
            f"concurrency={settings.chunk_parallel_concurrency}"
        )

        # --- Async helpers ---
        semaphore = asyncio.Semaphore(settings.chunk_parallel_concurrency)

        async def _extract_header() -> tuple[VendorQuoteHeader | None, int, bool]:
            """Extract header with provider fallback (not semaphore-limited)."""
            errors: list[tuple[str, Exception]] = []
            for i, provider in enumerate(providers):
                try:
                    api_start = time.time()
                    hdr = cast(
                        VendorQuoteHeader,
                        await provider.extract_text(
                            header_prompt, response_model=VendorQuoteHeader
                        ),
                    )
                    ms = int((time.time() - api_start) * 1000)
                    logger.info(
                        f"Header extraction complete: provider={provider.name}, "
                        f"time={ms}ms"
                    )
                    return hdr, ms, i > 0
                except Exception as e:
                    logger.warning(f"{provider.name} header extraction failed: {e}")
                    errors.append((provider.name, e))
            return None, 0, len(errors) > 1

        async def _extract_single_chunk(
            idx: int, prompt: str
        ) -> tuple[int, LineItemBatch | None, int]:
            """Extract one chunk under semaphore control."""
            if not prompt.strip():
                logger.warning(f"Chunk {idx + 1}/{len(chunks)}: empty, skipping")
                return idx, None, 0
            try:
                async with semaphore:
                    api_start = time.time()
                    batch = cast(
                        LineItemBatch,
                        await primary.extract_text(
                            prompt, response_model=LineItemBatch
                        ),
                    )
                    ms = int((time.time() - api_start) * 1000)
                    logger.info(
                        f"Chunk {idx + 1}/{len(chunks)}: "
                        f"extracted {len(batch.line_items)} items in {ms}ms"
                    )
                    return idx, batch, ms
            except Exception as e:
                logger.warning(f"Chunk {idx + 1}/{len(chunks)} failed: {e}")
                return idx, None, 0

        # --- Fire header + all chunks concurrently ---
        # Separate gathers for type safety: header starts as a task,
        # chunks gather in parallel, then we await the header.
        header_task = asyncio.create_task(_extract_header())
        chunk_results: list[tuple[int, LineItemBatch | None, int]] = (
            list(
                await asyncio.gather(
                    *[
                        _extract_single_chunk(i, cp)
                        for i, cp in enumerate(chunk_prompts)
                    ]
                )
            )
            if chunk_prompts
            else []
        )
        header, header_api_ms, header_fallback_used = await header_task

        if header is None:
            return ExtractionResult(
                quote=None,
                confidence=0.0,
                error=("Header extraction failed across all providers"),
                fallback_used=header_fallback_used,
            )

        # --- Collect chunk results (in original index order) ---
        all_line_items: list[LineItem] = []
        total_llm_ms = header_api_ms

        for _idx, batch, chunk_ms in chunk_results:
            total_llm_ms += chunk_ms
            if batch is not None:
                all_line_items.extend(batch.line_items)

        # --- Merge: Combine header + line items ---
        merged_items = self._merge_line_items(all_line_items)
        prompt_prep_ms = max(0, int((time.time() - prep_start) * 1000) - total_llm_ms)

        logger.info(
            f"Chunked extraction complete: rows_expected={total_rows}, "
            f"extracted={len(merged_items)}, "
            f"raw_items={len(all_line_items)}, "
            f"provider={primary.name}"
        )

        if not merged_items:
            return ExtractionResult(
                quote=None,
                confidence=0.0,
                error="Chunked extraction produced no line items",
                fallback_used=header_fallback_used,
            )

        # Build VendorQuote from header + merged line items
        quote = VendorQuote(
            quote_id=header.quote_id,
            quote_date=header.quote_date,
            valid_until=header.valid_until,
            currency=header.currency,
            vendor=header.vendor,
            customer=header.customer,
            line_items=merged_items,
            amounts=header.amounts,
            commercial_terms=header.commercial_terms,
        )

        # H2: Reconciliation — re-extract missing rows BEFORE post-processing
        # so that newly added items get confidence enrichment + source linking
        quote = await self._reconcile_missing_rows(quote, tables, total_rows, primary)

        # Auto-correct quantities where math doesn't match
        if self.settings.quantity_auto_correction:
            quote = quote.model_copy(
                update={"line_items": self._auto_correct_quantities(quote.line_items)}
            )

        # Post-processing (after reconciliation so all items get enriched)
        post_start = time.time()
        quote = self._enrich_line_items_confidence(quote)
        quote = self._link_line_items_to_source_regions(quote, tables)
        confidence = self._calculate_confidence(quote)

        if not quote.quote_id and pdf_bytes:
            quote = self._add_placeholder_quote_id(quote, pdf_bytes)
        post_process_ms = int((time.time() - post_start) * 1000)

        timing = ExtractTimingBreakdown(
            prompt_prep_ms=prompt_prep_ms,
            llm_api_ms=total_llm_ms,
            post_process_ms=post_process_ms,
        )

        return ExtractionResult(
            quote=quote,
            confidence=confidence,
            error=None,
            fallback_used=header_fallback_used,
            timing=timing,
            provider_used=primary.name,
        )

    def _build_table_summary(self, tables: list[TableData], total_rows: int) -> str:
        """Build abbreviated table summary for header extraction context."""
        if not tables:
            return "No tables detected."

        parts = []
        for table in tables:
            if table.headers:
                parts.append(
                    f"[Page {table.page}] {len(table.rows)} rows: "
                    + " | ".join(table.headers)
                )
            else:
                parts.append(f"[Page {table.page}] {len(table.rows)} rows")
        parts.append(f"\nTotal data rows: {total_rows}")
        return "\n".join(parts)

    def _build_header_prompt(self, text: str, table_summary: str) -> str:
        """Build prompt for header-only extraction (no line items)."""
        return (
            "You are a document extraction specialist. "
            "Extract ONLY the quote-level header fields from this vendor quote.\n\n"
            "Extract: quote_id, quote_date, valid_until, currency, vendor, "
            "customer, amounts (grand_total, subtotal, tax, etc.), "
            "and commercial_terms.\n\n"
            "Do NOT extract individual line items — only header/summary data.\n\n"
            "AMOUNT RULES:\n"
            "- subtotal = sum of line item prices BEFORE tax/shipping\n"
            "- tax = sales tax / VAT (look for SALES TAX, TAX, VAT)\n"
            "- shipping = shipping/freight (look for SHIPPING, FREIGHT)\n"
            "- grand_total = subtotal + tax + shipping (final amount)\n"
            "- net_price_total = grand_total minus credits/trade-ins\n"
            "- Use DISCOUNTED prices, not MSRP/list prices\n"
            "- IMPORTANT: Look for explicit SUBTOTAL, SHIPPING, SALES TAX,\n"
            "  GRAND TOTAL labels — extract shown values, don't calculate\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- ALL dates MUST be ISO 8601 format: YYYY-MM-DD\n"
            "- If shipping, tax, or discounts are not found, use 0.0\n\n"
            f"DOCUMENT TEXT:\n{text}\n\n"
            f"TABLE SUMMARY:\n{table_summary}\n\n"
            "Extract the quote header data according to the schema."
        )

    def _build_line_item_chunk_prompt(self, chunk_text: str) -> str:
        """Build prompt for a single line-item chunk extraction."""
        # Count [RNN] tags to give exact expected output count
        row_count = sum(
            1 for line in chunk_text.split("\n") if line.strip().startswith("[R")
        )
        return (
            "You are a document extraction specialist. "
            "Extract ALL line items from the table rows below.\n\n"
            "RULES:\n"
            f"1. You MUST output EXACTLY {row_count} line items — "
            "one for each [RNN] tag below\n"
            "2. line_number MUST equal the [RNN] tag number "
            "(e.g., [R07] → line_number=7)\n"
            "3. NEVER merge or deduplicate rows — identical descriptions with "
            "different [RNN] tags are SEPARATE purchases for different locations\n"
            "4. For item_type: hardware, software, services, discount/credit, "
            "fee, tax, shipping\n"
            "5. For credits: unit_price >= 0, extended_price is NEGATIVE\n"
            "6. Use DISCOUNTED prices, not MSRP/list prices\n\n"
            f"TABLE ROWS ({row_count} items to extract):\n{chunk_text}\n\n"
            f"Extract exactly {row_count} line items, one per [RNN] tag."
        )

    def _chunk_table_rows(
        self, tables: list[TableData], rows_per_chunk: int
    ) -> list[str]:
        """
        Split table rows into chunks for parallel extraction.

        Each chunk includes the table headers for context and preserves
        the original [RNN] tags (no resetting across chunks).
        """
        if rows_per_chunk <= 0:
            raise ValueError(f"rows_per_chunk must be > 0, got {rows_per_chunk}")

        # First, build all formatted rows with headers
        all_rows: list[tuple[str, str | None]] = []  # (formatted_row, header_block)
        row_num = 0
        current_header_block: str | None = None

        for table in tables:
            if table.headers:
                header_line = " | ".join(table.headers)
                current_header_block = f"[Page {table.page}]\n{header_line}"
            else:
                # Reset header state so headerless tables don't inherit
                # the previous table's headers
                current_header_block = f"[Page {table.page}]"

            for row in table.rows:
                row_num += 1
                formatted = f"[R{row_num:02d}] " + " | ".join(row)
                all_rows.append((formatted, current_header_block))

        # Now chunk them
        chunks: list[str] = []
        i = 0
        while i < len(all_rows):
            chunk_end = min(i + rows_per_chunk, len(all_rows))
            chunk_lines: list[str] = []

            # Include header from the first row in this chunk
            first_header = all_rows[i][1]
            if first_header:
                chunk_lines.append(first_header)

            # Track if we need to re-emit headers when they change
            last_header = first_header
            for j in range(i, chunk_end):
                row_text, row_header = all_rows[j]
                if row_header != last_header and row_header is not None:
                    chunk_lines.append(row_header)
                    last_header = row_header
                chunk_lines.append(row_text)

            chunks.append("\n".join(chunk_lines))
            i = chunk_end

        return chunks

    def _merge_line_items(self, items: list[LineItem]) -> list[LineItem]:
        """
        Merge line items from multiple chunks, deduplicating by line_number.

        Uses line_number (= [RNN] tag) as merge key. Keeps first occurrence
        of each line_number.
        """
        seen: dict[int, LineItem] = {}
        for item in items:
            if item.line_number not in seen:
                seen[item.line_number] = item
            else:
                logger.debug(
                    f"Duplicate line_number {item.line_number} — keeping first"
                )

        # Sort by line_number for consistent output
        merged = sorted(seen.values(), key=lambda x: x.line_number)
        if len(items) != len(merged):
            logger.info(
                f"Merged {len(items)} raw items → {len(merged)} unique "
                f"(removed {len(items) - len(merged)} duplicates)"
            )
        return merged

    def _auto_correct_quantities(self, items: list[LineItem]) -> list[LineItem]:
        """
        Auto-correct quantities where math doesn't match.

        If quantity * unit_price != extended_price (within tolerance),
        compute the expected quantity from extended_price / unit_price.
        Only correct when the ratio is a clean integer >= 1.

        Skips credits (negative extended_price) and zero unit_price to
        avoid division by zero and false corrections on bundle pricing.
        """
        corrected = []
        for item in items:
            if item.unit_price > 0 and item.extended_price > 0:
                expected = item.quantity * item.unit_price
                if abs(expected - item.extended_price) > self.settings.math_tolerance:
                    ratio = item.extended_price / item.unit_price
                    rounded = round(ratio)
                    if rounded >= 1 and abs(ratio - rounded) < 0.01:
                        logger.debug(
                            f"Quantity auto-correction: line {item.line_number} "
                            f"qty {item.quantity} → {rounded}"
                        )
                        item = item.model_copy(update={"quantity": rounded})
            corrected.append(item)
        return corrected

    # ------------------------------------------------------------------ #
    # Two-pass reconciliation (Phase 4: H2)
    # ------------------------------------------------------------------ #

    async def _reconcile_missing_rows(
        self,
        quote: VendorQuote,
        tables: list[TableData],
        expected_rows: int,
        provider: BaseLLMProvider,
    ) -> VendorQuote:
        """
        Detect and re-extract missing rows via gap analysis on line_numbers.

        Uses gap detection on [RNN] line_number tags to find missing rows.
        Invalid line_numbers (< 1 or > expected_rows) are filtered before
        gap detection to prevent false positives from LLM mislabeling.
        """
        if expected_rows == 0:
            return quote

        # Filter invalid line_numbers before gap detection
        extracted_numbers = {
            item.line_number
            for item in quote.line_items
            if 1 <= item.line_number <= expected_rows
        }
        expected_numbers = set(range(1, expected_rows + 1))
        missing_numbers = sorted(expected_numbers - extracted_numbers)

        if not missing_numbers:
            return quote

        logger.info(
            f"Gap detection: {len(missing_numbers)} missing rows detected "
            f"(line_numbers: {missing_numbers[:10]}"
            f"{'...' if len(missing_numbers) > 10 else ''})"
        )

        # Build a prompt with only the missing rows
        missing_rows_text = self._extract_rows_by_number(tables, missing_numbers)
        if not missing_rows_text:
            logger.warning("Reconciliation: could not find missing rows in table data")
            return quote

        recon_prompt = (
            "You are a document extraction specialist. "
            "Extract the following specific line items that were missed.\n\n"
            "RULES:\n"
            "1. line_number MUST equal the [RNN] tag number\n"
            "2. Extract EVERY row shown — do not skip any\n"
            "3. For credits: unit_price >= 0, extended_price is NEGATIVE\n"
            "4. Use DISCOUNTED prices, not MSRP\n\n"
            f"MISSING ROWS:\n{missing_rows_text}\n\n"
            "Extract these line items according to the schema."
        )

        try:
            batch = cast(
                LineItemBatch,
                await provider.extract_text(recon_prompt, response_model=LineItemBatch),
            )

            # Merge reconciled items into existing quote
            existing_by_num = {item.line_number: item for item in quote.line_items}
            added = 0
            for item in batch.line_items:
                if item.line_number not in existing_by_num:
                    existing_by_num[item.line_number] = item
                    added += 1

            if added > 0:
                all_items = sorted(
                    existing_by_num.values(), key=lambda x: x.line_number
                )
                quote = quote.model_copy(update={"line_items": all_items})
                logger.info(
                    f"Reconciliation added {added} items "
                    f"(now {len(all_items)}/{expected_rows})"
                )

        except Exception as e:
            logger.warning(f"Reconciliation failed: {e}")

        return quote

    def _extract_rows_by_number(
        self, tables: list[TableData], target_numbers: list[int]
    ) -> str:
        """Extract specific [RNN]-tagged rows from tables by row number."""
        target_set = set(target_numbers)
        parts: list[str] = []
        row_num = 0
        emitted_table_headers: set[int] = set()  # Track which table indices got headers

        for table_idx, table in enumerate(tables):
            header_text: str | None = None
            if table.headers:
                header_text = " | ".join(table.headers)

            for row in table.rows:
                row_num += 1
                if row_num in target_set:
                    # Emit header once per table (not per row)
                    if table_idx not in emitted_table_headers:
                        emitted_table_headers.add(table_idx)
                        if header_text:
                            parts.append(f"[Page {table.page}]\n{header_text}")
                        else:
                            parts.append(f"[Page {table.page}]")
                    parts.append(f"[R{row_num:02d}] " + " | ".join(row))

        return "\n".join(parts)

    async def extract_from_images(
        self, images: list[bytes], pdf_bytes: bytes | None = None
    ) -> ExtractionResult:
        """
        Extract data from scanned PDF page images using vision API.

        Args:
            images: List of PNG image bytes (one per page)
            pdf_bytes: Original PDF bytes (for placeholder ID generation)

        Returns:
            ExtractionResult with quote data from image analysis
        """
        # Track prompt preparation time
        prep_start = time.time()

        # H4: Append few-shot examples from user corrections
        few_shot_context = await self._get_few_shot_context()
        if few_shot_context:
            prompt = VISION_EXTRACTION_PROMPT + "\n\n" + few_shot_context
        else:
            prompt = VISION_EXTRACTION_PROMPT

        prompt_prep_ms = int((time.time() - prep_start) * 1000)

        # Try each provider in order until one succeeds
        errors: list[tuple[str, Exception]] = []

        for i, provider in enumerate(self._providers):
            is_fallback = i > 0
            try:
                # Track LLM API call time
                api_start = time.time()
                quote = await provider.extract_vision(images, prompt)
                llm_api_ms = int((time.time() - api_start) * 1000)

                # Track post-processing time
                post_start = time.time()

                # Enrich line items with per-item confidence scores
                quote = self._enrich_line_items_confidence(quote)

                confidence = self._calculate_confidence(quote)

                if not quote.quote_id and pdf_bytes:
                    quote = self._add_placeholder_quote_id(quote, pdf_bytes)
                post_process_ms = int((time.time() - post_start) * 1000)

                timing = ExtractTimingBreakdown(
                    prompt_prep_ms=prompt_prep_ms,
                    llm_api_ms=llm_api_ms,
                    post_process_ms=post_process_ms,
                )

                if is_fallback:
                    logger.info(f"Vision succeeded with fallback: {provider.name}")

                return ExtractionResult(
                    quote=quote,
                    confidence=confidence,
                    error=None,
                    fallback_used=is_fallback,
                    timing=timing,
                    provider_used=provider.name,
                )

            except Exception as e:
                logger.warning(f"{provider.name} vision failed: {e}")
                errors.append((provider.name, e))
                continue

        # All providers failed
        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        if len(errors) > 1:
            error_msg = f"Both vision providers failed. {error_details}"
        else:
            error_msg = f"Vision extraction failed: {error_details}"

        return ExtractionResult(
            quote=None,
            confidence=0.0,
            error=error_msg,
            fallback_used=len(errors) > 1,
        )

    # Regex matching embedded per-item tax disclosure lines in cell text.
    # E.g.: "TAX: SANTA MONICA, CA TAX: 10.2500% $341.23"
    _TAX_LINE_RE = re.compile(r"^\s*TAX:", re.IGNORECASE)

    def _preprocess_tables(self, tables: list[TableData]) -> list[TableData]:
        """
        Clean table data before extraction to remove noise that confuses the LLM.

        Two cleanups:
        1. Remove rows that are repeated column headers (pdfplumber artifact
           at page breaks — e.g., ['ITEM', 'QTY', 'SKU#', 'UNIT PRICE', 'EXT. PRICE']).
        2. Strip embedded per-item tax disclosure lines from cell text
           (e.g., 'TAX: SANTA MONICA, CA TAX: 10.2500% $341.23') that contain
           dollar amounts the LLM misinterprets as separate line items.
        """
        cleaned: list[TableData] = []
        total_header_rows_removed = 0
        total_tax_lines_stripped = 0

        for table in tables:
            clean_rows: list[list[str]] = []
            for row in table.rows:
                # Skip rows that are repeated column headers
                if table.headers and self._is_repeated_header_row(row, table.headers):
                    total_header_rows_removed += 1
                    continue

                # Strip embedded tax lines from cell text
                clean_row: list[str] = []
                for cell in row:
                    cleaned_cell, stripped = self._strip_tax_lines(cell)
                    total_tax_lines_stripped += stripped
                    clean_row.append(cleaned_cell)
                clean_rows.append(clean_row)

            cleaned.append(
                TableData(
                    page=table.page,
                    rows=clean_rows,
                    headers=table.headers,
                    bbox=table.bbox,
                )
            )

        if total_header_rows_removed or total_tax_lines_stripped:
            logger.info(
                f"Table preprocessing: removed {total_header_rows_removed} "
                f"repeated header rows, stripped {total_tax_lines_stripped} "
                f"embedded tax lines"
            )

        return cleaned

    # Column-header keywords: if ≥3 appear in a row, it's a header row
    _COLUMN_HEADER_KEYWORDS: tuple[str, ...] = (
        "ITEM",
        "QTY",
        "QUANTITY",
        "SKU",
        "UNIT PRICE",
        "EXT. PRICE",
        "EXT PRICE",
        "EXTENDED PRICE",
        "DESCRIPTION",
        "PART #",
        "PART NO",
        "LINE",
    )

    @staticmethod
    def _is_repeated_header_row(row: list[str], headers: list[str]) -> bool:
        """
        Check if a data row is actually a repeated column header.

        Detects two patterns:
        1. Row matches the table.headers (pdfplumber-assigned headers)
        2. Row contains ≥3 column-header keywords like ITEM, QTY, SKU, etc.
           (sub-headers reprinted at page breaks within merged multi-page tables)
        """
        # Pattern 1: exact match against table headers
        if len(row) == len(headers) and all(
            cell.strip().upper() == header.strip().upper()
            for cell, header in zip(row, headers)
        ):
            return True

        # Pattern 2: column-header keyword detection
        row_text = " ".join(c.strip().upper() for c in row)
        matches = sum(
            1 for kw in LLMExtractor._COLUMN_HEADER_KEYWORDS if kw in row_text
        )
        return matches >= 3

    def _strip_tax_lines(self, text: str) -> tuple[str, int]:
        """
        Remove embedded per-item tax disclosure lines from cell text.

        Returns:
            Tuple of (cleaned_text, number_of_lines_stripped).
        """
        if not text:
            return ("", 0)
        if "\n" not in text:
            # Single-line cell — only strip if the entire cell is a TAX line
            if self._TAX_LINE_RE.match(text):
                return ("", 1)
            return (text, 0)

        lines = text.split("\n")
        kept: list[str] = []
        stripped = 0
        for line in lines:
            if self._TAX_LINE_RE.match(line):
                stripped += 1
            else:
                kept.append(line)

        return ("\n".join(kept), stripped)

    # Keywords indicating a table holds document-level summary data,
    # NOT individual line items. Checked case-insensitively.
    _SUMMARY_KEYWORDS: tuple[str, ...] = (
        "grand total",
        "subtotal",
        "sub total",
        "total contract",
        "amount due",
        "net amount",
        "quote total",
        "total amount",
        "quote #",
        "quote no",
        "quote number",
        "quotation number",
        "customer #",
        "customer no",
        "customer number",
        "bill to",
        "ship to",
        "sold to",
        "payment terms",
        "terms and conditions",
    )

    def _is_summary_table(self, table: TableData) -> bool:
        """
        Detect tables that contain document-level summary/metadata, not line items.

        A table is classified as summary if it has very few rows (≤3) AND its
        headers or cell values contain keywords like 'grand total', 'quote #', etc.
        Real line-item tables typically have many more rows.

        Returns:
            True if the table is a summary/metadata table.
        """
        if len(table.rows) > 3:
            return False

        # Collect all text from headers and rows
        text_parts: list[str] = []
        if table.headers:
            text_parts.extend(table.headers)
        for row in table.rows:
            text_parts.extend(row)

        combined = " ".join(text_parts).lower()
        return any(kw in combined for kw in self._SUMMARY_KEYWORDS)

    def _filter_line_item_tables(
        self, tables: list[TableData]
    ) -> tuple[list[TableData], list[TableData]]:
        """
        Separate line-item tables from summary/metadata tables.

        Summary tables (quote headers, grand totals, terms) are excluded from
        line-item extraction so they don't receive [RNN] tags and get mistakenly
        extracted as line items. They remain available for header extraction.

        Returns:
            Tuple of (line_item_tables, summary_tables).
        """
        line_item_tables: list[TableData] = []
        summary_tables: list[TableData] = []

        for table in tables:
            if self._is_summary_table(table):
                summary_tables.append(table)
                logger.debug(
                    f"Filtered summary table on page {table.page} "
                    f"({len(table.rows)} rows, headers={table.headers})"
                )
            else:
                line_item_tables.append(table)

        if summary_tables:
            summary_rows = sum(len(t.rows) for t in summary_tables)
            logger.info(
                f"Filtered {len(summary_tables)} summary table(s) "
                f"({summary_rows} rows) from line-item extraction"
            )

        return line_item_tables, summary_tables

    def _format_tables(self, tables: list[TableData], compact: bool = False) -> str:
        """
        Format extracted tables as text for the prompt.

        Converts TableData objects to pipe-delimited format with page numbers.
        Each data row gets a sequential [RNN] prefix so that even identical
        rows are visually distinct, preventing LLM deduplication.

        Args:
            tables: List of TableData objects to format
            compact: If True, uses tighter formatting (no separators, no blank
                     lines between tables, compact pipe delimiters)
        """
        if not tables:
            return "No tables detected."

        parts = []
        row_num = 0
        for table in tables:
            if table.headers:
                header_line = " | ".join(table.headers)
                parts.append(f"[Page {table.page}]\n{header_line}")
                if not compact:
                    parts.append("-" * len(header_line))

            for row in table.rows:
                row_num += 1
                parts.append(f"[R{row_num:02d}] " + " | ".join(row))

            if not compact:
                parts.append("")  # Blank line between tables

        return "\n".join(parts)

    def _calculate_confidence(self, quote: VendorQuote) -> float:
        """
        Calculate extraction confidence based on field completeness.

        Confidence formula:
        - 70% weight on required fields (vendor name, grand total, line items)
        - 30% weight on optional fields (quote_id, date, customer, terms)
        """
        required_score = 0.0
        optional_score = 0.0

        # Required fields (70% weight)
        required_checks = [
            quote.vendor is not None and bool(quote.vendor.name),
            quote.amounts is not None and quote.amounts.grand_total > 0,
            len(quote.line_items) > 0,
        ]
        required_score = sum(required_checks) / len(required_checks)

        # Optional fields (30% weight)
        optional_checks = [
            quote.quote_id is not None,
            quote.quote_date is not None,
            quote.customer is not None,
            quote.commercial_terms is not None,
        ]
        optional_score = sum(optional_checks) / len(optional_checks)

        confidence = 0.7 * required_score + 0.3 * optional_score
        return round(confidence, 2)

    def _calculate_line_item_confidence(self, item) -> float:
        """
        Calculate confidence for a single line item based on field completeness.

        Confidence formula:
        - 40% weight: item_type classification present (critical for finance routing)
        - 25% weight: math consistency (quantity * unit_price ≈ extended_price)
        - 20% weight: SKU present (helps procurement matching)
        - 15% weight: manufacturer present (vendor identification)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Classification present (40% weight) - most important for finance
        classification_score = 1.0 if item.item_type is not None else 0.0

        # Math consistency (25% weight) - within configurable tolerance (default $0.01)
        settings = get_settings()
        expected_extended = item.quantity * item.unit_price
        actual_extended = abs(item.extended_price)  # Use abs for credits
        math_diff = abs(expected_extended - actual_extended)
        # Perfect match or within tolerance
        math_score = 1.0 if math_diff <= settings.math_tolerance else 0.0

        # SKU present (20% weight)
        sku_score = 1.0 if item.sku is not None and item.sku.strip() else 0.0

        # Manufacturer present (15% weight)
        manufacturer_score = (
            1.0 if item.manufacturer is not None and item.manufacturer.strip() else 0.0
        )

        confidence = (
            0.40 * classification_score
            + 0.25 * math_score
            + 0.20 * sku_score
            + 0.15 * manufacturer_score
        )
        return round(confidence, 2)

    def _enrich_line_items_confidence(self, quote: VendorQuote) -> VendorQuote:
        """
        Calculate and set confidence for each line item in the quote.

        This replaces the default confidence=1.0 with a calculated value
        based on field completeness for each individual line item.

        Returns:
            Updated VendorQuote with per-item confidence scores
        """
        if not quote.line_items:
            return quote

        updated_items = []
        low_confidence_count = 0
        settings = get_settings()
        threshold = settings.line_item_low_confidence_threshold

        for item in quote.line_items:
            confidence = self._calculate_line_item_confidence(item)
            # Create updated item with calculated confidence
            updated_item = item.model_copy(update={"confidence": confidence})
            updated_items.append(updated_item)

            if confidence < threshold:
                low_confidence_count += 1

        if low_confidence_count > 0:
            logger.info(
                f"{low_confidence_count}/{len(updated_items)} line items "
                f"have confidence < {threshold:.2f} (may need review)"
            )

        return quote.model_copy(update={"line_items": updated_items})

    def _add_placeholder_quote_id(
        self, quote: VendorQuote, pdf_bytes: bytes
    ) -> VendorQuote:
        """
        Generate placeholder quote_id when missing.

        Format: VQE-{sha256(pdf_bytes)[:8]}
        """
        hash_prefix = hashlib.sha256(pdf_bytes).hexdigest()[:8].upper()
        placeholder_id = f"VQE-{hash_prefix}"

        logger.warning(f"Generated placeholder quote_id: {placeholder_id}")

        # Create new quote with placeholder ID
        return quote.model_copy(update={"quote_id": placeholder_id})

    def _link_line_items_to_source_regions(
        self, quote: VendorQuote, tables: list[TableData]
    ) -> VendorQuote:
        """
        Link extracted line items to their source table regions.

        Why this exists: Enables UI highlighting by connecting each line item
        to the table region it was extracted from. When users hover over a line
        item in the UI, we can highlight the corresponding PDF region.

        Algorithm:
        1. Build a map of table content (descriptions, SKUs) to region IDs
        2. For each line item, find the best matching table
        3. Assign source_region_id to link the item to its source

        Args:
            quote: Extracted VendorQuote with line items
            tables: Original TableData objects with region info

        Returns:
            Updated VendorQuote with source_region_id assigned to line items
        """
        if not tables or not quote.line_items:
            return quote

        # Build lookup: normalize text -> (region_id, table_idx)
        # We match on first cell (usually description) of each row
        table_lookup: dict[str, str] = {}
        for idx, table in enumerate(tables):
            region_id = f"table_{idx + 1}_p{table.page}"
            for row in table.rows:
                if row and row[0]:
                    # Normalize: lowercase, strip whitespace, remove common noise
                    normalized = self._normalize_for_matching(row[0])
                    if normalized:
                        table_lookup[normalized] = region_id
            # Also index headers if they might contain descriptions
            if table.headers:
                for header in table.headers:
                    if header:
                        normalized = self._normalize_for_matching(header)
                        if normalized:
                            table_lookup[normalized] = region_id

        # Match line items to tables
        updated_items = []
        matched_count = 0

        for item in quote.line_items:
            # Try to find matching table by description
            normalized_desc = self._normalize_for_matching(item.description)
            matched_region = None

            # Exact match first
            if normalized_desc in table_lookup:
                matched_region = table_lookup[normalized_desc]
            else:
                # Fuzzy match: check if desc contains/is contained in table text
                for table_text, region_id in table_lookup.items():
                    if table_text in normalized_desc or normalized_desc in table_text:
                        matched_region = region_id
                        break

            if matched_region:
                updated_item = item.model_copy(
                    update={"source_region_id": matched_region}
                )
                matched_count += 1
            else:
                updated_item = item

            updated_items.append(updated_item)

        if matched_count > 0:
            logger.info(
                f"Linked {matched_count}/{len(quote.line_items)} line items "
                "to source regions"
            )

        return quote.model_copy(update={"line_items": updated_items})

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for fuzzy matching between table cells and line items.

        Handles common variations:
        - Case differences
        - Extra whitespace
        - Common punctuation

        Args:
            text: Text to normalize

        Returns:
            Normalized text for comparison
        """
        if not text:
            return ""
        # Lowercase, collapse whitespace, remove some punctuation
        normalized = text.lower().strip()
        # Collapse multiple spaces
        normalized = " ".join(normalized.split())
        return normalized

    def _get_provider_text_limit(self, provider_name: str) -> int:
        """
        Get provider-specific text truncation limit.

        H6: Different providers have different context window sizes.
        Use provider-specific limits to maximize context utilization
        while avoiding truncation for providers with large windows.

        Args:
            provider_name: Name of the provider (groq, openai, gemini, etc.)

        Returns:
            Maximum characters for text content
        """
        provider_limits = self.settings.provider_max_text_chars
        return provider_limits.get(provider_name, self.settings.max_text_chars)

    def _get_provider_table_limit(self, provider_name: str) -> int:
        """
        Get provider-specific table truncation limit.

        H6: Different providers have different context window sizes.
        Use provider-specific limits to maximize table data extraction.

        Args:
            provider_name: Name of the provider (groq, openai, gemini, etc.)

        Returns:
            Maximum characters for table content
        """
        provider_limits = self.settings.provider_max_table_chars
        return provider_limits.get(provider_name, self.settings.max_table_chars)

    # Headers that can be safely removed to reduce table size
    _NON_ESSENTIAL_HEADERS = {
        "notes",
        "note",
        "comments",
        "comment",
        "weight",
        "color",
        "warranty",
        "warranty info",
        "warranty terms",
        "lead time",
        "delivery",
        "delivery date",
        "availability",
        "stock",
        "image",
        "photo",
        "dimensions",
        "size",
    }

    def _prune_non_financial_columns(self, tables: list[TableData]) -> list[TableData]:
        """
        Remove non-essential columns from tables to reduce character count.

        Only removes columns whose headers match known non-essential names
        (case-insensitive exact match). Preserves all financial and
        identification columns.

        Args:
            tables: List of TableData objects to prune

        Returns:
            New list of TableData with non-essential columns removed
        """
        pruned_tables = []
        for table in tables:
            if not table.headers:
                pruned_tables.append(table)
                continue

            # Find indices of columns to keep
            keep_indices = []
            removed_headers = []
            for i, header in enumerate(table.headers):
                if header.lower().strip() in self._NON_ESSENTIAL_HEADERS:
                    removed_headers.append(header)
                else:
                    keep_indices.append(i)

            if not removed_headers or not keep_indices:
                # Nothing to prune, or pruning would remove all columns
                pruned_tables.append(table)
                continue

            logger.info(
                f"Pruned {len(removed_headers)} non-essential "
                f"columns from page {table.page}: "
                f"{removed_headers}"
            )

            pruned_headers = [table.headers[i] for i in keep_indices]
            pruned_rows = [
                [row[i] for i in keep_indices if i < len(row)] for row in table.rows
            ]
            pruned_tables.append(
                TableData(
                    page=table.page,
                    rows=pruned_rows,
                    headers=pruned_headers,
                    bbox=table.bbox,
                )
            )

        return pruned_tables

    def _truncate_tables_row_aware(
        self, formatted_tables: str, max_chars: int
    ) -> tuple[str, int, int]:
        """
        Truncate formatted table text at row boundaries instead of mid-row.

        Preserves complete header blocks (header + separator) per table and
        only truncates data rows. Tracks character budget across multiple
        tables separated by [Page N] markers.

        The formatted table structure per table is:
            [Page N]          <- page marker
            Header | Cols     <- header line (follows page marker)
            ----------        <- separator
            Data | Row        <- data rows
            ...
            (blank line)      <- between tables

        Args:
            formatted_tables: Pipe-delimited table text from _format_tables()
            max_chars: Maximum characters allowed

        Returns:
            Tuple of (truncated_text, rows_kept, rows_total)
        """
        lines = formatted_tables.split("\n")

        # Identify which lines are data rows vs structural lines.
        # A line is structural if it's a page marker, separator, blank,
        # or the header line immediately following a page marker.
        is_data_row = []
        after_page_marker = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[Page"):
                is_data_row.append(False)
                after_page_marker = True
            elif after_page_marker and stripped:
                # Header line following [Page N]
                is_data_row.append(False)
                after_page_marker = False
            elif re.fullmatch(r"[-|+ ]+", stripped) or stripped == "":
                # Separator line (all dashes/pipes) or blank line
                is_data_row.append(False)
                after_page_marker = False
            else:
                is_data_row.append(True)
                after_page_marker = False

        rows_total = sum(is_data_row)

        if len(formatted_tables) <= max_chars:
            return formatted_tables, rows_total, rows_total

        # Group lines into atomic blocks: [Page] + header + separator form one
        # block that must be included or excluded together to avoid orphaned
        # page markers confusing the LLM.
        blocks: list[tuple[list[int], bool]] = []  # (line indices, is_data)
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("[Page"):
                # Collect page marker + header + optional separator as one block
                block_indices = [i]
                i += 1
                # Header line follows page marker
                if i < len(lines) and not is_data_row[i]:
                    block_indices.append(i)
                    i += 1
                # Optional separator line
                if i < len(lines) and not is_data_row[i]:
                    s = lines[i].strip()
                    if re.fullmatch(r"[-|+ ]+", s) or s == "":
                        block_indices.append(i)
                        i += 1
                blocks.append((block_indices, False))
            else:
                blocks.append(([i], is_data_row[i]))
                i += 1

        kept_lines: list[str] = []
        budget = max_chars
        rows_kept = 0

        for block_indices, is_data in blocks:
            block_cost = sum(len(lines[j]) + 1 for j in block_indices)
            if block_cost > budget:
                break
            for j in block_indices:
                kept_lines.append(lines[j])
            budget -= block_cost
            if is_data:
                rows_kept += 1

        result = "\n".join(kept_lines)
        return result, rows_kept, rows_total

    async def _get_few_shot_context(self) -> str:
        """
        Retrieve and format few-shot examples from user corrections.

        Queries the database for correction examples on commonly-misextracted
        fields and formats them into a prompt section that guides the LLM
        to avoid similar mistakes.

        Target fields (based on common extraction errors):
        - amounts.grand_total: TCV vs net confusion in service-order quotes
        - line_items.item_type: Hardware vs software vs services classification
        - line_items.extended_price: Credit/discount sign handling

        Returns:
            Formatted few-shot prompt text, or empty string if no examples.
        """
        # Phase A4: Skip few-shot lookup if disabled (saves ~0.3s DB queries)
        if not self.settings.enable_few_shot:
            return ""

        # Fields commonly needing correction (prioritized by impact)
        target_fields = [
            "amounts.grand_total",
            "line_items.item_type",
            "line_items.extended_price",
            "line_items.description",
        ]

        all_examples = []

        try:
            # Get session from factory for database queries
            factory = get_session_factory()
            async with factory() as session:
                for field in target_fields:
                    examples = await get_examples_for_field(
                        session=session,
                        field_path=field,
                        limit=2,  # Max 2 examples per field to keep prompt concise
                    )
                    all_examples.extend(examples)

            if not all_examples:
                return ""

            # Format examples into prompt text
            few_shot_text = format_few_shot_prompt(all_examples)

            if few_shot_text:
                logger.info(
                    f"Included {len(all_examples)} few-shot examples from corrections"
                )

            return few_shot_text

        except Exception as e:
            # Don't fail extraction if few-shot lookup fails
            logger.warning(f"Few-shot example retrieval failed: {e}")
            return ""
