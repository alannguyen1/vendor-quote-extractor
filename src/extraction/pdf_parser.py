"""
PDF parsing module for extracting text and tables from vendor quotes.

Uses PyMuPDF (fitz) for text extraction and pdfplumber for table detection.
Handles both digital and scanned PDFs with OCR-ready image extraction.

Speed Phase 5 optimizations:
- Configurable DPI via settings (lower DPI = faster scanned PDF processing)
- Parallel text/table extraction using ThreadPoolExecutor
"""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pdfplumber

from config import get_settings
from src.schemas.models import SourceRegion

logger = logging.getLogger("vendor_quote_extractor.pdf_parser")


@dataclass
class TableData:
    """
    Represents a table extracted from a PDF page.

    Attributes:
        page: 1-indexed page number where table was found
        rows: List of rows, each row is a list of cell values
        headers: Optional first row as headers (if detected)
        bbox: Bounding box coordinates (x0, y0, x1, y1)
    """

    page: int
    rows: list[list[str]]
    headers: list[str] | None = None
    bbox: tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )


@dataclass
class ParseTimingBreakdown:
    """Timing breakdown for PDF parsing sub-steps."""

    text_extract_ms: int
    table_extract_ms: int
    image_render_ms: int | None = None


@dataclass
class ParseResult:
    """
    Result of parsing a PDF document.

    Attributes:
        text: Concatenated text from all pages with page markers
        tables: List of detected tables with their data
        page_count: Total number of pages in the document
        is_scanned: True if document appears to be a scanned image
        images: Optional list of page images (PNG bytes) for scanned PDFs
        error: Error message if parsing failed
        timing: Timing breakdown for parsing sub-steps
        source_regions: List of SourceRegion objects for UI highlighting
    """

    text: str
    tables: list[TableData]
    page_count: int
    is_scanned: bool
    images: list[bytes] | None = None
    error: str | None = None
    timing: ParseTimingBreakdown | None = None
    source_regions: list[SourceRegion] = field(default_factory=list)


class PDFParser:
    """
    Extracts text and tables from PDF documents.

    Handles both digital PDFs (direct text extraction) and scanned PDFs
    (renders pages to images for vision API processing).

    Speed Phase 5 optimizations:
    - Configurable DPI via settings (default 300, lower for speed)
    - Parallel text/table extraction when enabled in settings
    """

    # Threshold for detecting scanned PDFs: pages with fewer chars are likely scans
    SCANNED_THRESHOLD = 50  # chars per page

    def __init__(self) -> None:
        """Initialize parser with settings from config."""
        self._settings = get_settings()

    @property
    def render_dpi(self) -> int:
        """Get render DPI from settings (default 300)."""
        return self._settings.render_dpi

    def extract(self, pdf_bytes: bytes) -> ParseResult:
        """
        Extract text and tables from a PDF document.

        Args:
            pdf_bytes: Raw PDF file content

        Returns:
            ParseResult with extracted text, tables, and metadata
        """
        try:
            # Extract text and tables (parallel or sequential based on settings)
            if self._settings.parallel_parsing:
                # Speed Phase 5: Run text and table extraction in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    text_start = time.time()
                    text_future = executor.submit(self._extract_text, pdf_bytes)
                    table_future = executor.submit(self._extract_tables, pdf_bytes)

                    # Wait for both to complete
                    text, page_count, is_scanned = text_future.result()
                    text_extract_ms = int((time.time() - text_start) * 1000)

                    tables = table_future.result()
                    # For parallel execution, measure total time as table time
                    table_extract_ms = text_extract_ms  # Both ran concurrently
            else:
                # Sequential extraction (original behavior)
                text_start = time.time()
                text, page_count, is_scanned = self._extract_text(pdf_bytes)
                text_extract_ms = int((time.time() - text_start) * 1000)

                table_start = time.time()
                tables = self._extract_tables(pdf_bytes)
                table_extract_ms = int((time.time() - table_start) * 1000)

            # If scanned, render pages to images for vision API
            images = None
            image_render_ms = None
            if is_scanned:
                logger.info("Scanned PDF detected, rendering pages to images")
                render_start = time.time()
                images = self._render_pages_to_images(pdf_bytes)
                image_render_ms = int((time.time() - render_start) * 1000)

            # Build timing breakdown
            timing = ParseTimingBreakdown(
                text_extract_ms=text_extract_ms,
                table_extract_ms=table_extract_ms,
                image_render_ms=image_render_ms,
            )

            # Check for empty PDF
            if not text.strip() and not tables:
                return ParseResult(
                    text="",
                    tables=[],
                    page_count=page_count,
                    is_scanned=is_scanned,
                    images=images,
                    error="No extractable content found in PDF",
                    timing=timing,
                    source_regions=[],
                )

            # Generate source regions from tables for UI highlighting
            source_regions = self._generate_source_regions(tables)

            return ParseResult(
                text=text,
                tables=tables,
                page_count=page_count,
                is_scanned=is_scanned,
                images=images,
                error=None,
                timing=timing,
                source_regions=source_regions,
            )

        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return ParseResult(
                text="",
                tables=[],
                page_count=0,
                is_scanned=False,
                images=None,
                error=f"PDF parsing failed: {str(e)}",
                source_regions=[],
            )

    def _extract_text(self, pdf_bytes: bytes) -> tuple[str, int, bool]:
        """
        Extract text from all pages using PyMuPDF.

        Returns:
            Tuple of (text, page_count, is_scanned)
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        text_parts = []
        is_scanned = False

        for page_num in range(page_count):
            page = doc[page_num]
            # get_text() returns str by default (pyright stubs may not reflect this)
            page_text: str = page.get_text()  # type: ignore[assignment]
            text_parts.append(f"--- PAGE {page_num + 1} ---\n{page_text}")

            # Check if this page looks scanned (low text content)
            if len(page_text.strip()) < self.SCANNED_THRESHOLD:
                is_scanned = True

        doc.close()

        full_text = "\n\n".join(text_parts)
        return full_text, page_count, is_scanned

    def _extract_tables(self, pdf_bytes: bytes) -> list[TableData]:
        """
        Extract tables from PDF using pdfplumber.

        Uses find_tables() to get actual table bounding boxes instead of
        full page dimensions. This enables future source region highlighting.

        Returns:
            List of TableData objects for each detected table
        """
        raw_tables = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Use find_tables() to get table objects with bounding boxes
                table_objects = page.find_tables()
                page_tables = page.extract_tables()

                for table_idx, table in enumerate(page_tables):
                    if not table or len(table) < 2:
                        continue  # Skip empty or single-row tables

                    # Clean up cell values
                    cleaned_rows = []
                    for row in table:
                        cleaned_row = [
                            str(cell).strip() if cell else "" for cell in row
                        ]
                        cleaned_rows.append(cleaned_row)

                    # First row is typically headers
                    headers = cleaned_rows[0] if cleaned_rows else None
                    data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

                    # Get actual bounding box from table object if available
                    # bbox format: (x0, top, x1, bottom) in PDF coordinates
                    if table_idx < len(table_objects):
                        table_obj = table_objects[table_idx]
                        bbox = table_obj.bbox
                    else:
                        # Fallback to full page if table object not found
                        bbox = (0.0, 0.0, page.width or 0.0, page.height or 0.0)
                        logger.warning(
                            f"Table {table_idx + 1} on page {page_num}: "
                            "bbox not found, using full page dimensions"
                        )

                    raw_tables.append(
                        TableData(
                            page=page_num,
                            rows=data_rows,
                            headers=headers,
                            bbox=bbox,
                        )
                    )

                    cols = len(headers) if headers else 0
                    logger.debug(
                        f"Extracted table {table_idx + 1} from page {page_num}: "
                        f"{len(data_rows)} rows, {cols} cols, bbox={bbox}"
                    )

        # Merge tables that span multiple pages
        merged_tables = self._merge_multi_page_tables(raw_tables)
        return merged_tables

    def _merge_multi_page_tables(self, tables: list[TableData]) -> list[TableData]:
        """
        Merge tables that span multiple consecutive pages.

        Detects continuation tables by:
        - Same column count as previous table
        - Table on consecutive page
        - First row looks like data, not headers (numeric/currency values)

        Returns:
            List of merged TableData objects
        """
        if len(tables) <= 1:
            return tables

        # Sort by page to ensure order
        sorted_tables = sorted(tables, key=lambda t: t.page)
        merged: list[TableData] = []

        i = 0
        while i < len(sorted_tables):
            current = sorted_tables[i]

            # Look for continuation tables on subsequent pages
            while i + 1 < len(sorted_tables):
                next_table = sorted_tables[i + 1]

                if self._is_continuation_table(current, next_table):
                    # Merge rows from continuation into current
                    current = self._merge_two_tables(current, next_table)
                    logger.info(
                        f"Merged continuation table from page {next_table.page} "
                        f"into table from page {current.page}"
                    )
                    i += 1
                else:
                    break

            merged.append(current)
            i += 1

        logger.debug(
            f"Table merging: {len(tables)} raw tables -> {len(merged)} merged tables"
        )
        return merged

    def _is_continuation_table(self, primary: TableData, candidate: TableData) -> bool:
        """
        Determine if candidate table is a continuation of primary table.

        Criteria:
        - Consecutive or nearby pages (within 2 pages)
        - Same column count
        - Candidate's header row looks like data (not actual headers)
        """
        # Check page proximity (consecutive or within 2 pages)
        if candidate.page - primary.page > 2:
            return False

        # Check column count matches
        primary_cols = len(primary.headers) if primary.headers else 0
        candidate_cols = len(candidate.headers) if candidate.headers else 0

        if primary_cols == 0 or candidate_cols == 0:
            return False

        if primary_cols != candidate_cols:
            return False

        # Check if candidate's "headers" look like data (continuation indicator)
        if candidate.headers:
            if self._looks_like_data_row(candidate.headers):
                return True

            # Also check if headers match primary (repeated headers on new page)
            if primary.headers and candidate.headers == primary.headers:
                return True

        return False

    def _looks_like_data_row(self, row: list[str]) -> bool:
        """
        Check if a row looks like data rather than headers.

        Data indicators:
        - Contains numeric values
        - Contains currency symbols ($)
        - Contains date-like patterns
        - Cells are mostly short (< 20 chars)
        """
        numeric_count = 0
        currency_count = 0

        for cell in row:
            if not cell:
                continue

            # Check for numeric values
            cleaned = cell.replace(",", "").replace(".", "").replace("-", "")
            if cleaned.isdigit():
                numeric_count += 1

            # Check for currency
            if "$" in cell:
                currency_count += 1

        # If more than 30% of cells are numeric or currency, likely data
        total_cells = len([c for c in row if c])
        if total_cells > 0:
            data_ratio = (numeric_count + currency_count) / total_cells
            if data_ratio >= 0.3:
                return True

        return False

    def _merge_two_tables(
        self, primary: TableData, continuation: TableData
    ) -> TableData:
        """
        Merge continuation table rows into primary table.

        If continuation has same headers as primary, skip them.
        Otherwise, treat continuation's "headers" as data.
        """
        merged_rows = list(primary.rows)

        # Check if continuation's headers are actual headers (same as primary)
        if continuation.headers and primary.headers:
            if continuation.headers != primary.headers:
                # Headers look like data - include them
                merged_rows.append(continuation.headers)

        # Add all continuation data rows
        merged_rows.extend(continuation.rows)

        return TableData(
            page=primary.page,  # Keep original page number
            rows=merged_rows,
            headers=primary.headers,
            bbox=primary.bbox,
        )

    def _render_pages_to_images(self, pdf_bytes: bytes) -> list[bytes]:
        """
        Render PDF pages to PNG images for vision API processing.

        Used when PDF is detected as scanned (no extractable text).

        Returns:
            List of PNG image bytes, one per page
        """
        images = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        # Calculate zoom factor for target DPI (72 is default PDF DPI)
        zoom = self.render_dpi / 72

        for page_num in range(page_count):
            page = doc[page_num]
            # Render page to pixmap at target DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")
            images.append(png_bytes)

            logger.debug(
                f"Rendered page {page_num + 1} to image: {pix.width}x{pix.height} px"
            )

        doc.close()
        return images

    def _generate_source_regions(self, tables: list[TableData]) -> list[SourceRegion]:
        """
        Generate SourceRegion objects from extracted tables.

        Each table becomes a source region that can be highlighted in the UI
        when users interact with line items extracted from that table.

        Why this exists: Enables visual verification by linking extracted
        line items back to their source location in the PDF.

        Args:
            tables: List of TableData objects from table extraction

        Returns:
            List of SourceRegion objects for UI highlighting
        """
        regions = []
        for idx, table in enumerate(tables):
            # Generate unique ID: table_<index>_p<page>
            region_id = f"table_{idx + 1}_p{table.page}"

            # Create summary of table content for debugging
            row_count = len(table.rows)
            col_count = len(table.headers) if table.headers else 0
            content_summary = f"Table with {row_count} rows, {col_count} columns"

            region = SourceRegion(
                region_id=region_id,
                page=table.page,
                bbox=list(table.bbox),  # Convert tuple to list for JSON schema compat
                region_type="table",
                content_summary=content_summary,
            )
            regions.append(region)
            logger.debug(f"Created source region: {region_id} on page {table.page}")

        return regions
