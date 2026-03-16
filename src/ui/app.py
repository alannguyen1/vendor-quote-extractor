"""
Streamlit UI for Vendor Quote Extractor.

Stateless demo deployment - calls extraction pipeline directly.
No API server required, no database persistence.

Provides:
- PDF upload with drag & drop
- Processing progress display
- Split-panel results view (PDF preview + extracted data)
- Review interface for low-confidence fields
- JSON export
"""

import os
import sys

# Ensure repo root is on sys.path for Streamlit Cloud deployment
# This allows imports like `from config.settings import ...` to work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import hashlib
import io
import json
import time
from typing import Any

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw

from config.settings import get_settings
from src.extraction.pipeline import ExtractionPipeline

# Load settings once
_settings = get_settings()

# Demo constraints - use settings values to avoid mismatch
MAX_FILE_SIZE_MB = _settings.max_file_size_mb
MAX_PDF_PAGES = _settings.max_pdf_pages
CONFIDENCE_THRESHOLD = _settings.confidence_threshold

# Page configuration
st.set_page_config(
    page_title="Vendor Quote Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_pipeline() -> ExtractionPipeline:
    """
    Create a fresh pipeline instance for each extraction.

    NOTE: We intentionally do NOT cache this with @st.cache_resource because:
    - asyncio.run() creates a new event loop per call
    - ExtractionPipeline uses a module-level semaphore bound to the first loop
    - Reusing a cached pipeline with a new loop causes
      "Future attached to different loop" errors
    """
    # Reset the module-level semaphore before creating a new pipeline
    # This ensures the semaphore is created fresh in the current event loop
    from src.extraction.pipeline import reset_extraction_semaphore

    reset_extraction_semaphore()
    return ExtractionPipeline()


def render_pdf_page(pdf_bytes: bytes, page_number: int = 0) -> bytes:
    """Render a PDF page to PNG image for display.

    Returns empty bytes if PDF is empty or corrupted.
    """
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return b""
        if page_number >= len(doc):
            page_number = len(doc) - 1
        page = doc[page_number]
        # Render at 150 DPI for display
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    except Exception:
        return b""
    finally:
        if doc is not None:
            doc.close()


def render_pdf_page_with_highlight(
    pdf_bytes: bytes,
    page_number: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """
    Render a PDF page with optional bounding box highlight overlay.

    Args:
        pdf_bytes: Raw PDF content
        page_number: 0-indexed page number
        bbox: Optional (x0, y0, x1, y1) in PDF coordinates (72 DPI)

    Returns:
        PNG image bytes with highlight overlay if bbox provided.
        Returns empty bytes if PDF is empty or corrupted.
    """
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return b""
        if page_number >= len(doc):
            page_number = len(doc) - 1
        page = doc[page_number]

        # Render at 150 DPI for display
        render_dpi = 150
        scale = render_dpi / 72
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        png_bytes_rendered = pix.tobytes("png")
    except Exception:
        return b""
    finally:
        if doc is not None:
            doc.close()

    if bbox is None:
        return png_bytes_rendered

    # Draw highlight overlay using PIL
    img = Image.open(io.BytesIO(png_bytes_rendered))
    draw = ImageDraw.Draw(img, "RGBA")

    # Scale bbox from PDF coordinates (72 DPI) to render coordinates (150 DPI)
    x0, y0, x1, y1 = bbox
    scaled_bbox = (
        int(x0 * scale),
        int(y0 * scale),
        int(x1 * scale),
        int(y1 * scale),
    )

    # Draw semi-transparent yellow highlight rectangle
    highlight_color = (255, 255, 0, 80)  # Yellow with 30% opacity
    draw.rectangle(scaled_bbox, fill=highlight_color)

    # Draw border for visibility
    border_color = (255, 165, 0, 200)  # Orange border
    draw.rectangle(scaled_bbox, outline=border_color, width=3)

    # Save back to PNG bytes
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Get total number of pages in a PDF.

    Returns 0 if the PDF is corrupted or cannot be opened.
    """
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return len(doc)
    except Exception:
        # Corrupted or invalid PDF
        return 0
    finally:
        if doc is not None:
            doc.close()


def _apply_highlight_overlay(
    png_bytes: bytes,
    bbox: tuple[float, float, float, float],
    render_dpi: int = 150,
) -> bytes:
    """Apply a highlight overlay to a rendered PNG image.

    Args:
        png_bytes: Raw PNG image bytes.
        bbox: Bounding box (x0, y0, x1, y1) in PDF coordinates (72 DPI).
        render_dpi: The DPI at which the page was rendered.

    Returns:
        PNG image bytes with highlight overlay applied.
    """
    img = Image.open(io.BytesIO(png_bytes))
    draw = ImageDraw.Draw(img, "RGBA")

    # Scale bbox from PDF coordinates (72 DPI) to render coordinates
    scale = render_dpi / 72
    x0, y0, x1, y1 = bbox

    # Clamp coordinates to image dimensions
    img_width, img_height = img.size
    scaled_bbox = (
        max(0, min(int(x0 * scale), img_width)),
        max(0, min(int(y0 * scale), img_height)),
        max(0, min(int(x1 * scale), img_width)),
        max(0, min(int(y1 * scale), img_height)),
    )

    # Draw semi-transparent yellow highlight rectangle
    highlight_color = (255, 255, 0, 80)  # Yellow with 30% opacity
    draw.rectangle(scaled_bbox, fill=highlight_color)

    # Draw border for visibility
    border_color = (255, 165, 0, 200)  # Orange border
    draw.rectangle(scaled_bbox, outline=border_color, width=3)

    # Save back to PNG bytes
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def render_all_pdf_pages(
    pdf_bytes: bytes,
    render_dpi: int = 150,
) -> list[tuple[int, bytes]]:
    """Render all PDF pages at once, returning list of (page_idx, png_bytes).

    Opens PDF once and iterates pages for efficiency.

    Args:
        pdf_bytes: Raw PDF content.
        render_dpi: Resolution for rendering (default 150 DPI).

    Returns:
        List of tuples (page_index, png_bytes) for each page.
    """
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(doc) == 0:
            return []

        mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
        pages = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat)
            png_bytes_page = pix.tobytes("png")
            pages.append((page_idx, png_bytes_page))

        return pages
    except Exception:
        return []
    finally:
        if doc is not None:
            doc.close()


@st.cache_data
def get_cached_pdf_pages(pdf_hash: str, pdf_bytes: bytes) -> list[tuple[int, bytes]]:
    """Cache rendered pages by PDF hash.

    Args:
        pdf_hash: MD5 hash of the PDF bytes (used as cache key).
        pdf_bytes: Raw PDF content.

    Returns:
        List of tuples (page_index, png_bytes) for each page.
    """
    return render_all_pdf_pages(pdf_bytes)


def extract_document(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Process PDF through extraction pipeline directly.

    Returns the extracted quote as a dict.
    Raises Exception on failure.
    """
    pipeline = get_pipeline()

    # Run async pipeline in sync context
    result = asyncio.run(pipeline.process(pdf_bytes, filename))

    if result.error:
        raise Exception(result.error)

    if result.quote is None:
        raise Exception("Extraction returned no data")

    # Convert Pydantic model to dict for session state
    return result.quote.model_dump()


def apply_correction_in_memory(
    quote: dict[str, Any], field_path: str, value: Any
) -> dict[str, Any]:
    """
    Apply a correction to the quote in memory.

    Args:
        quote: The extraction result dict
        field_path: Dot-notation path (e.g., "line_items[0].unit_price")
        value: The new value to set

    Returns:
        Updated quote dict
    """
    # Parse field path like "line_items[0].unit_price" or "amounts.grand_total"
    parts = field_path.replace("[", ".").replace("]", "").split(".")

    # Navigate to parent and set value
    obj: Any = quote
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = obj[part]

    # Set the final value
    final_key = parts[-1]
    if final_key.isdigit():
        obj[int(final_key)] = value
    else:
        obj[final_key] = value

    return quote


def format_currency(value: float) -> str:
    """Format a number as USD currency."""
    return f"${value:,.2f}"


def get_confidence_badge(confidence: float) -> str:
    """Return a styled confidence badge."""
    if confidence >= CONFIDENCE_THRESHOLD:
        return f"🟢 {confidence:.1%}"
    return f"🟡 {confidence:.1%}"


def render_financial_summary(quote: dict, metadata: dict):
    """
    Render financial reconciliation summary — always shown.

    Displays:
    - Sum of line items vs subtotal
    - Calculated total vs grand total
    - Difference amounts with color indicators (when mismatched)
    """
    amounts = quote.get("amounts", {})
    line_items = quote.get("line_items", [])

    # Calculate financial breakdown
    line_items_sum = sum(item.get("extended_price", 0) for item in line_items)
    subtotal = amounts.get("subtotal", 0)
    tax = amounts.get("tax", 0)
    shipping = amounts.get("shipping", 0)
    discounts = amounts.get("discounts", 0)
    grand_total = amounts.get("grand_total", 0)

    # Calculated expected grand total
    calculated_total = subtotal + tax + shipping - discounts

    # Differences
    line_sum_diff = abs(line_items_sum - subtotal)
    total_diff = abs(calculated_total - grand_total)

    # Determine which checks pass/fail
    has_line_sum_error = line_sum_diff > 0.01
    has_total_error = total_diff > 0.01

    # Render structured breakdown
    st.markdown("#### Financial Breakdown")

    # Line Items Section
    st.markdown("**Line Items Calculation:**")
    status_icon = "✅" if not has_line_sum_error else "❌"
    st.markdown(f"Sum of Line Items: **{format_currency(line_items_sum)}**")
    st.markdown(f"Expected Subtotal: **{format_currency(subtotal)}** {status_icon}")
    if has_line_sum_error:
        st.markdown(
            f"<span style='color: #ff4b4b;'>Difference: "
            f"**{format_currency(line_sum_diff)}**</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Grand Total Section
    st.markdown("**Grand Total Calculation:**")
    st.markdown(f"Subtotal: {format_currency(subtotal)}")
    st.markdown(f"+ Tax: {format_currency(tax)}")
    st.markdown(f"+ Shipping: {format_currency(shipping)}")
    st.markdown(f"- Discounts: {format_currency(discounts)}")
    st.markdown("---")
    status_icon = "✅" if not has_total_error else "❌"
    st.markdown(f"Calculated Total: **{format_currency(calculated_total)}**")
    st.markdown(
        f"Extracted Grand Total: **{format_currency(grand_total)}** {status_icon}"
    )
    if has_total_error:
        st.markdown(
            f"<span style='color: #ff4b4b;'>Difference: "
            f"**{format_currency(total_diff)}**</span>",
            unsafe_allow_html=True,
        )


def render_validation_error_details(metadata: dict):
    """
    Render error-specific details — only shown when validation fails.

    Displays possible causes and raw validation errors.
    """
    validation_errors = metadata.get("validation_errors", [])
    if not validation_errors:
        return

    st.markdown("**Possible Causes:**")
    causes = []

    is_scanned = metadata.get("is_scanned", False)
    error_text = " ".join(validation_errors).lower()

    if any(kw in error_text for kw in ("subtotal", "line item", "sum")):
        causes.append("- Missing line items in extraction")
        causes.append("- Duplicate line items detected")
        if is_scanned:
            causes.append("- OCR error on line item prices")

    if any(kw in error_text for kw in ("grand total", "total")):
        causes.append("- LLM misread grand total field")
        causes.append("- Tax/shipping not extracted correctly")
        if "credit" in error_text:
            causes.append("- Credit/discount calculation issue")

    if "quantity" in error_text or "unit_price" in error_text:
        causes.append("- Line item math error (qty x price)")

    if is_scanned:
        causes.append("- Scanned PDF quality issue")

    if not causes:
        causes.append("- Extraction accuracy issue")
        causes.append("- Document format variation")

    for cause in causes[:5]:
        st.caption(cause)

    with st.expander("Raw Validation Errors", expanded=False):
        for error in validation_errors:
            st.text(f"- {error}")


def render_upload_zone():
    """Render the file upload zone with demo constraints."""
    st.markdown("### Upload Vendor Quote")
    st.markdown("Drag & drop a PDF file or click to browse")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB, max {MAX_PDF_PAGES} pages",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()

        # Validate file size
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
            return None

        # Validate page count (demo constraint)
        page_count = get_pdf_page_count(pdf_bytes)
        if page_count == 0:
            st.error("Invalid PDF: File appears to be empty or corrupted.")
            return None
        if page_count > MAX_PDF_PAGES:
            st.error(
                f"PDF has {page_count} pages. "
                f"Demo limit is {MAX_PDF_PAGES} pages to prevent timeouts."
            )
            return None

        st.success(
            f"Uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB, {page_count} pages)"
        )
        return uploaded_file

    return None


def render_entity(entity: dict, title: str):
    """Render an entity (vendor/customer) section."""
    with st.expander(title, expanded=True):
        st.write(f"**Name:** {entity.get('name', 'N/A')}")
        if entity.get("address"):
            st.write(f"**Address:** {entity['address']}")
        if entity.get("phone"):
            st.write(f"**Phone:** {entity['phone']}")
        if entity.get("email"):
            st.write(f"**Email:** {entity['email']}")


def render_line_items(
    line_items: list[dict],
    requires_review: bool = False,
    source_regions: list[dict] | None = None,
):
    """Render the line items table with source highlighting support."""
    # Build region lookup by ID for quick access
    region_map: dict[str, dict] = {}
    if source_regions:
        for region in source_regions:
            region_map[region.get("region_id", "")] = region

    with st.expander(f"Line Items ({len(line_items)} items)", expanded=True):
        # Create a dataframe-like view
        for item_idx, item in enumerate(line_items):
            # Add extra column for source button if regions available
            if source_regions:
                cols = st.columns([1, 3, 1, 2, 2, 2, 1])
            else:
                cols = st.columns([1, 4, 1, 2, 2, 2])

            # Highlight low confidence items
            confidence = item.get("confidence", 1.0)
            row_style = "" if confidence >= CONFIDENCE_THRESHOLD else "⚠️ "

            with cols[0]:
                st.write(f"{row_style}{item.get('line_number', '-')}")
            with cols[1]:
                desc = item.get("description", "N/A")
                item_type = item.get("item_type", "")
                if item_type:
                    st.write(f"{desc} [{item_type}]")
                else:
                    st.write(f"{desc} [unclassified]")
            with cols[2]:
                st.write(str(item.get("quantity", 1)))
            with cols[3]:
                st.write(format_currency(item.get("unit_price", 0)))
            with cols[4]:
                st.write(format_currency(item.get("extended_price", 0)))
            with cols[5]:
                st.write(get_confidence_badge(confidence))

            # Source highlight button
            if source_regions and len(cols) > 6:
                with cols[6]:
                    source_region_id = item.get("source_region_id")
                    # Only show button if item has a valid region ID (not empty string)
                    region = (
                        region_map.get(source_region_id) if source_region_id else None
                    )
                    if region:
                        # Check if this item is currently highlighted
                        current_highlight = st.session_state.get("highlighted_region")
                        region_bbox = region.get("bbox")
                        is_highlighted = (
                            current_highlight is not None
                            and region_bbox is not None
                            and current_highlight[0] == region.get("page", 1) - 1
                            and current_highlight[1] == tuple(region_bbox)
                        )
                        btn_label = "🎯" if is_highlighted else "📍"
                        # item_idx from enumerate is used for stable widget key
                        if st.button(
                            btn_label,
                            key=f"src_btn_{item_idx}",
                            help="Show source in PDF",
                        ):
                            if is_highlighted:
                                # Toggle off
                                st.session_state.highlighted_region = None
                            elif region_bbox is not None:
                                # Set highlight (0-indexed page)
                                page_idx = region.get("page", 1) - 1
                                st.session_state.highlighted_region = (
                                    page_idx,
                                    tuple(region_bbox),
                                )
                            st.rerun()
                    else:
                        st.write("")  # Empty placeholder


def render_amounts(amounts: dict, requires_review: bool = False):
    """Render the financial totals section with multiple totals override UI.

    Args:
        amounts: Financial amounts dictionary from extraction.
        requires_review: Whether the extraction requires review.
    """
    with st.expander("Totals", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Subtotal", format_currency(amounts.get("subtotal", 0)))
            st.metric("Tax", format_currency(amounts.get("tax", 0)))
            st.metric("Shipping", format_currency(amounts.get("shipping", 0)))

        with col2:
            st.metric("Discounts", format_currency(amounts.get("discounts", 0)))

            # Grand total with highlight
            grand_total = amounts.get("grand_total", 0)
            st.metric("**Grand Total**", format_currency(grand_total))

            # Show MSRP vs Net if both exist and differ - with override selector
            list_price = amounts.get("list_price_total")
            net_price = amounts.get("net_price_total")
            # Use 'is not None' to allow 0 as a valid price
            if list_price is not None and net_price is not None:
                diff = abs(list_price - net_price)
                # Only show variance UI if there's actually a price difference
                variance = (
                    diff / list_price if list_price > 0 else (1.0 if diff > 0 else 0)
                )
                threshold = _settings.msrp_variance_threshold
                if variance > threshold and diff > 0.01:
                    pct = int(threshold * 100)
                    st.warning(f"⚠️ Multiple totals detected (>{pct}% variance)")

                    # Determine current selection (default to net_price per spec)
                    current_selection = amounts.get(
                        "selected_finance_total", "net_price_total"
                    )

                    # Initialize session state for total selection
                    if "selected_total_type" not in st.session_state:
                        st.session_state.selected_total_type = current_selection

                    # Radio button selector for user override
                    st.markdown("**Select Total for Finance:**")
                    selected_total = st.radio(
                        "Choose which total to use:",
                        options=["net_price_total", "grand_total"],
                        format_func=lambda x: (
                            f"Net/Discounted Price: {format_currency(net_price)}"
                            if x == "net_price_total"
                            else f"Grand Total: {format_currency(grand_total)}"
                        ),
                        index=(
                            0
                            if st.session_state.selected_total_type == "net_price_total"
                            else 1
                        ),
                        key="total_selection_radio",
                        help="Choose which total to send to the Finance system",
                        label_visibility="collapsed",
                    )

                    # Update session state
                    st.session_state.selected_total_type = selected_total
                    is_net = selected_total == "net_price_total"
                    st.session_state.selected_finance_amount = (
                        net_price if is_net else grand_total
                    )

                    # Show MSRP (list price) for reference
                    st.caption(f"📋 List/MSRP Price: {format_currency(list_price)}")

                    # Visual indicator of selection
                    selected_amount = net_price if is_net else grand_total
                    msg = f"✅ Selected for Finance: {format_currency(selected_amount)}"
                    st.success(msg)


def render_commercial_terms(terms: dict | None):
    """Render commercial terms if present."""
    if not terms:
        return

    with st.expander("Commercial Terms", expanded=False):
        if terms.get("payment_terms"):
            st.write(f"**Payment Terms:** {terms['payment_terms']}")
        if terms.get("subscription_term_months"):
            months = terms["subscription_term_months"]
            st.write(f"**Subscription Term:** {months} months")
        if terms.get("auto_renew") is not None:
            st.write(f"**Auto-Renew:** {'Yes' if terms['auto_renew'] else 'No'}")
        if terms.get("coverage_start"):
            st.write(f"**Coverage Start:** {terms['coverage_start']}")
        if terms.get("coverage_end"):
            st.write(f"**Coverage End:** {terms['coverage_end']}")


def render_timing_breakdown(metadata: dict):
    """Render the detailed timing breakdown panel."""
    st.markdown("#### ⏱️ Timing Breakdown")

    # Use `or 0` to handle None values (keys may exist with None value)
    total_ms = metadata.get("processing_time_ms") or 0
    parse_ms = metadata.get("parse_time_ms") or 0
    extract_ms = metadata.get("extract_time_ms") or 0
    validate_ms = metadata.get("validate_time_ms") or 0

    # Check if we have stage timing data
    has_stage_timing = parse_ms > 0 or extract_ms > 0 or validate_ms > 0

    if not has_stage_timing:
        st.caption("Stage timing breakdown not available for this extraction.")
        return

    # Main stage bars
    if total_ms > 0:
        # Calculate percentages for visual bar
        parse_pct = (parse_ms / total_ms) * 100
        extract_pct = (extract_ms / total_ms) * 100
        validate_pct = (validate_ms / total_ms) * 100

        # Visual progress bar representation
        bar_style = "display:flex;height:24px;border-radius:4px;overflow:hidden"
        st.markdown(
            f"""
            <div style="{bar_style};margin-bottom:8px">
                <div style="background:#3b82f6;width:{parse_pct}%"></div>
                <div style="background:#f59e0b;width:{extract_pct}%"></div>
                <div style="background:#10b981;width:{validate_pct}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Legend
        st.markdown(
            f"🔵 Parse: {parse_ms}ms ({parse_pct:.1f}%) · "
            f"🟠 Extract: {extract_ms}ms ({extract_pct:.1f}%) · "
            f"🟢 Validate: {validate_ms}ms ({validate_pct:.1f}%)"
        )

    # Parse breakdown details
    parse_breakdown = metadata.get("parse_breakdown")
    if parse_breakdown:
        with st.container():
            st.markdown("**Parse Stage Breakdown:**")
            cols = st.columns(3)
            with cols[0]:
                st.caption(
                    f"Text Extract: {parse_breakdown.get('text_extract_ms', 0)}ms"
                )
            with cols[1]:
                st.caption(
                    f"Table Extract: {parse_breakdown.get('table_extract_ms', 0)}ms"
                )
            with cols[2]:
                img_render = parse_breakdown.get("image_render_ms")
                if img_render is not None:
                    st.caption(f"Image Render: {img_render}ms")
                else:
                    st.caption("Image Render: N/A")

    # Extract breakdown details
    extract_breakdown = metadata.get("extract_breakdown")
    if extract_breakdown:
        with st.container():
            st.markdown("**Extract Stage Breakdown:**")
            cols = st.columns(3)
            with cols[0]:
                st.caption(
                    f"Prompt Prep: {extract_breakdown.get('prompt_prep_ms', 0)}ms"
                )
            with cols[1]:
                llm_ms = extract_breakdown.get("llm_api_ms", 0)
                st.caption(f"LLM API: {llm_ms}ms")
            with cols[2]:
                st.caption(
                    f"Post-Process: {extract_breakdown.get('post_process_ms', 0)}ms"
                )

            # Highlight LLM as bottleneck if it exceeds configured threshold
            if extract_ms > 0 and llm_ms > 0:
                llm_pct = (llm_ms / extract_ms) * 100
                if llm_pct > _settings.llm_bottleneck_threshold:
                    st.caption(
                        f"💡 LLM API accounts for {llm_pct:.0f}% of extraction time"
                    )


def render_extraction_metadata(metadata: dict | None, quote: dict | None = None):
    """Render extraction metadata with optional structured error breakdown."""
    if not metadata:
        return

    with st.expander("Extraction Details", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Processing Time", f"{metadata.get('processing_time_ms', 0)}ms")
        with col2:
            st.metric("Model", metadata.get("model_used", "N/A"))
        with col3:
            confidence = metadata.get("overall_confidence", 0)
            st.metric("Confidence", get_confidence_badge(confidence))

        if metadata.get("validation_passed"):
            st.success("✅ Validation Passed")
        else:
            st.error("❌ Validation Failed")

        # Always show financial breakdown summary
        if quote:
            render_financial_summary(quote, metadata)

        # Show error details only when validation failed
        if not metadata.get("validation_passed"):
            if quote:
                render_validation_error_details(metadata)
            elif metadata.get("validation_errors"):
                st.write("**Errors:**")
                for error in metadata["validation_errors"]:
                    st.write(f"- {error}")

        # Add timing breakdown section
        st.markdown("---")
        render_timing_breakdown(metadata)


def render_review_interface(quote: dict):
    """Render the review interface for low-confidence fields (in-memory only)."""
    st.markdown("### Review Required")

    metadata = quote.get("extraction_metadata", {})

    # Show why review is required
    reasons = []
    if metadata.get("overall_confidence", 1.0) < CONFIDENCE_THRESHOLD:
        reasons.append(f"Low confidence: {metadata.get('overall_confidence', 0):.1%}")

    validation_failed = not metadata.get("validation_passed", True)
    if validation_failed:
        reasons.append("Validation failed (see breakdown below)")

    if reasons:
        st.warning("This extraction requires review:")
        for reason in reasons:
            st.write(reason)

    # Always show financial summary in review interface
    st.markdown("---")
    render_financial_summary(quote, metadata)

    # Show error details when validation failed
    if validation_failed and metadata.get("validation_errors"):
        render_validation_error_details(metadata)

    # Session-only note
    st.info("ℹ️ Changes made here are session-only and won't be persisted.")

    # Action buttons for review mode
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Re-Extract button
        if st.button("🔄 Re-Extract", help="Run extraction again on the same document"):
            pdf_bytes = st.session_state.get("pdf_bytes")
            filename = st.session_state.get("filename", "document.pdf")

            if pdf_bytes:
                # Reset UI state before re-extraction
                st.session_state.selected_total_type = None
                st.session_state.selected_finance_amount = None
                st.session_state.highlighted_region = None

                with st.spinner("Re-extracting..."):
                    try:
                        result = extract_document(pdf_bytes, filename)
                        st.session_state.extraction_result = result
                        st.success("✅ Re-extraction complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Re-extraction failed: {e}")
            else:
                st.error("Cannot re-extract: PDF not available in session.")

    with col2:
        # Manual review mode indicator
        st.info("📝 Manual Review Mode")

    # Show low confidence line items
    line_items = quote.get("line_items", [])
    low_confidence_items = [
        (i, item)
        for i, item in enumerate(line_items)
        if item.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    ]

    if low_confidence_items:
        st.markdown("#### Low Confidence Items")

        # Use a form for grouped submission
        with st.form(key="correction_form"):
            for list_index, item in low_confidence_items:
                with st.container():
                    line_num = item.get("line_number")
                    desc = item.get("description", "N/A")
                    st.write(f"**Line {line_num}:** {desc}")
                    conf_badge = get_confidence_badge(item.get("confidence", 0))
                    st.write(f"Confidence: {conf_badge}")

                    # Editable fields - use list_index for stable widget keys
                    col1, col2 = st.columns(2)
                    with col1:
                        st.number_input(
                            "Unit Price",
                            value=item.get("unit_price", 0.0),
                            key=f"price_{list_index}",
                            format="%.2f",
                        )
                    with col2:
                        item_types = [
                            "hardware",
                            "software",
                            "services",
                            "fee",
                            "tax",
                            "shipping",
                            "discount",
                            "credit",
                        ]
                        current_type = item.get("item_type") or ""
                        type_index = 0
                        if current_type and current_type in item_types:
                            type_index = item_types.index(current_type) + 1
                        st.selectbox(
                            "Item Type",
                            options=[""] + item_types,
                            index=type_index,
                            key=f"type_{list_index}",
                        )

                    st.divider()

            # Submit button (in-memory updates)
            submitted = st.form_submit_button(
                "💾 Apply Corrections (Session Only)",
                type="primary",
            )

            if submitted:
                # Apply corrections in memory
                corrections_applied = 0

                for list_index, item in low_confidence_items:
                    # Use list_index for widget keys (matches keys above)
                    new_price = st.session_state.get(f"price_{list_index}")
                    original_price = item.get("unit_price", 0.0)
                    if new_price is not None:
                        price_changed = abs(new_price - original_price) > 0.001
                        if price_changed:
                            apply_correction_in_memory(
                                st.session_state.extraction_result,
                                f"line_items.{list_index}.unit_price",
                                new_price,
                            )
                            corrections_applied += 1

                    # Check for type change
                    new_type = st.session_state.get(f"type_{list_index}")
                    original_type = item.get("item_type") or ""
                    if new_type and new_type != original_type:
                        apply_correction_in_memory(
                            st.session_state.extraction_result,
                            f"line_items.{list_index}.item_type",
                            new_type,
                        )
                        corrections_applied += 1

                if corrections_applied > 0:
                    st.success(
                        f"✅ Applied {corrections_applied} correction(s) to session!"
                    )
                    st.rerun()
                else:
                    st.info("No changes to apply.")


def render_action_buttons(quote: dict, pdf_bytes: bytes | None, filename: str):
    """Render the action buttons with selected finance total support."""
    col1, col2 = st.columns(2)

    # Get the selected total from session state (if user made an override selection)
    selected_total_type = st.session_state.get("selected_total_type")
    selected_amount = st.session_state.get("selected_finance_amount")

    # Prepare export data - include user's total selection if present
    export_quote = quote.copy()
    if selected_total_type and "amounts" in export_quote:
        export_quote["amounts"] = export_quote["amounts"].copy()
        export_quote["amounts"]["selected_finance_total"] = selected_total_type

    with col1:
        # Download JSON with user's selected total
        json_str = json.dumps(export_quote, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{filename.replace('.pdf', '')}_extracted.json",
            mime="application/json",
        )

    with col2:
        # Send to Finance (mock) - shows which total is being sent
        if st.button("💰 Send to Finance System"):
            with st.spinner("Sending..."):
                time.sleep(1)  # Mock delay

            # Show what was sent
            if selected_amount is not None:
                st.success(
                    f"✅ Successfully sent to Finance System!\n\n"
                    f"**Amount sent:** {format_currency(selected_amount)} "
                    f"({selected_total_type})"
                )
            else:
                grand_total = quote.get("amounts", {}).get("grand_total", 0)
                st.success(
                    f"✅ Successfully sent to Finance System!\n\n"
                    f"**Amount sent:** {format_currency(grand_total)} (grand_total)"
                )
            st.balloons()


def render_results_view(quote: dict, pdf_bytes: bytes | None, filename: str):
    """Render the split-panel results view."""
    st.markdown("---")
    st.markdown("## Extraction Results")

    # Confidence badge at top
    metadata = quote.get("extraction_metadata", {})
    confidence = metadata.get("overall_confidence", 0)
    requires_review = metadata.get("requires_review", False)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### Quote: {quote.get('quote_id', 'N/A')}")
    with col2:
        st.markdown(f"**Confidence:** {get_confidence_badge(confidence)}")
        processing_ms = metadata.get("processing_time_ms") or 0
        processing_s = processing_ms / 1000
        st.markdown(f"**Processing time (target: 7 seconds):** {processing_s:.1f}s")
    with col3:
        if requires_review:
            st.warning("⚠️ Review Required")
        else:
            st.success("✅ Ready")

    # Split panel layout
    left_col, right_col = st.columns([1, 1])

    # Left panel: PDF preview (scrollable)
    with left_col:
        st.markdown("### Original Document")
        if pdf_bytes:
            # Generate hash for caching
            pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

            # Get highlighted region from session state
            highlighted = st.session_state.get("highlighted_region")

            # Validate highlight index against page count
            page_count = get_pdf_page_count(pdf_bytes)
            if highlighted and (highlighted[0] < 0 or highlighted[0] >= page_count):
                highlighted = None
                st.session_state.highlighted_region = None

            # Clear stale highlight if PDF changed (different hash)
            last_pdf_hash = st.session_state.get("last_pdf_hash")
            if last_pdf_hash and last_pdf_hash != pdf_hash:
                st.session_state.highlighted_region = None
                highlighted = None
            st.session_state.last_pdf_hash = pdf_hash

            # Clear highlight button (persistent highlight UX)
            if highlighted:
                if st.button("Clear highlight", key="clear_highlight"):
                    st.session_state.highlighted_region = None
                    st.rerun()

            # Get cached pages (without highlight - highlight applied dynamically)
            pages = get_cached_pdf_pages(pdf_hash, pdf_bytes)

            if not pages:
                st.warning("PDF preview not available (0 pages or corrupted)")
            else:
                # Fixed-height scrollable container for PDF preview
                # This prevents scrolling the PDF from scrolling the entire page
                with st.container(height=700):
                    # Render all pages in scrollable view
                    for page_idx, png_bytes_page in pages:
                        # Apply highlight overlay dynamically if needed
                        display_bytes = png_bytes_page
                        caption = f"Page {page_idx + 1}"

                        if highlighted and highlighted[0] == page_idx:
                            display_bytes = _apply_highlight_overlay(
                                png_bytes_page, highlighted[1]
                            )
                            caption = f"Page {page_idx + 1} 🎯 Highlighted"

                        st.image(
                            display_bytes, use_container_width=True, caption=caption
                        )
        else:
            st.info("PDF preview not available")

    # Right panel: Extracted data
    with right_col:
        st.markdown("### Extracted Data")

        # Extraction metadata
        render_extraction_metadata(metadata, quote)

        # Quote header
        with st.expander("Quote Header", expanded=True):
            st.write(f"**Quote ID:** {quote.get('quote_id', 'N/A')}")
            st.write(f"**Quote Date:** {quote.get('quote_date', 'N/A')}")
            st.write(f"**Valid Until:** {quote.get('valid_until', 'N/A')}")
            st.write(f"**Currency:** {quote.get('currency', 'USD')}")

        # Vendor
        if quote.get("vendor"):
            render_entity(quote["vendor"], "Vendor")

        # Customer
        if quote.get("customer"):
            render_entity(quote["customer"], "Customer")

        # Line items (with source region highlighting support)
        source_regions = quote.get("source_regions", [])
        render_line_items(quote.get("line_items", []), requires_review, source_regions)

        # Amounts (in-memory only)
        render_amounts(quote.get("amounts", {}), requires_review)

        # Commercial terms
        render_commercial_terms(quote.get("commercial_terms"))

    # Review interface if needed
    if requires_review:
        st.markdown("---")
        render_review_interface(quote)

    # Action buttons
    st.markdown("---")
    render_action_buttons(quote, pdf_bytes, filename)


def main():
    """Main application entry point."""
    st.title("📄 Vendor Quote Extractor")
    st.markdown("AI-powered extraction of structured data from vendor quotes")
    st.caption("Demo mode - stateless, session-only processing")

    # Initialize session state
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "selected_total_type" not in st.session_state:
        st.session_state.selected_total_type = None
    if "selected_finance_amount" not in st.session_state:
        st.session_state.selected_finance_amount = None
    if "highlighted_region" not in st.session_state:
        st.session_state.highlighted_region = None  # (page, bbox) tuple
    if "last_pdf_hash" not in st.session_state:
        st.session_state.last_pdf_hash = None  # For detecting PDF changes

    # Upload zone
    uploaded_file = render_upload_zone()

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # Store in session state
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.filename = filename

        # Extract button
        extract_clicked = st.button("🚀 Extract Data", type="primary")

        # Process button
        if extract_clicked:
            # Reset finance selection state before new extraction
            # (prevents stale amounts from previous extractions)
            st.session_state.selected_total_type = None
            st.session_state.selected_finance_amount = None
            st.session_state.highlighted_region = None

            # Show processing spinner during extraction
            try:
                with st.spinner(
                    "🔄 Processing document... (Parse → Extract → Validate)"
                ):
                    start_time = time.time()
                    result = extract_document(pdf_bytes, filename)
                    elapsed_seconds = time.time() - start_time

                st.session_state.extraction_result = result

                # Show completion with actual elapsed time
                st.success(f"✅ Extraction complete in {elapsed_seconds:.1f}s")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")

    # New document button (clear state)
    if st.session_state.extraction_result:
        if st.button("🔄 Process New Document"):
            st.session_state.extraction_result = None
            st.session_state.pdf_bytes = None
            st.session_state.filename = None
            st.session_state.selected_total_type = None
            st.session_state.selected_finance_amount = None
            st.session_state.highlighted_region = None
            st.session_state.last_pdf_hash = None
            st.rerun()

    # Show results if available
    if st.session_state.extraction_result:
        render_results_view(
            st.session_state.extraction_result,
            st.session_state.pdf_bytes,
            st.session_state.filename or "document.pdf",
        )


if __name__ == "__main__":
    main()
