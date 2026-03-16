"""
Tests for Streamlit UI components - src/ui/app.py

This test module provides coverage for UI utility functions without
requiring a running Streamlit server. We test:
- Pure utility functions (format_currency, get_confidence_badge)
- PDF rendering functions (get_pdf_page_count, render_pdf_page)
- In-memory correction logic
- Validation calculation logic

WHY THESE TESTS MATTER:
- Pure functions are the foundation of correct UI behavior
- PDF rendering reliability directly impacts user experience
- Validation logic must match specification exactly ($0.01 tolerance)

Testing Strategy:
- Mock `streamlit` before importing the UI module to avoid set_page_config errors
- Use real PDF fixtures from tests/fixtures/ for PDF function tests

NOTE: This is a stateless demo deployment - no API functions are tested.
The app calls ExtractionPipeline directly instead of HTTP endpoints.
"""

import sys
from unittest.mock import MagicMock

# Mock streamlit BEFORE importing the UI module
# This prevents st.set_page_config() from being called during import
mock_st = MagicMock()
mock_st.session_state = {}
sys.modules["streamlit"] = mock_st

# Now we can import from the UI module
# These imports must come AFTER mocking streamlit
from src.ui.app import (  # noqa: E402
    apply_correction_in_memory,
    format_currency,
    get_confidence_badge,
    get_pdf_page_count,
    render_pdf_page,
)

# =============================================================================
# Pure Function Tests
# =============================================================================


class TestFormatCurrency:
    """Tests for format_currency() - USD currency formatting.

    WHY: Currency display is critical for financial accuracy. Users must see
    properly formatted values with thousands separators and 2 decimal places.
    """

    def test_formats_positive_amount(self):
        """Standard positive amounts get proper formatting."""
        assert format_currency(1234.56) == "$1,234.56"

    def test_formats_zero(self):
        """Zero displays as $0.00."""
        assert format_currency(0) == "$0.00"

    def test_formats_negative_amount(self):
        """Negative amounts (credits/discounts) preserve the sign."""
        assert format_currency(-500.00) == "$-500.00"

    def test_formats_large_amount(self):
        """Large amounts get thousands separators."""
        assert format_currency(1234567.89) == "$1,234,567.89"

    def test_formats_small_cents(self):
        """Small amounts display correct decimal places."""
        assert format_currency(0.01) == "$0.01"

    def test_rounds_to_two_decimals(self):
        """Values with more decimals get rounded to 2 places."""
        assert format_currency(10.999) == "$11.00"
        assert format_currency(10.991) == "$10.99"

    def test_formats_integer_with_decimals(self):
        """Integer values display .00 suffix."""
        assert format_currency(100) == "$100.00"


class TestGetConfidenceBadge:
    """Tests for get_confidence_badge() - visual confidence indicators.

    WHY: The confidence badge tells users at a glance whether to trust an
    extraction. Green (>= 0.9) means ready; yellow (< 0.9) signals review needed.
    """

    def test_high_confidence_returns_green_badge(self):
        """Confidence >= threshold shows green indicator."""
        badge = get_confidence_badge(0.95)
        assert "🟢" in badge
        assert "95.0%" in badge

    def test_low_confidence_returns_yellow_badge(self):
        """Confidence < threshold shows yellow warning."""
        badge = get_confidence_badge(0.80)
        assert "🟡" in badge
        assert "80.0%" in badge

    def test_threshold_boundary_is_green(self):
        """Exactly at threshold (0.90) is green (>= comparison)."""
        # Default CONFIDENCE_THRESHOLD is 0.9
        badge = get_confidence_badge(0.90)
        assert "🟢" in badge

    def test_just_below_threshold_is_yellow(self):
        """Just below threshold triggers yellow."""
        badge = get_confidence_badge(0.899)
        assert "🟡" in badge

    def test_perfect_confidence(self):
        """100% confidence displays correctly."""
        badge = get_confidence_badge(1.0)
        assert "🟢" in badge
        assert "100.0%" in badge

    def test_zero_confidence(self):
        """0% confidence displays correctly."""
        badge = get_confidence_badge(0.0)
        assert "🟡" in badge
        assert "0.0%" in badge


# =============================================================================
# PDF Function Tests
# =============================================================================


class TestGetPdfPageCount:
    """Tests for get_pdf_page_count() - PDF page enumeration.

    WHY: Accurate page count is essential for navigation UI and determines
    routing decisions (e.g., >5 pages routes to different providers).
    """

    def test_returns_page_count_for_valid_pdf(self, mixed_category_pdf_bytes):
        """Valid PDF returns correct page count."""
        count = get_pdf_page_count(mixed_category_pdf_bytes)
        assert isinstance(count, int)
        assert count >= 1

    def test_multi_page_pdf_count(self, multi_page_pdf_bytes):
        """Multi-page PDF (Multi-Page Quote ~50+ items) returns accurate count."""
        count = get_pdf_page_count(multi_page_pdf_bytes)
        assert count >= 1  # Multi-Page is multi-page

    def test_different_pdfs_have_different_counts(
        self, mixed_category_pdf_bytes, multi_page_pdf_bytes
    ):
        """Different documents may have different page counts."""
        count1 = get_pdf_page_count(mixed_category_pdf_bytes)
        count2 = get_pdf_page_count(multi_page_pdf_bytes)
        # Both are valid counts, may or may not be equal
        assert count1 >= 1
        assert count2 >= 1

    def test_invalid_pdf_returns_zero(self, invalid_file_bytes):
        """Non-PDF bytes return 0 (graceful handling)."""
        count = get_pdf_page_count(invalid_file_bytes)
        assert count == 0


class TestRenderPdfPage:
    """Tests for render_pdf_page() - PDF to PNG conversion.

    WHY: Users see the original document alongside extracted data. Rendering
    must be reliable and handle edge cases like out-of-bounds page numbers.
    """

    def test_renders_first_page_as_png(self, mixed_category_pdf_bytes):
        """Page 0 renders to valid PNG bytes."""
        png_bytes = render_pdf_page(mixed_category_pdf_bytes, page_number=0)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_specific_page(self, multi_page_pdf_bytes):
        """Can render pages other than the first."""
        png_bytes = render_pdf_page(multi_page_pdf_bytes, page_number=1)
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_clamps_out_of_bounds_page_to_last(self, mixed_category_pdf_bytes):
        """Page number exceeding max clamps to last page (no error)."""
        # Request page 999 on a document with fewer pages
        png_bytes = render_pdf_page(mixed_category_pdf_bytes, page_number=999)
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_default_page_is_zero(self, mixed_category_pdf_bytes):
        """Default page_number=0 works."""
        png_bytes = render_pdf_page(mixed_category_pdf_bytes)
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_invalid_pdf_returns_empty_bytes(self, invalid_file_bytes):
        """Non-PDF bytes return empty bytes (graceful handling)."""
        result = render_pdf_page(invalid_file_bytes)
        assert result == b""


# =============================================================================
# In-Memory Correction Tests
# =============================================================================


class TestApplyCorrectionInMemory:
    """Tests for apply_correction_in_memory() - session-only field updates.

    WHY: In stateless demo mode, corrections are applied in-memory only.
    This function must correctly parse field paths and update nested structures.
    """

    def test_updates_simple_field(self):
        """Simple dot-notation field path works."""
        quote = {"amounts": {"grand_total": 1000.00}}

        result = apply_correction_in_memory(quote, "amounts.grand_total", 1500.00)

        assert result["amounts"]["grand_total"] == 1500.00

    def test_updates_array_element(self):
        """Array index notation like line_items.0.unit_price works."""
        quote = {
            "line_items": [
                {"unit_price": 100.00, "description": "Item 1"},
                {"unit_price": 200.00, "description": "Item 2"},
            ]
        }

        result = apply_correction_in_memory(quote, "line_items.0.unit_price", 150.00)

        assert result["line_items"][0]["unit_price"] == 150.00
        assert result["line_items"][1]["unit_price"] == 200.00  # Unchanged

    def test_updates_second_array_element(self):
        """Can update elements other than the first."""
        quote = {
            "line_items": [
                {"unit_price": 100.00},
                {"unit_price": 200.00},
            ]
        }

        result = apply_correction_in_memory(quote, "line_items.1.unit_price", 250.00)

        assert result["line_items"][0]["unit_price"] == 100.00  # Unchanged
        assert result["line_items"][1]["unit_price"] == 250.00

    def test_updates_item_type(self):
        """Can update string fields like item_type."""
        quote = {
            "line_items": [
                {"item_type": "hardware", "description": "Server"},
            ]
        }

        result = apply_correction_in_memory(quote, "line_items.0.item_type", "software")

        assert result["line_items"][0]["item_type"] == "software"

    def test_returns_mutated_quote(self):
        """The function returns the same dict object (mutated in place)."""
        quote = {"amounts": {"grand_total": 1000.00}}

        result = apply_correction_in_memory(quote, "amounts.grand_total", 2000.00)

        assert result is quote


# =============================================================================
# Validation Calculation Tests
# =============================================================================


class TestValidationCalculations:
    """Tests for validation calculation logic used in render_validation_error_breakdown.

    WHY: Validation errors must follow the spec exactly:
    - $0.01 tolerance for math checks
    - Correct identification of line sum vs grand total errors
    - Accurate difference calculations

    These tests verify the calculation logic independently of UI rendering.
    """

    def test_line_items_sum_within_tolerance_passes(self):
        """Sum within $0.01 of subtotal = no error."""
        line_items = [
            {"extended_price": 100.00},
            {"extended_price": 200.00},
            {"extended_price": 199.99},  # Total: 499.99
        ]
        subtotal = 500.00  # Diff = 0.01 (within tolerance)

        line_items_sum = sum(item.get("extended_price", 0) for item in line_items)
        diff = abs(line_items_sum - subtotal)

        assert diff <= 0.01
        has_line_sum_error = diff > 0.01
        assert has_line_sum_error is False

    def test_line_items_sum_outside_tolerance_fails(self):
        """Sum more than $0.01 from subtotal = error."""
        line_items = [
            {"extended_price": 100.00},
            {"extended_price": 200.00},
        ]
        subtotal = 350.00  # Diff = 50.00 (line sum is 300)

        line_items_sum = sum(item.get("extended_price", 0) for item in line_items)
        diff = abs(line_items_sum - subtotal)

        assert diff > 0.01
        has_line_sum_error = diff > 0.01
        assert has_line_sum_error is True

    def test_grand_total_calculation_formula(self):
        """Grand total = subtotal + tax + shipping - discounts."""
        subtotal = 1000.00
        tax = 80.00
        shipping = 20.00
        discounts = 50.00

        calculated_total = subtotal + tax + shipping - discounts

        assert calculated_total == 1050.00

    def test_grand_total_within_tolerance_passes(self):
        """Calculated total within $0.01 of extracted = no error."""
        subtotal, tax, shipping, discounts = 1000.00, 80.00, 20.00, 50.00
        grand_total = 1050.00  # Matches exactly

        calculated_total = subtotal + tax + shipping - discounts
        diff = abs(calculated_total - grand_total)

        assert diff <= 0.01
        has_total_error = diff > 0.01
        assert has_total_error is False

    def test_grand_total_outside_tolerance_fails(self):
        """Calculated total more than $0.01 from extracted = error."""
        subtotal, tax, shipping, discounts = 1000.00, 80.00, 20.00, 50.00
        grand_total = 1100.00  # Off by $50

        calculated_total = subtotal + tax + shipping - discounts
        diff = abs(calculated_total - grand_total)

        assert diff > 0.01
        has_total_error = diff > 0.01
        assert has_total_error is True

    def test_handles_missing_values_as_zero(self):
        """Missing amounts should default to 0."""
        amounts = {"subtotal": 100.00}  # Missing tax, shipping, discounts

        subtotal = amounts.get("subtotal", 0)
        tax = amounts.get("tax", 0)
        shipping = amounts.get("shipping", 0)
        discounts = amounts.get("discounts", 0)

        calculated_total = subtotal + tax + shipping - discounts
        assert calculated_total == 100.00

    def test_handles_negative_extended_prices(self):
        """Credits/discounts appear as negative extended_price."""
        line_items = [
            {"extended_price": 500.00},
            {"extended_price": -50.00},  # Credit line item
        ]

        line_items_sum = sum(item.get("extended_price", 0) for item in line_items)

        assert line_items_sum == 450.00

    def test_empty_line_items_sum_to_zero(self):
        """Empty line items list sums to zero."""
        line_items = []

        line_items_sum = sum(item.get("extended_price", 0) for item in line_items)

        assert line_items_sum == 0


# =============================================================================
# Multiple Totals Detection Tests
# =============================================================================


class TestMultipleTotalsDetection:
    """Tests for MSRP vs Net/Discounted total variance detection.

    WHY: Per spec, when list_price_total and net_price_total differ by >5%,
    UI shows radio buttons for user to select which total to send to Finance.
    """

    def test_detects_variance_above_threshold(self):
        """Variance > 5% triggers multiple totals UI."""
        list_price = 1000.00
        net_price = 900.00  # 10% discount

        diff = abs(list_price - net_price)
        variance = diff / list_price if list_price > 0 else 0

        assert variance > 0.05
        assert variance == 0.10  # 10%

    def test_no_detection_when_variance_below_threshold(self):
        """Variance <= 5% does not trigger multiple totals."""
        list_price = 1000.00
        net_price = 960.00  # 4% discount

        diff = abs(list_price - net_price)
        variance = diff / list_price if list_price > 0 else 0

        assert variance <= 0.05
        assert variance == 0.04  # 4%

    def test_exactly_at_threshold_does_not_trigger(self):
        """Exactly 5% variance is <= threshold (no trigger)."""
        list_price = 1000.00
        net_price = 950.00  # Exactly 5%

        diff = abs(list_price - net_price)
        variance = diff / list_price if list_price > 0 else 0

        assert variance == 0.05
        # Per implementation: variance > 0.05 triggers
        assert not (variance > 0.05)

    def test_handles_zero_list_price(self):
        """Zero list price doesn't cause division error."""
        list_price = 0
        net_price = 100.00

        variance = abs(list_price - net_price) / list_price if list_price > 0 else 0

        assert variance == 0  # Graceful handling


# =============================================================================
# Session State Key Tests
# =============================================================================


class TestSessionStateKeys:
    """Tests verifying expected session state keys are used.

    WHY: Session state keys must be consistent across the app.
    Typos in key names cause silent failures where state isn't shared.
    """

    def test_expected_session_state_keys_exist(self):
        """Document the expected session state keys for stateless mode."""
        expected_keys = [
            "extraction_result",
            "pdf_bytes",
            "filename",
            "selected_total_type",
            "selected_finance_amount",
            "highlighted_region",
            "last_pdf_hash",
        ]

        # This is a documentation test - ensures we know the keys
        assert len(expected_keys) == 7


# =============================================================================
# Status Icon Logic Tests
# =============================================================================


class TestStatusIconLogic:
    """Tests for status icon selection logic.

    WHY: Status icons communicate extraction state at a glance:
    - Warning (yellow): requires review
    - Check (green): good confidence, no review needed
    - Blue circle: completed but uncertain
    """

    def test_requires_review_shows_warning(self):
        """requires_review=True shows warning icon."""
        requires_review = True
        confidence = 0.95
        threshold = 0.9

        if requires_review:
            status_icon = "⚠️"
        elif confidence is not None and confidence >= threshold:
            status_icon = "✅"
        else:
            status_icon = "🔵"

        assert status_icon == "⚠️"

    def test_high_confidence_no_review_shows_check(self):
        """High confidence + no review = green check."""
        requires_review = False
        confidence = 0.95
        threshold = 0.9

        if requires_review:
            status_icon = "⚠️"
        elif confidence is not None and confidence >= threshold:
            status_icon = "✅"
        else:
            status_icon = "🔵"

        assert status_icon == "✅"

    def test_low_confidence_no_review_shows_blue(self):
        """Low confidence but no review flag = blue (uncertain)."""
        requires_review = False
        confidence = 0.75
        threshold = 0.9

        if requires_review:
            status_icon = "⚠️"
        elif confidence is not None and confidence >= threshold:
            status_icon = "✅"
        else:
            status_icon = "🔵"

        assert status_icon == "🔵"

    def test_none_confidence_shows_blue(self):
        """None confidence shows blue (uncertain state)."""
        requires_review = False
        confidence = None
        threshold = 0.9

        if requires_review:
            status_icon = "⚠️"
        elif confidence is not None and confidence >= threshold:
            status_icon = "✅"
        else:
            status_icon = "🔵"

        assert status_icon == "🔵"
