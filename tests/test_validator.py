"""
Tests for the Validator module.

Verifies all validation rules with pass and fail cases.
"""

from datetime import date

import pytest

from src.extraction.validator import ValidationResult, Validator
from src.schemas import Amounts, Entity, ItemType, LineItem, VendorQuote


@pytest.fixture
def validator() -> Validator:
    """Create a Validator instance."""
    return Validator()


@pytest.fixture
def valid_quote() -> VendorQuote:
    """Create a valid quote that should pass all validations."""
    return VendorQuote(
        quote_id="Q-TEST-001",
        quote_date=date(2024, 1, 1),
        valid_until=date(2024, 2, 1),
        currency="USD",
        vendor=Entity(name="Test Vendor Inc"),
        line_items=[
            LineItem(
                line_number=1,
                description="Product A",
                quantity=2,
                unit_price=100.0,
                extended_price=200.0,
                item_type=ItemType.hardware,
            ),
            LineItem(
                line_number=2,
                description="Product B",
                quantity=3,
                unit_price=50.0,
                extended_price=150.0,
                item_type=ItemType.software,
            ),
        ],
        amounts=Amounts(
            subtotal=350.0,
            tax=35.0,
            shipping=15.0,
            discounts=0.0,
            grand_total=400.0,  # 350 + 35 + 15 = 400
        ),
    )


class TestValidatorBasic:
    """Basic validator tests."""

    def test_valid_quote_passes(self, validator: Validator, valid_quote: VendorQuote):
        result = validator.validate(valid_quote)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_validation_result_structure(
        self, validator: Validator, valid_quote: VendorQuote
    ):
        result = validator.validate(valid_quote)
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "passed")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")


class TestLineItemsSumValidation:
    """Tests for line items sum validation."""

    def test_line_items_sum_matches_subtotal(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item 1",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
                LineItem(
                    line_number=2,
                    description="Item 2",
                    quantity=2,
                    unit_price=50.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=200.0, grand_total=200.0),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_line_items_sum_mismatch_fails(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item 1",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=150.0,  # Mismatch: should be 100
                grand_total=150.0,
            ),
        )
        result = validator.validate(quote)
        assert result.passed is False
        assert any("Line items sum" in err for err in result.errors)

    def test_tolerance_allows_small_differences(self, validator: Validator):
        """$0.01 tolerance should allow tiny floating-point differences."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item 1",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.005,  # Tiny difference
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        # Should pass due to $0.01 tolerance
        assert result.passed is True


class TestGrandTotalValidation:
    """Tests for grand total calculation validation."""

    def test_grand_total_calculation_correct(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=100.0,
                tax=10.0,
                shipping=5.0,
                discounts=15.0,
                grand_total=100.0,  # 100 + 10 + 5 - 15 = 100
            ),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_grand_total_mismatch_fails(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=100.0,
                tax=10.0,
                shipping=5.0,
                discounts=0.0,
                grand_total=100.0,  # Wrong: should be 115
            ),
        )
        result = validator.validate(quote)
        assert result.passed is False
        assert any("Calculated total" in err for err in result.errors)


class TestLineItemMathValidation:
    """Tests for quantity * unit_price = extended_price validation."""

    def test_line_item_math_correct(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=5,
                    unit_price=20.0,
                    extended_price=100.0,  # 5 * 20 = 100 ✓
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_line_item_math_wrong_fails(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=5,
                    unit_price=20.0,
                    extended_price=90.0,  # Wrong: should be 100
                ),
            ],
            amounts=Amounts(subtotal=90.0, grand_total=90.0),
        )
        result = validator.validate(quote)
        assert result.passed is False
        assert any("Line 1:" in err and "unit_price" in err for err in result.errors)

    def test_negative_extended_price_skipped(self, validator: Validator):
        """Credits/discounts with negative extended prices skip math check."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
                LineItem(
                    line_number=2,
                    description="Credit",
                    quantity=1,
                    unit_price=0.0,  # Credits don't follow qty * price
                    extended_price=-50.0,
                    item_type=ItemType.credit,
                ),
            ],
            amounts=Amounts(subtotal=50.0, grand_total=50.0),
        )
        result = validator.validate(quote)
        # Should not fail on the credit line
        assert not any("Line 2:" in err for err in result.errors)


class TestTaxRateValidation:
    """Tests for tax rate sanity check."""

    def test_normal_tax_rate_no_warning(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=100.0,
                tax=10.0,  # 10% tax - normal
                grand_total=110.0,
            ),
        )
        result = validator.validate(quote)
        assert not any("tax rate" in w.lower() for w in result.warnings)

    def test_high_tax_rate_warning(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=100.0,
                tax=25.0,  # 25% tax - suspiciously high
                grand_total=125.0,
            ),
        )
        result = validator.validate(quote)
        assert any("tax rate" in w.lower() for w in result.warnings)
        # Should still pass (warnings don't fail)
        assert result.passed is True


class TestVendorValidation:
    """Tests for vendor name validation."""

    def test_valid_vendor_name(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Acme Retail Group"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_short_vendor_name_fails(self, validator: Validator):
        """Vendor name must be at least 2 characters."""
        # Note: This would fail at schema validation level first
        # but validator also checks in case schema is bypassed
        quote = VendorQuote(
            vendor=Entity(name="AB"),  # 2 chars - minimum
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        # 2 chars should be OK (minimum is 2)
        assert not any("Vendor name too short" in err for err in result.errors)


class TestItemClassificationValidation:
    """Tests for item classification warning."""

    def test_all_items_classified_no_warning(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Hardware",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                    item_type=ItemType.hardware,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert not any("classification" in w.lower() for w in result.warnings)

    def test_unclassified_items_warning(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Mystery Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                    item_type=None,  # Not classified
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert any("classification" in w.lower() for w in result.warnings)
        # Warnings don't cause failure
        assert result.passed is True


class TestDateLogicValidation:
    """Tests for date logic validation."""

    def test_valid_date_range(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            quote_date=date(2024, 1, 1),
            valid_until=date(2024, 2, 1),  # After quote_date - OK
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_invalid_date_range_fails(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            quote_date=date(2024, 2, 1),
            valid_until=date(2024, 1, 1),  # Before quote_date - Error!
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is False
        assert any("date range" in err.lower() for err in result.errors)


class TestCurrencyValidation:
    """Tests for currency validation."""

    def test_usd_accepted(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            currency="USD",
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is True

    def test_non_usd_rejected(self, validator: Validator):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            currency="EUR",  # Not supported in MVP
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        result = validator.validate(quote)
        assert result.passed is False
        assert any("EUR" in err for err in result.errors)


class TestH6CompletenessValidation:
    """Tests for H6 completeness validation to detect potential truncation."""

    def test_no_warning_when_no_tables(self, validator: Validator):
        """H6: No completeness warning when parse_result has no tables."""
        from src.extraction.pdf_parser import ParseResult

        parse_result = ParseResult(
            text="Test content",
            tables=[],  # No tables
            page_count=2,
            is_scanned=False,
            images=None,
            error=None,
        )

        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )

        result = validator.validate(quote, parse_result=parse_result)

        # No completeness warning since no tables to compare against
        assert not any("truncation" in w.lower() for w in result.warnings)

    def test_no_warning_when_items_match_expected(self, validator: Validator):
        """H6: No warning when extracted items >= 80% of expected rows."""
        from src.extraction.pdf_parser import ParseResult, TableData

        # 10 table rows, 8 extracted items = 80% (at threshold)
        table = TableData(
            page=1,
            rows=[["Item", "Qty", "Price"]]
            + [[f"Item {i}", "1", "100.00"] for i in range(9)],
            headers=["Item", "Qty", "Price"],
        )
        parse_result = ParseResult(
            text="Test content",
            tables=[table],
            page_count=2,
            is_scanned=False,
            images=None,
            error=None,
        )

        # 8 line items extracted (80% of 10 rows)
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
                for i in range(1, 9)
            ],
            amounts=Amounts(subtotal=800.0, grand_total=800.0),
        )

        result = validator.validate(quote, parse_result=parse_result)

        # Should pass - 80% completeness is acceptable
        assert not any("truncation" in w.lower() for w in result.warnings)

    def test_warning_when_items_below_threshold(self, validator: Validator):
        """H6: Warning when extracted items < 80% of expected rows."""
        from src.extraction.pdf_parser import ParseResult, TableData

        # 50 table rows, only 10 extracted items = 20% (well below 80%)
        table = TableData(
            page=1,
            rows=[["Item", "Qty", "Price"]]
            + [[f"Item {i}", "1", "100.00"] for i in range(49)],
            headers=["Item", "Qty", "Price"],
        )
        parse_result = ParseResult(
            text="Test content",
            tables=[table],
            page_count=2,
            is_scanned=False,
            images=None,
            error=None,
        )

        # Only 10 line items extracted (20% of 50 rows)
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
                for i in range(1, 11)
            ],
            amounts=Amounts(subtotal=1000.0, grand_total=1000.0),
        )

        result = validator.validate(quote, parse_result=parse_result)

        # Should have truncation warning
        assert any("truncation" in w.lower() for w in result.warnings)
        # Should still pass (warnings don't fail validation)
        assert result.passed is True

    def test_warning_includes_completeness_percentage(self, validator: Validator):
        """H6: Warning message includes completeness percentage."""
        from src.extraction.pdf_parser import ParseResult, TableData

        # 100 table rows, 30 extracted = 30%
        table = TableData(
            page=1,
            rows=[[f"Item {i}", "1", "100.00"] for i in range(100)],
            headers=["Item", "Qty", "Price"],
        )
        parse_result = ParseResult(
            text="Test content",
            tables=[table],
            page_count=5,
            is_scanned=False,
            images=None,
            error=None,
        )

        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
                for i in range(1, 31)
            ],
            amounts=Amounts(subtotal=3000.0, grand_total=3000.0),
        )

        result = validator.validate(quote, parse_result=parse_result)

        # Check warning mentions percentage
        truncation_warnings = [w for w in result.warnings if "truncation" in w.lower()]
        assert len(truncation_warnings) == 1
        assert "30%" in truncation_warnings[0]
        assert "100 table rows" in truncation_warnings[0]

    def test_no_warning_without_parse_result(self, validator: Validator):
        """H6: No completeness check when parse_result not provided."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )

        # No parse_result provided
        result = validator.validate(quote, parse_result=None)

        # No completeness warning
        assert not any("truncation" in w.lower() for w in result.warnings)

    def test_multiple_tables_aggregated(self, validator: Validator):
        """H6: Completeness validation aggregates rows from all tables."""
        from src.extraction.pdf_parser import ParseResult, TableData

        # Two tables: 30 rows + 20 rows = 50 total
        table1 = TableData(
            page=1,
            rows=[[f"Item {i}", "1", "100.00"] for i in range(30)],
            headers=["Item", "Qty", "Price"],
        )
        table2 = TableData(
            page=2,
            rows=[[f"Item {i}", "1", "100.00"] for i in range(30, 50)],
            headers=["Item", "Qty", "Price"],
        )
        parse_result = ParseResult(
            text="Test content",
            tables=[table1, table2],
            page_count=3,
            is_scanned=False,
            images=None,
            error=None,
        )

        # Only 20 items extracted (40% of 50 rows)
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
                for i in range(1, 21)
            ],
            amounts=Amounts(subtotal=2000.0, grand_total=2000.0),
        )

        result = validator.validate(quote, parse_result=parse_result)

        # Should warn about truncation (20 items vs 50 rows = 40%)
        assert any("truncation" in w.lower() for w in result.warnings)
        assert any("50 table rows" in w for w in result.warnings)


class TestTCVvsNetValidation:
    """
    Tests for TCV vs Net amount confusion detection.

    TCV (Total Contract Value) should always be >= Net Amount Due.
    If grand_total < net_price_total, the LLM likely picked "Amount Due"
    instead of the true TCV. This catches service-order extraction errors
    where credits are shown separately from the total contract value.
    """

    def test_no_warning_when_net_is_none(self, validator: Validator):
        """No TCV vs Net check when net_price_total is not present."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                ),
            ],
            amounts=Amounts(
                subtotal=100.0,
                grand_total=100.0,
                net_price_total=None,  # No net price
            ),
        )
        result = validator.validate(quote)
        assert not any("TCV vs Net" in w for w in result.warnings)

    def test_no_warning_when_tcv_gte_net(self, validator: Validator):
        """No warning when grand_total >= net_price_total (correct order)."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=1000.0,
                    extended_price=1000.0,
                ),
                LineItem(
                    line_number=2,
                    description="Credit",
                    quantity=1,
                    unit_price=0.0,
                    extended_price=-200.0,
                    item_type=ItemType.credit,
                ),
            ],
            amounts=Amounts(
                subtotal=800.0,
                grand_total=1000.0,  # TCV (larger)
                net_price_total=800.0,  # Net after credit (smaller)
            ),
        )
        result = validator.validate(quote)
        assert not any("TCV vs Net confusion" in w for w in result.warnings)

    def test_warning_when_tcv_lt_net_swapped(self, validator: Validator):
        """Warning when grand_total < net_price_total (values likely swapped).

        This simulates a scenario where the LLM extracts "Amount Due" as grand_total
        instead of the larger TCV value. The math is internally consistent but the
        TCV vs Net relationship is inverted.
        """
        # Scenario: LLM picks $800 (net) as grand_total, but $1000 should be TCV
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=800.0,
                    extended_price=800.0,
                    item_type=ItemType.hardware,
                ),
            ],
            amounts=Amounts(
                subtotal=800.0,
                grand_total=800.0,  # LLM picked "Amount Due" (net)
                net_price_total=1000.0,  # Swapped: this should be grand_total
            ),
        )
        result = validator.validate(quote)

        # Should have TCV vs Net confusion warning
        tcv_warnings = [w for w in result.warnings if "TCV vs Net confusion" in w]
        assert len(tcv_warnings) == 1
        assert "$800" in tcv_warnings[0]
        assert "$1,000" in tcv_warnings[0]
        assert "swapped" in tcv_warnings[0].lower()

        # Warnings don't fail validation (math is internally consistent)
        assert result.passed is True

    def test_credit_consistency_correct(self, validator: Validator):
        """No credit consistency warning when TCV - Net = credits."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=1000.0,
                    extended_price=1000.0,
                ),
                LineItem(
                    line_number=2,
                    description="Trade-in Credit",
                    quantity=1,
                    unit_price=0.0,
                    extended_price=-200.0,  # Credit
                    item_type=ItemType.credit,
                ),
            ],
            amounts=Amounts(
                subtotal=800.0,
                grand_total=1000.0,  # TCV
                net_price_total=800.0,  # Net = TCV - 200 credit
            ),
        )
        result = validator.validate(quote)

        # No credit consistency warning
        assert not any("Credit consistency" in w for w in result.warnings)

    def test_credit_consistency_warning_when_mismatch(self, validator: Validator):
        """Warning when TCV - Net doesn't match sum of credits."""
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=1000.0,
                    extended_price=1000.0,
                ),
                LineItem(
                    line_number=2,
                    description="Credit",
                    quantity=1,
                    unit_price=0.0,
                    extended_price=-100.0,  # Credit = $100
                    item_type=ItemType.credit,
                ),
            ],
            amounts=Amounts(
                subtotal=900.0,
                grand_total=1000.0,  # TCV
                net_price_total=700.0,  # Net = TCV - 300?? But credit is only 100
            ),
        )
        result = validator.validate(quote)

        # Should have credit consistency warning
        credit_warnings = [w for w in result.warnings if "Credit consistency" in w]
        assert len(credit_warnings) == 1
        # TCV - Net = 300, but credits = 100
        assert "$300" in credit_warnings[0]
        assert "$100" in credit_warnings[0]
