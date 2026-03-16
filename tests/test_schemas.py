"""
Tests for Pydantic schema models.

Verifies model instantiation, field validation, and serialization.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.schemas import (
    Amounts,
    CommercialTerms,
    Entity,
    ExtractionMetadata,
    ItemType,
    LineItem,
    TermType,
    VendorQuote,
)


class TestItemType:
    """Tests for ItemType enum."""

    def test_hardware_value(self):
        assert ItemType.hardware.value == "hardware"

    def test_software_value(self):
        assert ItemType.software.value == "software"

    def test_all_values_exist(self):
        expected = {
            "hardware",
            "software",
            "services",
            "fee",
            "tax",
            "shipping",
            "discount",
            "credit",
        }
        actual = {item.value for item in ItemType}
        assert actual == expected


class TestTermType:
    """Tests for TermType enum."""

    def test_all_values_exist(self):
        expected = {"one_time", "monthly", "annual", "multi_year"}
        actual = {term.value for term in TermType}
        assert actual == expected


class TestEntity:
    """Tests for Entity model."""

    def test_minimal_entity(self):
        entity = Entity(name="Test Corp")
        assert entity.name == "Test Corp"
        assert entity.address is None
        assert entity.phone is None
        assert entity.email is None

    def test_full_entity(self):
        entity = Entity(
            name="Acme Inc",
            address="123 Main St",
            phone="555-1234",
            email="info@acme.com",
        )
        assert entity.name == "Acme Inc"
        assert entity.address == "123 Main St"

    def test_name_min_length(self):
        with pytest.raises(ValidationError) as exc_info:
            Entity(name="A")  # Too short - min 2 chars
        assert "string_too_short" in str(exc_info.value)

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Entity(name="")


class TestLineItem:
    """Tests for LineItem model."""

    def test_minimal_line_item(self):
        item = LineItem(
            line_number=1,
            description="Test Product",
            quantity=1,
            unit_price=100.0,
            extended_price=100.0,
        )
        assert item.line_number == 1
        assert item.item_type is None  # Optional
        assert item.confidence == 1.0  # Default

    def test_full_line_item(self):
        item = LineItem(
            line_number=5,
            description="Enterprise License",
            sku="LIC-ENT-001",
            manufacturer="Vendor Corp",
            quantity=10,
            unit_price=500.0,
            extended_price=5000.0,
            item_type=ItemType.software,
            term_type=TermType.annual,
            term_length_months=12,
            confidence=0.95,
        )
        assert item.sku == "LIC-ENT-001"
        assert item.item_type == ItemType.software

    def test_negative_extended_price_allowed(self):
        """Credits/discounts have negative extended prices."""
        item = LineItem(
            line_number=1,
            description="Credit Adjustment",
            quantity=1,
            unit_price=0.0,
            extended_price=-500.0,
            item_type=ItemType.credit,
        )
        assert item.extended_price == -500.0

    def test_line_number_min_value(self):
        with pytest.raises(ValidationError):
            LineItem(
                line_number=0,  # Must be >= 1
                description="Test",
                quantity=1,
                unit_price=100.0,
                extended_price=100.0,
            )

    def test_quantity_min_value(self):
        with pytest.raises(ValidationError):
            LineItem(
                line_number=1,
                description="Test",
                quantity=0,  # Must be >= 1
                unit_price=100.0,
                extended_price=0.0,
            )

    def test_unit_price_min_value(self):
        with pytest.raises(ValidationError):
            LineItem(
                line_number=1,
                description="Test",
                quantity=1,
                unit_price=-10.0,  # Must be >= 0
                extended_price=-10.0,
            )

    def test_confidence_range(self):
        # Valid range
        item = LineItem(
            line_number=1,
            description="Test",
            quantity=1,
            unit_price=100.0,
            extended_price=100.0,
            confidence=0.5,
        )
        assert item.confidence == 0.5

        # Below range
        with pytest.raises(ValidationError):
            LineItem(
                line_number=1,
                description="Test",
                quantity=1,
                unit_price=100.0,
                extended_price=100.0,
                confidence=-0.1,
            )

        # Above range
        with pytest.raises(ValidationError):
            LineItem(
                line_number=1,
                description="Test",
                quantity=1,
                unit_price=100.0,
                extended_price=100.0,
                confidence=1.1,
            )


class TestAmounts:
    """Tests for Amounts model."""

    def test_minimal_amounts(self):
        amounts = Amounts(subtotal=1000.0, grand_total=1100.0)
        assert amounts.subtotal == 1000.0
        assert amounts.discounts == 0.0  # Default
        assert amounts.tax == 0.0  # Default

    def test_full_amounts(self):
        amounts = Amounts(
            subtotal=1000.0,
            discounts=100.0,
            shipping=50.0,
            tax=85.5,
            grand_total=1035.5,
            list_price_total=1200.0,
            net_price_total=1035.5,
            selected_finance_total="net_price_total",
        )
        assert amounts.selected_finance_total == "net_price_total"

    def test_subtotal_non_negative(self):
        with pytest.raises(ValidationError):
            Amounts(subtotal=-100.0, grand_total=100.0)

    def test_selected_finance_total_valid_values(self):
        """Verify only 'grand_total' or 'net_price_total' are accepted."""
        # Valid: grand_total
        amounts = Amounts(
            subtotal=1000.0,
            grand_total=1100.0,
            selected_finance_total="grand_total",
        )
        assert amounts.selected_finance_total == "grand_total"

        # Valid: net_price_total
        amounts = Amounts(
            subtotal=1000.0,
            grand_total=1100.0,
            selected_finance_total="net_price_total",
        )
        assert amounts.selected_finance_total == "net_price_total"

        # Valid: None (default)
        amounts = Amounts(subtotal=1000.0, grand_total=1100.0)
        assert amounts.selected_finance_total is None

    def test_selected_finance_total_rejects_invalid_values(self):
        """Verify invalid values for selected_finance_total are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Amounts(
                subtotal=1000.0,
                grand_total=1100.0,
                selected_finance_total="invalid_total",
            )
        assert "selected_finance_total" in str(exc_info.value)

        with pytest.raises(ValidationError):
            Amounts(
                subtotal=1000.0,
                grand_total=1100.0,
                selected_finance_total="list_price_total",  # Not a valid option
            )


class TestCommercialTerms:
    """Tests for CommercialTerms model."""

    def test_empty_terms(self):
        terms = CommercialTerms()
        assert terms.payment_terms is None
        assert terms.subscription_term_months is None

    def test_full_terms(self):
        terms = CommercialTerms(
            payment_terms="Net 30",
            subscription_term_months=36,
            auto_renew=True,
            coverage_start=date(2024, 1, 1),
            coverage_end=date(2026, 12, 31),
        )
        assert terms.subscription_term_months == 36

    def test_term_months_min_value(self):
        with pytest.raises(ValidationError):
            CommercialTerms(subscription_term_months=0)  # Must be >= 1


class TestExtractionMetadata:
    """Tests for ExtractionMetadata model."""

    def test_creation(self):
        metadata = ExtractionMetadata(
            processing_time_ms=5000,
            model_used="gpt-4o-2024-08-06",
            overall_confidence=0.92,
            validation_passed=True,
            validation_errors=[],
            requires_review=False,
        )
        assert metadata.processing_time_ms == 5000
        assert metadata.overall_confidence == 0.92

    def test_with_errors(self):
        metadata = ExtractionMetadata(
            processing_time_ms=3000,
            model_used="gpt-4o-2024-08-06",
            overall_confidence=0.75,
            validation_passed=False,
            validation_errors=["Line items sum mismatch", "Tax too high"],
            requires_review=True,
        )
        assert len(metadata.validation_errors) == 2


class TestVendorQuote:
    """Tests for VendorQuote root model."""

    def test_minimal_quote(self):
        quote = VendorQuote(
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Product",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        assert quote.vendor.name == "Test Vendor"
        assert quote.currency == "USD"  # Default
        assert quote.extraction_metadata is None  # Optional

    def test_full_quote(self):
        quote = VendorQuote(
            quote_id="Q-2024-001",
            quote_date=date(2024, 1, 15),
            valid_until=date(2024, 2, 15),
            currency="USD",
            vendor=Entity(name="Acme Corp", email="sales@acme.com"),
            customer=Entity(name="Customer Inc"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Widget",
                    quantity=10,
                    unit_price=50.0,
                    extended_price=500.0,
                    item_type=ItemType.hardware,
                )
            ],
            amounts=Amounts(
                subtotal=500.0,
                tax=50.0,
                grand_total=550.0,
            ),
            commercial_terms=CommercialTerms(payment_terms="Net 30"),
        )
        assert quote.quote_id == "Q-2024-001"
        assert quote.customer.name == "Customer Inc"

    def test_empty_line_items_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            VendorQuote(
                vendor=Entity(name="Test"),
                line_items=[],  # Must have at least 1
                amounts=Amounts(subtotal=0.0, grand_total=0.0),
            )
        assert "too_short" in str(exc_info.value)

    def test_serialization(self):
        quote = VendorQuote(
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        data = quote.model_dump()
        assert "vendor" in data
        assert "line_items" in data
        assert data["vendor"]["name"] == "Test"

    def test_json_serialization(self):
        quote = VendorQuote(
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        json_str = quote.model_dump_json()
        assert '"name":"Test"' in json_str
