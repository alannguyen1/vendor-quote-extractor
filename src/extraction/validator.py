"""
Validation module for extracted vendor quote data.

Implements math checks, entity validation, and business rules
to verify extraction accuracy and flag issues for review.

H6: Adds completeness validation to detect potential truncation
by comparing extracted line items to expected table rows.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import get_settings
from src.schemas import VendorQuote

if TYPE_CHECKING:
    from src.extraction.pdf_parser import ParseResult

logger = logging.getLogger("vendor_quote_extractor.validator")


@dataclass
class ValidationResult:
    """
    Result of validating an extracted vendor quote.

    Attributes:
        passed: True if all error checks passed (warnings don't fail)
        errors: List of validation error messages (blocking)
        warnings: List of validation warning messages (non-blocking)
    """

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Validator:
    """
    Validates extracted vendor quote data for accuracy and completeness.

    Implements the validation rules from validation-rules.md spec:
    - Math tolerance: $0.01 for all calculations
    - Line items sum must equal subtotal
    - Grand total must equal subtotal + tax + shipping - discounts
    - Extended price must equal quantity * unit_price per line item
    - Tax rate sanity check (warning if > 20%)
    - Vendor name required (min 2 chars)
    - Date logic check (valid_until >= quote_date)
    """

    # Tolerance for floating-point math comparisons
    TOLERANCE = 0.01

    def validate(
        self,
        quote: VendorQuote,
        parse_result: "ParseResult | None" = None,
    ) -> ValidationResult:
        """
        Validate an extracted vendor quote.

        Args:
            quote: The extracted VendorQuote to validate
            parse_result: Optional ParseResult for completeness validation (H6)

        Returns:
            ValidationResult with passed status, errors, and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Run all validation checks
        self._validate_line_items_sum(quote, errors)
        self._validate_grand_total(quote, errors)
        self._validate_line_item_math(quote, errors)
        self._validate_tax_rate(quote, warnings)
        self._validate_vendor(quote, errors)
        self._validate_item_classification(quote, warnings)
        self._validate_date_logic(quote, errors)
        self._validate_currency(quote, errors)
        self._validate_tcv_vs_net(quote, warnings)

        # H6: Completeness validation - detect potential truncation
        if parse_result is not None:
            self._validate_line_item_completeness(quote, parse_result, warnings)

        passed = len(errors) == 0

        if errors:
            logger.warning(f"Validation failed with {len(errors)} errors: {errors}")
        if warnings:
            logger.info(f"Validation has {len(warnings)} warnings: {warnings}")

        return ValidationResult(passed=passed, errors=errors, warnings=warnings)

    def _validate_line_items_sum(self, quote: VendorQuote, errors: list[str]) -> None:
        """Check that sum of line item extended prices equals subtotal."""
        line_sum = sum(item.extended_price for item in quote.line_items)
        subtotal = quote.amounts.subtotal

        if abs(line_sum - subtotal) > self.TOLERANCE:
            diff = abs(line_sum - subtotal)
            errors.append(
                f"Line items sum (${line_sum:.2f}) does not match "
                f"subtotal (${subtotal:.2f}), difference: ${diff:.2f}"
            )

    def _validate_grand_total(self, quote: VendorQuote, errors: list[str]) -> None:
        """Check grand total = subtotal + tax + shipping - discounts."""
        amounts = quote.amounts
        calculated = (
            amounts.subtotal + amounts.tax + amounts.shipping - amounts.discounts
        )

        if abs(calculated - amounts.grand_total) > self.TOLERANCE:
            errors.append(
                f"Calculated total (${calculated:.2f}) does not match "
                f"grand_total (${amounts.grand_total:.2f}), "
                f"difference: ${abs(calculated - amounts.grand_total):.2f}"
            )

    def _validate_line_item_math(self, quote: VendorQuote, errors: list[str]) -> None:
        """Check quantity * unit_price = extended_price for each line item."""
        for item in quote.line_items:
            expected = item.quantity * item.unit_price

            # Skip validation for credits/discounts (negative extended prices)
            if item.extended_price < 0:
                continue

            if abs(expected - item.extended_price) > self.TOLERANCE:
                errors.append(
                    f"Line {item.line_number}: quantity ({item.quantity}) × "
                    f"unit_price (${item.unit_price:.2f}) = ${expected:.2f}, "
                    f"but extended_price is ${item.extended_price:.2f}"
                )

    def _validate_tax_rate(self, quote: VendorQuote, warnings: list[str]) -> None:
        """Warn if tax rate exceeds configured threshold (default 20%) of subtotal."""
        settings = get_settings()
        if quote.amounts.subtotal > 0:
            tax_rate = quote.amounts.tax / quote.amounts.subtotal
            if tax_rate > settings.tax_rate_warning_threshold:
                tax = quote.amounts.tax
                subtotal = quote.amounts.subtotal
                warnings.append(
                    f"High tax rate detected: {tax_rate:.1%} "
                    f"(tax ${tax:.2f} / subtotal ${subtotal:.2f})"
                )

    def _validate_vendor(self, quote: VendorQuote, errors: list[str]) -> None:
        """Require vendor name with minimum 2 characters."""
        if not quote.vendor or not quote.vendor.name:
            errors.append("Vendor name is required but missing")
        elif len(quote.vendor.name.strip()) < 2:
            errors.append(
                f"Vendor name too short: '{quote.vendor.name}' (minimum 2 characters)"
            )

    def _validate_item_classification(
        self, quote: VendorQuote, warnings: list[str]
    ) -> None:
        """Flag items missing item_type classification."""
        unclassified = [
            item.line_number for item in quote.line_items if item.item_type is None
        ]

        if unclassified:
            warnings.append(
                f"{len(unclassified)} line item(s) missing classification: "
                f"lines {unclassified[:5]}{'...' if len(unclassified) > 5 else ''}"
            )

    def _validate_date_logic(self, quote: VendorQuote, errors: list[str]) -> None:
        """Check that valid_until date is not before quote_date."""
        if quote.quote_date and quote.valid_until:
            if quote.valid_until < quote.quote_date:
                errors.append(
                    f"Invalid date range: valid_until ({quote.valid_until}) "
                    f"is before quote_date ({quote.quote_date})"
                )

    def _validate_currency(self, quote: VendorQuote, errors: list[str]) -> None:
        """Reject non-USD currency quotes for MVP."""
        if quote.currency.upper() != "USD":
            errors.append(
                f"Currency '{quote.currency}' not supported. "
                "Only USD quotes are accepted."
            )

    def _validate_tcv_vs_net(self, quote: VendorQuote, warnings: list[str]) -> None:
        """
        Detect potential TCV vs Net confusion in extracted amounts.

        TCV (Total Contract Value) should always be >= Net Amount Due because:
        - TCV = full contract value including all line items
        - Net = TCV minus credits/trade-ins (what's actually payable)

        If grand_total < net_price_total, the values are likely swapped
        (common LLM error when documents prominently show "Amount Due").

        Example:
        - TCV (correct grand_total): $597,312
        - Net (correct net_price_total): $552,012
        - Credit: $45,300

        If extracted as grand_total=$552,012, net_price_total=$597,312,
        the LLM picked the "Amount Due" label instead of the TCV.
        """
        amounts = quote.amounts

        # Only check if both values exist
        if amounts.net_price_total is None:
            return

        grand_total = amounts.grand_total
        net_total = amounts.net_price_total

        # Check 1: TCV should be >= Net (if credits exist)
        if grand_total < net_total:
            diff = net_total - grand_total
            warnings.append(
                f"TCV vs Net confusion detected: grand_total (${grand_total:,.2f}) "
                f"< net_price_total (${net_total:,.2f}). "
                f"Values may be swapped. Difference: ${diff:,.2f}. "
                f"Review document - grand_total should be Total Contract Value, "
                f"not 'Amount Due' or 'Net'."
            )
            logger.warning(
                f"TCV vs Net warning: grand_total ${grand_total:,.2f} < "
                f"net_price_total ${net_total:,.2f} - likely swapped values"
            )

        # Check 2: If credits exist, verify relationship
        credit_sum = sum(
            abs(item.extended_price)
            for item in quote.line_items
            if item.extended_price < 0
        )

        if credit_sum > 0 and grand_total >= net_total:
            # Credits exist - verify TCV - Net ≈ credits
            expected_diff = grand_total - net_total
            if abs(expected_diff - credit_sum) > self.TOLERANCE * 100:
                # Allow larger tolerance for credit consistency (1% of credit value)
                credit_tolerance = max(credit_sum * 0.01, self.TOLERANCE)
                if abs(expected_diff - credit_sum) > credit_tolerance:
                    warnings.append(
                        f"Credit consistency: grand_total - net_price_total = "
                        f"${expected_diff:,.2f}, but credits = ${credit_sum:,.2f}. "
                        f"Verify credit handling is correct."
                    )

    def _validate_line_item_completeness(
        self,
        quote: VendorQuote,
        parse_result: "ParseResult",
        warnings: list[str],
    ) -> None:
        """
        H6: Check for potential truncation by comparing extracted line items
        to expected table rows from PDF parsing.

        Warns if extracted line items are significantly fewer than expected,
        which may indicate context window truncation during LLM extraction.

        Args:
            quote: The extracted VendorQuote
            parse_result: ParseResult containing table data
            warnings: List to append warnings to

        Uses 80% threshold - warns if extracted items < 80% of expected rows.
        This accounts for header rows and summary rows in tables.
        """
        # Count total table rows from parse result
        expected_rows = sum(len(t.rows) for t in parse_result.tables)

        if expected_rows == 0:
            # No tables detected - can't validate completeness
            return

        actual_items = len(quote.line_items)

        # Use 80% threshold to account for header/summary rows
        completeness_threshold = 0.80
        min_expected = int(expected_rows * completeness_threshold)

        if actual_items < min_expected:
            if expected_rows > 0:
                completeness_pct = (actual_items / expected_rows) * 100
            else:
                completeness_pct = 0
            warnings.append(
                f"Possible truncation: extracted {actual_items} line items "
                f"but found {expected_rows} table rows "
                f"({completeness_pct:.0f}% completeness). "
                f"Consider using a provider with larger context window."
            )
            logger.warning(
                f"H6 completeness warning: {actual_items}/{expected_rows} items "
                f"({completeness_pct:.0f}%) - possible truncation"
            )
