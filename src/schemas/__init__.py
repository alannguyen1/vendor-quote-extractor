"""
Pydantic schemas for Vendor Quote Extractor.

Provides data models for vendor quote extraction:
- Enums: ItemType, TermType
- Models: Entity, LineItem, Amounts, CommercialTerms, ExtractionMetadata, VendorQuote

Usage:
    from src.schemas import VendorQuote, LineItem, ItemType

    quote = VendorQuote(
        vendor=Entity(name="Acme Corp"),
        line_items=[
            LineItem(
                line_number=1, description="Widget",
                quantity=1, unit_price=100.0, extended_price=100.0
            )
        ],
        amounts=Amounts(subtotal=100.0, grand_total=100.0)
    )
"""

from src.schemas.enums import ItemType, TermType
from src.schemas.models import (
    Amounts,
    CommercialTerms,
    Entity,
    ExtractBreakdown,
    ExtractionMetadata,
    LineItem,
    LineItemBatch,
    ParseBreakdown,
    VendorQuote,
    VendorQuoteHeader,
)

__all__ = [
    # Enums
    "ItemType",
    "TermType",
    # Models
    "Entity",
    "LineItem",
    "LineItemBatch",
    "Amounts",
    "CommercialTerms",
    "ParseBreakdown",
    "ExtractBreakdown",
    "ExtractionMetadata",
    "VendorQuote",
    "VendorQuoteHeader",
]
