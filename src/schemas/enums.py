"""
Enumeration types for vendor quote data models.

These enums categorize line items and billing terms extracted from vendor quotes.
"""

from enum import Enum


class ItemType(str, Enum):
    """
    Classification of line items in vendor quotes.

    Used to categorize items for financial analysis and reporting.
    LLM extracts based on:
    - hardware: -HW suffix, physical equipment
    - software: LIC- prefix, licenses, subscriptions, SaaS
    - services: implementation, support, labor, professional services
    - discount/credit: negative amounts, rebates
    """

    hardware = "hardware"
    software = "software"
    services = "services"
    fee = "fee"
    tax = "tax"
    shipping = "shipping"
    discount = "discount"
    credit = "credit"


class TermType(str, Enum):
    """
    Billing period types for line items.

    Indicates how items are billed over time.
    """

    one_time = "one_time"
    monthly = "monthly"
    annual = "annual"
    multi_year = "multi_year"
