"""
Mock data and fixtures for testing.

This module provides pre-built VendorQuote responses that match expected
extractions from the 5 test fixture PDFs. Using these mocks in unit tests:
- Eliminates API costs (no real LLM calls)
- Removes rate limiting concerns
- Provides deterministic, fast test execution
- Allows testing edge cases without actual documents

Usage:
    from tests.mocks import (
        MIXED_CATEGORY_QUOTE,
        DISCOUNTED_HARDWARE_QUOTE,
        get_mock_extractor,
    )

    # Use pre-built fixtures directly
    assert MIXED_CATEGORY_QUOTE.amounts.grand_total == 32198.00

    # Or get a fully mocked LLMExtractor
    extractor = get_mock_extractor(MIXED_CATEGORY_QUOTE)
"""

from tests.mocks.llm_responses import (
    DISCOUNTED_HARDWARE_QUOTE,
    ENTERPRISE_TERM_QUOTE,
    MINIMAL_QUOTE,
    MIXED_CATEGORY_QUOTE,
    MULTI_PAGE_QUOTE,
    SERVICE_ORDER_WITH_CREDITS_QUOTE,
    get_mock_extractor,
)

__all__ = [
    "MIXED_CATEGORY_QUOTE",
    "DISCOUNTED_HARDWARE_QUOTE",
    "ENTERPRISE_TERM_QUOTE",
    "SERVICE_ORDER_WITH_CREDITS_QUOTE",
    "MULTI_PAGE_QUOTE",
    "MINIMAL_QUOTE",
    "get_mock_extractor",
]
