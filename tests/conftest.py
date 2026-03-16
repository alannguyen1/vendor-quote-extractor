"""
Pytest configuration and shared fixtures for Vendor Quote Extractor tests.

Provides:
- PDF fixture loading
- Mock LLM responses
- FastAPI TestClient
- Expected JSON output loading
"""

import os
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from src.api.routes import app

# Fixture directories
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_FIXTURES_DIR = FIXTURES_DIR
EXPECTED_DIR = FIXTURES_DIR / "expected"


def integration_provider_available() -> bool:
    """Return True when at least one real LLM provider key is configured."""
    if os.environ.get("PYTEST_RUN_INTEGRATION") == "1":
        return True
    return len(Settings().get_available_providers()) > 0


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless provider credentials are configured."""
    if integration_provider_available():
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests require configured LLM provider credentials"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient for API testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def pdf_fixtures_dir() -> Path:
    """Path to PDF fixtures directory."""
    return PDF_FIXTURES_DIR


@pytest.fixture
def mixed_category_pdf_bytes() -> bytes:
    """Load mixed_category_quote.pdf fixture."""
    pdf_path = PDF_FIXTURES_DIR / "mixed_category_quote.pdf"
    return pdf_path.read_bytes()


@pytest.fixture
def discounted_hardware_pdf_bytes() -> bytes:
    """Load discounted_hardware_quote.pdf fixture."""
    pdf_path = PDF_FIXTURES_DIR / "discounted_hardware_quote.pdf"
    return pdf_path.read_bytes()


@pytest.fixture
def enterprise_term_pdf_bytes() -> bytes:
    """Load enterprise_term_quote.pdf fixture."""
    pdf_path = PDF_FIXTURES_DIR / "enterprise_term_quote.pdf"
    return pdf_path.read_bytes()


@pytest.fixture
def service_order_with_credits_pdf_bytes() -> bytes:
    """Load service_order_with_credits.pdf fixture."""
    pdf_path = PDF_FIXTURES_DIR / "service_order_with_credits.pdf"
    return pdf_path.read_bytes()


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Load multi_page_quote.pdf fixture."""
    pdf_path = PDF_FIXTURES_DIR / "multi_page_quote.pdf"
    return pdf_path.read_bytes()


@pytest.fixture
def all_pdf_fixtures() -> dict[str, bytes]:
    """Load all PDF fixtures as a dictionary."""
    fixtures = {}
    for pdf_file in PDF_FIXTURES_DIR.glob("*.pdf"):
        fixtures[pdf_file.stem] = pdf_file.read_bytes()
    return fixtures


@pytest.fixture
def invalid_file_bytes() -> bytes:
    """Return bytes that are not a valid PDF."""
    return b"This is not a PDF file"


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Return minimal valid PDF structure but with no content."""
    # Minimal PDF that is technically valid but has no pages
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj
xref
0 3
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
trailer << /Size 3 /Root 1 0 R >>
startxref
115
%%EOF"""
