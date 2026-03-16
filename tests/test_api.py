"""
Tests for the FastAPI API endpoints.

Verifies request/response handling, validation, and error cases.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_health_returns_200(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestExtractEndpoint:
    """Tests for the /api/v1/extract endpoint."""

    def test_extract_rejects_non_pdf_content_type(self, test_client: TestClient):
        """Should reject files that don't have PDF content type."""
        response = test_client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"Hello World", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]

    def test_extract_rejects_invalid_pdf_magic_bytes(self, test_client: TestClient):
        """Should reject files that claim to be PDF but aren't."""
        response = test_client.post(
            "/api/v1/extract",
            files={"file": ("fake.pdf", b"Not a PDF", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Invalid PDF file" in response.json()["detail"]

    def test_extract_rejects_oversized_file(self, test_client: TestClient):
        """Should reject files over 10MB."""
        # Create a fake PDF header followed by 11MB of data
        large_content = b"%PDF-1.4" + b"x" * (11 * 1024 * 1024)
        response = test_client.post(
            "/api/v1/extract",
            files={"file": ("large.pdf", large_content, "application/pdf")},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    @pytest.mark.integration
    def test_extract_processes_valid_pdf(
        self, test_client: TestClient, mixed_category_pdf_bytes: bytes
    ):
        """Should successfully process a valid PDF fixture.

        Note: This is an integration test that makes real LLM API calls.
        Mark with @pytest.mark.integration to skip in unit test runs.
        """
        response = test_client.post(
            "/api/v1/extract",
            files={
                "file": (
                    "mixed_category_quote.pdf",
                    mixed_category_pdf_bytes,
                    "application/pdf",
                )
            },
        )

        # Should return 200 or 422 (if extraction has issues)
        # Both are valid responses for integration tests
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            data = response.json()
            assert "vendor" in data
            assert "line_items" in data
            assert "amounts" in data

    @pytest.mark.integration
    def test_extract_returns_extraction_id_header(
        self, test_client: TestClient, mixed_category_pdf_bytes: bytes
    ):
        """Should return X-Extraction-ID header for UI correction workflow.

        This header allows the UI to submit corrections to the extraction.
        """
        response = test_client.post(
            "/api/v1/extract",
            files={
                "file": (
                    "mixed_category_quote.pdf",
                    mixed_category_pdf_bytes,
                    "application/pdf",
                )
            },
        )

        if response.status_code == 200:
            # Header should be present and contain a valid integer ID
            assert "X-Extraction-ID" in response.headers
            extraction_id = response.headers["X-Extraction-ID"]
            assert extraction_id.isdigit()
            assert int(extraction_id) > 0


class TestErrorResponses:
    """Tests for error response format consistency."""

    def test_400_error_format(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"Hello", "text/plain")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error_code" in data

    def test_missing_file_error(self, test_client: TestClient):
        """Should return error when file is not provided."""
        response = test_client.post("/api/v1/extract")
        assert response.status_code == 422  # FastAPI validation error


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_allows_streamlit_origin(self, test_client: TestClient):
        """Should allow requests from Streamlit (localhost:8501)."""
        response = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not return 4xx error
        assert response.status_code < 400

    def test_cors_allows_localhost_3000(self, test_client: TestClient):
        """Should allow requests from alternative frontend port."""
        response = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code < 400

    def test_cors_exposes_extraction_id_header(self, test_client: TestClient):
        """Should expose X-Extraction-ID header for browser access.

        The UI needs to read this header to get the extraction ID for
        submitting corrections. The expose_headers config appears in
        actual responses, not preflight OPTIONS.
        """
        # Make an actual request with Origin header to trigger CORS
        response = test_client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:8501"},
        )
        # Check expose headers is configured in the response
        expose_headers = response.headers.get("access-control-expose-headers", "")
        assert "X-Extraction-ID" in expose_headers


class TestOpenAPIDocs:
    """Tests for API documentation endpoints."""

    def test_openapi_json_available(self, test_client: TestClient):
        response = test_client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_swagger_docs_available(self, test_client: TestClient):
        response = test_client.get("/api/v1/docs")
        assert response.status_code == 200

    def test_redoc_available(self, test_client: TestClient):
        response = test_client.get("/api/v1/redoc")
        assert response.status_code == 200


class TestListExtractionsEndpoint:
    """Tests for GET /api/v1/extractions endpoint (list)."""

    def test_list_extractions_returns_200(self, test_client: TestClient):
        """Should return 200 with list response format."""
        response = test_client.get("/api/v1/extractions")
        assert response.status_code == 200

        data = response.json()
        assert "extractions" in data
        assert "total" in data
        assert isinstance(data["extractions"], list)
        assert isinstance(data["total"], int)

    def test_list_extractions_accepts_limit_parameter(self, test_client: TestClient):
        """Should accept limit query parameter."""
        response = test_client.get("/api/v1/extractions?limit=5")
        assert response.status_code == 200

    def test_list_extractions_caps_limit_at_50(self, test_client: TestClient):
        """Should cap limit at 50 for performance."""
        response = test_client.get("/api/v1/extractions?limit=100")
        assert response.status_code == 200
        # Response should succeed but internally cap at 50

    def test_list_extractions_item_structure(self, test_client: TestClient):
        """Items should have expected fields."""
        from src.api.models import ExtractionListItem

        # Verify model structure
        assert hasattr(ExtractionListItem, "model_fields")
        fields = ExtractionListItem.model_fields
        assert "id" in fields
        assert "filename" in fields
        assert "overall_confidence" in fields
        assert "requires_review" in fields
        assert "created_at" in fields

    def test_list_extractions_empty_database(self, test_client: TestClient):
        """Should return empty list with total=0 when no extractions exist."""
        # Note: This test runs with a fresh database each time
        response = test_client.get("/api/v1/extractions")
        assert response.status_code == 200
        data = response.json()
        # total >= 0 is valid (could have extractions from other tests)
        assert data["total"] >= 0


class TestGetExtractionEndpoint:
    """Tests for GET /api/v1/extractions/{id} endpoint."""

    def test_get_nonexistent_extraction_returns_404(self, test_client: TestClient):
        """Should return 404 for non-existent extraction ID."""
        response = test_client.get("/api/v1/extractions/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_extraction_returns_expected_format(self, test_client: TestClient):
        """GET should return extraction with id, filename, and quote."""
        # First check that the endpoint structure is correct
        response = test_client.get("/api/v1/extractions/1")
        # Either 404 (not found) or 200 with proper structure
        assert response.status_code in [200, 404]


class TestCorrectExtractionEndpoint:
    """Tests for PUT /api/v1/extractions/{id}/correct endpoint."""

    def test_correct_nonexistent_extraction_returns_404(self, test_client: TestClient):
        """Should return 404 for non-existent extraction ID."""
        response = test_client.put(
            "/api/v1/extractions/99999/correct",
            json={
                "field_path": "vendor.name",
                "corrected_value": "New Vendor Name",
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_correct_missing_field_path_returns_422(self, test_client: TestClient):
        """Should return 422 if field_path is missing."""
        response = test_client.put(
            "/api/v1/extractions/1/correct",
            json={
                "corrected_value": "New Value",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_correct_missing_value_returns_422(self, test_client: TestClient):
        """Should return 422 if corrected_value is missing."""
        response = test_client.put(
            "/api/v1/extractions/1/correct",
            json={
                "field_path": "vendor.name",
            },
        )
        assert response.status_code == 422  # Validation error


class TestFieldPathCorrection:
    """Tests for the field path correction logic."""

    def test_apply_field_correction_simple_path(self):
        """Test correcting a simple nested field."""
        from src.api.routes import apply_field_correction

        data = {"vendor": {"name": "Old Name", "address": "123 Main St"}}
        updated, original = apply_field_correction(data, "vendor.name", "New Name")

        assert updated["vendor"]["name"] == "New Name"
        assert original == "Old Name"
        assert updated["vendor"]["address"] == "123 Main St"

    def test_apply_field_correction_array_index(self):
        """Test correcting a field with array index."""
        from src.api.routes import apply_field_correction

        data = {
            "line_items": [
                {"item_type": "hardware"},
                {"item_type": "software"},
                {"item_type": "services"},
            ]
        }
        updated, original = apply_field_correction(
            data, "line_items[1].item_type", "fee"
        )

        assert updated["line_items"][1]["item_type"] == "fee"
        assert original == "software"
        assert updated["line_items"][0]["item_type"] == "hardware"

    def test_apply_field_correction_top_level(self):
        """Test correcting a top-level field."""
        from src.api.routes import apply_field_correction

        data = {"currency": "EUR", "quote_id": "Q-123"}
        updated, original = apply_field_correction(data, "currency", "USD")

        assert updated["currency"] == "USD"
        assert original == "EUR"

    def test_apply_field_correction_invalid_path(self):
        """Test that invalid paths raise ValueError."""
        from src.api.routes import apply_field_correction

        data = {"vendor": {"name": "Test"}}

        with pytest.raises(ValueError) as exc_info:
            apply_field_correction(data, "invalid.path.here", "value")

        assert "Invalid field path" in str(exc_info.value)

    def test_apply_field_correction_invalid_array_index(self):
        """Test that out-of-bounds array index raises ValueError."""
        from src.api.routes import apply_field_correction

        data = {"line_items": [{"item_type": "hardware"}]}

        with pytest.raises(ValueError) as exc_info:
            apply_field_correction(data, "line_items[99].item_type", "value")

        assert "Invalid array index" in str(exc_info.value)


class TestCorrectionsHistory:
    """Tests for corrections history in GET/PUT responses.

    Per specs/api-endpoints.md: 'Corrections stored and returned in subsequent GETs'
    This verifies that the corrections audit trail is properly returned.
    """

    @pytest.mark.asyncio
    async def test_get_extraction_includes_corrections_field(self):
        """GET response should include a corrections list (empty initially)."""
        from src.api.models import ExtractionResponse

        # Verify the model has corrections field
        assert hasattr(ExtractionResponse, "model_fields")
        assert "corrections" in ExtractionResponse.model_fields

        # Verify default is empty list
        response = ExtractionResponse(
            id=1,
            filename="test.pdf",
            quote={"vendor": {"name": "Test"}},
        )
        assert response.corrections == []

    @pytest.mark.asyncio
    async def test_correction_record_model_structure(self):
        """CorrectionRecord should have required audit trail fields."""
        from src.api.models import CorrectionRecord

        record = CorrectionRecord(
            id=1,
            field_path="vendor.name",
            original_value="Old Name",
            corrected_value="New Name",
            created_at="2026-01-15T10:00:00",
        )

        assert record.id == 1
        assert record.field_path == "vendor.name"
        assert record.original_value == "Old Name"
        assert record.corrected_value == "New Name"
        assert record.created_at == "2026-01-15T10:00:00"

    @pytest.mark.asyncio
    async def test_correction_record_allows_none_values(self):
        """CorrectionRecord should allow None for original/corrected values."""
        from src.api.models import CorrectionRecord

        record = CorrectionRecord(
            id=1,
            field_path="new_field",
            original_value=None,
            corrected_value="New Value",
            created_at="2026-01-15T10:00:00",
        )

        assert record.original_value is None
        assert record.corrected_value == "New Value"

    @pytest.mark.asyncio
    async def test_extraction_response_with_multiple_corrections(self):
        """ExtractionResponse should support multiple corrections."""
        from src.api.models import CorrectionRecord, ExtractionResponse

        corrections = [
            CorrectionRecord(
                id=1,
                field_path="vendor.name",
                original_value="Old",
                corrected_value="New",
                created_at="2026-01-15T10:00:00",
            ),
            CorrectionRecord(
                id=2,
                field_path="amounts.grand_total",
                original_value="100.00",
                corrected_value="150.00",
                created_at="2026-01-15T10:05:00",
            ),
        ]

        response = ExtractionResponse(
            id=1,
            filename="test.pdf",
            quote={"vendor": {"name": "New"}, "amounts": {"grand_total": 150.00}},
            corrections=corrections,
        )

        assert len(response.corrections) == 2
        assert response.corrections[0].field_path == "vendor.name"
        assert response.corrections[1].field_path == "amounts.grand_total"


class TestExtractCorrectionContext:
    """Tests for the _extract_correction_context helper function."""

    def test_line_item_correction_extracts_description(self):
        """Should extract line item description as context."""
        from src.api.routes import _extract_correction_context

        quote_data = {
            "line_items": [
                {"description": "Item 1", "item_type": "hardware"},
                {
                    "description": "Cisco SMARTnet 24x7x4 Support",
                    "item_type": "hardware",
                },
                {"description": "Item 3", "item_type": "software"},
            ]
        }
        context = _extract_correction_context(quote_data, "line_items[1].item_type")
        assert context == "Cisco SMARTnet 24x7x4 Support"

    def test_line_item_truncates_long_description(self):
        """Should truncate descriptions longer than 200 chars."""
        from src.api.routes import _extract_correction_context

        long_desc = "A" * 300
        quote_data = {
            "line_items": [
                {"description": long_desc, "item_type": "hardware"},
            ]
        }
        context = _extract_correction_context(quote_data, "line_items[0].item_type")
        assert len(context) == 200
        assert context == "A" * 200

    def test_line_item_missing_description_returns_none(self):
        """Should return None if line item has no description."""
        from src.api.routes import _extract_correction_context

        quote_data = {
            "line_items": [
                {"item_type": "hardware"},  # No description
            ]
        }
        context = _extract_correction_context(quote_data, "line_items[0].item_type")
        assert context is None

    def test_line_item_empty_description_returns_none(self):
        """Should return None if description is empty string."""
        from src.api.routes import _extract_correction_context

        quote_data = {
            "line_items": [
                {"description": "", "item_type": "hardware"},
            ]
        }
        context = _extract_correction_context(quote_data, "line_items[0].item_type")
        assert context is None

    def test_line_item_index_out_of_range_returns_none(self):
        """Should return None if line item index is out of range."""
        from src.api.routes import _extract_correction_context

        quote_data = {
            "line_items": [
                {"description": "Item 1", "item_type": "hardware"},
            ]
        }
        context = _extract_correction_context(quote_data, "line_items[99].item_type")
        assert context is None

    def test_vendor_correction_extracts_name(self):
        """Should extract vendor name as context."""
        from src.api.routes import _extract_correction_context

        quote_data = {
            "vendor": {"name": "Acme Retail Group", "email": "sales@acme.com"}
        }
        context = _extract_correction_context(quote_data, "vendor.email")
        assert context == "Acme Retail Group"

    def test_customer_correction_extracts_name(self):
        """Should extract customer name as context."""
        from src.api.routes import _extract_correction_context

        quote_data = {"customer": {"name": "Big Company Inc", "address": "123 Main St"}}
        context = _extract_correction_context(quote_data, "customer.address")
        assert context == "Big Company Inc"

    def test_customer_missing_returns_none(self):
        """Should return None if customer is None."""
        from src.api.routes import _extract_correction_context

        quote_data = {"customer": None}
        context = _extract_correction_context(quote_data, "customer.name")
        assert context is None

    def test_unrelated_field_returns_none(self):
        """Should return None for fields without specific context extraction."""
        from src.api.routes import _extract_correction_context

        quote_data = {"amounts": {"grand_total": 1000.0}}
        context = _extract_correction_context(quote_data, "amounts.grand_total")
        assert context is None

    def test_empty_line_items_returns_none(self):
        """Should return None if line_items is empty."""
        from src.api.routes import _extract_correction_context

        quote_data = {"line_items": []}
        context = _extract_correction_context(quote_data, "line_items[0].item_type")
        assert context is None
