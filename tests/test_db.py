"""
Tests for the database layer (src/db/).

Covers:
- Database initialization and connection
- Extraction model CRUD operations
- Correction model and audit trail
- File hash deduplication
"""

import json
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base, Correction, Extraction

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_session() -> AsyncSession:
    """Create a test database session with in-memory SQLite."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    # Cleanup
    await engine.dispose()


class TestExtractionModel:
    """Tests for the Extraction ORM model."""

    async def test_create_extraction(self, test_session: AsyncSession):
        """Test creating a new extraction record."""
        extraction = Extraction(
            filename="test.pdf",
            file_hash="abc123def456",
            extracted_json='{"vendor": {"name": "Test Vendor"}}',
            overall_confidence=0.95,
            requires_review=False,
        )
        test_session.add(extraction)
        await test_session.commit()

        assert extraction.id is not None
        assert extraction.filename == "test.pdf"
        assert extraction.file_hash == "abc123def456"
        assert extraction.overall_confidence == 0.95
        assert extraction.requires_review is False
        assert isinstance(extraction.created_at, datetime)

    async def test_extraction_json_storage(self, test_session: AsyncSession):
        """Test that JSON is stored and retrieved correctly."""
        quote_data = {
            "quote_id": "Q-123",
            "vendor": {"name": "Test Vendor"},
            "line_items": [{"description": "Item 1", "quantity": 1}],
        }
        extraction = Extraction(
            filename="test.pdf",
            file_hash="abc123def456",
            extracted_json=json.dumps(quote_data),
        )
        test_session.add(extraction)
        await test_session.commit()

        # Retrieve and verify
        result = await test_session.execute(
            select(Extraction).where(Extraction.id == extraction.id)
        )
        retrieved = result.scalar_one()
        parsed_json = json.loads(retrieved.extracted_json)

        assert parsed_json["quote_id"] == "Q-123"
        assert parsed_json["vendor"]["name"] == "Test Vendor"
        assert len(parsed_json["line_items"]) == 1

    async def test_file_hash_unique_constraint(self, test_session: AsyncSession):
        """Test that duplicate file hashes are rejected."""
        extraction1 = Extraction(
            filename="test1.pdf",
            file_hash="unique_hash_123",
            extracted_json="{}",
        )
        test_session.add(extraction1)
        await test_session.commit()

        extraction2 = Extraction(
            filename="test2.pdf",
            file_hash="unique_hash_123",  # Same hash
            extracted_json="{}",
        )
        test_session.add(extraction2)

        with pytest.raises(Exception):  # IntegrityError wrapped
            await test_session.commit()

    async def test_find_by_hash(self, test_session: AsyncSession):
        """Test finding extraction by file hash."""
        extraction = Extraction(
            filename="test.pdf",
            file_hash="findable_hash",
            extracted_json="{}",
        )
        test_session.add(extraction)
        await test_session.commit()

        result = await test_session.execute(
            select(Extraction).where(Extraction.file_hash == "findable_hash")
        )
        found = result.scalar_one_or_none()

        assert found is not None
        assert found.filename == "test.pdf"


class TestCorrectionModel:
    """Tests for the Correction ORM model."""

    async def test_create_correction(self, test_session: AsyncSession):
        """Test creating a correction record."""
        # First create an extraction
        extraction = Extraction(
            filename="test.pdf",
            file_hash="correction_test_hash",
            extracted_json="{}",
        )
        test_session.add(extraction)
        await test_session.commit()

        # Then create a correction
        correction = Correction(
            extraction_id=extraction.id,
            field_path="vendor.name",
            original_value="Old Vendor",
            corrected_value="New Vendor",
        )
        test_session.add(correction)
        await test_session.commit()

        assert correction.id is not None
        assert correction.extraction_id == extraction.id
        assert correction.field_path == "vendor.name"
        assert correction.original_value == "Old Vendor"
        assert correction.corrected_value == "New Vendor"

    async def test_correction_relationship(self, test_session: AsyncSession):
        """Test the relationship between extraction and corrections."""
        extraction = Extraction(
            filename="test.pdf",
            file_hash="relationship_test_hash",
            extracted_json="{}",
        )
        test_session.add(extraction)
        await test_session.commit()

        # Add multiple corrections
        correction1 = Correction(
            extraction_id=extraction.id,
            field_path="vendor.name",
            original_value="Old",
            corrected_value="New",
        )
        correction2 = Correction(
            extraction_id=extraction.id,
            field_path="amounts.grand_total",
            original_value="100.00",
            corrected_value="110.00",
        )
        test_session.add_all([correction1, correction2])
        await test_session.commit()

        # Refresh to load relationship
        await test_session.refresh(extraction, ["corrections"])

        assert len(extraction.corrections) == 2
        field_paths = {c.field_path for c in extraction.corrections}
        assert "vendor.name" in field_paths
        assert "amounts.grand_total" in field_paths

    async def test_array_field_path(self, test_session: AsyncSession):
        """Test correction with array index in field path."""
        extraction = Extraction(
            filename="test.pdf",
            file_hash="array_path_test_hash",
            extracted_json="{}",
        )
        test_session.add(extraction)
        await test_session.commit()

        correction = Correction(
            extraction_id=extraction.id,
            field_path="line_items[3].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        test_session.add(correction)
        await test_session.commit()

        assert correction.field_path == "line_items[3].item_type"


class TestDatabaseOperations:
    """Tests for database-level operations."""

    async def test_cascade_delete(self, test_session: AsyncSession):
        """Test that corrections are deleted when extraction is deleted."""
        extraction = Extraction(
            filename="test.pdf",
            file_hash="cascade_test_hash",
            extracted_json="{}",
        )
        test_session.add(extraction)
        await test_session.commit()

        correction = Correction(
            extraction_id=extraction.id,
            field_path="vendor.name",
            original_value="Old",
            corrected_value="New",
        )
        test_session.add(correction)
        await test_session.commit()

        correction_id = correction.id

        # Delete extraction
        await test_session.delete(extraction)
        await test_session.commit()

        # Verify correction is also deleted
        result = await test_session.execute(
            select(Correction).where(Correction.id == correction_id)
        )
        assert result.scalar_one_or_none() is None

    async def test_extraction_optional_fields(self, test_session: AsyncSession):
        """Test extraction with minimal required fields."""
        extraction = Extraction(
            filename="minimal.pdf",
            file_hash="minimal_hash",
            extracted_json="{}",
            # overall_confidence and requires_review are optional
        )
        test_session.add(extraction)
        await test_session.commit()

        assert extraction.id is not None
        assert extraction.overall_confidence is None
        assert extraction.requires_review is None
