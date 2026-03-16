"""Tests for few-shot learning service.

Tests cover:
- Field path normalization
- Adding and retrieving correction examples
- Duplicate example handling
- Few-shot prompt formatting
- Example usage tracking
- Success rate updates
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.db.fewshot_service import (
    add_correction_example,
    format_few_shot_prompt,
    get_all_examples,
    get_example_count,
    get_examples_for_field,
    increment_example_usage,
    normalize_field_path,
    update_example_success,
)
from src.db.models import Base, CorrectionExample


# Test fixtures for in-memory database
@pytest.fixture
async def async_engine():
    """Create an async in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    """Create an async session for testing."""
    async_session_factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:
        yield session
        await session.rollback()


class TestNormalizeFieldPath:
    """Tests for field path normalization."""

    def test_normalize_simple_path(self):
        """Simple paths without indices remain unchanged."""
        assert normalize_field_path("amounts.grand_total") == "amounts.grand_total"
        assert normalize_field_path("vendor.name") == "vendor.name"

    def test_normalize_with_single_index(self):
        """Paths with single index are normalized."""
        assert normalize_field_path("line_items[3].item_type") == "line_items.item_type"
        result = normalize_field_path("line_items[0].description")
        assert result == "line_items.description"

    def test_normalize_with_large_index(self):
        """Paths with large indices are normalized."""
        assert normalize_field_path("line_items[42].sku") == "line_items.sku"
        assert normalize_field_path("line_items[999].quantity") == "line_items.quantity"

    def test_normalize_with_multiple_indices(self):
        """Paths with multiple indices are all normalized."""
        path = "data[0].items[5].nested[10].value"
        assert normalize_field_path(path) == "data.items.nested.value"

    def test_normalize_root_level_array(self):
        """Root-level array indices are removed."""
        assert normalize_field_path("[0].field") == ".field"

    def test_normalize_empty_path(self):
        """Empty paths return empty string."""
        assert normalize_field_path("") == ""


class TestAddCorrectionExample:
    """Tests for adding correction examples."""

    @pytest.mark.asyncio
    async def test_add_new_example(self, async_session):
        """Adding a new example creates a database record."""
        example = await add_correction_example(
            session=async_session,
            field_path="line_items[3].item_type",
            original_value="hardware",
            corrected_value="services",
            document_context="Cisco SMARTnet 24x7x4",
        )
        await async_session.commit()

        assert example.id is not None
        assert example.field_path == "line_items.item_type"  # Normalized
        assert example.original_value == "hardware"
        assert example.corrected_value == "services"
        assert example.document_context == "Cisco SMARTnet 24x7x4"
        assert example.used_count == 1
        assert example.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_add_duplicate_increments_count(self, async_session):
        """Adding duplicate example increments used_count."""
        # Add first example
        example1 = await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        await async_session.commit()
        original_id = example1.id

        # Add duplicate (same normalized path and values)
        example2 = await add_correction_example(
            session=async_session,
            field_path="line_items[5].item_type",  # Different index
            original_value="hardware",
            corrected_value="services",
        )
        await async_session.commit()

        # Should return the same record with incremented count
        assert example2.id == original_id
        assert example2.used_count == 2

    @pytest.mark.asyncio
    async def test_duplicate_updates_context_if_missing(self, async_session):
        """Duplicate with context updates record if context was missing."""
        # Add example without context
        await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="services",
            document_context=None,
        )
        await async_session.commit()

        # Add duplicate with context
        example2 = await add_correction_example(
            session=async_session,
            field_path="line_items[1].item_type",
            original_value="hardware",
            corrected_value="services",
            document_context="SMARTnet support",
        )
        await async_session.commit()

        assert example2.document_context == "SMARTnet support"

    @pytest.mark.asyncio
    async def test_different_values_creates_new(self, async_session):
        """Different values create separate examples."""
        example1 = await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        await async_session.commit()

        example2 = await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="software",  # Different corrected value
        )
        await async_session.commit()

        assert example1.id != example2.id

    @pytest.mark.asyncio
    async def test_add_example_with_none_original(self, async_session):
        """Examples can have None as original value."""
        example = await add_correction_example(
            session=async_session,
            field_path="vendor.email",
            original_value=None,
            corrected_value="vendor@example.com",
        )
        await async_session.commit()

        assert example.original_value is None
        assert example.corrected_value == "vendor@example.com"


class TestGetExamplesForField:
    """Tests for retrieving examples by field."""

    @pytest.mark.asyncio
    async def test_get_examples_returns_matching(self, async_session):
        """Returns examples for the specified field."""
        # Add examples for different fields
        await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        await add_correction_example(
            session=async_session,
            field_path="vendor.name",
            original_value="old",
            corrected_value="new",
        )
        await async_session.commit()

        examples = await get_examples_for_field(
            async_session, "line_items[5].item_type"
        )
        assert len(examples) == 1
        assert examples[0].field_path == "line_items.item_type"

    @pytest.mark.asyncio
    async def test_get_examples_sorted_by_count(self, async_session):
        """Examples are sorted by used_count descending."""
        # Add example and increment its count
        await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        # Add duplicates to increment count
        await add_correction_example(
            session=async_session,
            field_path="line_items[1].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        await add_correction_example(
            session=async_session,
            field_path="line_items[2].item_type",
            original_value="hardware",
            corrected_value="services",
        )
        # Add different example with lower count
        await add_correction_example(
            session=async_session,
            field_path="line_items[0].item_type",
            original_value="software",
            corrected_value="services",
        )
        await async_session.commit()

        examples = await get_examples_for_field(async_session, "line_items.item_type")
        assert len(examples) == 2
        assert examples[0].used_count >= examples[1].used_count

    @pytest.mark.asyncio
    async def test_get_examples_respects_limit(self, async_session):
        """Limit parameter restricts number of examples returned."""
        # Add multiple different examples
        for i in range(10):
            await add_correction_example(
                session=async_session,
                field_path="line_items[0].item_type",
                original_value=f"value{i}",
                corrected_value="services",
            )
        await async_session.commit()

        examples = await get_examples_for_field(
            async_session, "line_items.item_type", limit=3
        )
        assert len(examples) == 3

    @pytest.mark.asyncio
    async def test_get_examples_empty_for_unknown_field(self, async_session):
        """Returns empty list for fields with no examples."""
        examples = await get_examples_for_field(async_session, "unknown.field")
        assert examples == []


class TestGetAllExamples:
    """Tests for getting all examples."""

    @pytest.mark.asyncio
    async def test_get_all_returns_sorted(self, async_session):
        """Returns all examples sorted by used_count."""
        await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        await add_correction_example(
            session=async_session,
            field_path="field2",
            original_value="c",
            corrected_value="d",
        )
        # Add duplicate to increase count
        await add_correction_example(
            session=async_session,
            field_path="field2",
            original_value="c",
            corrected_value="d",
        )
        await async_session.commit()

        examples = await get_all_examples(async_session)
        assert len(examples) == 2
        assert examples[0].field_path == "field2"  # Higher count first

    @pytest.mark.asyncio
    async def test_get_all_respects_limit(self, async_session):
        """Limit parameter restricts results."""
        for i in range(25):
            await add_correction_example(
                session=async_session,
                field_path=f"field{i}",
                original_value="a",
                corrected_value="b",
            )
        await async_session.commit()

        examples = await get_all_examples(async_session, limit=10)
        assert len(examples) == 10


class TestGetExampleCount:
    """Tests for getting example count."""

    @pytest.mark.asyncio
    async def test_count_zero_initially(self, async_session):
        """Count is zero when no examples exist."""
        count = await get_example_count(async_session)
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_reflects_additions(self, async_session):
        """Count reflects added examples."""
        await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        await add_correction_example(
            session=async_session,
            field_path="field2",
            original_value="c",
            corrected_value="d",
        )
        await async_session.commit()

        count = await get_example_count(async_session)
        assert count == 2


class TestFormatFewShotPrompt:
    """Tests for few-shot prompt formatting."""

    def test_format_empty_list(self):
        """Empty list returns empty string."""
        result = format_few_shot_prompt([])
        assert result == ""

    def test_format_single_example_with_context(self):
        """Single example with context formats correctly."""
        example = CorrectionExample(
            field_path="line_items.item_type",
            original_value="hardware",
            corrected_value="services",
            document_context="Cisco SMARTnet 24x7x4",
            used_count=1,
            success_rate=0.0,
        )
        result = format_few_shot_prompt([example])

        assert "EXTRACTION CORRECTIONS - Learn from these past mistakes:" in result
        assert "Cisco SMARTnet 24x7x4" in result
        assert 'WRONG (do not use): "hardware"' in result
        assert 'CORRECT (use this): "services"' in result
        assert "IMPORTANT: Use only CORRECT values above" in result

    def test_format_example_without_context(self):
        """Example without context omits context line but has WRONG/CORRECT markers."""
        example = CorrectionExample(
            field_path="line_items.item_type",
            original_value="hardware",
            corrected_value="services",
            document_context=None,
            used_count=1,
            success_rate=0.0,
        )
        result = format_few_shot_prompt([example])

        assert "Correction #1 for item_type:" in result
        assert "Context:" not in result  # No context line when context is None
        assert 'WRONG (do not use): "hardware"' in result
        assert 'CORRECT (use this): "services"' in result

    def test_format_multiple_examples(self):
        """Multiple examples are numbered with explicit WRONG/CORRECT markers."""
        examples = [
            CorrectionExample(
                field_path="line_items.item_type",
                original_value="hardware",
                corrected_value="services",
                document_context="Support contract",
                used_count=1,
                success_rate=0.0,
            ),
            CorrectionExample(
                field_path="line_items.item_type",
                original_value="services",
                corrected_value="software",
                document_context="License agreement",
                used_count=1,
                success_rate=0.0,
            ),
        ]
        result = format_few_shot_prompt(examples)

        assert "Correction #1 for item_type:" in result
        assert "Correction #2 for item_type:" in result
        assert 'WRONG (do not use): "hardware"' in result
        assert 'CORRECT (use this): "software"' in result


class TestIncrementExampleUsage:
    """Tests for incrementing example usage."""

    @pytest.mark.asyncio
    async def test_increment_single(self, async_session):
        """Incrementing usage updates count."""
        example = await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        await async_session.commit()
        original_count = example.used_count

        await increment_example_usage(async_session, [example.id])
        await async_session.commit()

        await async_session.refresh(example)
        assert example.used_count == original_count + 1

    @pytest.mark.asyncio
    async def test_increment_multiple(self, async_session):
        """Can increment multiple examples at once."""
        example1 = await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        example2 = await add_correction_example(
            session=async_session,
            field_path="field2",
            original_value="c",
            corrected_value="d",
        )
        await async_session.commit()

        await increment_example_usage(async_session, [example1.id, example2.id])
        await async_session.commit()

        await async_session.refresh(example1)
        await async_session.refresh(example2)
        assert example1.used_count == 2
        assert example2.used_count == 2

    @pytest.mark.asyncio
    async def test_increment_empty_list(self, async_session):
        """Empty list does nothing."""
        await increment_example_usage(async_session, [])
        # Should not raise


class TestUpdateExampleSuccess:
    """Tests for updating example success rate."""

    @pytest.mark.asyncio
    async def test_update_success_true(self, async_session):
        """Successful extraction increases rate."""
        example = await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        await async_session.commit()
        assert example.success_rate == 0.0

        await update_example_success(async_session, example.id, was_successful=True)
        await async_session.commit()

        await async_session.refresh(example)
        # With alpha=0.2: 0.2 * 1.0 + 0.8 * 0.0 = 0.2
        assert example.success_rate == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_update_success_false(self, async_session):
        """Failed extraction decreases rate."""
        example = await add_correction_example(
            session=async_session,
            field_path="field1",
            original_value="a",
            corrected_value="b",
        )
        example.success_rate = 1.0
        await async_session.commit()

        await update_example_success(async_session, example.id, was_successful=False)
        await async_session.commit()

        await async_session.refresh(example)
        # With alpha=0.2: 0.2 * 0.0 + 0.8 * 1.0 = 0.8
        assert example.success_rate == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_update_nonexistent_id(self, async_session):
        """Updating nonexistent ID does nothing."""
        await update_example_success(async_session, 99999, was_successful=True)
        # Should not raise


class TestCorrectionExampleModel:
    """Tests for the CorrectionExample model."""

    @pytest.mark.asyncio
    async def test_model_defaults(self, async_session):
        """Model has correct defaults."""
        example = CorrectionExample(
            field_path="test.field",
            original_value="old",
            corrected_value="new",
        )
        async_session.add(example)
        await async_session.commit()

        assert example.used_count == 0
        assert example.success_rate == 0.0
        assert example.document_context is None
        assert example.created_at is not None

    @pytest.mark.asyncio
    async def test_model_index_exists(self, async_session):
        """Field path index is created."""
        # Query using the indexed field
        result = await async_session.execute(
            select(CorrectionExample).where(
                CorrectionExample.field_path == "nonexistent"
            )
        )
        assert result.scalars().all() == []
