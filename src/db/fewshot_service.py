"""Few-shot learning service for correction examples.

Provides functions to manage correction examples that can be used
for few-shot prompting to improve LLM extraction accuracy.

Key operations:
    - normalize_field_path: Convert specific paths to general patterns
    - add_correction_example: Store a new correction example
    - get_examples_for_field: Retrieve examples for a field type
    - format_few_shot_prompt: Generate prompt text from examples
"""

import re
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CorrectionExample


def normalize_field_path(field_path: str) -> str:
    """Normalize a specific field path to a general pattern.

    Converts indexed paths like "line_items[3].item_type" to
    pattern paths like "line_items.item_type" for grouping
    similar corrections together.

    Args:
        field_path: The specific field path with possible indices.

    Returns:
        Normalized path without array indices.

    Examples:
        >>> normalize_field_path("line_items[3].item_type")
        'line_items.item_type'
        >>> normalize_field_path("amounts.grand_total")
        'amounts.grand_total'
        >>> normalize_field_path("line_items[0].description")
        'line_items.description'
    """
    # Remove array indices like [0], [3], [42]
    return re.sub(r"\[\d+\]", "", field_path)


async def add_correction_example(
    session: AsyncSession,
    field_path: str,
    original_value: Optional[str],
    corrected_value: str,
    document_context: Optional[str] = None,
) -> CorrectionExample:
    """Add or update a correction example for few-shot learning.

    If an identical example (same field_path, original_value, corrected_value)
    already exists, increments its used_count. Otherwise creates a new example.

    Args:
        session: Database session.
        field_path: The specific field path (will be normalized).
        original_value: The incorrectly extracted value.
        corrected_value: The correct value provided by user.
        document_context: Optional surrounding text for context.

    Returns:
        The created or updated CorrectionExample.
    """
    normalized_path = normalize_field_path(field_path)

    # Check if an identical example already exists
    result = await session.execute(
        select(CorrectionExample).where(
            CorrectionExample.field_path == normalized_path,
            CorrectionExample.original_value == original_value,
            CorrectionExample.corrected_value == corrected_value,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Increment used_count for existing example
        existing.used_count += 1
        # Update context if provided and current is empty
        if document_context and not existing.document_context:
            existing.document_context = document_context
        return existing

    # Create new example
    example = CorrectionExample(
        field_path=normalized_path,
        original_value=original_value,
        corrected_value=corrected_value,
        document_context=document_context,
        used_count=1,
        success_rate=0.0,
    )
    session.add(example)
    return example


async def get_examples_for_field(
    session: AsyncSession,
    field_path: str,
    limit: int = 5,
) -> list[CorrectionExample]:
    """Retrieve correction examples for a specific field type.

    Returns examples sorted by used_count (most common first),
    limited to the specified number of examples.

    Args:
        session: Database session.
        field_path: The field path (will be normalized).
        limit: Maximum number of examples to return.

    Returns:
        List of CorrectionExample objects.
    """
    normalized_path = normalize_field_path(field_path)

    result = await session.execute(
        select(CorrectionExample)
        .where(CorrectionExample.field_path == normalized_path)
        .order_by(CorrectionExample.used_count.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_all_examples(
    session: AsyncSession,
    limit: int = 20,
) -> list[CorrectionExample]:
    """Retrieve all correction examples sorted by frequency.

    Useful for generating a comprehensive few-shot prompt section.

    Args:
        session: Database session.
        limit: Maximum number of examples to return.

    Returns:
        List of CorrectionExample objects.
    """
    result = await session.execute(
        select(CorrectionExample)
        .order_by(CorrectionExample.used_count.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_example_count(session: AsyncSession) -> int:
    """Get total count of correction examples.

    Args:
        session: Database session.

    Returns:
        Total number of correction examples in database.
    """
    result = await session.execute(select(func.count(CorrectionExample.id)))
    return result.scalar() or 0


def format_few_shot_prompt(examples: list[CorrectionExample]) -> str:
    """Format correction examples into a few-shot prompt section.

    Generates text that can be appended to LLM prompts to provide
    examples of common extraction mistakes and their corrections.

    Uses explicit WRONG/CORRECT markers to prevent LLMs from anchoring
    on the incorrect value. This is especially important for models
    with limited context windows (like Groq).

    Args:
        examples: List of CorrectionExample objects.

    Returns:
        Formatted prompt text, or empty string if no examples.

    Example output:
        EXTRACTION CORRECTIONS - Learn from these past mistakes:

        Correction #1 for grand_total:
          Context: "TCV vs net total confusion"
          WRONG (do not use): "552012.00"
          CORRECT (use this): "597312.00"
    """
    if not examples:
        return ""

    lines = ["EXTRACTION CORRECTIONS - Learn from these past mistakes:"]

    for i, example in enumerate(examples, 1):
        # Format field path for readability
        field_name = example.field_path.split(".")[-1]

        lines.append(f"\nCorrection #{i} for {field_name}:")
        if example.document_context:
            lines.append(f'  Context: "{example.document_context}"')
        lines.append(f'  WRONG (do not use): "{example.original_value}"')
        lines.append(f'  CORRECT (use this): "{example.corrected_value}"')

    lines.append("\nIMPORTANT: Use only CORRECT values above, never the WRONG ones.")

    return "\n".join(lines)


async def increment_example_usage(
    session: AsyncSession,
    example_ids: list[int],
) -> None:
    """Increment used_count for examples included in a prompt.

    Called after examples are used in an extraction prompt.

    Args:
        session: Database session.
        example_ids: List of example IDs that were used.
    """
    if not example_ids:
        return

    result = await session.execute(
        select(CorrectionExample).where(CorrectionExample.id.in_(example_ids))
    )
    for example in result.scalars():
        example.used_count += 1


async def update_example_success(
    session: AsyncSession,
    example_id: int,
    was_successful: bool,
) -> None:
    """Update success rate for an example after extraction.

    Uses exponential moving average to track success rate.

    Args:
        session: Database session.
        example_id: The example ID to update.
        was_successful: Whether the extraction was correct.
    """
    result = await session.execute(
        select(CorrectionExample).where(CorrectionExample.id == example_id)
    )
    example = result.scalar_one_or_none()

    if example:
        # Exponential moving average with alpha=0.2
        alpha = 0.2
        success_value = 1.0 if was_successful else 0.0
        new_rate = alpha * success_value + (1 - alpha) * example.success_rate
        example.success_rate = new_rate
