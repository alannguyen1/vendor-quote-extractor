"""
Groq provider implementation for LLM extraction.

Uses the Groq API with instructor for schema-enforced extraction.
Groq provides fast inference with Llama models.

Key constraints:
- Vision API limited to 5 images per request
- Uses OpenAI-compatible API format via groq SDK
"""

import base64
import logging

import instructor
from groq import AsyncGroq
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider, ProviderConfig
from src.schemas import VendorQuote

logger = logging.getLogger("vendor_quote_extractor.providers.groq")

# Groq Vision API constraint: maximum 5 images per request
MAX_VISION_IMAGES = 5

# H1: Default max tokens for Groq responses - prevents truncation on large documents
# Groq default is lower, which can truncate large multi-page quotes
# 16384 tokens needed for 50+ line items in JSON (~300 tokens per item)
DEFAULT_MAX_TOKENS = 16384

# H5: Groq-specific system prompt with critical extraction rules
# Generic prompts don't guide complex document handling properly
GROQ_SYSTEM_PROMPT = """\
You are a precise document extraction system specialized in vendor quotes.

CRITICAL RULES:
1. Extract ALL line items - do NOT consolidate, merge, or deduplicate
   - Rows are tagged [R01], [R02], etc. — every tag is a SEPARATE item
   - Items may look identical but represent different locations/groups
   - Count tables carefully, do not truncate
2. line_number MUST equal the [RNN] tag number (e.g., [R07] → line_number=7)
   - This enables gap detection: missing line_numbers mean dropped rows
   - If you see [R01] through [R51], your output MUST have line_numbers 1-51
3. For multi-page tables: continue extracting until you see subtotals/totals
4. grand_total = Total Contract Value (TCV) = SUM of all extended_prices + tax
   - TCV is the FULL contract value BEFORE applying credits
   - If document shows both TCV and "Net Amount Due", TCV is grand_total
5. Credits/trade-ins: Include as line items with NEGATIVE extended_price
   - Credit of $45,300 → extended_price: -45300.00
6. net_price_total = grand_total minus credits (may be labeled "Amount Due" or "Net")
   - Example: TCV $597,312 minus $45,300 credit = net_price_total $552,012
7. When in doubt, include more data rather than less
"""


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider for structured extraction.

    Uses Llama models via Groq's fast inference API with instructor library
    for schema-enforced JSON extraction.

    Note: Vision extraction is limited to 5 images per request. For documents
    with more than 5 pages, consider using a different provider (e.g., Fireworks)
    or implementing page batching with result merging.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Groq provider.

        Args:
            config: Provider configuration with API key and model
        """
        super().__init__(config)
        self._validate_config()

        # Setup Groq client with instructor wrapper
        # Groq uses OpenAI-compatible API, so instructor.from_openai works
        base_client = AsyncGroq(
            api_key=config.api_key,
        )
        # H2: Use JSON mode for better structured output (like Gemini's JSON mode)
        self._client = instructor.from_groq(base_client, mode=instructor.Mode.JSON)
        logger.info(f"Groq provider initialized with model: {config.model}")

    @property
    def name(self) -> str:
        """Provider name."""
        return "groq"

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt using Groq.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Pydantic model for structured output.
                           Defaults to VendorQuote if None.

        Returns:
            Instance of response_model with extracted data
        """
        model_cls = response_model or VendorQuote
        return await self._client.chat.completions.create(
            model=self._config.model,
            response_model=model_cls,
            max_tokens=DEFAULT_MAX_TOKENS,  # H1: Prevent truncation on large documents
            messages=[
                {
                    "role": "system",
                    "content": GROQ_SYSTEM_PROMPT,  # H5: Groq-specific extraction rules
                },
                {"role": "user", "content": prompt},
            ],
            max_retries=self._config.max_retries,
        )

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images using Groq Vision.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data

        Raises:
            ValueError: If more than 5 images are provided (Groq limit)
        """
        if len(images) > MAX_VISION_IMAGES:
            logger.warning(
                f"Groq Vision limited to {MAX_VISION_IMAGES} images, "
                f"received {len(images)}. Use Fireworks for larger documents."
            )
            raise ValueError(
                f"Groq Vision API supports maximum {MAX_VISION_IMAGES} images. "
                f"Received {len(images)}. Use a different provider."
            )

        # Build vision API message with images in OpenAI-compatible format
        image_contents: list[dict] = []
        for img_bytes in images:
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                        "detail": "high",
                    },
                }
            )

        # Add extraction instructions as text content
        image_contents.append({"type": "text", "text": prompt})

        return await self._client.chat.completions.create(
            model=self.vision_model,
            response_model=VendorQuote,
            max_tokens=DEFAULT_MAX_TOKENS,  # H1: Prevent truncation on large documents
            messages=[
                {
                    "role": "system",
                    # H5: Enhanced system prompt for vision extraction
                    "content": (
                        GROQ_SYSTEM_PROMPT
                        + "\nExtract structured data from the scanned document images."
                    ),
                },
                {"role": "user", "content": image_contents},
            ],
            max_retries=self._config.max_retries,
        )
