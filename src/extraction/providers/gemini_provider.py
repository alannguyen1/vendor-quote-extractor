"""
Gemini provider implementation for LLM extraction.

Uses the Google GenAI SDK with instructor for schema-enforced extraction.
Gemini Flash-Lite provides fast inference with large context window support.

Key features:
- Large context window (1M+ tokens for Flash models)
- Fast inference with Flash-Lite variant
- Native multimodal support

Note: Migrated from deprecated google-generativeai to google-genai in v46.5.
"""

import base64
import logging
import os

import instructor
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider, ProviderConfig
from src.schemas import VendorQuote

logger = logging.getLogger("vendor_quote_extractor.providers.gemini")


class GeminiProvider(BaseLLMProvider):
    """
    Gemini LLM provider for structured extraction.

    Uses Google's Gemini models via the new GenAI SDK with instructor library
    for schema-enforced JSON extraction.

    Key advantages:
    - Large context window (1M+ tokens)
    - Fast inference with Flash-Lite
    - Good for large text documents (11+ pages)
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Gemini provider.

        Args:
            config: Provider configuration with API key and model
        """
        super().__init__(config)
        self._validate_config()

        # Set API key in environment for google-genai SDK
        os.environ["GOOGLE_API_KEY"] = config.api_key

        # Build provider string for instructor (e.g., "google/gemini-2.0-flash-lite")
        # instructor.from_provider handles client creation internally
        provider_string = f"google/{config.model}"
        self._client = instructor.from_provider(
            provider_string,
            async_client=True,  # Required for proper async/await support (G19 fix)
        )
        logger.info(f"Gemini provider initialized with model: {config.model}")

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt using Gemini.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Pydantic model class for structured output.
                Defaults to VendorQuote when omitted.

        Returns:
            Instance of the response_model with extracted data
        """
        model_cls = response_model or VendorQuote
        # instructor.from_provider wraps the GenAI SDK for structured output
        system_prompt = "You are a precise document extraction system."
        full_prompt = f"{system_prompt}\n\n{prompt}"

        return await self._client.create(
            response_model=model_cls,
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            max_retries=self._config.max_retries,
        )

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images using Gemini Vision.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data
        """
        # Build content with images and text for Gemini
        # The new GenAI SDK uses a similar multimodal content structure
        content_parts: list[dict] = []

        # Add system instruction
        system_instruction = (
            "You are a precise document extraction system. "
            "Extract structured data from the scanned document images."
        )
        content_parts.append({"type": "text", "text": system_instruction})

        # Add images using OpenAI-compatible format (instructor normalizes this)
        for img_bytes in images:
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                }
            )

        # Add extraction prompt
        content_parts.append({"type": "text", "text": prompt})

        # Use instructor-wrapped client for structured output
        return await self._client.create(
            response_model=VendorQuote,
            messages=[
                {"role": "user", "content": content_parts},
            ],
            max_retries=self._config.max_retries,
        )
