"""
Anthropic provider implementation for LLM extraction.

Uses the Anthropic API with instructor for schema-enforced extraction.
Supports both text and vision extraction (Claude 3.5 has native vision support).
"""

import base64
import logging

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider, ProviderConfig
from src.schemas import VendorQuote

logger = logging.getLogger("vendor_quote_extractor.providers.anthropic")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM provider for structured extraction.

    Uses Claude 3.5 Sonnet (or configured model) with instructor library
    for schema-enforced JSON extraction.
    """

    # Anthropic requires explicit max_tokens
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Anthropic provider.

        Args:
            config: Provider configuration with API key and model
        """
        super().__init__(config)
        self._validate_config()

        # Setup Anthropic client with instructor wrapper
        base_client = AsyncAnthropic(api_key=config.api_key)
        self._client = instructor.from_anthropic(base_client)
        logger.info(f"Anthropic provider initialized with model: {config.model}")

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt using Anthropic.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Optional Pydantic model class for the response schema.
                Defaults to VendorQuote if not provided.

        Returns:
            BaseModel instance with extracted data
        """
        model_cls = response_model or VendorQuote
        # Anthropic uses messages.create instead of chat.completions.create
        # and requires max_tokens to be specified
        return await self._client.messages.create(
            model=self._config.model,
            response_model=model_cls,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_retries=self._config.max_retries,
        )

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images using Anthropic Vision.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data
        """
        # Build vision API message with images in Anthropic format
        # Anthropic uses a different structure than OpenAI
        anthropic_content: list[dict] = []

        for img_bytes in images:
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            anthropic_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_image,
                    },
                }
            )

        # Add extraction instructions as text content
        anthropic_content.append({"type": "text", "text": prompt})

        return await self._client.messages.create(
            model=self.vision_model,
            response_model=VendorQuote,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            messages=[
                {"role": "user", "content": anthropic_content},
            ],
            max_retries=self._config.max_retries,
        )
