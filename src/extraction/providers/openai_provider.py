"""
OpenAI provider implementation for LLM extraction.

Uses the OpenAI API with instructor for schema-enforced extraction.
Supports both text and vision extraction (GPT-4o has native vision support).
"""

import base64
import logging

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider, ProviderConfig
from src.schemas import VendorQuote

logger = logging.getLogger("vendor_quote_extractor.providers.openai")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider for structured extraction.

    Uses GPT-4o (or configured model) with instructor library
    for schema-enforced JSON extraction.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize OpenAI provider.

        Args:
            config: Provider configuration with API key and model
        """
        super().__init__(config)
        self._validate_config()

        # Setup OpenAI client with instructor wrapper
        base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,  # None uses default
        )
        self._client = instructor.from_openai(base_client)
        logger.info(f"OpenAI provider initialized with model: {config.model}")

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt using OpenAI.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Pydantic model class for structured output.
                Defaults to VendorQuote if not provided.

        Returns:
            Instance of the response_model with extracted data
        """
        model_cls = response_model or VendorQuote
        return await self._client.chat.completions.create(
            model=self._config.model,
            response_model=model_cls,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document extraction system.",
                },
                {"role": "user", "content": prompt},
            ],
            max_retries=self._config.max_retries,
        )

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images using OpenAI Vision.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data
        """
        # Build vision API message with images in OpenAI format
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
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document extraction system. "
                        "Extract structured data from the scanned document images."
                    ),
                },
                {"role": "user", "content": image_contents},
            ],
            max_retries=self._config.max_retries,
        )
