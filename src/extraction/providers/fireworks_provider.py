"""
Fireworks provider implementation for LLM extraction.

Uses the Fireworks AI API with instructor for schema-enforced extraction.
Fireworks provides fast inference with support for large vision workloads.

Key features:
- Vision API supports up to 30 images per request (vs Groq's 5)
- Prompt caching for reduced TTFT (80% reduction) - Speed Phase 5
- OpenAI-compatible API format via fireworks-ai SDK

Speed Phase 5 optimizations:
- Prompt caching via extra_body parameter reduces TTFT by ~80%
- Static system prompts are cached across requests
"""

import base64
import logging

import instructor
from fireworks.client import AsyncFireworks
from pydantic import BaseModel

from config import get_settings
from src.extraction.providers.base import BaseLLMProvider, ProviderConfig
from src.schemas import VendorQuote

logger = logging.getLogger("vendor_quote_extractor.providers.fireworks")

# Fireworks Vision API constraint: maximum 30 images per request
MAX_VISION_IMAGES = 30

# Static system prompts for caching (Speed Phase 5)
# Keeping these as module-level constants ensures consistent caching
SYSTEM_PROMPT_TEXT = "You are a precise document extraction system."
SYSTEM_PROMPT_VISION = (
    "You are a precise document extraction system. "
    "Extract structured data from the scanned document images."
)


class FireworksProvider(BaseLLMProvider):
    """
    Fireworks LLM provider for structured extraction.

    Uses Llama models via Fireworks' inference API with instructor library
    for schema-enforced JSON extraction.

    Key advantages over Groq:
    - Supports up to 30 images per request (6x more than Groq)
    - Prompt caching reduces TTFT by 80% (Speed Phase 5)
    - Better for multi-page scanned documents
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Fireworks provider.

        Args:
            config: Provider configuration with API key and model
        """
        super().__init__(config)
        self._validate_config()

        # Load settings for prompt caching configuration
        self._settings = get_settings()

        # Setup Fireworks client with instructor wrapper
        base_client = AsyncFireworks(
            api_key=config.api_key,
        )
        # Use instructor's native Fireworks integration
        self._client = instructor.from_fireworks(base_client)

        # Log initialization with caching status
        cache_status = "enabled" if self._settings.enable_prompt_caching else "disabled"
        logger.info(
            f"Fireworks provider initialized with model: {config.model} "
            f"(prompt caching: {cache_status})"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "fireworks"

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt using Fireworks.

        Uses prompt caching when enabled for ~80% TTFT reduction on
        repeated requests with the same system prompt prefix.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Optional Pydantic model class for the response schema.
                Defaults to VendorQuote if not provided.

        Returns:
            BaseModel instance with extracted data
        """
        model_cls = response_model or VendorQuote

        # Use module-level constant for consistent caching
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user", "content": prompt},
        ]

        # Fireworks automatically caches repeated prompt prefixes
        # The static system prompt enables efficient caching across requests
        return await self._client.chat.completions.create(
            model=self._config.model,
            response_model=model_cls,
            messages=messages,
            max_retries=self._config.max_retries,
        )

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images using Fireworks Vision.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data

        Raises:
            ValueError: If more than 30 images are provided (Fireworks limit)
        """
        if len(images) > MAX_VISION_IMAGES:
            logger.warning(
                f"Fireworks Vision limited to {MAX_VISION_IMAGES} images, "
                f"received {len(images)}. Consider OpenAI for larger documents."
            )
            raise ValueError(
                f"Fireworks Vision API supports maximum {MAX_VISION_IMAGES} images. "
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

        # Use module-level constant for consistent caching
        # Fireworks automatically caches repeated prompt prefixes
        return await self._client.chat.completions.create(
            model=self.vision_model,
            response_model=VendorQuote,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_VISION},
                {"role": "user", "content": image_contents},
            ],
            max_retries=self._config.max_retries,
        )
