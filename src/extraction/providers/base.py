"""
Base provider protocol and shared types for LLM extraction.

Defines the interface that all LLM providers must implement for
structured extraction from vendor quote documents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from src.schemas import VendorQuote


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: str
    model: str
    vision_model: str | None = None  # Some providers use separate vision models
    base_url: str | None = None  # For custom endpoints
    max_retries: int = 2


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol defining the interface for LLM extraction providers.

    All providers must implement both text and vision extraction methods.
    The protocol uses async methods to support concurrent extraction.
    """

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        ...

    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """
        Extract structured data from text prompt.

        Args:
            prompt: Formatted extraction prompt with document text and tables
            response_model: Pydantic model class for structured output.
                           Defaults to VendorQuote if not specified.

        Returns:
            Instance of response_model with extracted data

        Raises:
            Exception: If extraction fails
        """
        ...

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """
        Extract structured data from document images.

        Args:
            images: List of PNG image bytes (one per page)
            prompt: Extraction instructions for the vision model

        Returns:
            VendorQuote with extracted data

        Raises:
            Exception: If extraction fails
        """
        ...


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers with shared functionality.

    Provides common initialization and utility methods while requiring
    subclasses to implement provider-specific extraction logic.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: Provider configuration including API key and model
        """
        self._config = config
        self._client: Any = None  # Subclasses set their specific client type

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        return bool(self._config.api_key) and self._client is not None

    @property
    def model(self) -> str:
        """Get the configured model name."""
        return self._config.model

    @property
    def vision_model(self) -> str:
        """Get the vision model (defaults to main model if not specified)."""
        return self._config.vision_model or self._config.model

    @abstractmethod
    async def extract_text(
        self, prompt: str, response_model: type[BaseModel] | None = None
    ) -> BaseModel:
        """Extract structured data from text prompt.

        Args:
            prompt: Formatted extraction prompt
            response_model: Pydantic model for structured output.
                           Defaults to VendorQuote if None.
        """
        ...

    @abstractmethod
    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        """Extract structured data from document images."""
        ...

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._config.api_key:
            raise ValueError(f"{self.name} API key not configured")
        if not self._config.model:
            raise ValueError(f"{self.name} model not configured")
