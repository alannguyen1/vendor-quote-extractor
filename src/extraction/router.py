"""
Smart Router for document-aware provider selection.

Routes documents to optimal LLM providers based on characteristics:
- Document type (text vs scanned)
- Page count
- Configured routing strategy

Designed for Speed Phase 3 of the optimization plan.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from config import Settings, get_settings
from src.extraction.pdf_parser import ParseResult
from src.extraction.providers import (
    BaseLLMProvider,
    get_provider,
    is_provider_available,
)

logger = logging.getLogger("vendor_quote_extractor.router")

# Routing strategy type
RoutingStrategy = Literal["auto", "speed", "accuracy", "cost"]


@dataclass
class RoutingDecision:
    """
    Represents a routing decision made by the Smart Router.

    Attributes:
        provider_name: Name of the selected provider
        reason: Human-readable explanation of why this provider was chosen
        fallback_chain: Ordered list of fallback providers if primary fails
        expected_latency_ms: Estimated latency in milliseconds
        is_optimal: True if this is the optimal choice, False if fallback
    """

    provider_name: str
    reason: str
    fallback_chain: list[str]
    expected_latency_ms: int
    is_optimal: bool = True


class SmartRouter:
    """
    Routes documents to optimal LLM providers based on characteristics.

    Routing Rules (Speed Strategy):
    - Text PDF, 1-10 pages, <40 table rows: Groq (fastest, <2s)
    - Text PDF, 11+ pages OR >40 table rows: Gemini Flash (large context, <4s)
    - Scanned PDF, 1-5 pages: Groq Vision (fast, <3s)
    - Scanned PDF, 6+ pages: Fireworks VLM (30 image limit, <6s)

    H6: Complexity-based routing ensures large documents (many table rows)
    are sent to providers with adequate context windows (Gemini > Groq).

    Falls back to OpenAI/Anthropic if optimal provider unavailable.
    """

    # Provider latency estimates (milliseconds)
    LATENCY_ESTIMATES = {
        "groq": {"text": 1500, "vision": 2500},
        "gemini": {"text": 3000, "vision": 4000},
        "fireworks": {"text": 2000, "vision": 5000},
        "openai": {"text": 4000, "vision": 6000},
        "anthropic": {"text": 4500, "vision": 6500},
    }

    # Provider constraints
    GROQ_VISION_IMAGE_LIMIT = 5
    FIREWORKS_VISION_IMAGE_LIMIT = 30

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the Smart Router.

        Args:
            settings: Optional Settings instance. Uses get_settings() if not provided.
        """
        self.settings = settings or get_settings()

    def select_provider(
        self,
        parse_result: ParseResult,
        strategy: RoutingStrategy | None = None,
    ) -> RoutingDecision:
        """
        Select the optimal provider for a document based on its characteristics.

        Args:
            parse_result: Result from PDF parsing with page count and scan detection
            strategy: Routing strategy override. Uses settings if not provided.

        Returns:
            RoutingDecision with selected provider and fallback chain
        """
        # Determine routing strategy
        effective_strategy = strategy or self._get_strategy_from_settings()

        # Check if routing is enabled
        if not self._is_routing_enabled():
            return self._default_routing(parse_result)

        # Route based on strategy
        if effective_strategy == "accuracy":
            return self._route_for_accuracy(parse_result)
        elif effective_strategy == "cost":
            return self._route_for_cost(parse_result)
        else:  # "auto" or "speed"
            return self._route_for_speed(parse_result)

    def _is_routing_enabled(self) -> bool:
        """Check if smart routing is enabled in settings."""
        return getattr(self.settings, "routing_enabled", False)

    def _get_strategy_from_settings(self) -> RoutingStrategy:
        """Get the routing strategy from settings."""
        return getattr(self.settings, "routing_strategy", "speed")

    def _default_routing(self, parse_result: ParseResult) -> RoutingDecision:
        """
        Default routing when smart routing is disabled.

        Uses the configured fallback chain starting with primary provider.
        """
        fallback_chain = list(self.settings.llm_fallback_chain)
        primary = fallback_chain[0] if fallback_chain else "openai"

        extraction_type = "vision" if parse_result.is_scanned else "text"
        latency = self.LATENCY_ESTIMATES.get(primary, {}).get(extraction_type, 4000)

        return RoutingDecision(
            provider_name=primary,
            reason="Smart routing disabled, using default provider chain",
            fallback_chain=fallback_chain[1:] if len(fallback_chain) > 1 else [],
            expected_latency_ms=latency,
            is_optimal=False,
        )

    def _calculate_complexity(self, parse_result: ParseResult) -> int:
        """
        Calculate document complexity based on table row count.

        H6: Documents with many table rows (such as large multi-page quotes) need
        providers with larger context windows to avoid truncation.

        Args:
            parse_result: Result from PDF parsing with tables

        Returns:
            Total number of table rows across all tables
        """
        total_rows = sum(len(t.rows) for t in parse_result.tables)
        return total_rows

    def _route_for_speed(self, parse_result: ParseResult) -> RoutingDecision:
        """
        Route for minimum latency (speed strategy).

        Routing Rules:
        - Text PDF, 1-10 pages, <40 rows: Groq
        - Text PDF, 11+ pages OR >40 rows: Gemini Flash (large context)
        - Scanned PDF, 1-5 pages: Groq Vision
        - Scanned PDF, 6+ pages: Fireworks VLM

        H6: Complexity-based routing uses table row count to detect large
        documents that need providers with bigger context windows.
        """
        page_count = parse_result.page_count
        is_scanned = parse_result.is_scanned
        complexity = self._calculate_complexity(parse_result)

        if is_scanned:
            return self._route_scanned_for_speed(page_count)
        else:
            return self._route_text_for_speed(page_count, complexity)

    def _route_text_for_speed(
        self, page_count: int, complexity: int = 0
    ) -> RoutingDecision:
        """
        Route text-based PDF for speed with complexity awareness.

        H6: Uses table row count (complexity) to detect documents that need
        larger context windows, even if page count is low.

        Args:
            page_count: Number of pages in the document
            complexity: Total table rows (H6 complexity metric)

        Returns:
            RoutingDecision with selected provider
        """
        threshold = self.settings.complexity_routing_threshold

        # H6: Check complexity first - large tables need large-context provider
        if complexity > threshold:
            # NOTE: Gemini routing disabled — instructor library cannot
            # serialize Pydantic StrEnum fields for Google's API (returns
            # raw strings instead of enum instances). Re-enable once
            # instructor adds Gemini enum support.
            # Route to OpenAI (128K context) with chunked extraction.
            if is_provider_available("openai", self.settings):
                reason = (
                    f"Complex document ({complexity} rows) - "
                    f"Gemini unavailable, using OpenAI"
                )
                logger.info(f"H6 complexity routing fallback: {reason}")
                return RoutingDecision(
                    provider_name="openai",
                    reason=reason,
                    fallback_chain=self._build_fallback_chain("openai"),
                    expected_latency_ms=4000,
                    is_optimal=False,
                )
            # Last resort - Groq with warning
            if is_provider_available("groq", self.settings):
                reason = (
                    f"Complex document ({complexity} rows) - "
                    f"WARNING: using Groq (may truncate)"
                )
                logger.warning(f"H6: {reason}")
                return RoutingDecision(
                    provider_name="groq",
                    reason=reason,
                    fallback_chain=self._build_fallback_chain("groq"),
                    expected_latency_ms=1500,
                    is_optimal=False,
                )

        # Standard page-based routing for non-complex documents
        if page_count <= 10:
            # Try Groq first (fastest)
            if is_provider_available("groq", self.settings):
                return RoutingDecision(
                    provider_name="groq",
                    reason=f"Text PDF with {page_count} pages - Groq is fastest",
                    fallback_chain=self._build_fallback_chain("groq"),
                    expected_latency_ms=1500,
                    is_optimal=True,
                )
            # Fall back to OpenAI
            return self._fallback_decision(
                "Text PDF - Groq unavailable, using OpenAI",
                "text",
                page_count,
            )
        else:
            # Large text PDF - try Gemini for large context
            if is_provider_available("gemini", self.settings):
                reason = f"Large text PDF ({page_count} pages) - Gemini best context"
                # Prefer OpenAI over Groq in fallback for large docs
                large_doc_fallback = [
                    p
                    for p in ["openai", "anthropic", "fireworks", "groq"]
                    if is_provider_available(p, self.settings)
                ]
                return RoutingDecision(
                    provider_name="gemini",
                    reason=reason,
                    fallback_chain=large_doc_fallback,
                    expected_latency_ms=3000,
                    is_optimal=True,
                )
            # Try Groq even for larger docs
            if is_provider_available("groq", self.settings):
                reason = f"Large text PDF ({page_count} pages) - Gemini unavailable"
                return RoutingDecision(
                    provider_name="groq",
                    reason=reason,
                    fallback_chain=self._build_fallback_chain("groq"),
                    expected_latency_ms=2000,
                    is_optimal=False,
                )
            # Fall back to OpenAI
            return self._fallback_decision(
                f"Large text PDF ({page_count} pages) - using OpenAI",
                "text",
                page_count,
            )

    def _route_scanned_for_speed(self, page_count: int) -> RoutingDecision:
        """Route scanned PDF for speed."""
        if page_count <= self.GROQ_VISION_IMAGE_LIMIT:
            # Within Groq's image limit
            if is_provider_available("groq", self.settings):
                reason = f"Scanned PDF ({page_count} pages) - within Groq 5 limit"
                return RoutingDecision(
                    provider_name="groq",
                    reason=reason,
                    fallback_chain=self._build_fallback_chain("groq"),
                    expected_latency_ms=2500,
                    is_optimal=True,
                )
            # Fall back to OpenAI for vision
            return self._fallback_decision(
                f"Scanned PDF ({page_count} pages) - Groq unavailable",
                "vision",
                page_count,
            )
        elif page_count <= self.FIREWORKS_VISION_IMAGE_LIMIT:
            # Exceeds Groq limit, try Fireworks
            if is_provider_available("fireworks", self.settings):
                reason = f"Scanned PDF ({page_count} pages) - exceeds Groq 5 limit"
                return RoutingDecision(
                    provider_name="fireworks",
                    reason=reason,
                    fallback_chain=self._build_fallback_chain("fireworks"),
                    expected_latency_ms=5000,
                    is_optimal=True,
                )
            # Fall back to OpenAI (unlimited images)
            return self._fallback_decision(
                f"Scanned PDF ({page_count} pages) - Fireworks unavailable",
                "vision",
                page_count,
            )
        else:
            # Very large scanned PDF - OpenAI is most reliable
            return RoutingDecision(
                provider_name="openai",
                reason=f"Large scanned PDF ({page_count} pages) - OpenAI most reliable",
                fallback_chain=self._build_fallback_chain("openai"),
                expected_latency_ms=8000,
                is_optimal=True,
            )

    def _route_for_accuracy(self, parse_result: ParseResult) -> RoutingDecision:
        """
        Route for maximum accuracy (accuracy strategy).

        Always prefers OpenAI GPT-4o or Anthropic Claude for best accuracy.
        """
        extraction_type = "vision" if parse_result.is_scanned else "text"

        if is_provider_available("openai", self.settings):
            return RoutingDecision(
                provider_name="openai",
                reason="Accuracy mode - GPT-4o has highest accuracy",
                fallback_chain=self._build_fallback_chain("openai"),
                expected_latency_ms=self.LATENCY_ESTIMATES["openai"][extraction_type],
                is_optimal=True,
            )
        elif is_provider_available("anthropic", self.settings):
            return RoutingDecision(
                provider_name="anthropic",
                reason="Accuracy mode - Claude 3.5 as fallback",
                fallback_chain=self._build_fallback_chain("anthropic"),
                expected_latency_ms=self.LATENCY_ESTIMATES["anthropic"][
                    extraction_type
                ],
                is_optimal=True,
            )
        else:
            # Fall back to any available provider
            return self._fallback_decision(
                "Accuracy mode - no premium providers available",
                extraction_type,
                parse_result.page_count,
            )

    def _route_for_cost(self, parse_result: ParseResult) -> RoutingDecision:
        """
        Route for minimum cost (cost strategy).

        Prefers Groq (often free/cheap) > Gemini > Fireworks > OpenAI/Anthropic.
        """
        page_count = parse_result.page_count
        is_scanned = parse_result.is_scanned
        extraction_type = "vision" if is_scanned else "text"

        # Cost order: Groq (cheapest) -> Gemini -> Fireworks -> OpenAI -> Anthropic
        cost_order = ["groq", "gemini", "fireworks", "openai", "anthropic"]

        # Filter by vision constraints for scanned PDFs
        if is_scanned and page_count > self.GROQ_VISION_IMAGE_LIMIT:
            # Remove Groq from options - exceeds image limit
            cost_order = [p for p in cost_order if p != "groq"]

        if is_scanned and page_count > self.FIREWORKS_VISION_IMAGE_LIMIT:
            # Remove Fireworks from options
            cost_order = [p for p in cost_order if p != "fireworks"]

        for provider in cost_order:
            if is_provider_available(provider, self.settings):
                return RoutingDecision(
                    provider_name=provider,
                    reason=f"Cost mode - {provider} is cheapest available",
                    fallback_chain=self._build_fallback_chain(provider),
                    expected_latency_ms=self.LATENCY_ESTIMATES.get(provider, {}).get(
                        extraction_type, 4000
                    ),
                    is_optimal=True,
                )

        # No providers available
        return self._fallback_decision(
            "Cost mode - no providers available",
            extraction_type,
            page_count,
        )

    def _build_fallback_chain(self, primary: str) -> list[str]:
        """Build fallback chain excluding the primary provider."""
        # Start with configured fallback chain
        chain = list(self.settings.llm_fallback_chain)

        # Add any available providers not in chain
        available = self.settings.get_available_providers()
        for provider in available:
            if provider not in chain:
                chain.append(provider)

        # Remove primary from chain and return
        return [
            p for p in chain if p != primary and is_provider_available(p, self.settings)
        ]

    def _fallback_decision(
        self,
        reason: str,
        extraction_type: str,
        page_count: int,
    ) -> RoutingDecision:
        """Create a fallback decision when optimal provider is unavailable."""
        # Try providers in configured fallback order
        for provider in self.settings.llm_fallback_chain:
            if is_provider_available(provider, self.settings):
                return RoutingDecision(
                    provider_name=provider,
                    reason=reason,
                    fallback_chain=self._build_fallback_chain(provider),
                    expected_latency_ms=self.LATENCY_ESTIMATES.get(provider, {}).get(
                        extraction_type, 4000
                    ),
                    is_optimal=False,
                )

        # Last resort - return openai even if not available (will fail later)
        return RoutingDecision(
            provider_name="openai",
            reason=f"{reason} - no providers configured",
            fallback_chain=[],
            expected_latency_ms=5000,
            is_optimal=False,
        )

    def get_providers_for_decision(
        self,
        decision: RoutingDecision,
    ) -> list[BaseLLMProvider]:
        """
        Get provider instances for a routing decision.

        Returns the primary provider followed by fallback providers.

        Args:
            decision: The routing decision from select_provider()

        Returns:
            List of provider instances in order of preference
        """
        providers = []

        # Primary provider
        try:
            primary = get_provider(decision.provider_name, self.settings)
            providers.append(primary)
        except Exception as e:
            logger.warning(f"Failed to create primary provider: {e}")

        # Fallback providers
        for fallback_name in decision.fallback_chain:
            try:
                fallback = get_provider(fallback_name, self.settings)
                providers.append(fallback)
            except Exception as e:
                msg = f"Failed to create fallback provider {fallback_name}: {e}"
                logger.warning(msg)
                continue

        return providers


# Module-level singleton for convenience
_router_instance: SmartRouter | None = None


def get_router(settings: Settings | None = None) -> SmartRouter:
    """
    Get or create the Smart Router singleton.

    Args:
        settings: Optional Settings instance. Uses get_settings() if not provided.

    Returns:
        SmartRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter(settings)
    return _router_instance


def reset_router() -> None:
    """Reset the router singleton (for testing)."""
    global _router_instance
    _router_instance = None
