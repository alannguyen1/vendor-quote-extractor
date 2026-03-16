#!/usr/bin/env python3
"""
Benchmark script to compare LLM provider performance.

Runs extractions against test fixtures using different providers and reports:
- Processing time (total, parse, extract, validate)
- Extraction accuracy (line items count, grand total)
- Success/failure rate
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from src.extraction.pipeline import ExtractionPipeline, reset_extraction_semaphore

# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"

# Provider configuration
PROVIDERS = {
    "openai": {"model_env": "LLM_MODEL", "model_attr": "llm_model"},
    "groq": {"model_env": "GROQ_MODEL", "model_attr": "groq_model"},
    "gemini": {"model_env": "GEMINI_MODEL", "model_attr": "gemini_model"},
    "fireworks": {"model_env": "FIREWORKS_MODEL", "model_attr": "fireworks_model"},
}


async def run_benchmark_for_provider(provider: str, model: str | None = None) -> dict:
    """
    Run benchmark for a specific provider.

    Returns dict with timing and extraction results.
    """
    # Get fresh settings (clear cache first)
    get_settings.cache_clear()
    reset_extraction_semaphore()

    # Set provider via environment
    os.environ["LLM_PROVIDER"] = provider
    if model:
        model_env = PROVIDERS[provider]["model_env"]
        os.environ[model_env] = model

    settings = get_settings()
    model_attr = PROVIDERS[provider]["model_attr"]
    model_name = getattr(settings, model_attr, "unknown")

    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING: {provider.upper()}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    # Get test PDFs
    pdf_files = list(FIXTURES_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} test files")

    results = {
        "provider": provider,
        "model": model_name,
        "files": [],
        "total_time_ms": 0,
        "avg_time_ms": 0,
        "successes": 0,
        "failures": 0,
    }

    pipeline = ExtractionPipeline()

    for pdf_path in pdf_files:
        pdf_bytes = pdf_path.read_bytes()
        filename = pdf_path.name

        print(f"\n  Processing: {filename}")
        start = time.time()

        try:
            result = await pipeline.process(pdf_bytes, filename)
            elapsed_ms = int((time.time() - start) * 1000)

            file_result = {
                "filename": filename,
                "success": result.quote is not None,
                "total_ms": elapsed_ms,
                "error": result.error,
            }

            if result.quote and result.quote.extraction_metadata:
                meta = result.quote.extraction_metadata
                file_result["parse_ms"] = meta.parse_time_ms
                file_result["extract_ms"] = meta.extract_time_ms
                file_result["validate_ms"] = meta.validate_time_ms
                file_result["line_items"] = len(result.quote.line_items)
                file_result["grand_total"] = (
                    float(result.quote.amounts.grand_total)
                    if result.quote.amounts and result.quote.amounts.grand_total
                    else None
                )
                file_result["confidence"] = meta.overall_confidence

                # Check for LLM timing breakdown
                if meta.extract_breakdown:
                    file_result["llm_api_ms"] = meta.extract_breakdown.llm_api_ms

                results["successes"] += 1
                print(
                    "    "
                    f"✓ {elapsed_ms}ms "
                    f"(parse={meta.parse_time_ms}ms, "
                    f"extract={meta.extract_time_ms}ms)"
                )
                print(
                    "      "
                    f"{len(result.quote.line_items)} items, "
                    f"total=${file_result['grand_total']:.2f}, "
                    f"conf={meta.overall_confidence:.2f}"
                )
            else:
                results["failures"] += 1
                print(f"    ✗ {elapsed_ms}ms - {result.error}")

            results["files"].append(file_result)
            results["total_time_ms"] += elapsed_ms

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            results["failures"] += 1
            results["files"].append(
                {
                    "filename": filename,
                    "success": False,
                    "total_ms": elapsed_ms,
                    "error": str(e),
                }
            )
            print(f"    ✗ {elapsed_ms}ms - ERROR: {e}")

    if results["files"]:
        results["avg_time_ms"] = results["total_time_ms"] // len(results["files"])

    return results


# Ground truth expected values for accuracy testing
EXPECTED_VALUES = {
    "mixed_category_quote.pdf": {
        "vendor_name": "Northwind Systems",
        "grand_total": 32198.00,
        "item_count": 3,
    },
    "discounted_hardware_quote.pdf": {
        "vendor_name": "Summit Equipment Supply",
        "grand_total": 26763.19,
        "item_count": 4,
    },
    "enterprise_term_quote.pdf": {
        "vendor_name": "Bluewave Software",
        "grand_total": 143391.60,
        "item_count": 3,
    },
    "service_order_with_credits.pdf": {
        "vendor_name": "Cedar Managed Services",
        "grand_total": 597312.00,
        "item_count": 4,
    },
    "multi_page_quote.pdf": {
        "vendor_name": "Riverstone Distribution",
        "grand_total": 314432.95,
        "item_count": 51,
    },
}


def print_multi_comparison(all_results: list[dict]) -> None:
    """Print a comparison table of all provider results."""
    print("\n" + "=" * 120)
    print("BENCHMARK COMPARISON - ALL PROVIDERS")
    print("=" * 120)

    # Get all filenames (short names for columns)
    if not all_results:
        return

    filenames = [f["filename"] for f in all_results[0]["files"]]
    short_names = [fn.replace(".pdf", "")[:12] for fn in filenames]

    # === SPEED TABLE (Providers as rows, PDFs as columns) ===
    print("\n📊 SPEED (ms per file):")
    print("-" * 120)

    # Header row with PDF names
    header = f"{'Provider':<12}"
    for short in short_names:
        header += f" {short:>12}"
    header += f" {'AVG':>10}"
    print(header)
    print("-" * 120)

    # Each provider as a row
    for r in all_results:
        row = f"{r['provider'].upper():<12}"
        for f in r["files"]:
            if f["success"]:
                row += f" {f['total_ms']:>12}"
            else:
                row += f" {'FAIL':>12}"
        row += f" {r['avg_time_ms']:>10}"
        print(row)

    # === ACCURACY TABLE (Grand Total comparison) ===
    print("\n📊 ACCURACY (Grand Total - expected vs extracted):")
    print("-" * 120)

    # Header row
    header = f"{'Provider':<12}"
    for short in short_names:
        header += f" {short:>12}"
    header += f" {'WITHIN $0.01':>14}"
    print(header)
    print("-" * 120)

    # Each provider as a row
    for r in all_results:
        row = f"{r['provider'].upper():<12}"
        accurate_count = 0
        total_count = 0

        for f in r["files"]:
            filename = f["filename"]
            expected = EXPECTED_VALUES.get(filename, {})

            if (
                f["success"]
                and f.get("grand_total") is not None
                and expected.get("grand_total")
            ):
                expected_total = expected["grand_total"]
                actual_total = f["grand_total"]
                diff = abs(expected_total - actual_total)
                total_count += 1

                if diff <= 0.01:
                    row += f" {'✓':>12}"
                    accurate_count += 1
                else:
                    row += f" {f'${diff:,.0f}':>12}"
            else:
                row += f" {'-':>12}"

        accuracy_pct = f"{accurate_count}/{total_count}" if total_count > 0 else "-"
        row += f" {accuracy_pct:>14}"
        print(row)

    # === LINE ITEM COUNT TABLE ===
    print("\n📊 LINE ITEMS (extracted count):")
    print("-" * 120)

    # Header row
    header = f"{'Provider':<12}"
    for short in short_names:
        header += f" {short:>12}"
    print(header)
    print("-" * 120)

    # Expected row
    row = f"{'Expected':<12}"
    for fn in filenames:
        expected = EXPECTED_VALUES.get(fn, {})
        exp_count = expected.get("item_count", "?")
        row += f" {exp_count:>12}"
    print(row)

    # Each provider as a row
    for r in all_results:
        row = f"{r['provider'].upper():<12}"
        for f in r["files"]:
            if f["success"] and f.get("line_items") is not None:
                row += f" {f['line_items']:>12}"
            else:
                row += f" {'-':>12}"
        print(row)

    # === SUMMARY ===
    print("\n" + "-" * 120)
    print("SUMMARY:")
    print("-" * 120)

    # Fastest average
    fastest = min(all_results, key=lambda r: r["avg_time_ms"])
    print(
        f"  ⚡ Fastest:    {fastest['provider'].upper()} "
        f"at {fastest['avg_time_ms']}ms avg"
    )

    # Highest success rate
    best_success = max(all_results, key=lambda r: r["successes"])
    print(
        f"  ✓ Best success: {best_success['provider'].upper()} "
        f"with {best_success['successes']}/{len(best_success['files'])}"
    )

    # Best accuracy (most within $0.01)
    accuracy_scores = []
    for r in all_results:
        accurate = 0
        total = 0
        for f in r["files"]:
            filename = f["filename"]
            expected = EXPECTED_VALUES.get(filename, {})
            if f["success"] and f.get("grand_total") and expected.get("grand_total"):
                total += 1
                if abs(expected["grand_total"] - f["grand_total"]) <= 0.01:
                    accurate += 1
        accuracy_scores.append((r["provider"], accurate, total))

    best_accuracy = max(accuracy_scores, key=lambda x: x[1])
    print(
        f"  🎯 Best accuracy: {best_accuracy[0].upper()} "
        f"with {best_accuracy[1]}/{best_accuracy[2]} within $0.01"
    )

    print("\n" + "=" * 120)


async def main():
    print("Vendor Quote Extractor - Multi-Provider Benchmark")
    print("Testing OpenAI, Groq, Gemini, Fireworks performance\n")

    # Check for available API keys
    settings = get_settings()

    available_providers = []
    print("Checking API keys...")

    if settings.openai_api_key:
        print(f"  ✓ OpenAI: {settings.openai_api_key[:20]}...")
        available_providers.append("openai")
    else:
        print("  ✗ OpenAI: OPENAI_API_KEY not set")

    if settings.has_groq_key():
        print(f"  ✓ Groq: {settings.groq_api_key[:20]}...")
        available_providers.append("groq")
    else:
        print("  ✗ Groq: GROQ_API_KEY not set")

    if settings.has_gemini_key():
        print(f"  ✓ Gemini: {settings.gemini_api_key[:20]}...")
        available_providers.append("gemini")
    else:
        print("  ✗ Gemini: GEMINI_API_KEY not set")

    if settings.has_fireworks_key():
        print(f"  ✓ Fireworks: {settings.fireworks_api_key[:20]}...")
        available_providers.append("fireworks")
    else:
        print("  ✗ Fireworks: FIREWORKS_API_KEY not set")

    if len(available_providers) < 2:
        print("\nERROR: Need at least 2 providers with API keys to benchmark")
        return

    print(
        f"\nBenchmarking {len(available_providers)} providers: "
        f"{', '.join(available_providers)}"
    )

    # Run benchmarks for all available providers
    all_results = []
    for provider in available_providers:
        results = await run_benchmark_for_provider(provider)
        all_results.append(results)

    # Print comparison
    print_multi_comparison(all_results)


if __name__ == "__main__":
    asyncio.run(main())
