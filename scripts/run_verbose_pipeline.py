#!/usr/bin/env python3
"""
Run customer requests through the full Munder Difflin pipeline with verbose
input/output at each step and optional pauses for manual review.

Formal tests live in tests/test_verbose_pipeline.py (scenarios FP-H1–FP-H3).

Usage:
  source /workspace/.venv/bin/activate
  PYTHONPATH=/workspace python scripts/run_verbose_pipeline.py

Options:
  --no-pause     Skip interactive pauses (for unattended runs)
  --indices 4,5,11   CSV row indices to run (default: 4,5,11)
"""

from __future__ import annotations

import argparse
import os
import sys

from project_starter import db_engine, generate_financial_report, init_database

from agents.order_recommendation_agent import (
    CustomerContext,
    OrderRecommendationRequest,
    call_order_recommendation_agent,
    can_recommend_order,
)
from agents.pricing_review_agent import (
    PricingReviewRequest,
    call_pricing_review_agent,
    can_review_pricing,
)
from agents.quoting_agent import call_quoting_agent
from pipeline.full_pipeline import (
    DEFAULT_PIPELINE_INDICES,
    PipelineContext,
    can_check_inventory,
    format_json,
    load_pipeline_requests,
    validate_inventory_output,
    validate_pricing_output,
    validate_pricing_review_output,
    validate_quote_output,
    validate_recommendation_output,
)
from tools.inventory_tool import InventoryTool, quote_to_inventory_request
from tools.pricing_tool import (
    ItemUnitPrices,
    PricingRequest,
    PricingTool,
    can_price,
)


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _pause(enabled: bool, message: str) -> None:
    if not enabled:
        return
    try:
        input(f"\n>>> {message}")
    except EOFError:
        print("\n(non-interactive terminal — continuing)")


def _report_validation(step: str, errors: list[str]) -> bool:
    if not errors:
        print(f"\n✓ {step} validation PASSED")
        return True
    print(f"\n✗ {step} validation FAILED:")
    for error in errors:
        print(f"  - {error}")
    return False


def step_1_quoting(ctx: PipelineContext, pause: bool) -> bool:
    _banner(f"RUN {ctx.run_number} (CSV index {ctx.csv_index}) — STEP 1: Quoting Agent")
    request_with_date = f"{ctx.request_text} (Date of request: {ctx.request_date})"
    print("\n--- INPUT ---")
    print(
        format_json(
            {
                "request_text": ctx.request_text,
                "request_date": ctx.request_date,
                "request_with_date": request_with_date,
                "metadata": {
                    "job": ctx.job,
                    "need_size": ctx.need_size,
                    "event": ctx.event,
                    "mood": ctx.mood,
                },
            }
        )
    )

    ctx.quote = call_quoting_agent(request_with_date)

    print("\n--- OUTPUT (QuoteResponse) ---")
    print(format_json(ctx.quote))

    ok = _report_validation("Quoting Agent", validate_quote_output(ctx.quote))
    _pause(
        pause,
        "Review Step 1 output, then press Enter for Step 2 (Inventory Tool).",
    )
    return ok and ctx.quote.success


def step_2_inventory(ctx: PipelineContext, pause: bool) -> bool:
    assert ctx.quote is not None
    _banner(
        f"RUN {ctx.run_number} (CSV index {ctx.csv_index}) — STEP 2: Inventory Tool"
    )

    if not can_check_inventory(ctx.quote):
        print("\n--- GATE ---")
        print("can_check_inventory = False — skipping Inventory Tool call.")
        _pause(pause, "Review gate failure, then press Enter to continue.")
        return False

    inv_request = quote_to_inventory_request(ctx.quote)
    print("\n--- INPUT (InventoryRequest) ---")
    print(format_json(inv_request))

    ctx.inventory = InventoryTool().check(inv_request)

    print("\n--- OUTPUT (InventoryResult) ---")
    print(format_json(ctx.inventory))

    ok = _report_validation(
        "Inventory Tool",
        validate_inventory_output(ctx.quote, ctx.inventory),
    )
    all_fulfilled = all(item.success for item in ctx.inventory.items)
    if not all_fulfilled:
        print(
            "\n⚠ One or more lines cannot be fulfilled — "
            "pipeline stops after this step."
        )
    _pause(
        pause,
        "Review Step 2 output, then press Enter for Step 3 (Pricing Review Agent).",
    )
    return ok and all_fulfilled


def step_3_pricing_review(ctx: PipelineContext, pause: bool) -> bool:
    assert ctx.quote is not None
    _banner(
        f"RUN {ctx.run_number} (CSV index {ctx.csv_index}) — "
        "STEP 3: Pricing Review Agent"
    )

    if not can_review_pricing(ctx.quote):
        print("\n--- GATE ---")
        print("can_review_pricing = False — skipping Pricing Review Agent call.")
        _pause(pause, "Review gate failure, then press Enter to continue.")
        return False

    review_request = PricingReviewRequest(quote=ctx.quote)
    print("\n--- INPUT (PricingReviewRequest) ---")
    print(format_json(review_request))

    ctx.pricing_review = call_pricing_review_agent(review_request)

    print("\n--- OUTPUT (PricingReviewResponse) ---")
    print(format_json(ctx.pricing_review))

    ok = _report_validation(
        "Pricing Review Agent",
        validate_pricing_review_output(ctx.quote, ctx.pricing_review),
    )
    _pause(
        pause,
        "Review Step 3 output, then press Enter for Step 4 (Pricing Tool).",
    )
    return ok and ctx.pricing_review.success


def step_4_pricing(ctx: PipelineContext, pause: bool) -> bool:
    assert ctx.quote is not None
    assert ctx.inventory is not None
    assert ctx.pricing_review is not None
    _banner(f"RUN {ctx.run_number} (CSV index {ctx.csv_index}) — STEP 4: Pricing Tool")

    if not can_price(ctx.quote, ctx.inventory):
        print("\n--- GATE ---")
        print("can_price = False — skipping Pricing Tool call.")
        _pause(pause, "Review gate failure, then press Enter to continue.")
        return False

    unit_prices = [
        ItemUnitPrices(
            product_name=item.product_name,
            min_unit_price=item.min_unit_price,
            avg_unit_price=item.avg_unit_price,
            max_unit_price=item.max_unit_price,
        )
        for item in ctx.pricing_review.items
    ]
    pricing_request = PricingRequest(
        quote=ctx.quote,
        inventory=ctx.inventory,
        unit_prices=unit_prices,
    )
    print("\n--- INPUT (PricingRequest) ---")
    print(format_json(pricing_request))

    cash = generate_financial_report(ctx.quote.date_of_request)["cash_balance"]
    print(
        f"\n--- CONTEXT ---\n"
        f"cash_balance as of {ctx.quote.date_of_request}: ${cash:.2f}"
    )

    ctx.pricing = PricingTool().price(pricing_request)

    print("\n--- OUTPUT (PricingResponse) ---")
    print(format_json(ctx.pricing))

    ok = _report_validation(
        "Pricing Tool",
        validate_pricing_output(ctx.quote, ctx.inventory, ctx.pricing),
    )
    _pause(
        pause,
        "Review Step 4 output, then press Enter for Step 5 "
        "(Order Recommendation Agent).",
    )
    return ok and ctx.pricing.success


def step_5_order_recommendation(ctx: PipelineContext, pause: bool) -> bool:
    assert ctx.quote is not None
    assert ctx.inventory is not None
    assert ctx.pricing is not None
    _banner(
        f"RUN {ctx.run_number} (CSV index {ctx.csv_index}) — "
        "STEP 5: Order Recommendation Agent"
    )

    if not can_recommend_order(ctx.quote, ctx.inventory, ctx.pricing):
        print("\n--- GATE ---")
        print("can_recommend_order = False — skipping Order Recommendation Agent call.")
        _pause(pause, "Review gate failure, then press Enter to continue.")
        return False

    order_request = OrderRecommendationRequest(
        quote=ctx.quote,
        inventory=ctx.inventory,
        pricing=ctx.pricing,
        customer=CustomerContext(
            original_request_text=ctx.request_text,
            job_type=ctx.job,
            need_size=ctx.need_size,
            event_type=ctx.event,
            mood=ctx.mood,
        ),
    )
    print("\n--- INPUT (OrderRecommendationRequest) ---")
    print(format_json(order_request))

    ctx.recommendation = call_order_recommendation_agent(order_request)

    print("\n--- OUTPUT (OrderRecommendationResponse) ---")
    print(format_json(ctx.recommendation))

    ok = _report_validation(
        "Order Recommendation Agent",
        validate_recommendation_output(ctx.quote, ctx.pricing, ctx.recommendation),
    )
    _pause(
        pause,
        "Review Step 5 output, then press Enter to finish this request.",
    )
    return ok and ctx.recommendation.success


def run_request(ctx: PipelineContext, pause: bool) -> dict[str, bool]:
    _banner(
        f"RUN {ctx.run_number} (CSV index {ctx.csv_index}): "
        f"{ctx.job} / {ctx.need_size} / {ctx.event}\n"
        f"Date of request: {ctx.request_date}"
    )
    print("\n--- CUSTOMER REQUEST (excerpt) ---")
    excerpt = ctx.request_text.strip().replace("\n", " ")
    print(excerpt[:300] + ("..." if len(excerpt) > 300 else ""))

    results = {
        "quoting": step_1_quoting(ctx, pause),
        "inventory": False,
        "pricing_review": False,
        "pricing": False,
        "order_recommendation": False,
    }

    if not results["quoting"]:
        return results

    results["inventory"] = step_2_inventory(ctx, pause)
    if not results["inventory"]:
        return results

    results["pricing_review"] = step_3_pricing_review(ctx, pause)
    if not results["pricing_review"]:
        return results

    results["pricing"] = step_4_pricing(ctx, pause)
    if not results["pricing"]:
        return results

    results["order_recommendation"] = step_5_order_recommendation(ctx, pause)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verbose full-pipeline walkthrough")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run without interactive pauses between steps",
    )
    default_indices = ",".join(map(str, DEFAULT_PIPELINE_INDICES))
    parser.add_argument(
        "--indices",
        default=",".join(str(i) for i in DEFAULT_PIPELINE_INDICES),
        help=f"Comma-separated CSV indices (default: {default_indices})",
    )
    args = parser.parse_args()
    pause = not args.no_pause
    indices = tuple(
        int(part.strip()) for part in args.indices.split(",") if part.strip()
    )

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. LLM steps (1, 3, 5) require it.")
        return 1

    _banner("Munder Difflin — Verbose Pipeline Demo")
    print(f"Running {len(indices)} request(s) at CSV indices: {indices}")
    print(
        "Steps per request: Quoting → Inventory → Pricing Review → "
        "Pricing → Order Recommendation"
    )
    print("Formal tests: PYTHONPATH=/workspace python tests/test_verbose_pipeline.py")
    if pause:
        print("Interactive mode: pauses between steps. Use --no-pause to disable.")

    print("\nInitializing database...")
    init_database(db_engine)

    contexts = load_pipeline_requests(indices)
    summary: list[tuple[PipelineContext, dict[str, bool]]] = []

    for ctx in contexts:
        step_results = run_request(ctx, pause)
        summary.append((ctx, step_results))
        if pause:
            _pause(
                True,
                f"Finished run {ctx.run_number} (CSV index {ctx.csv_index}). "
                "Press Enter for the next request.",
            )

    _banner("SUMMARY")
    for ctx, step_results in summary:
        passed = [name for name, ok in step_results.items() if ok]
        failed = [name for name, ok in step_results.items() if not ok]
        status = "FULL PIPELINE" if all(step_results.values()) else "PARTIAL / STOPPED"
        print(
            f"\nRun {ctx.run_number} (CSV index {ctx.csv_index}, {ctx.event}): {status}"
        )
        print(f"  Passed: {', '.join(passed) if passed else '(none)'}")
        if failed:
            print(f"  Stopped at/failed: {', '.join(failed)}")

    all_complete = all(all(results.values()) for _, results in summary)
    return 0 if all_complete else 1


if __name__ == "__main__":
    sys.exit(main())
