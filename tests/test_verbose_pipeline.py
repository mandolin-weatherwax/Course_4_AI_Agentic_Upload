"""Full pipeline integration tests — quoting through order recommendation."""

from __future__ import annotations

import os
import sys
from typing import Callable

from project_starter import db_engine, init_database

from agents.quoting_agent import QuoteItem, QuoteResponse
from pipeline.full_pipeline import (
    DEFAULT_PIPELINE_INDICES,
    format_step_failure,
    load_pipeline_requests,
    pipeline_passed,
    run_full_pipeline,
    validate_inventory_output,
    validate_pricing_output,
    validate_pricing_review_output,
    validate_quote_output,
    validate_recommendation_output,
)

MAX_PIPELINE_ATTEMPTS = 2


def _run_scenario(scenario_id: str, test_fn: Callable[[], None]) -> bool:
    try:
        test_fn()
    except AssertionError as exc:
        print(f"{scenario_id}: FAIL - {exc}")
        return False
    except Exception as exc:
        print(f"{scenario_id}: FAIL - {type(exc).__name__}: {exc}")
        return False

    print(f"{scenario_id}: PASS")
    return True


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise AssertionError("OPENAI_API_KEY required for full pipeline LLM tests")


def _assert_full_pipeline(csv_index: int, event_label: str) -> None:
    _require_api_key()

    last_failure = ""
    for attempt in range(1, MAX_PIPELINE_ATTEMPTS + 1):
        init_database(db_engine)

        contexts = load_pipeline_requests((csv_index,))
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.event == event_label

        results = run_full_pipeline(ctx)
        if pipeline_passed(results):
            assert ctx.quote is not None and ctx.quote.success
            assert ctx.inventory is not None and all(
                item.success for item in ctx.inventory.items
            )
            assert ctx.pricing_review is not None and ctx.pricing_review.success
            assert ctx.pricing is not None and ctx.pricing.success
            assert ctx.recommendation is not None and ctx.recommendation.success

            assert not validate_quote_output(ctx.quote)
            assert not validate_inventory_output(ctx.quote, ctx.inventory)
            assert not validate_pricing_review_output(ctx.quote, ctx.pricing_review)
            assert not validate_pricing_output(ctx.quote, ctx.inventory, ctx.pricing)
            assert not validate_recommendation_output(
                ctx.quote, ctx.pricing, ctx.recommendation
            )

            assert ctx.recommendation.customer_response.strip()
            assert ctx.recommendation.pricing_justification.strip()
            assert ctx.recommendation.sales is not None
            assert len(ctx.recommendation.sales.transactions) >= 1
            return

        last_failure = format_step_failure(results)

    raise AssertionError(
        f"pipeline failed after {MAX_PIPELINE_ATTEMPTS} attempt(s): {last_failure}"
    )


def test_fp_t1_validators_accept_well_formed_outputs() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            )
        ],
    )
    assert not validate_quote_output(quote)


def test_fp_h1_party_csv_index_4() -> None:
    _assert_full_pipeline(4, "party")


def test_fp_h2_assembly_csv_index_5() -> None:
    _assert_full_pipeline(5, "assembly")


def test_fp_h3_show_csv_index_11() -> None:
    _assert_full_pipeline(11, "show")


def test_fp_v1_default_indices_match_runner() -> None:
    assert DEFAULT_PIPELINE_INDICES == (4, 5, 11)
    contexts = load_pipeline_requests(DEFAULT_PIPELINE_INDICES)
    assert [ctx.csv_index for ctx in contexts] == [4, 5, 11]
    assert [ctx.event for ctx in contexts] == ["party", "assembly", "show"]
    assert [ctx.mood for ctx in contexts] == [
        "stressed",
        "pissed off",
        "happy",
    ]


def main() -> int:
    scenarios = [
        ("FP-T1", test_fp_t1_validators_accept_well_formed_outputs),
        ("FP-V1", test_fp_v1_default_indices_match_runner),
        ("FP-H1", test_fp_h1_party_csv_index_4),
        ("FP-H2", test_fp_h2_assembly_csv_index_5),
        ("FP-H3", test_fp_h3_show_csv_index_11),
    ]

    passed = sum(_run_scenario(scenario_id, fn) for scenario_id, fn in scenarios)
    total = len(scenarios)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
