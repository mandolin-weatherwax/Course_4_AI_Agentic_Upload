"""Pricing review agent test scenarios."""

from __future__ import annotations

import sys
from typing import Callable
from unittest.mock import patch

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.pricing_review_agent import (
    MIN_HISTORY_ORDERS,
    ComputeItemUnitPricesInput,
    HistoricalLineCost,
    ItemPriceReview,
    PricingReviewRequest,
    PricingReviewResponse,
    _normalize_review_response,
    call_pricing_review_agent,
    can_review_pricing,
    compute_item_unit_prices,
    pricing_review_agent,
    tool_compute_item_unit_prices,
    tool_search_quote_history,
)
from agents.quoting_agent import QuoteItem, QuoteResponse
from tools.pricing_tool import MIN_UNIT_PRICE_FLOOR, default_unit_prices


def _build_item_price_review(
    *,
    product_name: str,
    unit_cost: float,
    line_costs: list[HistoricalLineCost],
) -> ItemPriceReview:
    """Simulate agent assembly from LLM-extracted line_costs."""
    history_count = len(line_costs)

    if history_count >= MIN_HISTORY_ORDERS:
        result = compute_item_unit_prices(
            ComputeItemUnitPricesInput(
                product_name=product_name,
                unit_cost=unit_cost,
                line_costs=line_costs,
            )
        )
        if hasattr(result, "error"):
            fallback = default_unit_prices(product_name, unit_cost)
            return ItemPriceReview(
                product_name=product_name,
                unit_cost=unit_cost,
                history_count=history_count,
                min_unit_price=fallback.min_unit_price,
                avg_unit_price=fallback.avg_unit_price,
                max_unit_price=fallback.max_unit_price,
                error=result.error,
            )
        return ItemPriceReview(
            product_name=product_name,
            unit_cost=unit_cost,
            history_count=result.history_count,
            min_unit_price=result.min_unit_price,
            avg_unit_price=result.avg_unit_price,
            max_unit_price=result.max_unit_price,
            error=None,
        )

    fallback = default_unit_prices(product_name, unit_cost)
    return ItemPriceReview(
        product_name=product_name,
        unit_cost=unit_cost,
        history_count=history_count,
        min_unit_price=fallback.min_unit_price,
        avg_unit_price=fallback.avg_unit_price,
        max_unit_price=fallback.max_unit_price,
        error=(
            f"Only {history_count} historical orders found; need at least "
            f"{MIN_HISTORY_ORDERS}. Used DEFAULT_STRATEGY_MULTIPLIERS."
        ),
    )


def _quote(
    *,
    items: list[QuoteItem],
    date_of_request: str = "2025-04-01",
    need_date: str = "2025-04-10",
) -> QuoteResponse:
    return QuoteResponse(
        success=True,
        date_of_request=date_of_request,
        need_date=need_date,
        items=items,
    )


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


def test_pr_t1_compute_three_line_costs() -> None:
    result = compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="A4 paper",
            unit_cost=0.05,
            line_costs=[
                HistoricalLineCost(total_cost=25.0, quantity=500),
                HistoricalLineCost(total_cost=30.0, quantity=500),
                HistoricalLineCost(total_cost=27.5, quantity=500),
            ],
        )
    )
    assert result.min_unit_price == 0.05
    assert result.max_unit_price == 0.06
    assert result.avg_unit_price == 0.06
    assert result.history_count == 3


def test_pr_t2_min_price_floor_clamped() -> None:
    unit_cost = 0.10
    result = compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="Deep discount paper",
            unit_cost=unit_cost,
            line_costs=[
                HistoricalLineCost(total_cost=5.0, quantity=100),
                HistoricalLineCost(total_cost=6.0, quantity=100),
                HistoricalLineCost(total_cost=7.0, quantity=100),
            ],
        )
    )
    floor = round(unit_cost * MIN_UNIT_PRICE_FLOOR, 2)
    assert result.min_unit_price == floor
    assert result.min_unit_price == 0.09


def test_pr_t2b_normalize_clamps_low_agent_bands() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            QuoteItem(
                product_name="Glossy paper",
                quantity_requested=200,
                unit_price=0.20,
            )
        ],
    )
    agent_response = PricingReviewResponse(
        success=True,
        date_of_request="2025-04-01",
        items=[
            ItemPriceReview(
                product_name="Glossy paper",
                unit_cost=0.20,
                history_count=3,
                min_unit_price=0.10,
                avg_unit_price=0.12,
                max_unit_price=0.15,
            )
        ],
    )
    normalized = _normalize_review_response(agent_response, quote)
    item = normalized.items[0]
    assert item.min_unit_price == round(0.20 * MIN_UNIT_PRICE_FLOOR, 2)
    assert item.avg_unit_price >= item.min_unit_price
    assert item.max_unit_price >= item.avg_unit_price


def test_pr_t3_compute_from_two_pairs() -> None:
    result = compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="Cardstock",
            unit_cost=0.15,
            line_costs=[
                HistoricalLineCost(total_cost=30.0, quantity=200),
                HistoricalLineCost(total_cost=32.0, quantity=200),
            ],
        )
    )
    assert result.history_count == 2
    assert result.min_unit_price == 0.15
    assert result.max_unit_price == 0.16
    assert result.avg_unit_price == 0.15


def test_pr_t3b_rejects_no_valid_pairs() -> None:
    result = compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="Cardstock",
            unit_cost=0.15,
            line_costs=[],
        )
    )
    assert hasattr(result, "error")
    dumped = tool_compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="Cardstock",
            unit_cost=0.15,
            line_costs=[HistoricalLineCost(total_cost=0.0, quantity=0)],
        )
    )
    assert "error" in dumped


def test_pr_t4_skips_zero_quantity_entries() -> None:
    result = compute_item_unit_prices(
        ComputeItemUnitPricesInput(
            product_name="Glossy paper",
            unit_cost=0.08,
            line_costs=[
                HistoricalLineCost(total_cost=20.0, quantity=200),
                HistoricalLineCost(total_cost=0.0, quantity=0),
                HistoricalLineCost(total_cost=22.0, quantity=200),
                HistoricalLineCost(total_cost=21.0, quantity=200),
            ],
        )
    )
    assert result.history_count == 3
    assert result.min_unit_price == 0.10
    assert result.max_unit_price == 0.11


def test_pr_h1_sufficient_history_from_extracted_pairs() -> None:
    review = _build_item_price_review(
        product_name="A4 paper",
        unit_cost=0.05,
        line_costs=[
            HistoricalLineCost(total_cost=25.0, quantity=500),
            HistoricalLineCost(total_cost=30.0, quantity=500),
            HistoricalLineCost(total_cost=35.0, quantity=500),
        ],
    )
    assert review.history_count >= MIN_HISTORY_ORDERS
    assert review.error is None
    assert review.min_unit_price < review.avg_unit_price < review.max_unit_price


def test_pr_h2_multi_item_mixed_history() -> None:
    a4 = _build_item_price_review(
        product_name="A4 paper",
        unit_cost=0.05,
        line_costs=[
            HistoricalLineCost(total_cost=25.0, quantity=500),
            HistoricalLineCost(total_cost=30.0, quantity=500),
            HistoricalLineCost(total_cost=35.0, quantity=500),
        ],
    )
    cardstock = _build_item_price_review(
        product_name="Cardstock",
        unit_cost=0.15,
        line_costs=[
            HistoricalLineCost(total_cost=30.0, quantity=200),
        ],
    )
    assert a4.error is None
    assert cardstock.error is not None
    assert cardstock.history_count == 1


def test_pr_h3_exactly_three_pairs() -> None:
    review = _build_item_price_review(
        product_name="A4 paper",
        unit_cost=0.05,
        line_costs=[
            HistoricalLineCost(total_cost=25.0, quantity=500),
            HistoricalLineCost(total_cost=30.0, quantity=500),
            HistoricalLineCost(total_cost=35.0, quantity=500),
        ],
    )
    assert review.history_count == 3
    assert review.error is None


def test_pr_i1_two_pairs_uses_fallback() -> None:
    review = _build_item_price_review(
        product_name="Cardstock",
        unit_cost=0.15,
        line_costs=[
            HistoricalLineCost(total_cost=30.0, quantity=200),
            HistoricalLineCost(total_cost=32.0, quantity=200),
        ],
    )
    fallback = default_unit_prices("Cardstock", 0.15)
    assert review.history_count == 2
    assert review.error is not None
    assert review.min_unit_price == fallback.min_unit_price
    assert review.avg_unit_price == fallback.avg_unit_price
    assert review.max_unit_price == fallback.max_unit_price


def test_pr_i2_zero_history_uses_fallback() -> None:
    review = _build_item_price_review(
        product_name="Banner paper",
        unit_cost=0.30,
        line_costs=[],
    )
    fallback = default_unit_prices("Banner paper", 0.30)
    assert review.history_count == 0
    assert review.error is not None
    assert review.min_unit_price == fallback.min_unit_price


def test_pr_i3_compute_not_used_when_under_threshold() -> None:
    with patch(
        "agents.pricing_review_agent.compute_item_unit_prices",
    ) as mock_compute:
        _build_item_price_review(
            product_name="Cardstock",
            unit_cost=0.15,
            line_costs=[
                HistoricalLineCost(total_cost=30.0, quantity=200),
            ],
        )
        mock_compute.assert_not_called()


def test_pr_e1_empty_items() -> None:
    response = call_pricing_review_agent(PricingReviewRequest(quote=_quote(items=[])))
    assert response.success is False
    assert response.error is not None


def test_pr_e2_missing_date() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request=None,
        need_date="2025-04-10",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            )
        ],
    )
    assert can_review_pricing(quote) is False
    response = call_pricing_review_agent(PricingReviewRequest(quote=quote))
    assert response.success is False


def test_tool_search_quote_history_delegate() -> None:
    with patch(
        "agents.pricing_review_agent.search_quote_history",
        return_value=[{"quote_explanation": "sample"}],
    ) as mock_search:
        result = tool_search_quote_history(["A4 paper"], limit=5)
        mock_search.assert_called_once_with(["A4 paper"], limit=5)
        assert result == [{"quote_explanation": "sample"}]


def _llm_extract_pairs(
    product_name: str, history: list[dict]
) -> list[HistoricalLineCost]:
    """Simulate LLM extraction from quote_explanation prose."""
    pairs: list[HistoricalLineCost] = []
    for record in history:
        text = record["quote_explanation"]
        if product_name == "A4 paper" and "A4 paper" in text:
            if "$0.05 each" in text:
                pairs.append(HistoricalLineCost(total_cost=25.0, quantity=500))
            elif "$0.06 each" in text:
                pairs.append(HistoricalLineCost(total_cost=30.0, quantity=500))
            elif "$0.07 each" in text:
                pairs.append(HistoricalLineCost(total_cost=35.0, quantity=500))
        elif product_name == "Cardstock" and "Cardstock" in text:
            pairs.append(HistoricalLineCost(total_cost=30.0, quantity=200))
    return pairs


def test_pr_a1_agent_calls_typed_compute_tool() -> None:
    history = {
        "A4 paper": [
            {"quote_explanation": "500 sheets of A4 paper at $0.05 each."},
            {"quote_explanation": "500 sheets of A4 paper at $0.06 each."},
            {"quote_explanation": "500 sheets of A4 paper at $0.07 each."},
        ]
    }
    compute_calls: list[ComputeItemUnitPricesInput] = []

    def simulated_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if hasattr(part, "tool_name")
        ]
        if not tool_returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_search_quote_history",
                        args={"search_terms": ["A4 paper"], "limit": 20},
                    )
                ]
            )

        last_return = tool_returns[-1]
        if last_return.tool_name == "tool_search_quote_history":
            line_costs = _llm_extract_pairs("A4 paper", history["A4 paper"])
            compute_calls.append(
                ComputeItemUnitPricesInput(
                    product_name="A4 paper",
                    unit_cost=0.05,
                    line_costs=line_costs,
                )
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_compute_item_unit_prices",
                        args={
                            "payload": {
                                "product_name": "A4 paper",
                                "unit_cost": 0.05,
                                "line_costs": [
                                    pair.model_dump() for pair in line_costs
                                ],
                            }
                        },
                    )
                ]
            )

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "success": True,
                        "date_of_request": "2025-04-01",
                        "items": [
                            {
                                "product_name": "A4 paper",
                                "unit_cost": 0.05,
                                "history_count": 3,
                                "min_unit_price": 0.05,
                                "avg_unit_price": 0.06,
                                "max_unit_price": 0.07,
                                "error": None,
                            }
                        ],
                    },
                )
            ]
        )

    request = PricingReviewRequest(
        quote=_quote(
            items=[
                QuoteItem(
                    product_name="A4 paper",
                    quantity_requested=100,
                    unit_price=0.05,
                )
            ]
        )
    )

    with patch(
        "agents.pricing_review_agent.search_quote_history",
        side_effect=lambda terms, limit=20: history[terms[0]],
    ):
        with pricing_review_agent.override(model=FunctionModel(simulated_llm)):
            response = call_pricing_review_agent(request)

    assert response.success is True
    assert len(compute_calls) == 1
    assert len(compute_calls[0].line_costs) == 3
    assert response.items[0].error is None


def main() -> int:
    scenarios = [
        ("PR-T1", test_pr_t1_compute_three_line_costs),
        ("PR-T2", test_pr_t2_min_price_floor_clamped),
        ("PR-T2b", test_pr_t2b_normalize_clamps_low_agent_bands),
        ("PR-T3", test_pr_t3_compute_from_two_pairs),
        ("PR-T3b", test_pr_t3b_rejects_no_valid_pairs),
        ("PR-T4", test_pr_t4_skips_zero_quantity_entries),
        ("PR-H1", test_pr_h1_sufficient_history_from_extracted_pairs),
        ("PR-H2", test_pr_h2_multi_item_mixed_history),
        ("PR-H3", test_pr_h3_exactly_three_pairs),
        ("PR-I1", test_pr_i1_two_pairs_uses_fallback),
        ("PR-I2", test_pr_i2_zero_history_uses_fallback),
        ("PR-I3", test_pr_i3_compute_not_used_when_under_threshold),
        ("PR-E1", test_pr_e1_empty_items),
        ("PR-E2", test_pr_e2_missing_date),
        ("PR-S1", test_tool_search_quote_history_delegate),
        ("PR-A1", test_pr_a1_agent_calls_typed_compute_tool),
    ]

    passed = sum(_run_scenario(scenario_id, fn) for scenario_id, fn in scenarios)
    total = len(scenarios)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
