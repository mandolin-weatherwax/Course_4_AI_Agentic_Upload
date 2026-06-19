"""Order recommendation agent tool tests."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from unittest.mock import patch

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.order_recommendation_agent import (
    BuildTransactionsInput,
    CompanyContextInput,
    CompanyContextResult,
    CustomerContext,
    OrderRecommendationRequest,
    OrderRecommendationResponse,
    PriceBand,
    PricingSignalsInput,
    RecommendedLineItem,
    TransactionBatch,
    TransactionRecord,
    ValidateOrderRecommendationPayload,
    _validation_error,
    build_fallback_recommendation,
    build_transaction_batches,
    call_order_recommendation_agent,
    can_recommend_order,
    compute_inventory_pressure,
    customer_context_from_csv_row,
    evaluate_pricing_signals,
    extract_price_bands,
    gather_company_context,
    order_recommendation_agent,
    tool_build_transaction_batches,
    tool_evaluate_pricing_signals,
    tool_extract_price_bands,
    tool_gather_company_context,
    tool_validate_order_recommendation,
    validate_order_recommendation,
)
from agents.quoting_agent import QuoteItem, QuoteResponse
from tools.inventory_tool import CheckedItem, InventoryResult
from tools.pricing_tool import (
    PricedLineItem,
    PricingRecommendation,
    PricingResponse,
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


def _priced_line(
    *,
    product_name: str = "A4 paper",
    quantity_requested: int = 500,
    quantity_fulfilled: int = 500,
    unit_cost: float = 0.05,
    unit_price: float,
    included: bool = True,
) -> PricedLineItem:
    return PricedLineItem(
        product_name=product_name,
        quantity_requested=quantity_requested,
        quantity_fulfilled=quantity_fulfilled,
        unit_cost=unit_cost,
        unit_price=unit_price,
        line_revenue=round(unit_price * quantity_fulfilled, 2),
        line_acquisition_cost=round(unit_cost * quantity_fulfilled, 2),
        included=included,
    )


def _pricing_response(
    *,
    floor: float = 0.05,
    mid: float = 0.06,
    ceiling: float = 0.07,
    quantity_fulfilled: int = 500,
    included: bool = True,
    product_name: str = "A4 paper",
    quantity_requested: int = 500,
    unit_cost: float = 0.05,
) -> PricingResponse:
    line_args = dict(
        product_name=product_name,
        quantity_requested=quantity_requested,
        quantity_fulfilled=quantity_fulfilled,
        unit_cost=unit_cost,
        included=included,
    )
    acquisition = round(unit_cost * quantity_fulfilled, 2)
    return PricingResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        recommendations=[
            PricingRecommendation(
                strategy="maximize_turnover",
                items=[_priced_line(unit_price=floor, **line_args)],
                total_acquisition_cost=acquisition,
                total_profit=round(floor * quantity_fulfilled - acquisition, 2),
            ),
            PricingRecommendation(
                strategy="average_pricing",
                items=[_priced_line(unit_price=mid, **line_args)],
                total_acquisition_cost=acquisition,
                total_profit=round(mid * quantity_fulfilled - acquisition, 2),
            ),
            PricingRecommendation(
                strategy="maximize_profit",
                items=[_priced_line(unit_price=ceiling, **line_args)],
                total_acquisition_cost=acquisition,
                total_profit=round(ceiling * quantity_fulfilled - acquisition, 2),
            ),
        ],
    )


def _pricing_response_multi(
    specs: list[dict[str, Any]],
) -> PricingResponse:
    recommendations_by_strategy: dict[str, list[PricedLineItem]] = {
        "maximize_turnover": [],
        "average_pricing": [],
        "maximize_profit": [],
    }
    totals: dict[str, dict[str, float]] = {
        "maximize_turnover": {"acquisition": 0.0, "profit": 0.0},
        "average_pricing": {"acquisition": 0.0, "profit": 0.0},
        "maximize_profit": {"acquisition": 0.0, "profit": 0.0},
    }

    for spec in specs:
        fulfilled = spec["quantity_fulfilled"]
        unit_cost = spec.get("unit_cost", 0.05)
        for strategy, price_key in (
            ("maximize_turnover", "floor"),
            ("average_pricing", "mid"),
            ("maximize_profit", "ceiling"),
        ):
            unit_price = spec[price_key]
            line = _priced_line(
                product_name=spec["product_name"],
                quantity_requested=spec["quantity_requested"],
                quantity_fulfilled=fulfilled,
                unit_cost=unit_cost,
                unit_price=unit_price,
                included=spec.get("included", True),
            )
            recommendations_by_strategy[strategy].append(line)
            totals[strategy]["acquisition"] += line.line_acquisition_cost
            totals[strategy]["profit"] += line.line_revenue - line.line_acquisition_cost

    return PricingResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        recommendations=[
            PricingRecommendation(
                strategy=strategy,  # type: ignore[arg-type]
                items=items,
                total_acquisition_cost=round(totals[strategy]["acquisition"], 2),
                total_profit=round(totals[strategy]["profit"], 2),
            )
            for strategy, items in recommendations_by_strategy.items()
        ],
    )


def _quote(
    *,
    items: list[QuoteItem] | None = None,
    date_of_request: str = "2025-04-01",
    need_date: str = "2025-04-10",
) -> QuoteResponse:
    if items is None:
        items = [
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=500,
                unit_price=0.05,
            )
        ]
    return QuoteResponse(
        success=True,
        date_of_request=date_of_request,
        need_date=need_date,
        items=items,
    )


def _inventory(
    *,
    lines: list[dict[str, Any]] | None = None,
) -> InventoryResult:
    if lines is None:
        lines = [
            {
                "product_name": "A4 paper",
                "quantity_requested": 500,
                "unit_price": 0.05,
                "quantity_in_stock": 100,
                "quantity_to_order": 400,
            }
        ]
    return InventoryResult(
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            CheckedItem(
                product_name=line["product_name"],
                quantity_requested=line["quantity_requested"],
                unit_price=line.get("unit_price", 0.05),
                success=True,
                quantity_in_stock=line["quantity_in_stock"],
                quantity_to_order=line["quantity_to_order"],
            )
            for line in lines
        ],
    )


@contextmanager
def _mock_company_services(
    *,
    inventory: dict[str, int],
    cash_balance: float = 1200.0,
    inventory_value: float = 2500.0,
) -> Iterator[None]:
    financial = {
        "cash_balance": cash_balance,
        "inventory_value": inventory_value,
        "total_assets": cash_balance + inventory_value,
        "inventory_summary": [],
        "top_selling_products": [],
    }
    with (
        patch(
            "agents.order_recommendation_agent.get_all_inventory",
            return_value=inventory,
        ),
        patch(
            "agents.order_recommendation_agent.generate_financial_report",
            return_value=financial,
        ),
    ):
        yield


def _tool_return_content(messages: list[ModelMessage], tool_name: str) -> Any:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if getattr(part, "tool_name", None) == tool_name:
                return part.content
    raise AssertionError(f"No tool return found for {tool_name!r}")


def test_or_t1_inventory_pressure_twenty_percent() -> None:
    pressure = compute_inventory_pressure(
        "2025-04-01",
        ["A4 paper"],
        inventory={"A4 paper": 200, "Cardstock": 100, "Glossy paper": 700},
    )
    assert pressure["inventory_pressure_pct"] == 20.0
    assert pressure["ordered_in_stock_units"] == 200
    assert pressure["total_units"] == 1000


def test_or_t2_empty_warehouse() -> None:
    pressure = compute_inventory_pressure(
        "2025-04-01",
        ["A4 paper"],
        inventory={},
    )
    assert pressure["inventory_pressure_pct"] == 0.0
    assert pressure["total_units"] == 0


def test_or_t3_unknown_products_treated_as_zero() -> None:
    pressure = compute_inventory_pressure(
        "2025-04-01",
        ["Unknown product"],
        inventory={"A4 paper": 100},
    )
    assert pressure["inventory_pressure_pct"] == 0.0
    assert pressure["ordered_in_stock_units"] == 0


def test_or_t4_gather_company_context() -> None:
    mock_inventory = {"A4 paper": 400, "Cardstock": 100}
    mock_financial = {
        "cash_balance": 1200.0,
        "inventory_value": 2500.0,
        "total_assets": 3700.0,
        "inventory_summary": [
            {
                "item_name": "A4 paper",
                "stock": 400,
                "unit_price": 0.05,
                "value": 20.0,
            }
        ],
        "top_selling_products": [
            {"item_name": "A4 paper", "total_units": 10, "total_revenue": 50.0}
        ],
    }

    with (
        patch(
            "agents.order_recommendation_agent.get_all_inventory",
            return_value=mock_inventory,
        ),
        patch(
            "agents.order_recommendation_agent.generate_financial_report",
            return_value=mock_financial,
        ),
    ):
        result = gather_company_context(
            CompanyContextInput(
                date_of_request="2025-04-01",
                ordered_products=["A4 paper"],
            )
        )
        dumped = tool_gather_company_context(
            CompanyContextInput(
                date_of_request="2025-04-01", ordered_products=["A4 paper"]
            )
        )

    assert result.inventory == mock_inventory
    assert result.cash_balance == 1200.0
    assert result.inventory_value == 2500.0
    assert result.inventory_pressure_pct == 80.0
    assert dumped["total_assets"] == 3700.0
    assert dumped["inventory_pressure_pct"] == 80.0


def test_or_t5_extract_price_bands() -> None:
    bands = extract_price_bands(_pricing_response(floor=0.05, mid=0.06, ceiling=0.07))
    assert len(bands) == 1
    band = bands[0]
    assert band.floor == 0.05
    assert band.mid == 0.06
    assert band.ceiling == 0.07
    assert band.floor <= band.mid <= band.ceiling
    assert band.included is True
    assert band.quantity_fulfilled == 500

    dumped = tool_extract_price_bands(_pricing_response())
    assert dumped[0]["product_name"] == "A4 paper"


def test_or_t6_build_transaction_batches() -> None:
    items = [
        RecommendedLineItem(
            product_name="A4 paper",
            quantity_requested=500,
            quantity_fulfilled=500,
            quantity_in_stock=100,
            quantity_to_order=400,
            unit_cost=0.05,
            unit_price=0.06,
            line_total=30.0,
            included=True,
        ),
        RecommendedLineItem(
            product_name="Cardstock",
            quantity_requested=200,
            quantity_fulfilled=200,
            quantity_in_stock=200,
            quantity_to_order=0,
            unit_cost=0.15,
            unit_price=0.16,
            line_total=32.0,
            included=True,
        ),
    ]

    result = build_transaction_batches(
        BuildTransactionsInput(
            date_of_request="2025-04-01",
            need_date="2025-04-10",
            recommended_items=items,
        )
    )

    assert result.stock_orders is not None
    assert result.stock_orders.transaction_date == "2025-04-01"
    assert len(result.stock_orders.transactions) == 1
    assert result.stock_orders.transactions[0].transaction_type == "stock_orders"
    assert result.stock_orders.transactions[0].quantity == 400
    assert result.stock_orders.transactions[0].price == 20.0

    assert result.sales is not None
    assert result.sales.transaction_date == "2025-04-10"
    assert len(result.sales.transactions) == 2
    assert result.sales.transactions[0].transaction_type == "sales"
    assert result.sales.transactions[0].price == 30.0

    dumped = tool_build_transaction_batches(
        BuildTransactionsInput(
            date_of_request="2025-04-01",
            need_date="2025-04-10",
            recommended_items=items,
        )
    )
    assert dumped["sales"]["transaction_date"] == "2025-04-10"


def test_or_t7_evaluate_pricing_signals_cash_stress() -> None:
    result = evaluate_pricing_signals(
        PricingSignalsInput(
            company_context=CompanyContextResult(
                as_of_date="2025-04-01",
                inventory={"A4 paper": 100},
                cash_balance=100.0,
                inventory_value=3000.0,
                total_assets=3100.0,
                inventory_summary=[],
                top_selling_products=[],
                inventory_pressure_pct=5.0,
                ordered_in_stock_units=100,
                total_units=100,
            ),
            customer=CustomerContext(
                original_request_text="Need paper",
                need_size="medium",
                event_type="meeting",
            ),
        )
    )
    assert result.lean_premium_cash_stress is True
    assert result.lean_discount_inventory is False
    assert "premium lean" in result.summary

    dumped = tool_evaluate_pricing_signals(
        PricingSignalsInput(
            company_context=CompanyContextResult(
                as_of_date="2025-04-01",
                inventory={},
                cash_balance=100.0,
                inventory_value=3000.0,
                total_assets=3100.0,
                inventory_summary=[],
                top_selling_products=[],
                inventory_pressure_pct=0.0,
                ordered_in_stock_units=0,
                total_units=0,
            ),
            customer=CustomerContext(original_request_text="Need paper"),
        )
    )
    assert dumped["lean_premium_cash_stress"] is True


def _company_context_for_signals() -> CompanyContextResult:
    return CompanyContextResult(
        as_of_date="2025-04-01",
        inventory={"A4 paper": 100},
        cash_balance=1200.0,
        inventory_value=500.0,
        total_assets=1700.0,
        inventory_summary=[],
        top_selling_products=[],
        inventory_pressure_pct=5.0,
        ordered_in_stock_units=100,
        total_units=100,
    )


def test_or_t7b_mood_happy_premium_lean() -> None:
    result = evaluate_pricing_signals(
        PricingSignalsInput(
            company_context=_company_context_for_signals(),
            customer=CustomerContext(
                original_request_text="Need paper",
                mood="happy",
                event_type="meeting",
            ),
        )
    )
    assert result.lean_premium_mood is True
    assert result.lean_discount_mood is False
    assert result.mood == "happy"
    assert "mood=happy (premium lean)" in result.summary


def test_or_i1_customer_context_from_csv_row() -> None:
    import pandas as pd

    row = pd.Series(
        {
            "mood": "happy",
            "job": "event manager",
            "need_size": "medium",
            "event": "party",
            "request": "Need napkins for a party.",
        }
    )
    customer = customer_context_from_csv_row(row)
    assert customer.mood == "happy"
    assert customer.job_type == "event manager"
    assert customer.need_size == "medium"
    assert customer.event_type == "party"
    assert customer.original_request_text == "Need napkins for a party."

    row_no_mood = pd.Series(
        {
            "job": "teacher",
            "need_size": "small",
            "event": "assembly",
            "request": "Need paper.",
        }
    )
    assert customer_context_from_csv_row(row_no_mood).mood is None


def test_or_t7c_mood_sad_discount_lean() -> None:
    result = evaluate_pricing_signals(
        PricingSignalsInput(
            company_context=_company_context_for_signals(),
            customer=CustomerContext(
                original_request_text="Need paper",
                mood="miserable",
                event_type="party",
            ),
        )
    )
    assert result.lean_discount_mood is True
    assert result.lean_premium_mood is False
    assert result.lean_discount_event is True
    assert result.mood == "miserable"
    assert "mood=miserable (discount lean)" in result.summary
    assert "event_type=party (discount lean)" in result.summary


def test_or_t8c_fallback_recommendation_passes_validation() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-17",
        need_date="2025-05-15",
        items=[
            QuoteItem(product_name="Flyers", quantity_requested=5000, unit_price=0.15),
            QuoteItem(
                product_name="Poster paper",
                quantity_requested=2000,
                unit_price=0.25,
            ),
        ],
        excluded_products=["tickets"],
    )
    inventory = InventoryResult(
        date_of_request="2025-04-17",
        need_date="2025-05-15",
        items=[
            CheckedItem(
                product_name="Flyers",
                quantity_requested=5000,
                unit_price=0.15,
                success=True,
                quantity_in_stock=0,
                quantity_to_order=5000,
            ),
            CheckedItem(
                product_name="Poster paper",
                quantity_requested=2000,
                unit_price=0.25,
                success=True,
                quantity_in_stock=299,
                quantity_to_order=1701,
            ),
        ],
    )
    pricing = _pricing_response(
        floor=0.14,
        mid=0.15,
        ceiling=0.18,
        quantity_fulfilled=5000,
        product_name="Flyers",
        quantity_requested=5000,
        unit_cost=0.15,
    )
    pricing = PricingResponse(
        success=True,
        date_of_request="2025-04-17",
        need_date="2025-05-15",
        recommendations=[
            PricingRecommendation(
                strategy="maximize_turnover",
                items=[
                    _priced_line(
                        product_name="Flyers",
                        unit_price=0.14,
                        quantity_fulfilled=5000,
                    ),
                    _priced_line(
                        product_name="Poster paper",
                        quantity_requested=2000,
                        quantity_fulfilled=2000,
                        unit_cost=0.25,
                        unit_price=0.23,
                    ),
                ],
                total_acquisition_cost=1000.0,
                total_profit=100.0,
            ),
            PricingRecommendation(
                strategy="average_pricing",
                items=[
                    _priced_line(
                        product_name="Flyers",
                        unit_price=0.15,
                        quantity_fulfilled=5000,
                    ),
                    _priced_line(
                        product_name="Poster paper",
                        quantity_requested=2000,
                        quantity_fulfilled=2000,
                        unit_cost=0.25,
                        unit_price=0.26,
                    ),
                ],
                total_acquisition_cost=1000.0,
                total_profit=150.0,
            ),
            PricingRecommendation(
                strategy="maximize_profit",
                items=[
                    _priced_line(
                        product_name="Flyers",
                        unit_price=0.18,
                        quantity_fulfilled=5000,
                    ),
                    _priced_line(
                        product_name="Poster paper",
                        quantity_requested=2000,
                        quantity_fulfilled=2000,
                        unit_cost=0.25,
                        unit_price=0.30,
                    ),
                ],
                total_acquisition_cost=1000.0,
                total_profit=200.0,
            ),
        ],
    )
    request = OrderRecommendationRequest(
        quote=quote,
        inventory=inventory,
        pricing=pricing,
        customer=CustomerContext(original_request_text="concert order"),
    )
    fallback = build_fallback_recommendation(request)
    assert fallback.success is True
    assert _validation_error(fallback, pricing) is None
    assert fallback.total_quote_amount == 1270.0


def test_or_t8_validate_rejects_price_above_ceiling() -> None:
    bands = [
        PriceBand(
            product_name="A4 paper",
            floor=0.05,
            mid=0.06,
            ceiling=0.07,
            unit_cost=0.05,
            quantity_fulfilled=500,
            included=True,
        )
    ]
    invalid = OrderRecommendationResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        recommended_items=[
            RecommendedLineItem(
                product_name="A4 paper",
                quantity_requested=500,
                quantity_fulfilled=500,
                quantity_in_stock=100,
                quantity_to_order=400,
                unit_cost=0.05,
                unit_price=0.10,
                line_total=50.0,
                included=True,
            )
        ],
        stock_orders=TransactionBatch(
            transaction_date="2025-04-01",
            transactions=[
                TransactionRecord(
                    item_name="A4 paper",
                    transaction_type="stock_orders",
                    quantity=400,
                    price=20.0,
                    transaction_date="2025-04-01",
                )
            ],
        ),
        sales=TransactionBatch(
            transaction_date="2025-04-10",
            transactions=[
                TransactionRecord(
                    item_name="A4 paper",
                    transaction_type="sales",
                    quantity=500,
                    price=50.0,
                    transaction_date="2025-04-10",
                )
            ],
        ),
        customer_response="Thank you for your order.",
        pricing_justification="Test justification.",
        total_quote_amount=50.0,
    )

    result = validate_order_recommendation(
        ValidateOrderRecommendationPayload(response=invalid, price_bands=bands)
    )
    assert hasattr(result, "error")

    dumped = tool_validate_order_recommendation(
        ValidateOrderRecommendationPayload(response=invalid, price_bands=bands)
    )
    assert "error" in dumped
    assert "outside" in dumped["error"]


def test_or_t8b_validate_accepts_in_band_response() -> None:
    bands = [
        PriceBand(
            product_name="A4 paper",
            floor=0.05,
            mid=0.06,
            ceiling=0.07,
            unit_cost=0.05,
            quantity_fulfilled=500,
            included=True,
        )
    ]
    valid = OrderRecommendationResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        recommended_items=[
            RecommendedLineItem(
                product_name="A4 paper",
                quantity_requested=500,
                quantity_fulfilled=500,
                quantity_in_stock=100,
                quantity_to_order=400,
                unit_cost=0.05,
                unit_price=0.06,
                line_total=30.0,
                included=True,
            )
        ],
        sales=TransactionBatch(
            transaction_date="2025-04-10",
            transactions=[
                TransactionRecord(
                    item_name="A4 paper",
                    transaction_type="sales",
                    quantity=500,
                    price=30.0,
                    transaction_date="2025-04-10",
                )
            ],
        ),
        customer_response="Thank you for your order.",
        pricing_justification="Priced at mid band.",
        total_quote_amount=30.0,
    )

    result = validate_order_recommendation(
        ValidateOrderRecommendationPayload(response=valid, price_bands=bands)
    )
    assert isinstance(result, OrderRecommendationResponse)
    assert result.success is True


def test_or_e3_gate_blocks_missing_need_date() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date=None,
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            )
        ],
    )
    inventory = InventoryResult(
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            CheckedItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
                success=True,
                quantity_in_stock=100,
                quantity_to_order=0,
            )
        ],
    )
    assert can_recommend_order(quote, inventory, _pricing_response()) is False
    response = call_order_recommendation_agent(
        OrderRecommendationRequest(
            quote=quote,
            inventory=inventory,
            pricing=_pricing_response(),
            customer=CustomerContext(original_request_text="Need paper"),
        )
    )
    assert response.success is False


def test_or_a1_agent_calls_validate_tool_chain() -> None:
    request = OrderRecommendationRequest(
        quote=_quote(),
        inventory=_inventory(),
        pricing=_pricing_response(),
        customer=CustomerContext(
            original_request_text="Need 500 sheets of A4 paper for a meeting.",
            need_size="medium",
            event_type="meeting",
        ),
    )
    tool_calls: list[str] = []

    def simulated_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if hasattr(part, "tool_name")
        ]
        if not tool_returns:
            tool_calls.append("tool_gather_company_context")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_gather_company_context",
                        args={
                            "payload": {
                                "date_of_request": "2025-04-01",
                                "ordered_products": ["A4 paper"],
                            }
                        },
                    )
                ]
            )

        last_return = tool_returns[-1]
        if last_return.tool_name == "tool_gather_company_context":
            tool_calls.append("tool_extract_price_bands")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_extract_price_bands",
                        args={"pricing": request.pricing.model_dump()},
                    )
                ]
            )

        if last_return.tool_name == "tool_extract_price_bands":
            tool_calls.append("tool_evaluate_pricing_signals")
            company_context = _tool_return_content(
                messages, "tool_gather_company_context"
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_evaluate_pricing_signals",
                        args={
                            "payload": {
                                "company_context": company_context,
                                "customer": request.customer.model_dump(),
                            }
                        },
                    )
                ]
            )

        if last_return.tool_name == "tool_evaluate_pricing_signals":
            tool_calls.append("tool_build_transaction_batches")
            recommended_items = [
                RecommendedLineItem(
                    product_name="A4 paper",
                    quantity_requested=500,
                    quantity_fulfilled=500,
                    quantity_in_stock=100,
                    quantity_to_order=400,
                    unit_cost=0.05,
                    unit_price=0.06,
                    line_total=30.0,
                    included=True,
                )
            ]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_build_transaction_batches",
                        args={
                            "payload": {
                                "date_of_request": "2025-04-01",
                                "need_date": "2025-04-10",
                                "recommended_items": [
                                    item.model_dump() for item in recommended_items
                                ],
                            }
                        },
                    )
                ]
            )

        if last_return.tool_name == "tool_build_transaction_batches":
            tool_calls.append("tool_validate_order_recommendation")
            batches = _tool_return_content(messages, "tool_build_transaction_batches")
            bands = [band.model_dump() for band in extract_price_bands(request.pricing)]
            draft = {
                "success": True,
                "date_of_request": "2025-04-01",
                "need_date": "2025-04-10",
                "recommended_items": [
                    {
                        "product_name": "A4 paper",
                        "quantity_requested": 500,
                        "quantity_fulfilled": 500,
                        "quantity_in_stock": 100,
                        "quantity_to_order": 400,
                        "unit_cost": 0.05,
                        "unit_price": 0.06,
                        "line_total": 30.0,
                        "included": True,
                    }
                ],
                "stock_orders": batches["stock_orders"],
                "sales": batches["sales"],
                "customer_response": (
                    "Thank you for your order. Delivery by 2025-04-10."
                ),
                "pricing_justification": "Priced at mid band with neutral signals.",
                "total_quote_amount": 30.0,
                "error": None,
            }
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_validate_order_recommendation",
                        args={
                            "payload": {
                                "response": draft,
                                "price_bands": bands,
                            }
                        },
                    )
                ]
            )

        validated = _tool_return_content(messages, "tool_validate_order_recommendation")
        tool_calls.append("final_result")
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_result", args=validated)]
        )

    with _mock_company_services(inventory={"A4 paper": 100, "Cardstock": 900}):
        with order_recommendation_agent.override(model=FunctionModel(simulated_llm)):
            response = call_order_recommendation_agent(request)

    assert response.success is True
    assert tool_calls == [
        "tool_gather_company_context",
        "tool_extract_price_bands",
        "tool_evaluate_pricing_signals",
        "tool_build_transaction_batches",
        "tool_validate_order_recommendation",
        "final_result",
    ]
    assert response.stock_orders is not None
    assert response.sales is not None
    assert response.total_quote_amount == 30.0


def test_or_h1_standard_multi_line_order() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise AssertionError("OPENAI_API_KEY required for OR-H1 live LLM test")

    request = OrderRecommendationRequest(
        quote=_quote(
            items=[
                QuoteItem(
                    product_name="A4 paper",
                    quantity_requested=500,
                    unit_price=0.05,
                ),
                QuoteItem(
                    product_name="Cardstock",
                    quantity_requested=200,
                    unit_price=0.15,
                ),
            ]
        ),
        inventory=_inventory(
            lines=[
                {
                    "product_name": "A4 paper",
                    "quantity_requested": 500,
                    "unit_price": 0.05,
                    "quantity_in_stock": 100,
                    "quantity_to_order": 400,
                },
                {
                    "product_name": "Cardstock",
                    "quantity_requested": 200,
                    "unit_price": 0.15,
                    "quantity_in_stock": 200,
                    "quantity_to_order": 0,
                },
            ]
        ),
        pricing=_pricing_response_multi(
            [
                {
                    "product_name": "A4 paper",
                    "quantity_requested": 500,
                    "quantity_fulfilled": 500,
                    "unit_cost": 0.05,
                    "floor": 0.05,
                    "mid": 0.06,
                    "ceiling": 0.07,
                },
                {
                    "product_name": "Cardstock",
                    "quantity_requested": 200,
                    "quantity_fulfilled": 200,
                    "unit_cost": 0.15,
                    "floor": 0.14,
                    "mid": 0.16,
                    "ceiling": 0.18,
                },
            ]
        ),
        customer=CustomerContext(
            original_request_text=(
                "Please quote 500 sheets of A4 paper and 200 cardstock for delivery "
                "by April 10, 2025."
            ),
            need_size="medium",
            event_type="meeting",
        ),
    )

    with _mock_company_services(
        inventory={"A4 paper": 100, "Cardstock": 200, "Envelopes": 50}
    ):
        response = call_order_recommendation_agent(request)

    assert response.success is True
    assert response.error is None
    assert response.stock_orders is not None
    assert len(response.stock_orders.transactions) >= 1
    assert response.sales is not None
    assert len(response.sales.transactions) == 2
    assert response.sales.transaction_date == "2025-04-10"
    prose = response.customer_response.lower()
    assert "2025-04-10" in prose or "april 10" in prose


def test_or_h2_all_stock_in_hand() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise AssertionError("OPENAI_API_KEY required for OR-H2 live LLM test")

    request = OrderRecommendationRequest(
        quote=_quote(
            items=[
                QuoteItem(
                    product_name="A4 paper",
                    quantity_requested=100,
                    unit_price=0.05,
                )
            ]
        ),
        inventory=_inventory(
            lines=[
                {
                    "product_name": "A4 paper",
                    "quantity_requested": 100,
                    "unit_price": 0.05,
                    "quantity_in_stock": 500,
                    "quantity_to_order": 0,
                }
            ]
        ),
        pricing=_pricing_response(
            floor=0.05,
            mid=0.06,
            ceiling=0.07,
            quantity_fulfilled=100,
            quantity_requested=100,
        ),
        customer=CustomerContext(
            original_request_text="Need 100 sheets of A4 paper.",
            need_size="small",
        ),
    )

    with _mock_company_services(inventory={"A4 paper": 500, "Cardstock": 100}):
        response = call_order_recommendation_agent(request)

    assert response.success is True
    assert response.stock_orders is None
    assert response.sales is not None
    assert len(response.sales.transactions) == 1


def test_or_h3_large_need_and_gathering_event() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise AssertionError("OPENAI_API_KEY required for OR-H3 live LLM test")

    request = OrderRecommendationRequest(
        quote=_quote(
            items=[
                QuoteItem(
                    product_name="A4 paper",
                    quantity_requested=600,
                    unit_price=0.05,
                )
            ]
        ),
        inventory=_inventory(
            lines=[
                {
                    "product_name": "A4 paper",
                    "quantity_requested": 600,
                    "unit_price": 0.05,
                    "quantity_in_stock": 200,
                    "quantity_to_order": 400,
                }
            ]
        ),
        pricing=_pricing_response(
            floor=0.05,
            mid=0.06,
            ceiling=0.07,
            quantity_fulfilled=600,
            quantity_requested=600,
        ),
        customer=CustomerContext(
            original_request_text="Large gathering needs 600 sheets of A4 paper.",
            need_size="large",
            event_type="gathering",
        ),
    )

    with _mock_company_services(
        inventory={"A4 paper": 800, "Cardstock": 200},
        cash_balance=1200.0,
        inventory_value=2500.0,
    ):
        response = call_order_recommendation_agent(request)

    assert response.success is True
    included = [item for item in response.recommended_items if item.included]
    assert len(included) == 1
    item = included[0]
    band_midpoint = (0.05 + 0.07) / 2
    assert item.unit_price <= band_midpoint

    combined = f"{response.customer_response} {response.pricing_justification}".lower()
    assert any(
        keyword in combined
        for keyword in ("bulk", "event", "gathering", "discount", "large")
    )


def test_or_t6b_stock_orders_none_when_nothing_to_order() -> None:
    items = [
        RecommendedLineItem(
            product_name="A4 paper",
            quantity_requested=100,
            quantity_fulfilled=100,
            quantity_in_stock=500,
            quantity_to_order=0,
            unit_cost=0.05,
            unit_price=0.06,
            line_total=6.0,
            included=True,
        ),
    ]
    result = build_transaction_batches(
        BuildTransactionsInput(
            date_of_request="2025-04-01",
            need_date="2025-04-10",
            recommended_items=items,
        )
    )
    assert result.stock_orders is None
    assert result.sales is not None


def main() -> int:
    scenarios = [
        ("OR-T1", test_or_t1_inventory_pressure_twenty_percent),
        ("OR-T2", test_or_t2_empty_warehouse),
        ("OR-T3", test_or_t3_unknown_products_treated_as_zero),
        ("OR-T4", test_or_t4_gather_company_context),
        ("OR-T5", test_or_t5_extract_price_bands),
        ("OR-T6", test_or_t6_build_transaction_batches),
        ("OR-T6b", test_or_t6b_stock_orders_none_when_nothing_to_order),
        ("OR-T7", test_or_t7_evaluate_pricing_signals_cash_stress),
        ("OR-I1", test_or_i1_customer_context_from_csv_row),
        ("OR-T7b", test_or_t7b_mood_happy_premium_lean),
        ("OR-T7c", test_or_t7c_mood_sad_discount_lean),
        ("OR-T8", test_or_t8_validate_rejects_price_above_ceiling),
        ("OR-T8b", test_or_t8b_validate_accepts_in_band_response),
        ("OR-T8c", test_or_t8c_fallback_recommendation_passes_validation),
        ("OR-E3", test_or_e3_gate_blocks_missing_need_date),
        ("OR-A1", test_or_a1_agent_calls_validate_tool_chain),
        ("OR-H1", test_or_h1_standard_multi_line_order),
        ("OR-H2", test_or_h2_all_stock_in_hand),
        ("OR-H3", test_or_h3_large_need_and_gathering_event),
    ]

    passed = sum(_run_scenario(scenario_id, fn) for scenario_id, fn in scenarios)
    total = len(scenarios)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
