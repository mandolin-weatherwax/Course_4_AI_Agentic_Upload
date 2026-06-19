"""Orchestrator agent tests from specification/agents/orchestrator_agent.md."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from unittest.mock import patch

from project_starter import db_engine, init_database
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.orchestrator_agent import (
    DELIVERY_TIMEFRAME_CUSTOMER_RESPONSE,
    ORCHESTRATOR_TOOL_NAMES,
    ORDER_TOO_LARGE_CUSTOMER_RESPONSE,
    PIPELINE_STEPS,
    STEP_POST_SALES,
    STEP_PRICING,
    STEP_QUOTING,
    OrchestratorDeps,
    OrchestratorRequest,
    OrchestratorResponse,
    PostTransactionsResult,
    _append_excluded_products_notice,
    _finalize_orchestrator_response,
    handle_quote_request,
    orchestrator_agent,
    post_sales_transactions,
    post_stock_transactions,
)
from agents.order_recommendation_agent import (
    CustomerContext,
    OrderRecommendationResponse,
    RecommendedLineItem,
    TransactionBatch,
    TransactionRecord,
)
from agents.pricing_review_agent import ItemPriceReview, PricingReviewResponse
from agents.quoting_agent import QuoteItem, QuoteResponse
from pipeline.full_pipeline import (
    DEFAULT_PIPELINE_INDICES,
    load_pipeline_requests,
)
from tools.inventory_tool import CheckedItem, InventoryResult
from tools.pricing_tool import (
    PricedLineItem,
    PricingRecommendation,
    PricingResponse,
)

MAX_PIPELINE_ATTEMPTS = 2


def _tool_return_content(
    messages: list[ModelMessage], tool_name: str
) -> dict[str, Any]:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if getattr(part, "tool_name", None) == tool_name:
                content = part.content
                if isinstance(content, str):
                    return json.loads(content)
                if isinstance(content, dict):
                    return content
                raise AssertionError(
                    f"Unexpected tool return type for {tool_name!r}: {type(content)}"
                )
    raise AssertionError(f"No tool return found for {tool_name!r}")


def _tool_return_parts(messages: list[ModelMessage]) -> list[Any]:
    return [
        part
        for message in messages
        for part in message.parts
        if getattr(part, "tool_name", None) is not None
        and getattr(part, "part_kind", "") == "tool-return"
    ]


def _parse_observation(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise AssertionError(f"Unexpected observation type: {type(content)}")


def _sequential_orchestrator_llm(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    """Simulate ReAct agent calling pipeline tools in fixed order."""
    tool_returns = _tool_return_parts(messages)

    if not tool_returns:
        return ModelResponse(
            parts=[ToolCallPart(tool_name=ORCHESTRATOR_TOOL_NAMES[0], args={})]
        )

    last_return = tool_returns[-1]
    observation = _parse_observation(last_return.content)

    if not observation.get("gate_passed", True):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "success": False,
                        "step_completed": observation.get("step_completed", ""),
                        "customer_response": "",
                        "error": observation.get("error"),
                    },
                )
            ]
        )

    tool_index = ORCHESTRATOR_TOOL_NAMES.index(last_return.tool_name)
    if tool_index + 1 >= len(ORCHESTRATOR_TOOL_NAMES):
        recommendation = _tool_return_content(
            messages, "tool_run_order_recommendation"
        )["recommendation"]
        quote = _tool_return_content(messages, "tool_run_quoting_agent")["quote"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "success": True,
                        "step_completed": STEP_POST_SALES,
                        "customer_response": recommendation["customer_response"],
                        "total_quote_amount": recommendation["total_quote_amount"],
                        "date_of_request": quote.get("date_of_request"),
                        "need_date": quote.get("need_date"),
                    },
                )
            ]
        )

    next_tool = ORCHESTRATOR_TOOL_NAMES[tool_index + 1]
    return ModelResponse(parts=[ToolCallPart(tool_name=next_tool, args={})])


@contextmanager
def _sequential_orchestrator_model() -> Iterator[None]:
    with orchestrator_agent.override(model=FunctionModel(_sequential_orchestrator_llm)):
        yield


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
        raise AssertionError("OPENAI_API_KEY required for orchestrator LLM tests")


def _recommendation(
    *,
    date_of_request: str = "2025-04-01",
    need_date: str = "2025-04-10",
    with_stock: bool = True,
) -> OrderRecommendationResponse:
    stock = (
        TransactionBatch(
            transaction_date=date_of_request,
            transactions=[
                TransactionRecord(
                    item_name="A4 paper",
                    transaction_type="stock_orders",
                    quantity=400,
                    price=20.0,
                    transaction_date=date_of_request,
                )
            ],
        )
        if with_stock
        else None
    )
    return OrderRecommendationResponse(
        success=True,
        date_of_request=date_of_request,
        need_date=need_date,
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
        stock_orders=stock,
        sales=TransactionBatch(
            transaction_date=need_date,
            transactions=[
                TransactionRecord(
                    item_name="A4 paper",
                    transaction_type="sales",
                    quantity=500,
                    price=30.0,
                    transaction_date=need_date,
                )
            ],
        ),
        customer_response="Thank you for your order.",
        pricing_justification="Priced at mid band.",
        total_quote_amount=30.0,
    )


def test_oc_t1_post_stock_on_request_date() -> None:
    calls: list[dict] = []

    def mock_create(**kwargs) -> int:
        calls.append(kwargs)
        return len(calls)

    result = post_stock_transactions(
        _recommendation(),
        create_transaction_fn=mock_create,
    )
    assert result.error is None
    assert result.posted_count == 1
    assert calls[0]["transaction_type"] == "stock_orders"
    assert calls[0]["date"] == "2025-04-01"


def test_oc_t2_post_stock_null_batch() -> None:
    calls: list[dict] = []

    def mock_create(**kwargs) -> int:
        calls.append(kwargs)
        return 1

    result = post_stock_transactions(
        _recommendation(with_stock=False),
        create_transaction_fn=mock_create,
    )
    assert result.posted_count == 0
    assert calls == []


def test_oc_t3_post_sales_on_need_date() -> None:
    calls: list[dict] = []

    def mock_create(**kwargs) -> int:
        calls.append(kwargs)
        return len(calls)

    result = post_sales_transactions(
        _recommendation(),
        create_transaction_fn=mock_create,
    )
    assert result.error is None
    assert result.posted_count == 1
    assert calls[0]["transaction_type"] == "sales"
    assert calls[0]["date"] == "2025-04-10"


def test_oc_t4_post_sales_wrong_date() -> None:
    rec = _recommendation()
    rec.sales.transactions[0].transaction_date = "2025-04-01"
    result = post_sales_transactions(rec)
    assert result.error is not None
    assert result.posted_count == 0


def test_oc_t5_stock_before_sales_order() -> None:
    calls: list[str] = []

    def mock_create(**kwargs) -> int:
        calls.append(kwargs["transaction_type"])
        return len(calls)

    rec = _recommendation()
    post_stock_transactions(rec, create_transaction_fn=mock_create)
    post_sales_transactions(rec, create_transaction_fn=mock_create)
    assert calls == ["stock_orders", "sales"]


def test_oc_e1_missing_date_suffix() -> None:
    _require_api_key()
    init_database(db_engine)
    response = handle_quote_request(
        OrchestratorRequest(request_with_date="Need 500 sheets of A4 paper.")
    )
    assert response.success is False
    assert response.debug is not None
    assert "tool_run_quoting_agent" in response.debug.tools_called
    assert response.debug.failure_kind == "business_failure"
    assert response.step_completed in ("", STEP_QUOTING, "quote")


def test_oc_i1_str_response_equals_customer_response() -> None:
    response = OrchestratorResponse(
        success=True,
        step_completed=STEP_POST_SALES,
        customer_response="Hello customer",
    )
    assert str(response) == "Hello customer"


def test_oc_i2_handle_quote_request_return_type() -> None:
    init_database(db_engine)
    quote = QuoteResponse(
        success=False,
        error="Could not parse.",
        date_of_request="2025-04-01",
    )
    with (
        _sequential_orchestrator_model(),
        patch("agents.orchestrator_agent.call_quoting_agent", return_value=quote),
    ):
        response = handle_quote_request(
            OrchestratorRequest(
                request_with_date="Need paper (Date of request: 2025-04-01)"
            )
        )
    assert isinstance(response, OrchestratorResponse)
    assert isinstance(str(response), str)


def test_oc_a1_pipeline_runs_all_steps() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=500,
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
                quantity_requested=500,
                unit_price=0.05,
                success=True,
                quantity_in_stock=100,
                quantity_to_order=400,
            )
        ],
    )
    pricing_review = PricingReviewResponse(
        success=True,
        date_of_request="2025-04-01",
        items=[
            ItemPriceReview(
                product_name="A4 paper",
                unit_cost=0.05,
                history_count=3,
                min_unit_price=0.05,
                avg_unit_price=0.05,
                max_unit_price=0.06,
            )
        ],
    )
    pricing = PricingResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        recommendations=[],
    )
    recommendation = _recommendation()

    request = OrchestratorRequest(
        request_with_date="Need paper (Date of request: 2025-04-01)"
    )

    with (
        _sequential_orchestrator_model(),
        patch("agents.orchestrator_agent.call_quoting_agent", return_value=quote),
        patch("agents.orchestrator_agent.InventoryTool") as mock_inv_tool,
        patch(
            "agents.orchestrator_agent.call_pricing_review_agent",
            return_value=pricing_review,
        ),
        patch("agents.orchestrator_agent.can_price", return_value=True),
        patch("agents.orchestrator_agent.PricingTool") as mock_pricing_tool,
        patch(
            "agents.orchestrator_agent.call_order_recommendation_agent",
            return_value=recommendation,
        ),
        patch("agents.orchestrator_agent.can_recommend_order", return_value=True),
        patch("agents.orchestrator_agent.can_review_pricing", return_value=True),
        patch("agents.orchestrator_agent.can_check_inventory", return_value=True),
        patch("agents.orchestrator_agent.post_stock_transactions") as mock_stock,
        patch("agents.orchestrator_agent.post_sales_transactions") as mock_sales,
    ):
        mock_inv_tool.return_value.check.return_value = inventory
        mock_pricing_tool.return_value.price.return_value = pricing
        mock_stock.return_value = PostTransactionsResult(
            posted_count=1, transactions=[]
        )
        mock_sales.return_value = PostTransactionsResult(
            posted_count=1, transactions=[]
        )

        response = handle_quote_request(request)

    assert response.success is True
    assert response.step_completed == STEP_POST_SALES
    assert response.debug is not None
    assert response.debug.tools_called == [
        "tool_run_quoting_agent",
        "tool_run_inventory_check",
        "tool_run_pricing_review",
        "tool_run_pricing_tool",
        "tool_run_order_recommendation",
        "tool_post_stock_transactions",
        "tool_post_sales_transactions",
    ]


def test_oc_a2_early_exit_after_quoting_failure() -> None:
    request = OrchestratorRequest(request_with_date="bad request")
    failed_quote = QuoteResponse(
        success=False,
        error="Could not parse quote.",
        date_of_request="2025-04-01",
    )

    with (
        _sequential_orchestrator_model(),
        patch(
            "agents.orchestrator_agent.call_quoting_agent",
            return_value=failed_quote,
        ),
    ):
        response = handle_quote_request(request)

    assert response.success is False
    assert response.debug is not None
    assert response.debug.failure_kind == "business_failure"
    assert response.debug.tools_called == ["tool_run_quoting_agent"]
    assert "tool_run_inventory_check" not in response.debug.tools_called
    assert response.debug.last_tool == "tool_run_quoting_agent"
    assert response.debug.expected_next_step == STEP_QUOTING


def test_oc_f1_inventory_failure_uses_timeframe_message() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-14",
        need_date="2025-04-15",
        items=[
            QuoteItem(
                product_name="Paper napkins",
                quantity_requested=2000,
                unit_price=0.02,
            )
        ],
    )
    inventory = InventoryResult(
        date_of_request="2025-04-14",
        need_date="2025-04-15",
        items=[
            CheckedItem(
                product_name="Paper napkins",
                quantity_requested=2000,
                unit_price=0.02,
                success=False,
                quantity_in_stock=100,
                quantity_to_order=1900,
                error=(
                    "Cannot fulfil 'Paper napkins': supplier lead time "
                    "misses need date."
                ),
            )
        ],
    )
    request = OrchestratorRequest(
        request_with_date="Need napkins (Date of request: 2025-04-14)"
    )

    with (
        _sequential_orchestrator_model(),
        patch("agents.orchestrator_agent.call_quoting_agent", return_value=quote),
        patch("agents.orchestrator_agent.InventoryTool") as mock_inv_tool,
    ):
        mock_inv_tool.return_value.check.return_value = inventory
        response = handle_quote_request(request)

    assert response.success is False
    assert response.customer_response == DELIVERY_TIMEFRAME_CUSTOMER_RESPONSE
    assert response.step_completed == STEP_QUOTING
    assert response.debug is not None
    assert response.debug.tools_called == [
        "tool_run_quoting_agent",
        "tool_run_inventory_check",
    ]


def test_oc_f2_cash_failure_uses_order_too_large_message() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-14",
        need_date="2025-04-15",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=10000,
                unit_price=0.05,
            )
        ],
    )
    inventory = InventoryResult(
        date_of_request="2025-04-14",
        need_date="2025-04-15",
        items=[
            CheckedItem(
                product_name="A4 paper",
                quantity_requested=10000,
                unit_price=0.05,
                success=True,
                quantity_in_stock=0,
                quantity_to_order=10000,
            )
        ],
    )
    pricing_review = PricingReviewResponse(
        success=True,
        date_of_request="2025-04-14",
        items=[
            ItemPriceReview(
                product_name="A4 paper",
                unit_cost=0.05,
                history_count=3,
                min_unit_price=0.05,
                avg_unit_price=0.05,
                max_unit_price=0.06,
            )
        ],
    )
    cash_blocked_pricing = PricingResponse(
        success=True,
        date_of_request="2025-04-14",
        need_date="2025-04-15",
        recommendations=[
            PricingRecommendation(
                strategy="average_pricing",
                items=[
                    PricedLineItem(
                        product_name="A4 paper",
                        quantity_requested=10000,
                        quantity_fulfilled=0,
                        unit_cost=0.05,
                        unit_price=0.0,
                        line_revenue=0.0,
                        line_acquisition_cost=0.0,
                        included=False,
                    )
                ],
                total_acquisition_cost=0.0,
                total_profit=0.0,
                error="Insufficient cash: excluded A4 paper.",
            )
        ],
    )
    request = OrchestratorRequest(
        request_with_date="Huge order (Date of request: 2025-04-14)"
    )

    with (
        _sequential_orchestrator_model(),
        patch("agents.orchestrator_agent.call_quoting_agent", return_value=quote),
        patch("agents.orchestrator_agent.InventoryTool") as mock_inv_tool,
        patch(
            "agents.orchestrator_agent.call_pricing_review_agent",
            return_value=pricing_review,
        ),
        patch("agents.orchestrator_agent.PricingTool") as mock_pricing_tool,
    ):
        mock_inv_tool.return_value.check.return_value = inventory
        mock_pricing_tool.return_value.price.return_value = cash_blocked_pricing
        response = handle_quote_request(request)

    assert response.success is False
    assert response.customer_response == ORDER_TOO_LARGE_CUSTOMER_RESPONSE
    assert response.step_completed == STEP_PRICING
    assert response.debug is not None
    assert "tool_run_order_recommendation" in response.debug.tools_called


def test_oc_p1_partial_quote_appends_excluded_notice() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-05",
        need_date="2025-04-20",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=200,
                unit_price=0.05,
            )
        ],
        excluded_products=["balloons"],
    )
    deps = OrchestratorDeps(
        request=OrchestratorRequest(request_with_date="mixed request"),
        quote=quote,
    )
    deps.step_completed = STEP_POST_SALES

    result = OrchestratorResponse(
        success=True,
        step_completed=STEP_POST_SALES,
        customer_response="Thank you for your order. Total: $10.00.",
        total_quote_amount=10.0,
    )
    finalized = _finalize_orchestrator_response(result, deps)

    assert finalized.success is True
    assert "balloons" in finalized.customer_response.lower()
    assert "unable to quote" in finalized.customer_response.lower()


def test_oc_p1_append_notice_skips_duplicate_mention() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-05",
        need_date="2025-04-20",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=200,
                unit_price=0.05,
            )
        ],
        excluded_products=["balloons"],
    )
    prose = (
        "Thank you. We could not quote balloons, but here is your quote for A4 paper."
    )
    appended = _append_excluded_products_notice(prose, quote)
    assert appended == prose


def test_oc_d1_debug_on_pipeline_incomplete() -> None:
    request = OrchestratorRequest(
        request_with_date="Need paper (Date of request: 2025-04-01)"
    )
    deps = OrchestratorDeps(request=request)
    deps.step_completed = STEP_QUOTING
    deps.tools_called = ["tool_run_quoting_agent"]

    result = OrchestratorResponse(
        success=True,
        step_completed=STEP_QUOTING,
        customer_response="Stopped early.",
    )
    finalized = _finalize_orchestrator_response(result, deps)

    assert finalized.success is False
    assert finalized.debug is not None
    assert finalized.debug.failure_kind == "react_incomplete"
    assert finalized.debug.expected_next_step == "inventory"
    assert "stopped at" in finalized.error.lower()


def _assert_full_orchestrator(csv_index: int, event_label: str) -> None:
    _require_api_key()
    last_failure = ""

    for _ in range(MAX_PIPELINE_ATTEMPTS):
        init_database(db_engine)
        contexts = load_pipeline_requests((csv_index,))
        ctx = contexts[0]
        assert ctx.event == event_label

        request = OrchestratorRequest(
            request_with_date=(
                f"{ctx.request_text} (Date of request: {ctx.request_date})"
            ),
            customer=CustomerContext(
                original_request_text=ctx.request_text,
                job_type=ctx.job,
                need_size=ctx.need_size,
                event_type=ctx.event,
                mood=ctx.mood,
            ),
        )
        response = handle_quote_request(request)
        if response.success:
            assert response.step_completed == STEP_POST_SALES
            assert response.customer_response.strip()
            assert response.sales_transactions_posted >= 1
            assert response.need_date is not None
            for posted in response.posted_transactions:
                if posted.transaction_type == "stock_orders":
                    assert posted.transaction_date == response.date_of_request
                if posted.transaction_type == "sales":
                    assert posted.transaction_date == response.need_date
            return
        last_failure = response.error or response.customer_response
        if response.debug is not None:
            last_failure = f"{last_failure}; debug={response.debug.model_dump()}"

    raise AssertionError(
        f"orchestrator failed after {MAX_PIPELINE_ATTEMPTS} attempt(s): {last_failure}"
    )


def test_oc_h1_party_csv_index_4() -> None:
    _assert_full_orchestrator(4, "party")


def test_oc_h2_assembly_csv_index_5() -> None:
    _assert_full_orchestrator(5, "assembly")


def test_oc_h3_show_csv_index_11() -> None:
    _assert_full_orchestrator(11, "show")


def test_oc_v1_default_indices() -> None:
    assert DEFAULT_PIPELINE_INDICES == (4, 5, 11)
    assert [s for s in PIPELINE_STEPS] == list(PIPELINE_STEPS)


def main() -> int:
    scenarios = [
        ("OC-T1", test_oc_t1_post_stock_on_request_date),
        ("OC-T2", test_oc_t2_post_stock_null_batch),
        ("OC-T3", test_oc_t3_post_sales_on_need_date),
        ("OC-T4", test_oc_t4_post_sales_wrong_date),
        ("OC-T5", test_oc_t5_stock_before_sales_order),
        ("OC-E1", test_oc_e1_missing_date_suffix),
        ("OC-I1", test_oc_i1_str_response_equals_customer_response),
        ("OC-I2", test_oc_i2_handle_quote_request_return_type),
        ("OC-A1", test_oc_a1_pipeline_runs_all_steps),
        ("OC-A2", test_oc_a2_early_exit_after_quoting_failure),
        ("OC-F1", test_oc_f1_inventory_failure_uses_timeframe_message),
        ("OC-F2", test_oc_f2_cash_failure_uses_order_too_large_message),
        ("OC-P1", test_oc_p1_partial_quote_appends_excluded_notice),
        ("OC-P2", test_oc_p1_append_notice_skips_duplicate_mention),
        ("OC-D1", test_oc_d1_debug_on_pipeline_incomplete),
        ("OC-V1", test_oc_v1_default_indices),
        ("OC-H1", test_oc_h1_party_csv_index_4),
        ("OC-H2", test_oc_h2_assembly_csv_index_5),
        ("OC-H3", test_oc_h3_show_csv_index_11),
    ]

    passed = sum(_run_scenario(scenario_id, fn) for scenario_id, fn in scenarios)
    total = len(scenarios)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
