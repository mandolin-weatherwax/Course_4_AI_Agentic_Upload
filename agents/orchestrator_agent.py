"""Orchestrator — ReAct agent for the seven-step quote-to-order pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from project_starter import create_transaction
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.order_recommendation_agent import (
    CustomerContext,
    OrderRecommendationRequest,
    OrderRecommendationResponse,
    call_order_recommendation_agent,
    can_recommend_order,
)
from agents.pricing_review_agent import (
    PricingReviewRequest,
    PricingReviewResponse,
    call_pricing_review_agent,
    can_review_pricing,
)
from agents.quoting_agent import QuoteResponse, call_quoting_agent
from tools.inventory_tool import (
    InventoryResult,
    InventoryTool,
    quote_to_inventory_request,
)
from tools.pricing_tool import (
    ItemUnitPrices,
    PricingRequest,
    PricingResponse,
    PricingTool,
    can_price,
)

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

STEP_QUOTING = "quoting"
STEP_INVENTORY = "inventory"
STEP_PRICING_REVIEW = "pricing_review"
STEP_PRICING = "pricing"
STEP_ORDER_RECOMMENDATION = "order_recommendation"
STEP_POST_STOCK = "post_stock"
STEP_POST_SALES = "post_sales"

DELIVERY_TIMEFRAME_CUSTOMER_RESPONSE = (
    "Thank you for your order. Unfortunately, we cannot deliver this order "
    "within your requested timeframe. Please contact us if you would like to "
    "discuss alternative delivery dates or smaller quantities."
)

ORDER_TOO_LARGE_CUSTOMER_RESPONSE = (
    "We truly appreciate your order. However, this order is too large for "
    "the size of our business at this time. Please contact us if you would "
    "like to discuss a smaller order or alternative arrangements."
)

PIPELINE_STEPS = (
    STEP_QUOTING,
    STEP_INVENTORY,
    STEP_PRICING_REVIEW,
    STEP_PRICING,
    STEP_ORDER_RECOMMENDATION,
    STEP_POST_STOCK,
    STEP_POST_SALES,
)

ORCHESTRATOR_TOOL_NAMES = (
    "tool_run_quoting_agent",
    "tool_run_inventory_check",
    "tool_run_pricing_review",
    "tool_run_pricing_tool",
    "tool_run_order_recommendation",
    "tool_post_stock_transactions",
    "tool_post_sales_transactions",
)

CreateTransactionFn = Callable[..., int]


class OrchestratorRequest(BaseModel):
    request_with_date: str
    customer: Optional[CustomerContext] = None


class PostedTransaction(BaseModel):
    transaction_id: int
    item_name: str
    transaction_type: str
    quantity: int
    price: float
    transaction_date: str


class PostTransactionsResult(BaseModel):
    posted_count: int
    transactions: list[PostedTransaction] = Field(default_factory=list)
    error: Optional[str] = None


FailureKind = Literal[
    "react_exception",
    "react_incomplete",
    "business_failure",
]


class OrchestratorDebugInfo(BaseModel):
    """Diagnostics when the ReAct agent fails or exits early."""

    tools_called: list[str] = Field(default_factory=list)
    last_tool: Optional[str] = None
    last_tool_error: Optional[str] = None
    agent_exception: Optional[str] = None
    failure_kind: Optional[FailureKind] = None
    expected_next_step: Optional[str] = None


class OrchestratorResponse(BaseModel):
    success: bool
    step_completed: str = ""
    customer_response: str = ""
    total_quote_amount: Optional[float] = None
    date_of_request: Optional[str] = None
    need_date: Optional[str] = None
    stock_transactions_posted: int = 0
    sales_transactions_posted: int = 0
    posted_transactions: list[PostedTransaction] = Field(default_factory=list)
    error: Optional[str] = None
    debug: Optional[OrchestratorDebugInfo] = None

    def __str__(self) -> str:
        return self.customer_response


@dataclass
class OrchestratorDeps:
    request: OrchestratorRequest
    create_transaction_fn: CreateTransactionFn = create_transaction
    quote: QuoteResponse | None = None
    inventory: InventoryResult | None = None
    pricing_review: PricingReviewResponse | None = None
    pricing: PricingResponse | None = None
    recommendation: OrderRecommendationResponse | None = None
    posted_transactions: list[PostedTransaction] = field(default_factory=list)
    step_completed: str = ""
    stock_transactions_posted: int = 0
    sales_transactions_posted: int = 0
    tools_called: list[str] = field(default_factory=list)
    last_tool: str = ""
    last_tool_error: Optional[str] = None


ORCHESTRATOR_DIRECTIVE = """\
You are the Munder Difflin Orchestrator Agent.
You run a fixed seven-step quote-to-order pipeline using ReAct
(Thought, Action, Observation).

For EACH step, before calling a tool:
  THOUGHT: State the step name, why it is next, and which gates must pass.

Then call exactly ONE tool (Action).
Read the tool result (Observation).
If the gate for the NEXT step fails, stop and return OrchestratorResponse
with success=false.

Steps (in order — never skip or reorder):

1. tool_run_quoting_agent
   - Input: request_with_date from OrchestratorRequest
   - Stop if quote.success is false

2. tool_run_inventory_check
   - Requires successful quote from step 1
   - Stop if any inventory line has success=false

3. tool_run_pricing_review
   - Requires fulfillable inventory from step 2
   - Stop if pricing_review.success is false

4. tool_run_pricing_tool
   - Requires pricing review from step 3
   - Stop if pricing.success is false

5. tool_run_order_recommendation
   - Requires pricing from step 4 and customer context from request
   - Stop if recommendation.success is false

6. tool_post_stock_transactions
   - Post stock_orders on date_of_request via create_transaction
   - Skip gracefully when stock_orders is null

7. tool_post_sales_transactions
   - Post sales on need_date via create_transaction
   - Then return OrchestratorResponse with success=true

Final output rules:
- customer_response = recommendation.customer_response on success
- total_quote_amount = recommendation.total_quote_amount on success
- On failure, customer_response = clear explanation for the customer (no raw JSON)
- Never call create_transaction except through posting tools
"""


def can_check_inventory(quote: QuoteResponse) -> bool:
    return (
        quote.success
        and quote.date_of_request is not None
        and quote.need_date is not None
        and len(quote.items) > 0
    )


def _customer_context(request: OrchestratorRequest) -> CustomerContext:
    if request.customer is not None:
        return request.customer
    return CustomerContext(original_request_text=request.request_with_date)


def _record_tool_start(deps: OrchestratorDeps, tool_name: str) -> None:
    deps.tools_called.append(tool_name)
    deps.last_tool = tool_name
    deps.last_tool_error = None


def _record_tool_error(deps: OrchestratorDeps, message: str) -> None:
    deps.last_tool_error = message


def _next_expected_step(step_completed: str) -> str | None:
    if not step_completed:
        return STEP_QUOTING
    try:
        index = PIPELINE_STEPS.index(step_completed)
    except ValueError:
        return None
    if index + 1 < len(PIPELINE_STEPS):
        return PIPELINE_STEPS[index + 1]
    return None


def _append_excluded_products_notice(
    customer_response: str,
    quote: QuoteResponse | None,
) -> str:
    """Append a notice when quoting excluded some requested products."""
    if quote is None or not quote.excluded_products:
        return customer_response
    excluded = ", ".join(quote.excluded_products)
    if any(
        name.lower() in customer_response.lower() for name in quote.excluded_products
    ):
        return customer_response
    notice = (
        f"\n\nWe were unable to quote the following items: {excluded}. "
        "Please contact us if you would like alternatives."
    )
    return customer_response.strip() + notice


def _inventory_timeline_failure(deps: OrchestratorDeps) -> bool:
    """True when stock or supplier lead time blocks delivery by need_date."""
    return (
        deps.inventory is not None
        and deps.inventory.items
        and not all(item.success for item in deps.inventory.items)
    )


def _order_blocked_by_cash(deps: OrchestratorDeps) -> bool:
    """True when cash cannot fund any line in the pricing tool's mid strategy."""
    if deps.pricing is None or not deps.pricing.success:
        return False

    mid_recommendations = [
        rec for rec in deps.pricing.recommendations if rec.strategy == "average_pricing"
    ]
    if not mid_recommendations:
        return False

    mid = mid_recommendations[0]
    if any(item.included for item in mid.items):
        return False

    return any(
        rec.error and "Insufficient cash" in rec.error
        for rec in deps.pricing.recommendations
    )


def _build_failure_message(deps: OrchestratorDeps) -> str:
    if _inventory_timeline_failure(deps):
        return DELIVERY_TIMEFRAME_CUSTOMER_RESPONSE
    if _order_blocked_by_cash(deps):
        return ORDER_TOO_LARGE_CUSTOMER_RESPONSE
    if deps.quote is not None and not deps.quote.success:
        return deps.quote.error or "Quote could not be parsed."
    if deps.pricing_review is not None and not deps.pricing_review.success:
        return deps.pricing_review.error or "Pricing review failed."
    if deps.pricing is not None and not deps.pricing.success:
        return "Pricing could not be completed for this order."
    if deps.recommendation is not None and not deps.recommendation.success:
        return deps.recommendation.error or "Order recommendation failed."
    if deps.last_tool_error:
        return deps.last_tool_error
    return "The orchestrator could not complete this request."


def _build_debug_info(
    deps: OrchestratorDeps,
    *,
    agent_exception: str | None = None,
    failure_kind: FailureKind | None = None,
) -> OrchestratorDebugInfo:
    return OrchestratorDebugInfo(
        tools_called=list(deps.tools_called),
        last_tool=deps.last_tool or None,
        last_tool_error=deps.last_tool_error,
        agent_exception=agent_exception,
        failure_kind=failure_kind,
        expected_next_step=_next_expected_step(deps.step_completed),
    )


def _normalize_step_completed(step: str, deps: OrchestratorDeps) -> str:
    """Prefer deps step name; ignore LLM tool-name placeholders."""
    if deps.step_completed:
        return deps.step_completed
    if step.startswith("tool_"):
        return ""
    return step


def _finalize_orchestrator_response(
    result: OrchestratorResponse,
    deps: OrchestratorDeps,
    *,
    agent_exception: str | None = None,
    failure_kind: FailureKind | None = None,
) -> OrchestratorResponse:
    """Attach deps state and debug diagnostics."""
    date_of_request = result.date_of_request or (
        deps.quote.date_of_request if deps.quote else None
    )
    need_date = result.need_date or (deps.quote.need_date if deps.quote else None)

    if result.success and deps.step_completed != STEP_POST_SALES:
        failure_kind = failure_kind or "react_incomplete"
        message = (
            "Pipeline returned success=true but stopped at "
            f"{deps.step_completed or 'unknown'}; expected {STEP_POST_SALES}."
        )
        return OrchestratorResponse(
            success=False,
            step_completed=deps.step_completed,
            customer_response=message,
            date_of_request=date_of_request,
            need_date=need_date,
            stock_transactions_posted=deps.stock_transactions_posted,
            sales_transactions_posted=deps.sales_transactions_posted,
            posted_transactions=list(deps.posted_transactions),
            error=message,
            debug=_build_debug_info(
                deps,
                agent_exception=agent_exception,
                failure_kind=failure_kind,
            ),
        )

    if not result.success:
        resolved_kind: FailureKind = failure_kind or "business_failure"
        customer_response = result.customer_response.strip() or _build_failure_message(
            deps
        )
        error = result.error or customer_response
        return OrchestratorResponse(
            success=False,
            step_completed=_normalize_step_completed(
                result.step_completed or deps.step_completed,
                deps,
            ),
            customer_response=customer_response,
            total_quote_amount=result.total_quote_amount,
            date_of_request=date_of_request,
            need_date=need_date,
            stock_transactions_posted=deps.stock_transactions_posted,
            sales_transactions_posted=deps.sales_transactions_posted,
            posted_transactions=list(deps.posted_transactions),
            error=error,
            debug=_build_debug_info(
                deps,
                agent_exception=agent_exception,
                failure_kind=resolved_kind,
            ),
        )

    return OrchestratorResponse(
        success=True,
        step_completed=_normalize_step_completed(
            deps.step_completed or result.step_completed,
            deps,
        ),
        customer_response=_append_excluded_products_notice(
            result.customer_response,
            deps.quote,
        ),
        total_quote_amount=result.total_quote_amount,
        date_of_request=date_of_request,
        need_date=need_date,
        stock_transactions_posted=deps.stock_transactions_posted,
        sales_transactions_posted=deps.sales_transactions_posted,
        posted_transactions=list(deps.posted_transactions),
        debug=_build_debug_info(deps),
    )


def post_stock_transactions(
    recommendation: OrderRecommendationResponse,
    *,
    create_transaction_fn: CreateTransactionFn = create_transaction,
) -> PostTransactionsResult:
    """Post supplier stock orders on date_of_request."""
    if recommendation.stock_orders is None:
        return PostTransactionsResult(posted_count=0, transactions=[])

    posted: list[PostedTransaction] = []
    for record in recommendation.stock_orders.transactions:
        if record.transaction_type != "stock_orders":
            return PostTransactionsResult(
                posted_count=0,
                transactions=[],
                error=f"Expected stock_orders, got {record.transaction_type!r}",
            )
        if record.transaction_date != recommendation.date_of_request:
            return PostTransactionsResult(
                posted_count=0,
                transactions=[],
                error="stock_orders.transaction_date must equal date_of_request",
            )
        tx_id = create_transaction_fn(
            item_name=record.item_name,
            transaction_type="stock_orders",
            quantity=record.quantity,
            price=record.price,
            date=record.transaction_date,
        )
        posted.append(
            PostedTransaction(
                transaction_id=tx_id,
                item_name=record.item_name,
                transaction_type="stock_orders",
                quantity=record.quantity,
                price=record.price,
                transaction_date=record.transaction_date,
            )
        )
    return PostTransactionsResult(posted_count=len(posted), transactions=posted)


def post_sales_transactions(
    recommendation: OrderRecommendationResponse,
    *,
    create_transaction_fn: CreateTransactionFn = create_transaction,
) -> PostTransactionsResult:
    """Post customer sales on need_date."""
    if recommendation.sales is None or not recommendation.sales.transactions:
        return PostTransactionsResult(
            posted_count=0,
            transactions=[],
            error="success path requires non-empty sales batch",
        )

    posted: list[PostedTransaction] = []
    for record in recommendation.sales.transactions:
        if record.transaction_type != "sales":
            return PostTransactionsResult(
                posted_count=0,
                transactions=[],
                error=f"Expected sales, got {record.transaction_type!r}",
            )
        if record.transaction_date != recommendation.need_date:
            return PostTransactionsResult(
                posted_count=0,
                transactions=[],
                error="sales.transaction_date must equal need_date",
            )
        tx_id = create_transaction_fn(
            item_name=record.item_name,
            transaction_type="sales",
            quantity=record.quantity,
            price=record.price,
            date=record.transaction_date,
        )
        posted.append(
            PostedTransaction(
                transaction_id=tx_id,
                item_name=record.item_name,
                transaction_type="sales",
                quantity=record.quantity,
                price=record.price,
                transaction_date=record.transaction_date,
            )
        )
    return PostTransactionsResult(posted_count=len(posted), transactions=posted)


def _inventory_failure_message(inventory: InventoryResult) -> str:
    for item in inventory.items:
        if not item.success and item.error.strip():
            return item.error
    return "One or more line items cannot be fulfilled by the required date."


def _observation(
    *,
    gate_passed: bool,
    step: str,
    error: str | None = None,
    **payload: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gate_passed": gate_passed,
        "step_completed": step,
    }
    if error:
        result["error"] = error
    result.update(payload)
    return result


def tool_run_quoting_agent(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 1: Parse customer request into QuoteResponse."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_run_quoting_agent")
    deps.quote = call_quoting_agent(deps.request.request_with_date)
    if deps.quote.success:
        deps.step_completed = STEP_QUOTING
        return _observation(
            gate_passed=True,
            step=STEP_QUOTING,
            quote=deps.quote.model_dump(),
        )
    error = deps.quote.error or "Quote could not be parsed."
    _record_tool_error(deps, error)
    return _observation(
        gate_passed=False,
        step=deps.step_completed,
        error=error,
        quote=deps.quote.model_dump(),
    )


def tool_run_inventory_check(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 2: Check stock and supplier lead times."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_run_inventory_check")
    if deps.quote is None or not can_check_inventory(deps.quote):
        error = "Quoting must succeed before inventory check."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    deps.inventory = InventoryTool().check(quote_to_inventory_request(deps.quote))
    if all(item.success for item in deps.inventory.items):
        deps.step_completed = STEP_INVENTORY
        return _observation(
            gate_passed=True,
            step=STEP_INVENTORY,
            inventory=deps.inventory.model_dump(),
        )
    error = _inventory_failure_message(deps.inventory)
    _record_tool_error(deps, error)
    return _observation(
        gate_passed=False,
        step=deps.step_completed,
        error=error,
        inventory=deps.inventory.model_dump(),
    )


def tool_run_pricing_review(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 3: Review historical pricing bands per line."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_run_pricing_review")
    if deps.quote is None or not can_review_pricing(deps.quote):
        error = "Inventory must succeed before pricing review."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    deps.pricing_review = call_pricing_review_agent(
        PricingReviewRequest(quote=deps.quote)
    )
    if deps.pricing_review.success:
        deps.step_completed = STEP_PRICING_REVIEW
        return _observation(
            gate_passed=True,
            step=STEP_PRICING_REVIEW,
            pricing_review=deps.pricing_review.model_dump(),
        )
    error = deps.pricing_review.error or "Pricing review failed."
    _record_tool_error(deps, error)
    return _observation(
        gate_passed=False,
        step=deps.step_completed,
        error=error,
        pricing_review=deps.pricing_review.model_dump(),
    )


def tool_run_pricing_tool(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 4: Validate cash and build strategy recommendations."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_run_pricing_tool")
    if (
        deps.quote is None
        or deps.inventory is None
        or deps.pricing_review is None
        or not can_price(deps.quote, deps.inventory)
    ):
        error = "Pricing review must succeed before pricing tool."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    unit_prices = [
        ItemUnitPrices(
            product_name=item.product_name,
            min_unit_price=item.min_unit_price,
            avg_unit_price=item.avg_unit_price,
            max_unit_price=item.max_unit_price,
        )
        for item in deps.pricing_review.items
    ]
    deps.pricing = PricingTool().price(
        PricingRequest(
            quote=deps.quote,
            inventory=deps.inventory,
            unit_prices=unit_prices,
        )
    )
    if deps.pricing.success:
        deps.step_completed = STEP_PRICING
        return _observation(
            gate_passed=True,
            step=STEP_PRICING,
            pricing=deps.pricing.model_dump(),
        )
    error = "Pricing could not be completed for this order."
    _record_tool_error(deps, error)
    return _observation(
        gate_passed=False,
        step=deps.step_completed,
        error=error,
        pricing=deps.pricing.model_dump(),
    )


def tool_run_order_recommendation(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 5: Final pricing, customer prose, and transaction batches."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_run_order_recommendation")
    if (
        deps.quote is None
        or deps.inventory is None
        or deps.pricing is None
        or not can_recommend_order(deps.quote, deps.inventory, deps.pricing)
    ):
        if _order_blocked_by_cash(deps):
            error = "Insufficient cash to fund this order."
        else:
            error = "Pricing must succeed before order recommendation."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    deps.recommendation = call_order_recommendation_agent(
        OrderRecommendationRequest(
            quote=deps.quote,
            inventory=deps.inventory,
            pricing=deps.pricing,
            customer=_customer_context(deps.request),
        )
    )
    if deps.recommendation.success:
        deps.step_completed = STEP_ORDER_RECOMMENDATION
        return _observation(
            gate_passed=True,
            step=STEP_ORDER_RECOMMENDATION,
            recommendation=deps.recommendation.model_dump(),
        )
    error = deps.recommendation.error or "Order recommendation failed."
    _record_tool_error(deps, error)
    return _observation(
        gate_passed=False,
        step=deps.step_completed,
        error=error,
        recommendation=deps.recommendation.model_dump(),
    )


def tool_post_stock_transactions(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 6: Post stock_orders via create_transaction on date_of_request."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_post_stock_transactions")
    if deps.recommendation is None or not deps.recommendation.success:
        error = "Order recommendation must succeed before posting stock."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    result = post_stock_transactions(
        deps.recommendation,
        create_transaction_fn=deps.create_transaction_fn,
    )
    if result.error:
        _record_tool_error(deps, result.error)
        return _observation(
            gate_passed=False,
            step=deps.step_completed,
            error=result.error,
            posting=result.model_dump(),
        )
    deps.posted_transactions.extend(result.transactions)
    deps.stock_transactions_posted = result.posted_count
    deps.step_completed = STEP_POST_STOCK
    return _observation(
        gate_passed=True,
        step=STEP_POST_STOCK,
        posting=result.model_dump(),
    )


def tool_post_sales_transactions(ctx: RunContext[OrchestratorDeps]) -> dict[str, Any]:
    """Step 7: Post sales via create_transaction on need_date."""
    deps = ctx.deps
    _record_tool_start(deps, "tool_post_sales_transactions")
    if deps.recommendation is None or not deps.recommendation.success:
        error = "Order recommendation must succeed before posting sales."
        _record_tool_error(deps, error)
        return _observation(gate_passed=False, step=deps.step_completed, error=error)
    result = post_sales_transactions(
        deps.recommendation,
        create_transaction_fn=deps.create_transaction_fn,
    )
    if result.error:
        _record_tool_error(deps, result.error)
        return _observation(
            gate_passed=False,
            step=deps.step_completed,
            error=result.error,
            posting=result.model_dump(),
        )
    deps.posted_transactions.extend(result.transactions)
    deps.sales_transactions_posted = result.posted_count
    deps.step_completed = STEP_POST_SALES
    return _observation(
        gate_passed=True,
        step=STEP_POST_SALES,
        posting=result.model_dump(),
    )


orchestrator_agent = Agent(
    MODEL,
    system_prompt=ORCHESTRATOR_DIRECTIVE,
    output_type=OrchestratorResponse,
    tools=[
        tool_run_quoting_agent,
        tool_run_inventory_check,
        tool_run_pricing_review,
        tool_run_pricing_tool,
        tool_run_order_recommendation,
        tool_post_stock_transactions,
        tool_post_sales_transactions,
    ],
    deps_type=OrchestratorDeps,
)


def _failure_response_from_deps(
    deps: OrchestratorDeps,
    *,
    agent_exception: str | None = None,
    failure_kind: FailureKind = "business_failure",
) -> OrchestratorResponse:
    message = _build_failure_message(deps)
    return OrchestratorResponse(
        success=False,
        step_completed=deps.step_completed,
        customer_response=message,
        date_of_request=deps.quote.date_of_request if deps.quote else None,
        need_date=deps.quote.need_date if deps.quote else None,
        stock_transactions_posted=deps.stock_transactions_posted,
        sales_transactions_posted=deps.sales_transactions_posted,
        posted_transactions=list(deps.posted_transactions),
        error=message,
        debug=_build_debug_info(
            deps,
            agent_exception=agent_exception,
            failure_kind=failure_kind,
        ),
    )


def handle_quote_request(request: OrchestratorRequest) -> OrchestratorResponse:
    """Run the seven-step ReAct pipeline and return a project_starter-ready response."""
    deps = OrchestratorDeps(request=request)
    try:
        result = orchestrator_agent.run_sync(
            request.model_dump_json(),
            deps=deps,
        ).output
        return _finalize_orchestrator_response(result, deps)
    except Exception as exc:
        _record_tool_error(deps, str(exc))
        return _finalize_orchestrator_response(
            _failure_response_from_deps(deps),
            deps,
            agent_exception=str(exc),
            failure_kind="react_exception",
        )


run_request = handle_quote_request
