"""Order recommendation agent — final pricing, customer response, and ledger batches."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from project_starter import generate_financial_report, get_all_inventory
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.quoting_agent import QuoteResponse
from tools.inventory_tool import InventoryResult
from tools.pricing_tool import PricingResponse, StrategyName

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

INVENTORY_PRESSURE_DISCOUNT_PCT = 15.0
LOW_CASH_THRESHOLD = 500.0
HIGH_INVENTORY_VALUE_THRESHOLD = 2000.0

DISCOUNT_EVENTS = {
    "party",
    "festival",
    "gathering",
    "concert",
    "reception",
    "celebration",
    "exhibition",
    "show",
}
DISCOUNT_NEED_SIZES = {"large"}
PREMIUM_NEED_SIZES = {"small"}

DISCOUNT_MOODS = {
    "angry",
    "sad",
    "miserable",
    "pissed off",
    "stressed",
    "upset",
    "frustrated",
    "unhappy",
    "depressed",
}
PREMIUM_MOODS = {"happy", "cheerful", "pleased", "excited", "delighted"}

LOW_QUANTITY_THRESHOLD = 200
HIGH_QUANTITY_THRESHOLD = 500

_STRATEGY_FLOOR: StrategyName = "maximize_turnover"
_STRATEGY_MID: StrategyName = "average_pricing"
_STRATEGY_CEILING: StrategyName = "maximize_profit"


class TransactionRecord(BaseModel):
    item_name: str
    transaction_type: Literal["stock_orders", "sales"]
    quantity: int = Field(gt=0)
    price: float = Field(gt=0, description="Total line price (not per-unit).")
    transaction_date: str


class TransactionBatch(BaseModel):
    transaction_date: str
    transactions: list[TransactionRecord] = Field(min_length=1)


class RecommendedLineItem(BaseModel):
    product_name: str
    quantity_requested: int
    quantity_fulfilled: int
    quantity_in_stock: int
    quantity_to_order: int
    unit_cost: float
    unit_price: float
    line_total: float
    included: bool


class OrderRecommendationResponse(BaseModel):
    success: bool
    date_of_request: str
    need_date: str
    recommended_items: list[RecommendedLineItem] = Field(default_factory=list)
    stock_orders: Optional[TransactionBatch] = None
    sales: Optional[TransactionBatch] = None
    customer_response: str = ""
    pricing_justification: str = ""
    total_quote_amount: float = 0.0
    error: Optional[str] = None


class CustomerContext(BaseModel):
    original_request_text: str
    job_type: Optional[str] = None
    need_size: Optional[str] = None
    event_type: Optional[str] = None
    mood: Optional[str] = None


def _optional_csv_field(row: Any, column: str) -> str | None:
    if isinstance(row, dict):
        if column not in row:
            return None
        value = row[column]
    elif hasattr(row, "index"):
        if column not in row.index:
            return None
        value = row[column]
    else:
        return None
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def customer_context_from_csv_row(
    row: Any,
    *,
    request_column: str = "request",
) -> CustomerContext:
    """Build CustomerContext from a quote_requests CSV row (Series or dict)."""
    request_text = _optional_csv_field(row, request_column)
    if request_text is None:
        raise ValueError(f"CSV row missing {request_column!r}")
    return CustomerContext(
        original_request_text=request_text,
        job_type=_optional_csv_field(row, "job"),
        need_size=_optional_csv_field(row, "need_size"),
        event_type=_optional_csv_field(row, "event"),
        mood=_optional_csv_field(row, "mood"),
    )


class OrderRecommendationRequest(BaseModel):
    quote: QuoteResponse
    inventory: InventoryResult
    pricing: PricingResponse
    customer: CustomerContext


class CompanyContextInput(BaseModel):
    date_of_request: str
    ordered_products: list[str] = Field(min_length=1)


class CompanyContextResult(BaseModel):
    as_of_date: str
    inventory: dict[str, int]
    cash_balance: float
    inventory_value: float
    total_assets: float
    inventory_summary: list[dict[str, Any]]
    top_selling_products: list[dict[str, Any]]
    inventory_pressure_pct: float
    ordered_in_stock_units: int
    total_units: int


class PriceBand(BaseModel):
    product_name: str
    floor: float
    mid: float
    ceiling: float
    unit_cost: float
    quantity_fulfilled: int
    included: bool


class BuildTransactionsInput(BaseModel):
    date_of_request: str
    need_date: str
    recommended_items: list[RecommendedLineItem] = Field(min_length=1)


class BuildTransactionsResult(BaseModel):
    stock_orders: Optional[TransactionBatch] = None
    sales: Optional[TransactionBatch] = None


class PricingSignalsInput(BaseModel):
    company_context: CompanyContextResult
    customer: CustomerContext


class PricingSignalsResult(BaseModel):
    lean_discount_inventory: bool
    lean_premium_cash_stress: bool
    lean_discount_need_size: bool
    lean_premium_need_size: bool
    lean_discount_event: bool
    lean_discount_mood: bool
    lean_premium_mood: bool
    mood: Optional[str] = None
    summary: str


class ValidateOrderRecommendationPayload(BaseModel):
    response: OrderRecommendationResponse
    price_bands: list[PriceBand] = Field(default_factory=list)


class OrderRecommendationValidationError(BaseModel):
    error: str


def compute_inventory_pressure(
    date_of_request: str,
    ordered_products: list[str],
    *,
    inventory: dict[str, int] | None = None,
) -> dict[str, float | int]:
    """Percentage of total on-hand units held in ordered SKUs."""
    all_inventory = (
        inventory if inventory is not None else get_all_inventory(date_of_request)
    )
    total_units = sum(all_inventory.values())
    if total_units == 0:
        return {
            "inventory_pressure_pct": 0.0,
            "ordered_in_stock_units": 0,
            "total_units": 0,
        }

    ordered_in_stock_units = sum(
        all_inventory.get(product, 0) for product in ordered_products
    )
    return {
        "inventory_pressure_pct": round(
            100.0 * ordered_in_stock_units / total_units, 2
        ),
        "ordered_in_stock_units": ordered_in_stock_units,
        "total_units": total_units,
    }


def gather_company_context(
    payload: CompanyContextInput,
) -> CompanyContextResult:
    """Inventory snapshot, financial report, and inventory pressure in one call."""
    inventory = get_all_inventory(payload.date_of_request)
    financial = generate_financial_report(payload.date_of_request)
    pressure = compute_inventory_pressure(
        payload.date_of_request,
        payload.ordered_products,
        inventory=inventory,
    )

    return CompanyContextResult(
        as_of_date=payload.date_of_request,
        inventory=inventory,
        cash_balance=float(financial["cash_balance"]),
        inventory_value=float(financial["inventory_value"]),
        total_assets=float(financial["total_assets"]),
        inventory_summary=list(financial["inventory_summary"]),
        top_selling_products=list(financial["top_selling_products"]),
        inventory_pressure_pct=float(pressure["inventory_pressure_pct"]),
        ordered_in_stock_units=int(pressure["ordered_in_stock_units"]),
        total_units=int(pressure["total_units"]),
    )


def extract_price_bands(pricing: PricingResponse) -> list[PriceBand]:
    """Flatten three strategy recommendations into one row per product."""
    by_strategy: dict[StrategyName, dict[str, Any]] = {}
    for recommendation in pricing.recommendations:
        by_strategy[recommendation.strategy] = {
            item.product_name: item for item in recommendation.items
        }

    missing = {
        strategy
        for strategy in (_STRATEGY_FLOOR, _STRATEGY_MID, _STRATEGY_CEILING)
        if strategy not in by_strategy
    }
    if missing:
        raise ValueError(f"PricingResponse missing strategies: {sorted(missing)}")

    anchor = by_strategy[_STRATEGY_MID]
    bands: list[PriceBand] = []
    for product_name, mid_item in anchor.items():
        floor_item = by_strategy[_STRATEGY_FLOOR][product_name]
        ceiling_item = by_strategy[_STRATEGY_CEILING][product_name]
        bands.append(
            PriceBand(
                product_name=product_name,
                floor=round(floor_item.unit_price, 2),
                mid=round(mid_item.unit_price, 2),
                ceiling=round(ceiling_item.unit_price, 2),
                unit_cost=round(mid_item.unit_cost, 2),
                quantity_fulfilled=mid_item.quantity_fulfilled,
                included=mid_item.included,
            )
        )

    return bands


def build_transaction_batches(
    payload: BuildTransactionsInput,
) -> BuildTransactionsResult:
    """Build stock_orders and sales batches from recommended line items."""
    stock_records: list[TransactionRecord] = []
    sales_records: list[TransactionRecord] = []

    for item in payload.recommended_items:
        if item.quantity_to_order > 0:
            stock_records.append(
                TransactionRecord(
                    item_name=item.product_name,
                    transaction_type="stock_orders",
                    quantity=item.quantity_to_order,
                    price=round(item.unit_cost * item.quantity_to_order, 2),
                    transaction_date=payload.date_of_request,
                )
            )

        if item.included and item.quantity_fulfilled > 0:
            sales_records.append(
                TransactionRecord(
                    item_name=item.product_name,
                    transaction_type="sales",
                    quantity=item.quantity_fulfilled,
                    price=round(item.unit_price * item.quantity_fulfilled, 2),
                    transaction_date=payload.need_date,
                )
            )

    return BuildTransactionsResult(
        stock_orders=(
            TransactionBatch(
                transaction_date=payload.date_of_request,
                transactions=stock_records,
            )
            if stock_records
            else None
        ),
        sales=(
            TransactionBatch(
                transaction_date=payload.need_date,
                transactions=sales_records,
            )
            if sales_records
            else None
        ),
    )


def tool_gather_company_context(
    payload: CompanyContextInput,
) -> dict[str, Any]:
    """
    Return inventory, financial health, and inventory pressure in one call.

    Args:
        payload: date_of_request and ordered product names.

    Returns:
        CompanyContextResult as a dict.
    """
    return gather_company_context(payload).model_dump()


def tool_extract_price_bands(pricing: PricingResponse) -> list[dict[str, Any]]:
    """
    Flatten PricingResponse strategy recommendations into per-product price bands.

    Args:
        pricing: PricingTool output with three strategy recommendations.

    Returns:
        List of PriceBand dicts with floor, mid, ceiling, and inclusion flags.
    """
    return [band.model_dump() for band in extract_price_bands(pricing)]


def tool_build_transaction_batches(
    payload: BuildTransactionsInput,
) -> dict[str, Any]:
    """
    Build stock_orders and sales transaction batches from recommended items.

    Args:
        payload: Dates and finalized recommended line items.

    Returns:
        BuildTransactionsResult as a dict with optional stock_orders and sales batches.
    """
    return build_transaction_batches(payload).model_dump()


def evaluate_pricing_signals(
    payload: PricingSignalsInput,
) -> PricingSignalsResult:
    """Evaluate pricing policy lean flags from company context and customer metadata."""
    context = payload.company_context
    customer = payload.customer

    lean_discount_inventory = (
        context.inventory_pressure_pct >= INVENTORY_PRESSURE_DISCOUNT_PCT
    )
    lean_premium_cash_stress = (
        context.cash_balance < LOW_CASH_THRESHOLD
        and context.inventory_value > HIGH_INVENTORY_VALUE_THRESHOLD
    )

    need_size = (customer.need_size or "").strip().lower()
    lean_discount_need_size = need_size in DISCOUNT_NEED_SIZES
    lean_premium_need_size = need_size in PREMIUM_NEED_SIZES

    event_type = (customer.event_type or "").strip().lower()
    lean_discount_event = event_type in DISCOUNT_EVENTS

    mood_raw = (customer.mood or "").strip()
    mood = mood_raw.lower() if mood_raw else ""
    lean_discount_mood = mood in DISCOUNT_MOODS
    lean_premium_mood = mood in PREMIUM_MOODS

    signals: list[str] = []
    if lean_discount_inventory:
        signals.append(
            f"inventory pressure {context.inventory_pressure_pct}% >= "
            f"{INVENTORY_PRESSURE_DISCOUNT_PCT}% (discount lean)"
        )
    if lean_premium_cash_stress:
        signals.append(
            f"low cash ({context.cash_balance}) + high inventory value "
            f"({context.inventory_value}) (premium lean)"
        )
    if lean_discount_need_size:
        signals.append(f"need_size={need_size} (discount lean)")
    if lean_premium_need_size:
        signals.append(f"need_size={need_size} (premium lean)")
    if lean_discount_event:
        signals.append(f"event_type={event_type} (discount lean)")
    if lean_discount_mood:
        signals.append(f"mood={mood_raw} (discount lean)")
    if lean_premium_mood:
        signals.append(f"mood={mood_raw} (premium lean)")

    summary = "; ".join(signals) if signals else "neutral pricing signals"

    return PricingSignalsResult(
        lean_discount_inventory=lean_discount_inventory,
        lean_premium_cash_stress=lean_premium_cash_stress,
        lean_discount_need_size=lean_discount_need_size,
        lean_premium_need_size=lean_premium_need_size,
        lean_discount_event=lean_discount_event,
        lean_discount_mood=lean_discount_mood,
        lean_premium_mood=lean_premium_mood,
        mood=mood_raw or None,
        summary=summary,
    )


def _is_valid_date(value: str) -> bool:
    return bool(_DATE_PATTERN.match(value))


def _validation_error(
    response: OrderRecommendationResponse,
    pricing: PricingResponse,
) -> str | None:
    """Return a validation error string, or None when the draft is acceptable."""
    try:
        bands = extract_price_bands(pricing)
    except ValueError as exc:
        return str(exc)
    result = validate_order_recommendation(
        ValidateOrderRecommendationPayload(response=response, price_bands=bands)
    )
    if isinstance(result, OrderRecommendationValidationError):
        return result.error
    return None


def build_fallback_recommendation(
    request: OrderRecommendationRequest,
) -> OrderRecommendationResponse:
    """Deterministic mid-band quote when the LLM draft fails validation."""
    quote = request.quote
    date_of_request = quote.date_of_request or ""
    need_date = quote.need_date or ""
    inv_by_name = {item.product_name: item for item in request.inventory.items}
    bands = extract_price_bands(request.pricing)

    recommended_items: list[RecommendedLineItem] = []
    prose_lines: list[str] = []

    for band in bands:
        inv = inv_by_name[band.product_name]
        included = band.included and band.quantity_fulfilled > 0
        if included:
            unit_price = band.mid
            quantity_fulfilled = band.quantity_fulfilled
            line_total = round(unit_price * quantity_fulfilled, 2)
            prose_lines.append(
                f"- {band.product_name}: {quantity_fulfilled} at "
                f"${unit_price:.2f} each = ${line_total:.2f}"
            )
        else:
            unit_price = 0.0
            quantity_fulfilled = 0
            line_total = 0.0

        recommended_items.append(
            RecommendedLineItem(
                product_name=band.product_name,
                quantity_requested=inv.quantity_requested,
                quantity_fulfilled=quantity_fulfilled,
                quantity_in_stock=inv.quantity_in_stock,
                quantity_to_order=inv.quantity_to_order,
                unit_cost=band.unit_cost,
                unit_price=unit_price,
                line_total=line_total,
                included=included,
            )
        )

    batches = build_transaction_batches(
        BuildTransactionsInput(
            date_of_request=date_of_request,
            need_date=need_date,
            recommended_items=recommended_items,
        )
    )
    sales = batches.sales
    total_quote_amount = round(
        sum(record.price for record in sales.transactions) if sales else 0.0,
        2,
    )

    customer_response = (
        "Thank you for your order. We have prepared the following quote:\n\n"
        + "\n".join(prose_lines)
        + f"\n\nTotal quote amount: ${total_quote_amount:,.2f}. "
        f"We can deliver by {need_date}."
    )
    if quote.excluded_products:
        excluded = ", ".join(quote.excluded_products)
        customer_response += (
            f"\n\nWe were unable to quote the following items: {excluded}. "
            "The total above covers only the items listed."
        )

    return OrderRecommendationResponse(
        success=True,
        date_of_request=date_of_request,
        need_date=need_date,
        recommended_items=recommended_items,
        stock_orders=batches.stock_orders,
        sales=sales,
        customer_response=customer_response,
        pricing_justification=(
            "Deterministic fallback: each included line priced at the mid "
            "strategy band from pricing tool output."
        ),
        total_quote_amount=total_quote_amount,
    )


def validate_order_recommendation(
    payload: ValidateOrderRecommendationPayload,
) -> OrderRecommendationResponse | OrderRecommendationValidationError:
    """Validate draft order recommendation against bands, dates, and totals."""
    response = payload.response
    bands_by_product = {band.product_name: band for band in payload.price_bands}

    if not _is_valid_date(response.date_of_request):
        return OrderRecommendationValidationError(
            error="date_of_request must be YYYY-MM-DD."
        )
    if not _is_valid_date(response.need_date):
        return OrderRecommendationValidationError(error="need_date must be YYYY-MM-DD.")

    if response.success:
        if not response.recommended_items:
            return OrderRecommendationValidationError(
                error="success=true requires non-empty recommended_items."
            )
        if not response.customer_response.strip():
            return OrderRecommendationValidationError(
                error="success=true requires non-empty customer_response."
            )
        if not response.pricing_justification.strip():
            return OrderRecommendationValidationError(
                error="success=true requires non-empty pricing_justification."
            )

        included_items = [item for item in response.recommended_items if item.included]
        if not included_items:
            return OrderRecommendationValidationError(
                error="success=true requires at least one included recommended item."
            )

        for item in included_items:
            band = bands_by_product.get(item.product_name)
            if band is None:
                return OrderRecommendationValidationError(
                    error=f"No price band for {item.product_name!r}."
                )
            if item.unit_price < band.floor or item.unit_price > band.ceiling:
                return OrderRecommendationValidationError(
                    error=(
                        f"{item.product_name}: unit_price {item.unit_price} outside "
                        f"[{band.floor}, {band.ceiling}]."
                    )
                )
            expected_line_total = round(item.unit_price * item.quantity_fulfilled, 2)
            if item.line_total != expected_line_total:
                return OrderRecommendationValidationError(
                    error=(
                        f"{item.product_name}: line_total {item.line_total} != "
                        f"{expected_line_total}."
                    )
                )

        if response.sales is None or not response.sales.transactions:
            return OrderRecommendationValidationError(
                error="success=true requires a non-empty sales batch."
            )

        sales_total = round(
            sum(record.price for record in response.sales.transactions),
            2,
        )
        if abs(sales_total - round(response.total_quote_amount, 2)) > 0.01:
            return OrderRecommendationValidationError(
                error=(
                    f"total_quote_amount {response.total_quote_amount} does not match "
                    f"sales sum {sales_total}."
                )
            )

        if response.sales.transaction_date != response.need_date:
            return OrderRecommendationValidationError(
                error="sales.transaction_date must equal need_date."
            )

        if response.stock_orders is not None:
            if response.stock_orders.transaction_date != response.date_of_request:
                return OrderRecommendationValidationError(
                    error="stock_orders.transaction_date must equal date_of_request."
                )

    return response


def tool_evaluate_pricing_signals(
    payload: PricingSignalsInput,
) -> dict[str, Any]:
    """
    Evaluate pricing policy lean flags from company context and customer metadata.

    Args:
        payload: CompanyContextResult and CustomerContext.

    Returns:
        PricingSignalsResult as a dict.
    """
    return evaluate_pricing_signals(payload).model_dump()


def tool_validate_order_recommendation(
    payload: ValidateOrderRecommendationPayload,
) -> dict[str, Any]:
    """
    Validate a draft OrderRecommendationResponse against price bands and invariants.

    Args:
        payload: Draft response and price bands from extract_price_bands.

    Returns:
        Validated OrderRecommendationResponse dict, or {"error": "..."} on failure.
    """
    result = validate_order_recommendation(payload)
    return result.model_dump()


ORDER_RECOMMENDATION_DIRECTIVE = """\
You are the Order Recommendation Agent for Munder Difflin.
You receive an OrderRecommendationRequest JSON and return an OrderRecommendationResponse
with final unit prices, customer prose, ledger batches, and pricing justification.

Follow these steps exactly, in order:

1. Validate quote.date_of_request, quote.need_date, non-empty quote.items,
   pricing.success, and inventory success for every line. On failure return
   success=false with error.

2. Call tool_gather_company_context once with date_of_request and all quote
   product names.

3. Call tool_extract_price_bands with pricing from the request.

4. Call tool_evaluate_pricing_signals with the company context from step 2 and customer
   from the request. Use the returned lean flags when applying pricing policy.

5. For each quote line, choose one unit_price per included line:
   - Hard bounds: floor <= unit_price <= ceiling from price bands.
   - Default anchor: mid band; adjust using lean flags and per-line quantity:
     * quantity_requested < 200: favor discount (closer to floor)
     * quantity_requested >= 500: favor premium (closer to ceiling)
   - When customer.mood is provided, apply mood pricing
     (from tool_evaluate_pricing_signals):
     * lean_premium_mood (e.g. happy): favor higher pricing (closer to ceiling)
     * lean_discount_mood (e.g. angry, sad, miserable, stressed, pissed off): favor
       discount (closer to floor)
     * If mood is absent or unrecognized, do not apply mood leans.
   - Prioritize profit when signals conflict but never exceed ceiling.
   - Excluded lines (included=false): set quantity_fulfilled=0, line_total=0;
     still list them.

6. Build recommended_items with inventory fields from the request (quantity_in_stock,
   quantity_to_order from inventory.items; quantity_fulfilled from price bands).

7. Call tool_build_transaction_batches with date_of_request, need_date, and
   recommended_items. Copy stock_orders and sales from the tool result.

8. Compose customer_response prose:
   - Thank the customer; reference event/need_size when known.
   - Line-item breakdown: quantity, unit price, subtotals.
   - Mention bulk discounts or event support when policy leans discount.
   - State total_quote_amount and confirm delivery by need_date.
   - If quote.excluded_products is non-empty, clearly state which requested items
     could not be quoted and that the total covers only the items listed above.
   - Match a warm professional tone (no JSON, no tool names).

9. Write pricing_justification citing: tool_evaluate_pricing_signals summary, per-line
   quantity tactic, and final unit_price vs floor/mid/ceiling for each included line.
   For every included line where unit_price differs from the mid band:
   - State the deviation explicitly (e.g. "priced at 0.05 vs mid 0.06").
   - Name the PRIMARY driver that moved price away from mid — choose one dominant
     reason and be specific:
     * If mood drove the move: cite customer.mood (e.g. "primary driver: happy mood
       → premium lean above mid").
     * If event drove the move: cite event_type (e.g. "primary driver: party event
       → discount lean below mid").
     * Otherwise cite inventory pressure, cash stress, need_size, or quantity tactic.
   - When both mood and event apply, state which you weighted more and why
     for that line.
   Lines priced exactly at mid still need a brief note that mid band was used.

10. Set total_quote_amount to the sum of sales transaction prices from step 7.

11. Call tool_validate_order_recommendation with your draft response and the price_bands
    from step 3. Use the validated result as your final output.

Rules:
- Never invent products not in the quote.
- Never skip tool_validate_order_recommendation.
- Never price outside [floor, ceiling] for included lines.
- stock_orders only when quantity_to_order > 0; sales only for included fulfilled lines.
"""


order_recommendation_agent = Agent(
    MODEL,
    system_prompt=ORDER_RECOMMENDATION_DIRECTIVE,
    output_type=OrderRecommendationResponse,
    tools=[
        tool_gather_company_context,
        tool_extract_price_bands,
        tool_evaluate_pricing_signals,
        tool_build_transaction_batches,
        tool_validate_order_recommendation,
    ],
)


def can_recommend_order(
    quote: QuoteResponse,
    inventory: InventoryResult,
    pricing: PricingResponse,
) -> bool:
    """Orchestrator gate before calling the order recommendation agent."""
    if not (
        quote.success
        and quote.date_of_request is not None
        and quote.need_date is not None
        and len(quote.items) > 0
        and pricing.success
        and all(item.success for item in inventory.items)
    ):
        return False

    return any(
        item.included
        for rec in pricing.recommendations
        for item in rec.items
        if rec.strategy == _STRATEGY_MID
    )


def _failure_response(
    *, date_of_request: str, need_date: str, message: str
) -> OrderRecommendationResponse:
    return OrderRecommendationResponse(
        success=False,
        date_of_request=date_of_request,
        need_date=need_date,
        error=message,
    )


def call_order_recommendation_agent(
    request: OrderRecommendationRequest,
) -> OrderRecommendationResponse:
    """Recommend final pricing and compose customer response via the LLM agent."""
    quote = request.quote
    date_of_request = quote.date_of_request or ""
    need_date = quote.need_date or ""

    if not can_recommend_order(quote, request.inventory, request.pricing):
        return _failure_response(
            date_of_request=date_of_request,
            need_date=need_date,
            message=(
                "Quote, inventory, or pricing is not ready for order recommendation."
            ),
        )

    llm_result = order_recommendation_agent.run_sync(request.model_dump_json()).output

    if llm_result.success and _validation_error(llm_result, request.pricing) is None:
        return llm_result

    llm_error = llm_result.error or _validation_error(llm_result, request.pricing)

    fallback = build_fallback_recommendation(request)
    fallback_error = _validation_error(fallback, request.pricing)
    if fallback_error is None:
        return fallback

    detail = llm_error or "LLM draft was invalid"
    return _failure_response(
        date_of_request=date_of_request,
        need_date=need_date,
        message=(
            f"Order recommendation validation failed: {detail}. "
            f"Fallback also failed: {fallback_error}"
        ),
    )
