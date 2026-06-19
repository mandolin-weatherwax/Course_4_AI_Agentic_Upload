"""Pricing review agent — historical min/avg/max unit prices per quote line."""

from __future__ import annotations

from typing import Any, Optional, Union

from dotenv import load_dotenv
from project_starter import search_quote_history
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.quoting_agent import QuoteResponse
from tools.pricing_tool import clamp_min_unit_price, default_unit_prices

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

MIN_HISTORY_ORDERS = 3
HISTORY_SEARCH_LIMIT = 20


class ItemPriceReview(BaseModel):
    product_name: str
    unit_cost: float
    history_count: int
    min_unit_price: float
    avg_unit_price: float
    max_unit_price: float
    error: Optional[str] = None


class PricingReviewResponse(BaseModel):
    success: bool
    date_of_request: str
    items: list[ItemPriceReview] = Field(default_factory=list)
    error: Optional[str] = None


class PricingReviewRequest(BaseModel):
    quote: QuoteResponse


class HistoricalLineCost(BaseModel):
    total_cost: float
    quantity: int = Field(ge=0)


class ComputeItemUnitPricesInput(BaseModel):
    product_name: str
    unit_cost: float = Field(gt=0)
    line_costs: list[HistoricalLineCost]


class ComputeItemUnitPricesResult(BaseModel):
    product_name: str
    history_count: int
    min_unit_price: float
    avg_unit_price: float
    max_unit_price: float


class ComputeItemUnitPricesError(BaseModel):
    error: str


PRICING_REVIEW_DIRECTIVE = """\
You are the Pricing Review Agent for Munder Difflin.
You receive a PricingReviewRequest JSON (quote) and return a PricingReviewResponse
with min, average, and max unit selling prices per quote line.

Follow these steps exactly, in order:

1. Validate date_of_request is set and items is non-empty. On failure return
   success=false with a non-null error.

2. For each QuoteItem.product_name, call tool_search_quote_history once with
   [product_name] and limit=20.

3. From each quote_explanation in the results, extract line-level
   {total_cost, quantity} pairs for that product only:
   - "500 sheets at $0.05 each" → {total_cost: 25.0, quantity: 500}
   - "$25 for A4 paper" with "500 sheets" in the same explanation →
     {total_cost: 25.0, quantity: 500}.
   - Ignore order-level rounded totals and amounts not tied to the product.
   - At most one pair per historical quote per product.
   - Skip entries with quantity=0.

4. Count pairs as history_count for the product.

5. If history_count >= 3: call tool_compute_item_unit_prices with
   product_name, unit_cost (from QuoteItem.unit_price), and line_costs
   (the list of extracted pairs). Copy min_unit_price, avg_unit_price,
   max_unit_price, and history_count from the tool result. Set error=null.
   If the tool returns an error field, set history_count from your extracted
   pairs and describe the error on the item.

6. If history_count < 3: do NOT call tool_compute_item_unit_prices. Set
   min/avg/max using DEFAULT_STRATEGY_MULTIPLIERS on unit_cost:
   min = max(unit_cost * 0.92, unit_cost * 0.85), avg = unit_cost * 1.05,
   max = unit_cost * 1.20 (round to 2 decimals). Set a non-null error explaining
   insufficient history.

7. Return PricingReviewResponse with success=true when all items are reviewed.

Rules:
- Never divide total_cost by quantity yourself.
- Never compute min, average, or max yourself when history_count >= 3.
- tool_compute_item_unit_prices only when history_count >= 3 for that item.
- One tool_search_quote_history call per distinct product_name.
- Pass extracted pairs as structured line_costs to the compute tool — not prose.
"""


def can_review_pricing(quote: QuoteResponse) -> bool:
    """Orchestrator gate before calling the pricing review agent."""
    return quote.success and quote.date_of_request is not None and len(quote.items) > 0


def compute_item_unit_prices(
    payload: ComputeItemUnitPricesInput,
) -> ComputeItemUnitPricesResult | ComputeItemUnitPricesError:
    """Derive min/avg/max unit prices from historical line costs and quantities."""
    unit_prices = [
        round(entry.total_cost / entry.quantity, 2)
        for entry in payload.line_costs
        if entry.quantity > 0
    ]

    if not unit_prices:
        return ComputeItemUnitPricesError(
            error="No valid line_costs with quantity > 0."
        )

    return ComputeItemUnitPricesResult(
        product_name=payload.product_name,
        history_count=len(unit_prices),
        min_unit_price=clamp_min_unit_price(payload.unit_cost, min(unit_prices)),
        avg_unit_price=round(sum(unit_prices) / len(unit_prices), 2),
        max_unit_price=round(max(unit_prices), 2),
    )


def _failure_response(*, date_of_request: str, message: str) -> PricingReviewResponse:
    return PricingReviewResponse(
        success=False,
        date_of_request=date_of_request,
        items=[],
        error=message,
    )


def _sanitize_item_price_review(item: ItemPriceReview) -> ItemPriceReview:
    """Clamp bands so pricing_tool.validate_unit_prices always passes."""
    min_price = clamp_min_unit_price(item.unit_cost, item.min_unit_price)
    avg_price = max(round(item.avg_unit_price, 2), min_price)
    max_price = max(round(item.max_unit_price, 2), avg_price)
    return item.model_copy(
        update={
            "min_unit_price": min_price,
            "avg_unit_price": avg_price,
            "max_unit_price": max_price,
        }
    )


def _normalize_review_response(
    response: PricingReviewResponse,
    quote: QuoteResponse,
) -> PricingReviewResponse:
    """Align agent output with quote lines and catalog unit_cost values."""
    if not response.success or not quote.items:
        return response

    items_by_name = {item.product_name: item for item in response.items}
    normalized: list[ItemPriceReview] = []

    for quote_item in quote.items:
        review = items_by_name.get(quote_item.product_name)
        if review is None:
            fallback = default_unit_prices(
                quote_item.product_name, quote_item.unit_price
            )
            normalized.append(
                ItemPriceReview(
                    product_name=quote_item.product_name,
                    unit_cost=quote_item.unit_price,
                    history_count=0,
                    min_unit_price=fallback.min_unit_price,
                    avg_unit_price=fallback.avg_unit_price,
                    max_unit_price=fallback.max_unit_price,
                    error="Agent did not return a review for this product.",
                )
            )
            continue

        normalized.append(
            _sanitize_item_price_review(
                review.model_copy(update={"unit_cost": quote_item.unit_price})
            )
        )

    return response.model_copy(update={"items": normalized})


def tool_search_quote_history(
    search_terms: list[str],
    limit: int = HISTORY_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search historical quotes matching any of the given terms.

    Args:
        search_terms: Keywords matched against requests and explanations.
        limit: Maximum records to return.

    Returns:
        List of quote history dicts from the database.
    """
    return search_quote_history(search_terms, limit=limit)


def tool_compute_item_unit_prices(
    payload: ComputeItemUnitPricesInput,
) -> dict[str, Any]:
    """
    Derive min/avg/max unit prices from historical line costs and quantities.

    Args:
        payload: product_name, unit_cost, and extracted line_costs pairs.

    Returns:
        ComputeItemUnitPricesResult or ComputeItemUnitPricesError as a dict.
    """
    result: Union[ComputeItemUnitPricesResult, ComputeItemUnitPricesError] = (
        compute_item_unit_prices(payload)
    )
    return result.model_dump()


pricing_review_agent = Agent(
    MODEL,
    system_prompt=PRICING_REVIEW_DIRECTIVE,
    output_type=PricingReviewResponse,
    tools=[
        tool_search_quote_history,
        tool_compute_item_unit_prices,
    ],
)


def call_pricing_review_agent(
    request: PricingReviewRequest,
) -> PricingReviewResponse:
    """Review historical pricing bands for each quote line via the LLM agent."""
    quote = request.quote

    if quote.date_of_request is None:
        return _failure_response(
            date_of_request="", message="Missing date_of_request on quote."
        )
    if not quote.items:
        return _failure_response(
            date_of_request=quote.date_of_request,
            message="Quote items must be non-empty for pricing review.",
        )

    result = pricing_review_agent.run_sync(request.model_dump_json()).output
    return _normalize_review_response(result, quote)
