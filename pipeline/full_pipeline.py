"""Full quote-to-recommendation pipeline runner and step validators."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from project_starter import paper_supplies

from agents.order_recommendation_agent import (
    OrderRecommendationRequest,
    OrderRecommendationResponse,
    call_order_recommendation_agent,
    can_recommend_order,
    customer_context_from_csv_row,
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

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CATALOG_NAMES = {item["item_name"] for item in paper_supplies}
_STRATEGIES = ("maximize_profit", "average_pricing", "maximize_turnover")
DEFAULT_PIPELINE_INDICES = (4, 5, 11)

STEP_NAMES = (
    "quoting",
    "inventory",
    "pricing_review",
    "pricing",
    "order_recommendation",
)


@dataclass
class PipelineContext:
    run_number: int
    csv_index: int
    job: str
    need_size: str
    event: str
    request_text: str
    request_date: str
    mood: str | None = None
    quote: QuoteResponse | None = None
    inventory: InventoryResult | None = None
    pricing_review: PricingReviewResponse | None = None
    pricing: PricingResponse | None = None
    recommendation: OrderRecommendationResponse | None = None


@dataclass
class PipelineStepResult:
    step: str
    passed: bool
    validation_errors: list[str] = field(default_factory=list)
    gate_blocked: bool = False


def format_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump()
    else:
        payload = obj
    return json.dumps(payload, indent=2, default=str)


def can_check_inventory(quote: QuoteResponse) -> bool:
    return (
        quote.success
        and quote.date_of_request is not None
        and quote.need_date is not None
        and len(quote.items) > 0
    )


def validate_quote_output(quote: QuoteResponse) -> list[str]:
    errors: list[str] = []

    if quote.success:
        if quote.error is not None:
            errors.append("success=true but error is set")
        if not quote.date_of_request or not _DATE_PATTERN.match(quote.date_of_request):
            errors.append("success=true requires date_of_request YYYY-MM-DD")
        if not quote.need_date or not _DATE_PATTERN.match(quote.need_date):
            errors.append("success=true requires need_date YYYY-MM-DD")
        if not quote.items:
            errors.append("success=true requires non-empty items")
        for item in quote.items:
            if item.product_name not in _CATALOG_NAMES:
                errors.append(f"unknown catalog product: {item.product_name!r}")
            if item.quantity_requested <= 0:
                errors.append(f"{item.product_name}: quantity_requested must be > 0")
            if item.unit_price <= 0:
                errors.append(f"{item.product_name}: unit_price must be > 0")
    else:
        if quote.error is None or not quote.error.strip():
            errors.append("success=false requires non-empty error")

    return errors


def validate_inventory_output(
    quote: QuoteResponse,
    inventory: InventoryResult,
) -> list[str]:
    errors: list[str] = []

    if inventory.date_of_request != quote.date_of_request:
        errors.append("inventory.date_of_request does not match quote")
    if inventory.need_date != quote.need_date:
        errors.append("inventory.need_date does not match quote")
    if len(inventory.items) != len(quote.items):
        errors.append(
            "item count mismatch: "
            f"quote={len(quote.items)} inventory={len(inventory.items)}"
        )

    quote_by_name = {item.product_name: item for item in quote.items}
    for checked in inventory.items:
        quote_item = quote_by_name.get(checked.product_name)
        if quote_item is None:
            errors.append(f"unexpected inventory line: {checked.product_name!r}")
            continue
        if checked.quantity_requested != quote_item.quantity_requested:
            errors.append(f"{checked.product_name}: quantity_requested mismatch")
        if checked.unit_price != quote_item.unit_price:
            errors.append(f"{checked.product_name}: unit_price mismatch")
        if checked.quantity_in_stock < 0 or checked.quantity_to_order < 0:
            errors.append(f"{checked.product_name}: negative stock fields")
        expected_shortfall = max(
            0, checked.quantity_requested - checked.quantity_in_stock
        )
        if checked.quantity_to_order != expected_shortfall:
            errors.append(
                f"{checked.product_name}: quantity_to_order "
                f"{checked.quantity_to_order} != expected shortfall "
                f"{expected_shortfall}"
            )
        if not checked.success and not checked.error.strip():
            errors.append(f"{checked.product_name}: success=false requires error text")

    return errors


def validate_pricing_review_output(
    quote: QuoteResponse,
    review: PricingReviewResponse,
) -> list[str]:
    errors: list[str] = []

    if not review.success:
        errors.append(f"pricing review failed: {review.error}")
        return errors

    if review.date_of_request != quote.date_of_request:
        errors.append("review.date_of_request does not match quote")

    quote_names = {item.product_name for item in quote.items}
    review_names = {item.product_name for item in review.items}
    if quote_names != review_names:
        errors.append(
            "product mismatch: "
            f"quote={sorted(quote_names)} review={sorted(review_names)}"
        )

    quote_costs = {item.product_name: item.unit_price for item in quote.items}
    for item in review.items:
        if item.min_unit_price > item.avg_unit_price:
            errors.append(f"{item.product_name}: min_unit_price > avg_unit_price")
        if item.avg_unit_price > item.max_unit_price:
            errors.append(f"{item.product_name}: avg_unit_price > max_unit_price")
        expected_cost = quote_costs.get(item.product_name)
        if expected_cost is not None and item.unit_cost != expected_cost:
            errors.append(
                f"{item.product_name}: unit_cost {item.unit_cost} != quote unit_price "
                f"{expected_cost}"
            )
        if item.min_unit_price <= 0 or item.max_unit_price <= 0:
            errors.append(f"{item.product_name}: price bands must be positive")

    return errors


def validate_pricing_output(
    quote: QuoteResponse,
    inventory: InventoryResult,
    pricing: PricingResponse,
) -> list[str]:
    errors: list[str] = []

    if not pricing.success:
        errors.append("pricing.success is false")
        return errors

    if pricing.date_of_request != quote.date_of_request:
        errors.append("pricing.date_of_request does not match quote")
    if pricing.need_date != quote.need_date:
        errors.append("pricing.need_date does not match quote")

    strategies = {rec.strategy for rec in pricing.recommendations}
    if strategies != set(_STRATEGIES):
        errors.append(f"expected strategies {_STRATEGIES}, got {sorted(strategies)}")

    quote_names = {item.product_name for item in quote.items}
    for rec in pricing.recommendations:
        rec_names = {item.product_name for item in rec.items}
        if rec_names != quote_names:
            errors.append(f"{rec.strategy}: product set mismatch")
        for item in rec.items:
            if item.unit_price <= 0:
                errors.append(f"{rec.strategy}/{item.product_name}: unit_price <= 0")
            if item.quantity_fulfilled < 0:
                errors.append(
                    f"{rec.strategy}/{item.product_name}: negative quantity_fulfilled"
                )

    inv_by_name = {item.product_name: item for item in inventory.items}
    mid_rec = next(
        (rec for rec in pricing.recommendations if rec.strategy == "average_pricing"),
        None,
    )
    if mid_rec and not any(item.included for item in mid_rec.items):
        errors.append("average_pricing strategy has no included lines")

    for quote_item in quote.items:
        inv_item = inv_by_name[quote_item.product_name]
        if inv_item.quantity_to_order > 0 and mid_rec:
            mid_line = next(
                item
                for item in mid_rec.items
                if item.product_name == quote_item.product_name
            )
            if mid_line.quantity_fulfilled != quote_item.quantity_requested:
                errors.append(
                    f"{quote_item.product_name}: fulfilled qty does not match request "
                    f"when stock is available or orderable"
                )

    return errors


def validate_recommendation_output(
    quote: QuoteResponse,
    pricing: PricingResponse,
    recommendation: OrderRecommendationResponse,
) -> list[str]:
    errors: list[str] = []

    if not recommendation.success:
        errors.append(f"recommendation failed: {recommendation.error}")
        return errors

    if recommendation.date_of_request != quote.date_of_request:
        errors.append("recommendation.date_of_request does not match quote")
    if recommendation.need_date != quote.need_date:
        errors.append("recommendation.need_date does not match quote")
    if not recommendation.customer_response.strip():
        errors.append("customer_response is empty")
    if not recommendation.pricing_justification.strip():
        errors.append("pricing_justification is empty")

    bands: dict[str, dict[str, float]] = {}
    for rec in pricing.recommendations:
        if rec.strategy == "maximize_turnover":
            for item in rec.items:
                bands[item.product_name] = {"floor": item.unit_price}
        elif rec.strategy == "average_pricing":
            for item in rec.items:
                bands.setdefault(item.product_name, {})["mid"] = item.unit_price
        elif rec.strategy == "maximize_profit":
            for item in rec.items:
                bands.setdefault(item.product_name, {})["ceiling"] = item.unit_price

    included = [item for item in recommendation.recommended_items if item.included]
    if not included:
        errors.append("no included recommended_items")

    for item in included:
        band = bands.get(item.product_name)
        if band is None:
            errors.append(f"no pricing band for {item.product_name!r}")
            continue
        floor = band.get("floor", 0.0)
        ceiling = band.get("ceiling", float("inf"))
        if item.unit_price < floor or item.unit_price > ceiling:
            errors.append(
                f"{item.product_name}: unit_price {item.unit_price} outside "
                f"[{floor}, {ceiling}]"
            )
        expected_line = round(item.unit_price * item.quantity_fulfilled, 2)
        if item.line_total != expected_line:
            errors.append(
                f"{item.product_name}: line_total {item.line_total} != {expected_line}"
            )

    if recommendation.sales is None or not recommendation.sales.transactions:
        errors.append("sales batch missing")
    else:
        sales_total = round(
            sum(tx.price for tx in recommendation.sales.transactions), 2
        )
        if abs(sales_total - round(recommendation.total_quote_amount, 2)) > 0.01:
            errors.append(
                f"total_quote_amount {recommendation.total_quote_amount} != "
                f"sales sum {sales_total}"
            )
        if recommendation.sales.transaction_date != quote.need_date:
            errors.append("sales.transaction_date must equal need_date")

    if recommendation.stock_orders is not None:
        if recommendation.stock_orders.transaction_date != quote.date_of_request:
            errors.append("stock_orders.transaction_date must equal date_of_request")

    return errors


def load_pipeline_requests(
    indices: tuple[int, ...],
    *,
    csv_path: str = "quote_requests_sample.csv",
) -> list[PipelineContext]:
    df = pd.read_csv(csv_path).copy()
    df.loc[:, "request_date"] = pd.to_datetime(
        df["request_date"], format="%m/%d/%y", errors="coerce"
    )
    df = (
        df.dropna(subset=["request_date"])
        .sort_values("request_date")
        .reset_index(drop=True)
    )

    contexts: list[PipelineContext] = []
    for idx in indices:
        if idx < 0 or idx >= len(df):
            raise ValueError(f"Index {idx} out of range (0..{len(df) - 1})")
        row = df.iloc[idx]
        customer = customer_context_from_csv_row(row)
        contexts.append(
            PipelineContext(
                run_number=len(contexts) + 1,
                csv_index=idx,
                job=str(row["job"]),
                need_size=str(row["need_size"]),
                event=str(row["event"]),
                request_text=str(row["request"]),
                request_date=row["request_date"].strftime("%Y-%m-%d"),
                mood=customer.mood,
            )
        )
    return contexts


def _step_result(
    step: str,
    *,
    validation_errors: list[str],
    gate_blocked: bool = False,
    success: bool = True,
) -> PipelineStepResult:
    passed = not gate_blocked and not validation_errors and success
    return PipelineStepResult(
        step=step,
        passed=passed,
        validation_errors=validation_errors,
        gate_blocked=gate_blocked,
    )


def run_full_pipeline(
    ctx: PipelineContext,
    *,
    verbose: Callable[[str], None] | None = None,
) -> dict[str, PipelineStepResult]:
    """Run all five pipeline steps; stop early when a gate or validation fails."""

    def log(message: str) -> None:
        if verbose is not None:
            verbose(message)

    results: dict[str, PipelineStepResult] = {}

    request_with_date = f"{ctx.request_text} (Date of request: {ctx.request_date})"
    log(f"STEP quoting: input request_date={ctx.request_date}")
    ctx.quote = call_quoting_agent(request_with_date)
    quote_errors = validate_quote_output(ctx.quote)
    log(
        f"STEP quoting: output success={ctx.quote.success} items={len(ctx.quote.items)}"
    )
    results["quoting"] = _step_result(
        "quoting",
        validation_errors=quote_errors,
        success=ctx.quote.success,
    )
    if not results["quoting"].passed:
        return _empty_tail(results)

    if not can_check_inventory(ctx.quote):
        results["inventory"] = PipelineStepResult(
            step="inventory", passed=False, gate_blocked=True
        )
        return _empty_tail(results)

    inv_request = quote_to_inventory_request(ctx.quote)
    ctx.inventory = InventoryTool().check(inv_request)
    inventory_errors = validate_inventory_output(ctx.quote, ctx.inventory)
    all_fulfilled = all(item.success for item in ctx.inventory.items)
    log(
        f"STEP inventory: output fulfilled={all_fulfilled} "
        f"lines={len(ctx.inventory.items)}"
    )
    results["inventory"] = _step_result(
        "inventory",
        validation_errors=inventory_errors,
        success=all_fulfilled,
    )
    if not results["inventory"].passed:
        return _empty_tail(results)

    if not can_review_pricing(ctx.quote):
        results["pricing_review"] = PipelineStepResult(
            step="pricing_review", passed=False, gate_blocked=True
        )
        return _empty_tail(results)

    ctx.pricing_review = call_pricing_review_agent(
        PricingReviewRequest(quote=ctx.quote)
    )
    review_errors = validate_pricing_review_output(ctx.quote, ctx.pricing_review)
    log(f"STEP pricing_review: output success={ctx.pricing_review.success}")
    results["pricing_review"] = _step_result(
        "pricing_review",
        validation_errors=review_errors,
        success=ctx.pricing_review.success,
    )
    if not results["pricing_review"].passed:
        return _empty_tail(results)

    if not can_price(ctx.quote, ctx.inventory):
        results["pricing"] = PipelineStepResult(
            step="pricing", passed=False, gate_blocked=True
        )
        return _empty_tail(results)

    unit_prices = [
        ItemUnitPrices(
            product_name=item.product_name,
            min_unit_price=item.min_unit_price,
            avg_unit_price=item.avg_unit_price,
            max_unit_price=item.max_unit_price,
        )
        for item in ctx.pricing_review.items
    ]
    ctx.pricing = PricingTool().price(
        PricingRequest(
            quote=ctx.quote,
            inventory=ctx.inventory,
            unit_prices=unit_prices,
        )
    )
    pricing_errors = validate_pricing_output(ctx.quote, ctx.inventory, ctx.pricing)
    log(f"STEP pricing: output success={ctx.pricing.success}")
    results["pricing"] = _step_result(
        "pricing",
        validation_errors=pricing_errors,
        success=ctx.pricing.success,
    )
    if not results["pricing"].passed:
        return _empty_tail(results)

    if not can_recommend_order(ctx.quote, ctx.inventory, ctx.pricing):
        results["order_recommendation"] = PipelineStepResult(
            step="order_recommendation", passed=False, gate_blocked=True
        )
        return results

    ctx.recommendation = call_order_recommendation_agent(
        OrderRecommendationRequest(
            quote=ctx.quote,
            inventory=ctx.inventory,
            pricing=ctx.pricing,
            customer=customer_context_from_csv_row(
                {
                    "request": ctx.request_text,
                    "job": ctx.job,
                    "need_size": ctx.need_size,
                    "event": ctx.event,
                    "mood": ctx.mood,
                }
            ),
        )
    )
    recommendation_errors = validate_recommendation_output(
        ctx.quote, ctx.pricing, ctx.recommendation
    )
    log(f"STEP order_recommendation: output success={ctx.recommendation.success}")
    results["order_recommendation"] = _step_result(
        "order_recommendation",
        validation_errors=recommendation_errors,
        success=ctx.recommendation.success,
    )
    return results


def _empty_tail(
    results: dict[str, PipelineStepResult],
) -> dict[str, PipelineStepResult]:
    for step in STEP_NAMES:
        if step not in results:
            results[step] = PipelineStepResult(step=step, passed=False)
    return results


def pipeline_passed(results: dict[str, PipelineStepResult]) -> bool:
    return all(result.passed for result in results.values())


def format_step_failure(results: dict[str, PipelineStepResult]) -> str:
    parts: list[str] = []
    for step in STEP_NAMES:
        result = results.get(step)
        if result is None:
            continue
        if result.passed:
            continue
        if result.gate_blocked:
            parts.append(f"{step}: gate blocked")
        elif result.validation_errors:
            parts.append(f"{step}: {result.validation_errors[0]}")
        else:
            parts.append(f"{step}: step failed")
    return "; ".join(parts) if parts else "unknown failure"
