"""Deterministic pricing — three strategy recommendations from quote + inventory."""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Literal, Optional

from project_starter import get_cash_balance
from pydantic import BaseModel, Field

from agents.quoting_agent import QuoteResponse
from tools.inventory_tool import InventoryResult

MAX_SUBSET_LINES = 20
MIN_UNIT_PRICE_FLOOR = 0.85

StrategyName = Literal["maximize_profit", "average_pricing", "maximize_turnover"]

DEFAULT_STRATEGY_MULTIPLIERS: dict[StrategyName, float] = {
    "maximize_profit": 1.20,
    "average_pricing": 1.05,
    "maximize_turnover": 0.92,
}

STRATEGY_ORDER: list[StrategyName] = [
    "maximize_profit",
    "average_pricing",
    "maximize_turnover",
]

STRATEGY_UNIT_PRICE_FIELD: dict[
    StrategyName, Literal["min_unit_price", "avg_unit_price", "max_unit_price"]
] = {
    "maximize_profit": "max_unit_price",
    "average_pricing": "avg_unit_price",
    "maximize_turnover": "min_unit_price",
}


class ItemUnitPrices(BaseModel):
    """Required per-product min / avg / max unit prices from Pricing Review Agent."""

    product_name: str
    min_unit_price: float
    avg_unit_price: float
    max_unit_price: float


class PricedLineItem(BaseModel):
    product_name: str
    quantity_requested: int
    quantity_fulfilled: int
    unit_cost: float
    unit_price: float
    line_revenue: float
    line_acquisition_cost: float
    included: bool


class PricingRecommendation(BaseModel):
    strategy: StrategyName
    items: list[PricedLineItem]
    total_acquisition_cost: float
    total_profit: float
    error: Optional[str] = None


class PricingResponse(BaseModel):
    success: bool
    date_of_request: str
    need_date: str
    recommendations: list[PricingRecommendation]


class PricingRequest(BaseModel):
    """Orchestrator handoff: quote + inventory + per-item unit price bands."""

    quote: QuoteResponse
    inventory: InventoryResult
    unit_prices: list[ItemUnitPrices] = Field(
        min_length=1,
        description="One entry per quote line; from Pricing Review Agent.",
    )


class ProcurementLine(BaseModel):
    product_name: str
    quantity_requested: int
    quantity_to_order: int
    unit_cost: float


class SelectItemsResult(BaseModel):
    included_by_product: dict[str, bool]
    error: Optional[str] = None


CashBalanceFn = Callable[[str], float]


def clamp_min_unit_price(unit_cost: float, min_unit_price: float) -> float:
    """Ensure min unit price is never below unit_cost × MIN_UNIT_PRICE_FLOOR."""
    floor = round(unit_cost * MIN_UNIT_PRICE_FLOOR, 2)
    return max(round(min_unit_price, 2), floor)


def default_unit_prices(product_name: str, unit_cost: float) -> ItemUnitPrices:
    """Fallback bands from DEFAULT_STRATEGY_MULTIPLIERS when history is insufficient."""
    raw_min = round(unit_cost * DEFAULT_STRATEGY_MULTIPLIERS["maximize_turnover"], 2)
    return ItemUnitPrices(
        product_name=product_name,
        min_unit_price=clamp_min_unit_price(unit_cost, raw_min),
        avg_unit_price=round(
            unit_cost * DEFAULT_STRATEGY_MULTIPLIERS["average_pricing"], 2
        ),
        max_unit_price=round(
            unit_cost * DEFAULT_STRATEGY_MULTIPLIERS["maximize_profit"], 2
        ),
    )


def unit_prices_by_product(
    unit_prices: list[ItemUnitPrices],
) -> dict[str, ItemUnitPrices]:
    return {entry.product_name: entry for entry in unit_prices}


def strategy_unit_price(strategy: StrategyName, prices: ItemUnitPrices) -> float:
    field = STRATEGY_UNIT_PRICE_FIELD[strategy]
    return getattr(prices, field)


def can_price(quote: QuoteResponse, inventory: InventoryResult) -> bool:
    """Orchestrator gate before calling PricingTool.price()."""
    return (
        quote.success
        and quote.date_of_request is not None
        and quote.need_date is not None
        and len(quote.items) > 0
        and len(inventory.items) == len(quote.items)
        and all(item.success for item in inventory.items)
    )


def build_procurement_lines(
    quote: QuoteResponse,
    inventory: InventoryResult,
) -> list[ProcurementLine]:
    """Join quote + inventory into procurement lines. Fail fast on mismatch."""
    if len(quote.items) != len(inventory.items):
        raise ValueError(
            f"Quote/inventory item count mismatch: "
            f"{len(quote.items)} vs {len(inventory.items)}"
        )

    inv_by_name = {item.product_name: item for item in inventory.items}
    lines: list[ProcurementLine] = []

    for quote_item in quote.items:
        checked = inv_by_name.get(quote_item.product_name)
        if checked is None:
            raise ValueError(f"No inventory row for {quote_item.product_name!r}")
        lines.append(
            ProcurementLine(
                product_name=quote_item.product_name,
                quantity_requested=quote_item.quantity_requested,
                quantity_to_order=checked.quantity_to_order,
                unit_cost=quote_item.unit_price,
            )
        )

    return lines


def validate_unit_prices(
    lines: list[ProcurementLine],
    prices: dict[str, ItemUnitPrices],
) -> None:
    """Ensure every procurement line has required min / avg / max unit prices."""
    for line in lines:
        if line.product_name not in prices:
            raise ValueError(f"No unit prices for {line.product_name!r}")
        item_prices = prices[line.product_name]
        floor = round(line.unit_cost * MIN_UNIT_PRICE_FLOOR, 2)
        if item_prices.min_unit_price < floor:
            raise ValueError(
                f"min_unit_price for {line.product_name!r} is below floor "
                f"{floor} (unit_cost × {MIN_UNIT_PRICE_FLOOR})"
            )


def _line_selection_profit(
    line: ProcurementLine,
    max_unit_price: float,
) -> float:
    revenue = max_unit_price * line.quantity_requested
    acquisition = line.unit_cost * line.quantity_to_order
    return revenue - acquisition


def _line_selection_revenue(
    line: ProcurementLine,
    max_unit_price: float,
) -> float:
    return max_unit_price * line.quantity_requested


def select_items_for_cash(
    cash_balance: float,
    lines: list[ProcurementLine],
    prices_by_product: dict[str, ItemUnitPrices],
) -> SelectItemsResult:
    """Pick the profit-maximizing affordable procurement subset."""
    if len(lines) > MAX_SUBSET_LINES:
        return SelectItemsResult(
            included_by_product={line.product_name: False for line in lines},
            error=(
                f"Too many line items ({len(lines)}) for subset enumeration; "
                f"maximum is {MAX_SUBSET_LINES}."
            ),
        )

    best_subset: tuple[ProcurementLine, ...] = ()
    best_profit = float("-inf")
    best_revenue = float("-inf")

    for size in range(1, len(lines) + 1):
        for subset in combinations(lines, size):
            total_cost = sum(item.quantity_to_order * item.unit_cost for item in subset)
            if total_cost > cash_balance:
                continue
            total_profit = sum(
                _line_selection_profit(
                    item,
                    prices_by_product[item.product_name].max_unit_price,
                )
                for item in subset
            )
            total_revenue = sum(
                _line_selection_revenue(
                    item,
                    prices_by_product[item.product_name].max_unit_price,
                )
                for item in subset
            )
            if (
                total_profit > best_profit
                or (total_profit == best_profit and len(subset) > len(best_subset))
                or (
                    total_profit == best_profit
                    and len(subset) == len(best_subset)
                    and total_revenue > best_revenue
                )
            ):
                best_profit = total_profit
                best_revenue = total_revenue
                best_subset = subset

    included_names = {item.product_name for item in best_subset}
    included_by_product = {
        line.product_name: line.product_name in included_names for line in lines
    }
    excluded = [name for name, included in included_by_product.items() if not included]

    error: Optional[str] = None
    if excluded:
        error = f"Insufficient cash: excluded {', '.join(excluded)}."

    return SelectItemsResult(included_by_product=included_by_product, error=error)


def _failure_response(
    *,
    date_of_request: str,
    need_date: str,
    message: str,
) -> PricingResponse:
    recommendations = [
        PricingRecommendation(
            strategy=strategy,
            items=[],
            total_acquisition_cost=0.0,
            total_profit=0.0,
            error=message,
        )
        for strategy in STRATEGY_ORDER
    ]
    return PricingResponse(
        success=False,
        date_of_request=date_of_request,
        need_date=need_date,
        recommendations=recommendations,
    )


def build_pricing_response(
    *,
    cash_balance: float,
    date_of_request: str,
    need_date: str,
    lines: list[ProcurementLine],
    prices_by_product: dict[str, ItemUnitPrices],
) -> PricingResponse:
    """Build a complete PricingResponse from cash balance and procurement lines."""
    validate_unit_prices(lines, prices_by_product)

    total_acquisition_cost = sum(
        line.quantity_to_order * line.unit_cost for line in lines
    )

    if cash_balance >= total_acquisition_cost:
        included_by_product = {line.product_name: True for line in lines}
        selection_error: Optional[str] = None
    else:
        selection = select_items_for_cash(cash_balance, lines, prices_by_product)
        included_by_product = selection.included_by_product
        selection_error = selection.error

    recommendations: list[PricingRecommendation] = []

    for strategy in STRATEGY_ORDER:
        items: list[PricedLineItem] = []
        total_acq = 0.0
        total_profit = 0.0

        for line in lines:
            included = included_by_product[line.product_name]
            if included:
                unit_price = strategy_unit_price(
                    strategy,
                    prices_by_product[line.product_name],
                )
                quantity_fulfilled = line.quantity_requested
                line_revenue = unit_price * quantity_fulfilled
                line_acquisition_cost = line.unit_cost * line.quantity_to_order
            else:
                unit_price = 0.0
                quantity_fulfilled = 0
                line_revenue = 0.0
                line_acquisition_cost = 0.0

            items.append(
                PricedLineItem(
                    product_name=line.product_name,
                    quantity_requested=line.quantity_requested,
                    quantity_fulfilled=quantity_fulfilled,
                    unit_cost=line.unit_cost,
                    unit_price=unit_price,
                    line_revenue=line_revenue,
                    line_acquisition_cost=line_acquisition_cost,
                    included=included,
                )
            )
            total_acq += line_acquisition_cost
            total_profit += line_revenue - line_acquisition_cost

        recommendations.append(
            PricingRecommendation(
                strategy=strategy,
                items=items,
                total_acquisition_cost=total_acq,
                total_profit=total_profit,
                error=selection_error,
            )
        )

    return PricingResponse(
        success=True,
        date_of_request=date_of_request,
        need_date=need_date,
        recommendations=recommendations,
    )


class PricingTool:
    """Build three strategy recommendations from quote + inventory."""

    def __init__(
        self,
        cash_balance_fn: CashBalanceFn | None = None,
    ) -> None:
        self._cash_balance_fn = cash_balance_fn or get_cash_balance

    def price(self, request: PricingRequest) -> PricingResponse:
        """Build three strategy recommendations from quote + inventory."""
        quote = request.quote
        inventory = request.inventory

        if quote.date_of_request is None or quote.need_date is None:
            return _failure_response(
                date_of_request=quote.date_of_request or "",
                need_date=quote.need_date or "",
                message="Missing date_of_request or need_date on quote.",
            )

        try:
            lines = build_procurement_lines(quote, inventory)
            prices = unit_prices_by_product(request.unit_prices)
            validate_unit_prices(lines, prices)
        except ValueError as exc:
            return _failure_response(
                date_of_request=quote.date_of_request or "",
                need_date=quote.need_date or "",
                message=str(exc),
            )

        cash_balance = self._cash_balance_fn(quote.date_of_request)
        return build_pricing_response(
            cash_balance=cash_balance,
            date_of_request=quote.date_of_request,
            need_date=quote.need_date,
            lines=lines,
            prices_by_product=prices,
        )
