"""Pricing tool test scenarios from specification/tools/pricing_tool.md."""

from __future__ import annotations

import sys
from typing import Callable

from agents.quoting_agent import QuoteItem, QuoteResponse
from tools.inventory_tool import CheckedItem, InventoryResult
from tools.pricing_tool import (
    MAX_SUBSET_LINES,
    ItemUnitPrices,
    PricingRequest,
    PricingTool,
    ProcurementLine,
    build_pricing_response,
    can_price,
    default_unit_prices,
    select_items_for_cash,
    unit_prices_by_product,
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


def _inventory(
    *,
    checked_items: list[CheckedItem],
    date_of_request: str = "2025-04-01",
    need_date: str = "2025-04-10",
) -> InventoryResult:
    return InventoryResult(
        need_date=need_date,
        date_of_request=date_of_request,
        items=checked_items,
    )


def _checked(
    *,
    product_name: str,
    quantity_requested: int,
    unit_price: float,
    quantity_to_order: int,
    quantity_in_stock: int = 0,
    success: bool = True,
) -> CheckedItem:
    return CheckedItem(
        product_name=product_name,
        quantity_requested=quantity_requested,
        unit_price=unit_price,
        success=success,
        quantity_in_stock=quantity_in_stock,
        quantity_to_order=quantity_to_order,
        error="" if success else "inventory failure",
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


def _procurement_line(
    *,
    product_name: str,
    quantity_requested: int,
    quantity_to_order: int,
    unit_cost: float,
) -> ProcurementLine:
    return ProcurementLine(
        product_name=product_name,
        quantity_requested=quantity_requested,
        quantity_to_order=quantity_to_order,
        unit_cost=unit_cost,
    )


def _default_prices_for_lines(
    lines: list[ProcurementLine],
) -> dict[str, ItemUnitPrices]:
    return unit_prices_by_product(
        [default_unit_prices(line.product_name, line.unit_cost) for line in lines]
    )


def test_p_h1_full_cash_multi_item() -> None:
    tool = PricingTool(cash_balance_fn=lambda _date: 10_000.0)
    quote = _quote(
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            ),
            QuoteItem(
                product_name="Cardstock",
                quantity_requested=200,
                unit_price=0.15,
            ),
        ]
    )
    inventory = _inventory(
        checked_items=[
            _checked(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
                quantity_to_order=100,
            ),
            _checked(
                product_name="Cardstock",
                quantity_requested=200,
                unit_price=0.15,
                quantity_to_order=200,
            ),
        ]
    )
    unit_prices = [
        default_unit_prices("A4 paper", 0.05),
        default_unit_prices("Cardstock", 0.15),
    ]
    response = tool.price(
        PricingRequest(quote=quote, inventory=inventory, unit_prices=unit_prices)
    )

    assert response.success is True
    assert len(response.recommendations) == 3
    for recommendation in response.recommendations:
        assert recommendation.error is None
        assert len(recommendation.items) == 2
        assert all(item.included for item in recommendation.items)


def test_p_h2_strategy_price_ordering() -> None:
    lines = [
        _procurement_line(
            product_name="A4 paper",
            quantity_requested=100,
            quantity_to_order=100,
            unit_cost=0.10,
        )
    ]
    response = build_pricing_response(
        cash_balance=1_000.0,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        lines=lines,
        prices_by_product=_default_prices_for_lines(lines),
    )

    prices = {rec.strategy: rec.items[0].unit_price for rec in response.recommendations}
    assert prices["maximize_profit"] == 0.12
    assert prices["average_pricing"] == 0.11
    assert prices["maximize_turnover"] == 0.09
    assert (
        prices["maximize_profit"]
        > prices["average_pricing"]
        > prices["maximize_turnover"]
    )


def test_p_h3_single_item_total_profit() -> None:
    lines = [
        _procurement_line(
            product_name="A4 paper",
            quantity_requested=100,
            quantity_to_order=50,
            unit_cost=0.10,
        )
    ]
    response = build_pricing_response(
        cash_balance=500.0,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        lines=lines,
        prices_by_product=_default_prices_for_lines(lines),
    )

    profit_rec = response.recommendations[0]
    assert profit_rec.strategy == "maximize_profit"
    item = profit_rec.items[0]
    assert item.unit_price == 0.12
    assert item.line_revenue == 12.0
    assert item.line_acquisition_cost == 5.0
    assert profit_rec.total_profit == 7.0


def test_p_e1_mismatched_item_count() -> None:
    tool = PricingTool(cash_balance_fn=lambda _date: 1_000.0)
    quote = _quote(
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            )
        ]
    )
    inventory = _inventory(checked_items=[])
    response = tool.price(
        PricingRequest(
            quote=quote,
            inventory=inventory,
            unit_prices=[default_unit_prices("A4 paper", 0.05)],
        )
    )

    assert response.success is False
    assert len(response.recommendations) == 3
    for recommendation in response.recommendations:
        assert recommendation.error is not None
        assert recommendation.items == []


def test_p_e2_can_price_gate_blocks_failed_inventory() -> None:
    quote = _quote(
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            )
        ]
    )
    inventory = _inventory(
        checked_items=[
            _checked(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
                quantity_to_order=100,
                success=False,
            )
        ]
    )
    assert can_price(quote, inventory) is False


def test_p_e3_too_many_lines_insufficient_cash() -> None:
    lines = [
        _procurement_line(
            product_name=f"Item {index}",
            quantity_requested=10,
            quantity_to_order=10,
            unit_cost=1.0,
        )
        for index in range(MAX_SUBSET_LINES + 1)
    ]
    response = build_pricing_response(
        cash_balance=5.0,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        lines=lines,
        prices_by_product=_default_prices_for_lines(lines),
    )

    for recommendation in response.recommendations:
        assert recommendation.error is not None
        assert all(not item.included for item in recommendation.items)


def test_p_c1_cash_covers_one_of_three() -> None:
    lines = [
        _procurement_line(
            product_name="A4 paper",
            quantity_requested=500,
            quantity_to_order=500,
            unit_cost=0.05,
        ),
        _procurement_line(
            product_name="Cardstock",
            quantity_requested=300,
            quantity_to_order=200,
            unit_cost=0.15,
        ),
        _procurement_line(
            product_name="Banner paper",
            quantity_requested=1000,
            quantity_to_order=1000,
            unit_cost=0.30,
        ),
    ]
    prices = _default_prices_for_lines(lines)
    selection = select_items_for_cash(50.0, lines, prices)
    assert selection.included_by_product["Cardstock"] is True
    assert selection.included_by_product["A4 paper"] is False
    assert selection.included_by_product["Banner paper"] is False
    assert selection.error is not None

    response = build_pricing_response(
        cash_balance=50.0,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        lines=lines,
        prices_by_product=prices,
    )
    for recommendation in response.recommendations:
        assert recommendation.error is not None
        included = {item.product_name: item.included for item in recommendation.items}
        assert included == selection.included_by_product


def test_p_c2_pair_beats_single() -> None:
    lines = [
        _procurement_line(
            product_name="Colored paper",
            quantity_requested=200,
            quantity_to_order=200,
            unit_cost=0.10,
        ),
        _procurement_line(
            product_name="Glossy paper",
            quantity_requested=100,
            quantity_to_order=100,
            unit_cost=0.20,
        ),
        _procurement_line(
            product_name="Poster paper",
            quantity_requested=500,
            quantity_to_order=500,
            unit_cost=0.25,
        ),
    ]
    prices = _default_prices_for_lines(lines)
    selection = select_items_for_cash(45.0, lines, prices)
    assert selection.included_by_product["Colored paper"] is True
    assert selection.included_by_product["Glossy paper"] is True
    assert selection.included_by_product["Poster paper"] is False


def test_p_c3_same_subset_different_prices_per_strategy() -> None:
    lines = [
        _procurement_line(
            product_name="Colored paper",
            quantity_requested=200,
            quantity_to_order=200,
            unit_cost=0.10,
        ),
        _procurement_line(
            product_name="Glossy paper",
            quantity_requested=100,
            quantity_to_order=100,
            unit_cost=0.20,
        ),
        _procurement_line(
            product_name="Poster paper",
            quantity_requested=500,
            quantity_to_order=500,
            unit_cost=0.25,
        ),
    ]
    response = build_pricing_response(
        cash_balance=45.0,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        lines=lines,
        prices_by_product=_default_prices_for_lines(lines),
    )

    included_sets = [
        {item.product_name: item.included for item in rec.items}
        for rec in response.recommendations
    ]
    assert included_sets[0] == included_sets[1] == included_sets[2]

    colored_prices = [
        next(
            item.unit_price
            for item in rec.items
            if item.product_name == "Colored paper"
        )
        for rec in response.recommendations
    ]
    assert colored_prices[0] > colored_prices[1] > colored_prices[2]


def test_min_price_floor_enforced() -> None:
    """min_unit_price below 0.85 × unit_cost is rejected."""
    lines = [
        _procurement_line(
            product_name="A4 paper",
            quantity_requested=100,
            quantity_to_order=100,
            unit_cost=0.10,
        )
    ]
    bad_prices = unit_prices_by_product(
        [
            ItemUnitPrices(
                product_name="A4 paper",
                min_unit_price=0.08,
                avg_unit_price=0.11,
                max_unit_price=0.12,
            )
        ]
    )
    try:
        build_pricing_response(
            cash_balance=1_000.0,
            date_of_request="2025-04-01",
            need_date="2025-04-10",
            lines=lines,
            prices_by_product=bad_prices,
        )
        raise AssertionError("expected ValueError for min price below floor")
    except ValueError as exc:
        assert "below floor" in str(exc)


def main() -> int:
    scenarios = [
        ("P-H1", test_p_h1_full_cash_multi_item),
        ("P-H2", test_p_h2_strategy_price_ordering),
        ("P-H3", test_p_h3_single_item_total_profit),
        ("P-E1", test_p_e1_mismatched_item_count),
        ("P-E2", test_p_e2_can_price_gate_blocks_failed_inventory),
        ("P-E3", test_p_e3_too_many_lines_insufficient_cash),
        ("P-C1", test_p_c1_cash_covers_one_of_three),
        ("P-C2", test_p_c2_pair_beats_single),
        ("P-C3", test_p_c3_same_subset_different_prices_per_strategy),
        ("P-F1", test_min_price_floor_enforced),
    ]

    passed = sum(_run_scenario(scenario_id, fn) for scenario_id, fn in scenarios)
    total = len(scenarios)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
