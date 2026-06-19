"""Inventory tool test scenarios from specification/tools/inventory_tool.md."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from agents.quoting_agent import QuoteItem, QuoteResponse
from tools.inventory_tool import (
    InventoryRequest,
    InventoryTool,
    RequestedItem,
    quote_to_inventory_request,
)


def make_stock_fn(stock_by_name: dict[str, int]):
    def _stock_fn(item_name: str, as_of_date: str) -> pd.DataFrame:
        qty = stock_by_name.get(item_name, 0)
        return pd.DataFrame({"item_name": [item_name], "current_stock": [qty]})

    return _stock_fn


def _single_item_request(
    *,
    product_name: str,
    quantity_requested: int,
    unit_price: float,
    date_of_request: str,
    need_date: str,
) -> InventoryRequest:
    return InventoryRequest(
        need_date=need_date,
        date_of_request=date_of_request,
        items=[
            RequestedItem(
                product_name=product_name,
                quantity_requested=quantity_requested,
                unit_price=unit_price,
            )
        ],
    )


@dataclass
class ScenarioExpectation:
    success: bool
    quantity_in_stock: int
    quantity_to_order: int


def _assert_checked_item(result, expected: ScenarioExpectation) -> None:
    assert len(result.items) == 1
    item = result.items[0]
    assert item.success is expected.success
    assert item.quantity_in_stock == expected.quantity_in_stock
    assert item.quantity_to_order == expected.quantity_to_order
    if expected.success:
        assert item.error == ""
    else:
        assert item.error != ""


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


def test_hp1_stock_covers() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"A4 paper": 500}))
    request = _single_item_request(
        product_name="A4 paper",
        quantity_requested=100,
        unit_price=0.05,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(True, 500, 0))


def test_hp2_stock_exactly_equal() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Cardstock": 200}))
    request = _single_item_request(
        product_name="Cardstock",
        quantity_requested=200,
        unit_price=0.15,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(True, 200, 0))


def test_hp3_supplier_order_on_time() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Colored paper": 100}))
    request = _single_item_request(
        product_name="Colored paper",
        quantity_requested=300,
        unit_price=0.10,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(True, 100, 200))


def test_hp4_small_shortfall_same_day_delivery() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Sticky notes": 50}))
    request = _single_item_request(
        product_name="Sticky notes",
        quantity_requested=55,
        unit_price=0.03,
        date_of_request="2025-04-01",
        need_date="2025-04-03",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(True, 50, 5))


def test_hp5_large_order_on_time() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Banner paper": 1000}))
    request = _single_item_request(
        product_name="Banner paper",
        quantity_requested=3000,
        unit_price=0.30,
        date_of_request="2025-04-01",
        need_date="2025-04-20",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(True, 1000, 2000))


def test_sp1_large_order_too_late() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Banner paper": 0}))
    request = _single_item_request(
        product_name="Banner paper",
        quantity_requested=3000,
        unit_price=0.30,
        date_of_request="2025-04-01",
        need_date="2025-04-05",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(False, 0, 3000))


def test_sp2_delivery_on_need_date_not_early_enough() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Colored paper": 100}))
    request = _single_item_request(
        product_name="Colored paper",
        quantity_requested=300,
        unit_price=0.10,
        date_of_request="2025-04-01",
        need_date="2025-04-05",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(False, 100, 200))


def test_sp3_same_day_need_date_misses_deadline() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Sticky notes": 10}))
    request = _single_item_request(
        product_name="Sticky notes",
        quantity_requested=20,
        unit_price=0.03,
        date_of_request="2025-04-01",
        need_date="2025-04-01",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(False, 10, 10))


def test_sp4_photo_paper_too_late() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Photo paper": 500}))
    request = _single_item_request(
        product_name="Photo paper",
        quantity_requested=5000,
        unit_price=0.25,
        date_of_request="2025-06-01",
        need_date="2025-06-05",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(False, 500, 4500))


def test_sp5_poster_paper_too_late() -> None:
    tool = InventoryTool(stock_fn=make_stock_fn({"Poster paper": 100}))
    request = _single_item_request(
        product_name="Poster paper",
        quantity_requested=900,
        unit_price=0.25,
        date_of_request="2025-07-01",
        need_date="2025-07-03",
    )
    result = tool.check(request)
    _assert_checked_item(result, ScenarioExpectation(False, 100, 800))


def test_mixed_request_from_spec_example() -> None:
    tool = InventoryTool(
        stock_fn=make_stock_fn(
            {
                "A4 paper": 500,
                "Cardstock": 100,
                "Banner paper": 0,
            }
        )
    )
    request = InventoryRequest(
        need_date="2025-04-06",
        date_of_request="2025-04-01",
        items=[
            RequestedItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            ),
            RequestedItem(
                product_name="Cardstock",
                quantity_requested=300,
                unit_price=0.15,
            ),
            RequestedItem(
                product_name="Banner paper",
                quantity_requested=3000,
                unit_price=0.30,
            ),
        ],
    )
    result = tool.check(request)
    assert len(result.items) == 3
    assert result.items[0].success is True
    assert result.items[0].quantity_to_order == 0
    assert result.items[1].success is True
    assert result.items[1].quantity_to_order == 200
    assert result.items[2].success is False
    assert result.items[2].quantity_to_order == 3000
    assert result.items[2].error != ""


def test_quote_to_inventory_request_maps_fields() -> None:
    quote = QuoteResponse(
        success=True,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[
            QuoteItem(
                product_name="A4 paper",
                quantity_requested=100,
                unit_price=0.05,
            ),
        ],
    )
    request = quote_to_inventory_request(quote)
    assert request.date_of_request == "2025-04-01"
    assert request.need_date == "2025-04-10"
    assert len(request.items) == 1
    assert request.items[0].product_name == "A4 paper"


def test_quote_to_inventory_request_rejects_missing_dates() -> None:
    quote = QuoteResponse(
        success=False,
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
    try:
        quote_to_inventory_request(quote)
    except ValueError as exc:
        assert "need_date" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing need_date")


def test_quote_to_inventory_request_rejects_empty_items() -> None:
    quote = QuoteResponse(
        success=False,
        date_of_request="2025-04-01",
        need_date="2025-04-10",
        items=[],
    )
    try:
        quote_to_inventory_request(quote)
    except ValueError as exc:
        assert "items" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty items")


def run_all_tests() -> int:
    scenarios = [
        ("HP-1", test_hp1_stock_covers),
        ("HP-2", test_hp2_stock_exactly_equal),
        ("HP-3", test_hp3_supplier_order_on_time),
        ("HP-4", test_hp4_small_shortfall_same_day_delivery),
        ("HP-5", test_hp5_large_order_on_time),
        ("SP-1", test_sp1_large_order_too_late),
        ("SP-2", test_sp2_delivery_on_need_date_not_early_enough),
        ("SP-3", test_sp3_same_day_need_date_misses_deadline),
        ("SP-4", test_sp4_photo_paper_too_late),
        ("SP-5", test_sp5_poster_paper_too_late),
        ("MIXED", test_mixed_request_from_spec_example),
        ("MAP", test_quote_to_inventory_request_maps_fields),
        ("GATE-ND", test_quote_to_inventory_request_rejects_missing_dates),
        ("GATE-ITEMS", test_quote_to_inventory_request_rejects_empty_items),
    ]

    passed = 0
    for scenario_id, test_fn in scenarios:
        if _run_scenario(scenario_id, test_fn):
            passed += 1

    total = len(scenarios)
    print(f"\nResults: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
