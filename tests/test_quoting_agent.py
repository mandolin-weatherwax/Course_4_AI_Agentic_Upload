"""Quoting agent test scenarios from specification/agents/quoting_agent.md."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from project_starter import paper_supplies

from agents.quoting_agent import QuoteResponse, call_quoting_agent

PAPER_SUPPLY_NAMES = {item["item_name"] for item in paper_supplies}


@dataclass
class ExpectedItem:
    product_name: str
    quantity_requested: int
    unit_price: float


def _items_by_name(response: QuoteResponse) -> dict[str, ExpectedItem]:
    return {
        item.product_name: ExpectedItem(
            product_name=item.product_name,
            quantity_requested=item.quantity_requested,
            unit_price=item.unit_price,
        )
        for item in response.items
    }


def _assert_items(
    response: QuoteResponse,
    expected_items: list[ExpectedItem],
) -> None:
    assert len(response.items) == len(expected_items)
    actual = _items_by_name(response)
    for expected in expected_items:
        assert expected.product_name in PAPER_SUPPLY_NAMES
        assert expected.product_name in actual, f"Missing item: {expected.product_name}"
        item = actual[expected.product_name]
        assert item.quantity_requested == expected.quantity_requested
        assert item.unit_price == expected.unit_price


def _assert_error_mentions(response: QuoteResponse, *keywords: str) -> None:
    assert response.error is not None
    lowered = response.error.lower()
    assert any(keyword.lower() in lowered for keyword in keywords)


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


def test_h1_standard_two_item_order() -> None:
    request_with_date = (
        "I need 200 sheets of A4 paper and 100 sheets of cardstock. "
        "Delivery needed by April 15, 2025. (Date of request: 2025-04-01)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is True
    assert response.error is None
    assert response.date_of_request == "2025-04-01"
    assert response.need_date == "2025-04-15"
    _assert_items(
        response,
        [
            ExpectedItem("A4 paper", 200, 0.05),
            ExpectedItem("Cardstock", 100, 0.15),
        ],
    )


def test_h2_multi_item_order() -> None:
    request_with_date = (
        "Please quote 150 sticky notes, 50 envelopes, and 80 paper napkins. "
        "We need delivery by May 1, 2025. (Date of request: 2025-04-03)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is True
    assert response.error is None
    assert response.date_of_request == "2025-04-03"
    assert response.need_date == "2025-05-01"
    _assert_items(
        response,
        [
            ExpectedItem("Sticky notes", 150, 0.03),
            ExpectedItem("Envelopes", 50, 0.05),
            ExpectedItem("Paper napkins", 80, 0.02),
        ],
    )


def test_h3_paraphrased_product_names() -> None:
    request_with_date = (
        "I would like 300 sheets of glossy paper and 200 kraft papers. "
        "Needed by April 20, 2025. (Date of request: 2025-04-05)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is True
    assert response.error is None
    assert response.date_of_request == "2025-04-05"
    assert response.need_date == "2025-04-20"
    _assert_items(
        response,
        [
            ExpectedItem("Glossy paper", 300, 0.20),
            ExpectedItem("Kraft paper", 200, 0.10),
        ],
    )


def test_e1_missing_date_of_request() -> None:
    request_with_date = (
        "I need 200 sheets of A4 paper and 100 sheets of cardstock. "
        "Delivery by April 15."
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is False
    _assert_error_mentions(response, "date of request", "missing")
    assert response.date_of_request is None
    assert response.need_date is None
    assert response.items == []


def test_e2_missing_need_date() -> None:
    request_with_date = (
        "I need 200 sheets of A4 paper and 100 sheets of cardstock. "
        "(Date of request: 2025-04-01)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is False
    assert response.error is not None


def test_e3_missing_quantities() -> None:
    request_with_date = (
        "I need some A4 paper and cardstock for our event. "
        "Delivery by April 15, 2025. (Date of request: 2025-04-01)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is False
    assert response.error is not None


def test_e4_misspelled_product_name() -> None:
    request_with_date = (
        "Please quote 100 sheets of Glosyy paper and 50 envelopes. "
        "Needed by April 20, 2025. (Date of request: 2025-04-05)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is True
    assert response.error is None
    assert response.date_of_request == "2025-04-05"
    assert response.need_date == "2025-04-20"
    _assert_items(
        response,
        [
            ExpectedItem("Glossy paper", 100, 0.20),
            ExpectedItem("Envelopes", 50, 0.05),
        ],
    )


def test_e5_unrecognized_product_mixed_with_valid() -> None:
    request_with_date = (
        "I need 200 sheets of A4 paper, 100 balloons, and 50 envelopes. "
        "Needed by April 20, 2025. (Date of request: 2025-04-05)"
    )
    response = call_quoting_agent(request_with_date)

    assert response.success is True
    assert response.error is None
    assert response.date_of_request == "2025-04-05"
    assert response.need_date == "2025-04-20"
    assert "balloons" in [name.lower() for name in response.excluded_products] or any(
        "balloon" in name.lower() for name in response.excluded_products
    ), f"Expected balloons in excluded_products, got {response.excluded_products}"
    _assert_items(
        response,
        [
            ExpectedItem("A4 paper", 200, 0.05),
            ExpectedItem("Envelopes", 50, 0.05),
        ],
    )


def run_all_tests() -> int:
    scenarios = [
        ("H1", test_h1_standard_two_item_order),
        ("H2", test_h2_multi_item_order),
        ("H3", test_h3_paraphrased_product_names),
        ("E1", test_e1_missing_date_of_request),
        ("E2", test_e2_missing_need_date),
        ("E3", test_e3_missing_quantities),
        ("E4", test_e4_misspelled_product_name),
        ("E5", test_e5_unrecognized_product_mixed_with_valid),
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
