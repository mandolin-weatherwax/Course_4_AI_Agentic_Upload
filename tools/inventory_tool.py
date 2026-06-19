"""Deterministic inventory check — stock and supplier lead time."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import pandas as pd
from project_starter import get_stock_level, get_supplier_delivery_date
from pydantic import BaseModel, Field

from agents.quoting_agent import QuoteResponse

DEFAULT_LEAD_BUFFER_DAYS = 1


class RequestedItem(BaseModel):
    """One line item from QuoteResponse.items."""

    product_name: str = Field(description="Exact catalog product name.")
    quantity_requested: int = Field(description="Units requested by the customer.")
    unit_price: float = Field(description="Per-unit price from the catalog.")


class InventoryRequest(BaseModel):
    """Subset of QuoteResponse passed to the inventory tool."""

    need_date: str = Field(description="Required delivery date, YYYY-MM-DD.")
    date_of_request: str = Field(
        description="Date the request was submitted, YYYY-MM-DD."
    )
    items: list[RequestedItem] = Field(default_factory=list)


class CheckedItem(BaseModel):
    """RequestedItem enriched with the inventory decision."""

    product_name: str
    quantity_requested: int
    unit_price: float
    success: bool = False
    quantity_in_stock: int = 0
    quantity_to_order: int = 0
    error: str = ""


class InventoryResult(BaseModel):
    """Full enriched envelope returned by InventoryTool.check()."""

    need_date: str
    date_of_request: str
    items: list[CheckedItem] = Field(default_factory=list)


def quote_to_inventory_request(quote: QuoteResponse) -> InventoryRequest:
    """Map QuoteResponse → InventoryRequest.

    Fail fast on invalid orchestrator handoff.
    """
    if quote.date_of_request is None:
        raise ValueError("date_of_request is required for inventory check")
    if quote.need_date is None:
        raise ValueError("need_date is required for inventory check")
    if not quote.items:
        raise ValueError("items must be non-empty for inventory check")

    return InventoryRequest(
        need_date=quote.need_date,
        date_of_request=quote.date_of_request,
        items=[
            RequestedItem(
                product_name=item.product_name,
                quantity_requested=item.quantity_requested,
                unit_price=item.unit_price,
            )
            for item in quote.items
        ],
    )


StockFn = Callable[[str, str], pd.DataFrame]
DeliveryDateFn = Callable[[str, int], str]


class InventoryTool:
    """Check whether quoted line items can be fulfilled by need_date."""

    def __init__(
        self,
        stock_fn: StockFn | None = None,
        delivery_date_fn: DeliveryDateFn | None = None,
        lead_buffer_days: int = DEFAULT_LEAD_BUFFER_DAYS,
    ) -> None:
        self._stock_fn = stock_fn or get_stock_level
        self._delivery_date_fn = delivery_date_fn or get_supplier_delivery_date
        self._lead_buffer_days = lead_buffer_days

    def check(self, request: InventoryRequest) -> InventoryResult:
        """Evaluate every item and return an InventoryResult."""
        checked_items = [
            self._check_item(
                item,
                date_of_request=request.date_of_request,
                need_date=request.need_date,
            )
            for item in request.items
        ]
        return InventoryResult(
            need_date=request.need_date,
            date_of_request=request.date_of_request,
            items=checked_items,
        )

    def _current_stock(self, product_name: str, date_of_request: str) -> int:
        df = self._stock_fn(product_name, date_of_request)
        if df.empty or "current_stock" not in df.columns:
            return 0
        return int(df["current_stock"].iloc[0])

    def _check_item(
        self,
        item: RequestedItem,
        *,
        date_of_request: str,
        need_date: str,
    ) -> CheckedItem:
        quantity_in_stock = self._current_stock(item.product_name, date_of_request)

        if item.quantity_requested <= quantity_in_stock:
            return CheckedItem(
                product_name=item.product_name,
                quantity_requested=item.quantity_requested,
                unit_price=item.unit_price,
                success=True,
                quantity_in_stock=quantity_in_stock,
                quantity_to_order=0,
                error="",
            )

        quantity_to_order = item.quantity_requested - quantity_in_stock
        estimated_delivery = self._delivery_date_fn(date_of_request, quantity_to_order)
        deadline = date.fromisoformat(need_date) - timedelta(
            days=self._lead_buffer_days
        )

        if date.fromisoformat(estimated_delivery) <= deadline:
            return CheckedItem(
                product_name=item.product_name,
                quantity_requested=item.quantity_requested,
                unit_price=item.unit_price,
                success=True,
                quantity_in_stock=quantity_in_stock,
                quantity_to_order=quantity_to_order,
                error="",
            )

        error = (
            f"Cannot fulfil '{item.product_name}': short by {quantity_to_order} units. "
            f"Supplier order placed {date_of_request} is estimated to arrive "
            f"{estimated_delivery}, which is not at least {self._lead_buffer_days} "
            f"day(s) before need_date {need_date}."
        )
        return CheckedItem(
            product_name=item.product_name,
            quantity_requested=item.quantity_requested,
            unit_price=item.unit_price,
            success=False,
            quantity_in_stock=quantity_in_stock,
            quantity_to_order=quantity_to_order,
            error=error,
        )
