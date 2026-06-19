import re
from typing import Any, Optional

from dotenv import load_dotenv
from project_starter import paper_supplies
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

load_dotenv()

MODEL = "openai:gpt-5.4-mini"

_DATE_SUFFIX_PATTERN = re.compile(r"\(Date of request:\s*(\d{4}-\d{2}-\d{2})\)\s*$")
_CATALOG_BY_LOWER = {item["item_name"].lower(): item for item in paper_supplies}


class QuoteItem(BaseModel):
    product_name: str
    quantity_requested: int
    unit_price: float


class QuoteResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    date_of_request: Optional[str] = None
    need_date: Optional[str] = None
    items: list[QuoteItem] = []
    excluded_products: list[str] = Field(default_factory=list)

    @field_validator("date_of_request", "need_date", mode="before")
    @classmethod
    def validate_date_format(cls, value: Any) -> Any:
        if value is None:
            return value
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
            raise ValueError(f"Date must be YYYY-MM-DD, got: {value!r}")
        return value


QUOTING_DIRECTIVE = """\
You are the Quoting Agent for Munder Difflin quote requests.
You parse the request, resolve catalog products via tools, assemble the final
QuoteResponse, and run a validation tool before returning. Never substitute
unrecognized products (e.g. do not map "balloons" to "Party streamers").

Follow these steps exactly, in order:

1. Call tool_extract_date_of_request with the full customer input text.

2. If found=False:
   - Return quote_response from the tool result as your final QuoteResponse.
   - Stop. Do not call any other tools.

3. Parse cleaned_quote from step 1:
   - need_date: delivery / "need by" date as YYYY-MM-DD, or null if absent.
   - items: every product as {raw_name, quantity}; quantity null for vague
     wording ("some", "a few") or no number. Never guess quantities.

4. Resolve catalog matches yourself using tool_validate_product_and_get_price:
   - Call with no arguments to get the full catalog.
   - For each item with an explicit quantity only, find the best catalog match.
   - Confirm each match by calling tool_validate_product_and_get_price(name).
     Retry with no more than 3 unique name attempts per item if found=False.

     -  Use deductive reasoning to infer the product name from the request. Examples:
     A4 glossy paper / glossy A4 -> Glossy paper
     A4 matte paper / matte A3 -> Matte paper
     colorful cardstock / heavy cardstock -> cardstock
     printer paper / printing paper / copy paper -> Printer paper
     poster boards (24x36) -> Poster board
     washi tape -> Decorative adhesive tape (washi tape)

   - Track recognized items (exact matched_name + unit_price from tool).
   - Put unrecognized raw_name values in excluded_products (do not add to items).
   - Never add items the tool did not confirm.

5. Assemble a draft QuoteResponse using these rules:
   - date_of_request from step 1.
   - need_date from step 3.
   - excluded_products: list of raw_name strings that could not be resolved.
   - If any item has quantity null: success=false, non-null error naming those
     products, items=[].
   - Else items[] = recognized catalog matches only (product_name, quantity,
     unit_price from tool results).
   - success=true when need_date is set, items is non-empty, and no missing
     quantities — even if excluded_products is non-empty (partial quote).
   - success=false only when need_date is missing OR zero items were recognized;
     then non-null error describing the problem.

6. Call tool_validate_quote_response with quote_json set to JSON of your draft
   QuoteResponse from step 5.

7. Return the tool result from step 6 as your final QuoteResponse output.

Rules:
- Use tool_validate_product_and_get_price for every catalog name and price.
- Do not skip step 6.
- product_name in items must be exact names confirmed by the validation tool.
"""


def tool_extract_date_of_request(text: str) -> dict:
    """
    Extract the 'Date of request' appended to the end of a quote string.

    Looks for the pattern '(Date of request: YYYY-MM-DD)' at the end of text.

    Args:
        text: The raw quote string ending with '(Date of request: YYYY-MM-DD)'.

    Returns:
        Dict with keys:
          - 'date_of_request': str in YYYY-MM-DD format, or None if not found
          - 'cleaned_quote': str with the date suffix removed
          - 'found': bool indicating whether the date pattern was present
          - 'quote_response': QuoteResponse fields as a dict when found=False;
            null when found=True
    """
    match = _DATE_SUFFIX_PATTERN.search(text)
    if not match:
        return {
            "date_of_request": None,
            "cleaned_quote": text,
            "found": False,
            "quote_response": QuoteResponse(
                success=False,
                error="Missing date of request",
                date_of_request=None,
                need_date=None,
                items=[],
            ).model_dump(),
        }

    return {
        "date_of_request": match.group(1),
        "cleaned_quote": text[: match.start()].rstrip(),
        "found": True,
        "quote_response": None,
    }


def tool_validate_product_and_get_price(
    product_name: str | None = None,
) -> dict | list:
    """
    Return the full product catalog, or validate a single product name.

    When called with no argument (or product_name=None), returns every entry
    in the paper_supplies catalog so the agent can reason about available products.

    When called with a product_name, performs an exact lookup against the catalog
    and returns the canonical name and unit price.

    Args:
        product_name: Optional. The exact product name to validate.
                      If None, the full catalog is returned instead.

    Returns:
        - If product_name is None:
            list of dicts, each with 'product_name' (str) and 'unit_price' (float),
            representing every item in paper_supplies.
        - If product_name is provided:
            dict with keys:
              - 'matched_name': str — exact item_name from paper_supplies, or None
              - 'unit_price': float — price per unit from paper_supplies, or None
              - 'found': bool — True if an exact match was found
    """
    if product_name is None:
        return [
            {
                "product_name": item["item_name"],
                "unit_price": item["unit_price"],
            }
            for item in paper_supplies
        ]

    item = _CATALOG_BY_LOWER.get(product_name.lower())
    if item is None:
        return {"matched_name": None, "unit_price": None, "found": False}

    return {
        "matched_name": item["item_name"],
        "unit_price": item["unit_price"],
        "found": True,
    }


def validate_quote_response(response: QuoteResponse) -> QuoteResponse:
    """Thin safety net: enforce catalog grounding and success/error invariants."""
    valid_items: list[QuoteItem] = []
    seen_names: set[str] = set()

    for item in response.items:
        catalog_item = _CATALOG_BY_LOWER.get(item.product_name.lower())
        if catalog_item is None:
            continue
        if item.product_name in seen_names:
            continue
        expected_price = catalog_item["unit_price"]
        if item.unit_price != expected_price:
            item = QuoteItem(
                product_name=catalog_item["item_name"],
                quantity_requested=item.quantity_requested,
                unit_price=expected_price,
            )
        else:
            item = QuoteItem(
                product_name=catalog_item["item_name"],
                quantity_requested=item.quantity_requested,
                unit_price=item.unit_price,
            )
        seen_names.add(item.product_name)
        valid_items.append(item)

    success = response.success
    error = response.error
    need_date = response.need_date
    excluded_products = list(response.excluded_products)

    # Partial quote: any recognized lines with both dates should succeed.
    if (
        not success
        and valid_items
        and need_date is not None
        and response.date_of_request is not None
    ):
        success = True
        error = None

    if success:
        if error is not None or need_date is None or not valid_items:
            success = False
            if error is None:
                if need_date is None:
                    error = "Missing delivery date"
                elif not valid_items:
                    error = "No valid items in quote"
                else:
                    error = "Quote validation failed"
    else:
        if error is None:
            error = "Quote request could not be fulfilled"

    if not success and error is None:
        error = "Quote request could not be fulfilled"

    if success and error is not None:
        error = None

    return QuoteResponse(
        success=success,
        error=error,
        date_of_request=response.date_of_request,
        need_date=need_date,
        items=valid_items,
        excluded_products=excluded_products,
    )


def tool_validate_quote_response(quote_json: str) -> dict:
    """
    Apply the thin safety-net validation to a QuoteResponse.

    Args:
        quote_json: JSON string of QuoteResponse fields from agent assembly.

    Returns:
        Validated QuoteResponse fields as a dict.
    """
    response = QuoteResponse.model_validate_json(quote_json)
    validated = validate_quote_response(response)
    return validated.model_dump()


quoting_agent = Agent(
    MODEL,
    system_prompt=QUOTING_DIRECTIVE,
    output_type=QuoteResponse,
    tools=[
        tool_extract_date_of_request,
        tool_validate_product_and_get_price,
        tool_validate_quote_response,
    ],
)


def call_quoting_agent(request_with_date: str) -> QuoteResponse:
    """Parse a customer quote request and return a validated QuoteResponse."""
    return quoting_agent.run_sync(request_with_date).output
