"""Reference third-party CrewAI tool.

The catalog is local on purpose. A real provider would call its SDK from `_run`
and read credentials from the environment.
"""

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

CATALOG = {
    "acme-100": {
        "name": "Acme Widget",
        "price_usd": "12.00",
        "stock": "42",
        "summary": "A small sample widget used in community-tool examples.",
    },
    "acme-200": {
        "name": "Acme Gadget",
        "price_usd": "24.50",
        "stock": "7",
        "summary": "A larger sample gadget used in community-tool examples.",
    },
}


class AcmeCatalogLookupInput(BaseModel):
    query: str = Field(
        ...,
        description="Product SKU (for example ACME-100) or product name.",
    )


class AcmeCatalogLookupTool(BaseTool):
    name: str = "acme_catalog_lookup"
    description: str = (
        "Look up price, stock, and a short description for a product in the "
        "Acme sample catalog. Use this when the user asks about an Acme sample "
        "product by SKU or name."
    )
    args_schema: type[BaseModel] = AcmeCatalogLookupInput

    def _run(self, query: str) -> str:
        key = query.strip().lower().replace(" ", "-")
        product = CATALOG.get(key)
        if product is None:
            for sku, item in CATALOG.items():
                if query.strip().lower() in item["name"].lower():
                    product = item
                    key = sku
                    break
        if product is None:
            known = ", ".join(sku.upper() for sku in CATALOG)
            return f"No Acme sample product matched {query!r}. Known SKUs: {known}."

        return (
            f"{product['name']} ({key.upper()})\n"
            f"Price: ${product['price_usd']}\n"
            f"Stock: {product['stock']}\n"
            f"{product['summary']}"
        )
