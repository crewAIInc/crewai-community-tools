"""Asin tools for CrewAI agents.

Copy this folder into your project and install `requirements.txt`. The tools read
ASIN_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://www.asin.dev/api/v1"
DEFAULT_TIMEOUT = 60.0
USER_AGENT = "crewai-asin"
ENV = "ASIN_API_KEY"

_ERROR_DETAIL_LIMIT = 300


class AsinError(Exception):
    """A Asin request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``ASIN_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get(ENV) or "").strip()
    if not key:
        raise AsinError(f"Set {ENV} or pass api_key. Get a key at https://www.asin.dev/app")
    return key


def required(value: Any, name: str) -> str:
    """Return ``value`` stripped, or raise if it is empty."""
    text = str(value or "").strip()
    if not text:
        raise AsinError(f"{name} is empty.")
    return text


def clean(values: dict[str, Any]) -> dict[str, Any]:
    """Drop unset values; send booleans as true/false."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        out[key] = ("true" if value else "false") if isinstance(value, bool) else value
    return out


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def _error(response: httpx.Response, api_key: str) -> AsinError:
    detail = ""
    try:
        body = response.json()
    except ValueError:
        detail = response.text.strip()
    else:
        if isinstance(body, dict):
            error = body.get("error") or body.get("message") or ""
            if isinstance(error, dict):
                error = error.get("message") or ""
            detail = str(error)
    detail = _redact(detail, api_key)[:_ERROR_DETAIL_LIMIT]
    status = response.status_code
    # Asin answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "Asin rate limit exceeded"
    elif status == 401:
        label = "Asin rejected the API key"
    elif status == 402:
        label = "Asin account is out of credits"
    else:
        label = "Asin API error"
    return AsinError(f"{label} (HTTP {status})" + (f": {detail}" if detail else ""), status)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _decode(response: httpx.Response, api_key: str) -> Any:
    if not response.is_success:
        raise _error(response, api_key)
    try:
        return response.json()
    except ValueError as exc:
        raise AsinError("Asin returned a response that is not JSON.") from exc


def request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Call ``path`` and return the decoded JSON body."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method, f"{base_url.rstrip('/')}{path}", params=params, json=json, headers=_headers(api_key)
            )
    except httpx.TimeoutException as exc:
        raise AsinError(f"Asin did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise AsinError(f"Could not reach Asin: {_redact(str(exc), api_key)}") from exc
    return _decode(response, api_key)


async def arequest(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Async version of :func:`request`."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method, f"{base_url.rstrip('/')}{path}", params=params, json=json, headers=_headers(api_key)
            )
    except httpx.TimeoutException as exc:
        raise AsinError(f"Asin did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise AsinError(f"Could not reach Asin: {_redact(str(exc), api_key)}") from exc
    return _decode(response, api_key)


class _AsinTool(BaseTool):
    """Shared settings and request handling for the Asin tools. Errors come back as text."""

    api_key: str | None = Field(default=None, description="Asin API key. Defaults to ASIN_API_KEY.")
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [EnvVar(name="ASIN_API_KEY", description="API key for Asin", required=True)]
    )

    def _output(self, data: Any) -> Any:
        return data

    def _send(self, method: str, path: str, params: dict[str, Any] | None, body: dict[str, Any] | None) -> Any:
        api_key = resolve_api_key(self.api_key)
        return request(
            method,
            path,
            params=clean(params) if params else None,
            json=body,
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @staticmethod
    def _text(data: Any) -> str:
        return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)


class AsinAmazonSearchToolSchema(BaseModel):
    """Input for AsinAmazonSearchTool."""

    query: str = Field(default=..., description="What to search for on Amazon.")


class AsinAmazonSearchTool(_AsinTool):
    """Search an Amazon store through Asin."""

    name: str = "Search Amazon with Asin"
    description: str = "Search an Amazon store with Asin and get up to 60 products with their ASIN, title, price, rating and review count. Input is the search keywords."
    args_schema: type[BaseModel] = AsinAmazonSearchToolSchema
    marketplace: str = "com"
    """Amazon store: com, co.uk, de, fr, it, es, ca, co.jp, com.br, com.mx or in."""

    def _request(self, query: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return "GET", "/search", {"query": required(query, "query"), "marketplace": self.marketplace}, None

    def _run(self, query: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(query))))
        except AsinError as exc:
            return str(exc)


class AsinPriceHistoryToolSchema(BaseModel):
    """Input for AsinPriceHistoryTool."""

    asin: str = Field(default=..., description="The 10-character ASIN, such as B09DFCB66S.")


class AsinPriceHistoryTool(_AsinTool):
    """Read the price history of an Amazon product through Asin."""

    name: str = "Get an Amazon price history with Asin"
    description: str = "Get the price history of an Amazon product with Asin: every price read over the period, with the lowest, highest and average. Use it to tell whether the current price is a good one. Input is the ASIN."
    args_schema: type[BaseModel] = AsinPriceHistoryToolSchema
    marketplace: str = "com"
    """Amazon store, such as com, co.uk, de or fr."""
    days: int = Field(default=90, ge=1)
    """History period in days."""

    def _request(self, asin: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "GET",
            "/history",
            {"asin": required(asin, "asin"), "marketplace": self.marketplace, "days": self.days},
            None,
        )

    def _run(self, asin: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(asin))))
        except AsinError as exc:
            return str(exc)


class AsinOffersToolSchema(BaseModel):
    """Input for AsinOffersTool."""

    asin: str = Field(default=..., description="The 10-character ASIN, such as B09DFCB66S.")


class AsinOffersTool(_AsinTool):
    """Read the sellers' offers for an Amazon product through Asin."""

    name: str = "Get Amazon offers with Asin"
    description: str = "Get the offers for an Amazon product with Asin: Amazon's own price, the lowest new and used prices, and each seller's price, cheapest first. Input is the ASIN."
    args_schema: type[BaseModel] = AsinOffersToolSchema
    marketplace: str = "com"
    """Amazon store, such as com, co.uk, de or fr."""
    live: bool = False
    """Read the offers live from Amazon (1 credit; a read from the last 3 hours is returned for free). Off: the offers Asin last read, free."""

    def _request(self, asin: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        body = {"asin": required(asin, "asin"), "marketplace": self.marketplace}
        return ("POST", "/offers", None, body) if self.live else ("GET", "/offers", body, None)

    def _run(self, asin: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(asin))))
        except AsinError as exc:
            return str(exc)
