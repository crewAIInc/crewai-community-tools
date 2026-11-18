"""ChartQuery tools for CrewAI agents.

Copy this folder into your project and install `requirements.txt`. The tools read
CHARTQUERY_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.chartquery.com"
DEFAULT_TIMEOUT = 120.0
USER_AGENT = "crewai-chartquery"
ENV = "CHARTQUERY_API_KEY"

_ERROR_DETAIL_LIMIT = 300


class ChartQueryError(Exception):
    """A ChartQuery request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``CHARTQUERY_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get(ENV) or "").strip()
    if not key:
        raise ChartQueryError(f"Set {ENV} or pass api_key. Get a key at https://app.chartquery.com")
    return key


def required(value: Any, name: str) -> str:
    """Return ``value`` stripped, or raise if it is empty."""
    text = str(value or "").strip()
    if not text:
        raise ChartQueryError(f"{name} is empty.")
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


def _error(response: httpx.Response, api_key: str) -> ChartQueryError:
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
    # ChartQuery answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "ChartQuery rate limit exceeded"
    elif status == 401:
        label = "ChartQuery rejected the API key"
    elif status == 402:
        label = "ChartQuery account is out of credits"
    else:
        label = "ChartQuery API error"
    return ChartQueryError(f"{label} (HTTP {status})" + (f": {detail}" if detail else ""), status)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    headers["x-api-key"] = api_key
    return headers


def _decode(response: httpx.Response, api_key: str) -> Any:
    if not response.is_success:
        raise _error(response, api_key)
    try:
        return response.json()
    except ValueError as exc:
        raise ChartQueryError("ChartQuery returned a response that is not JSON.") from exc


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
        raise ChartQueryError(f"ChartQuery did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise ChartQueryError(f"Could not reach ChartQuery: {_redact(str(exc), api_key)}") from exc
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
        raise ChartQueryError(f"ChartQuery did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise ChartQueryError(f"Could not reach ChartQuery: {_redact(str(exc), api_key)}") from exc
    return _decode(response, api_key)


class _ChartQueryTool(BaseTool):
    """Shared settings and request handling for the ChartQuery tools. Errors come back as text."""

    api_key: str | None = Field(default=None, description="ChartQuery API key. Defaults to CHARTQUERY_API_KEY.")
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [EnvVar(name="CHARTQUERY_API_KEY", description="API key for ChartQuery", required=True)]
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


class ChartQueryChartToolSchema(BaseModel):
    """Input for ChartQueryChartTool."""

    chart_config: str = Field(
        default=..., description="The Chart.js config as a JSON object with type, data and optional options."
    )


class ChartQueryChartTool(_ChartQueryTool):
    """Render a Chart.js chart through ChartQuery and return links to the image."""

    name: str = "Render a chart with ChartQuery"
    description: str = 'Render a Chart.js chart with ChartQuery and get public links to the image. Returns JSON with render_url (the image), view_url (a page to open) and expires_in. Input is the Chart.js config as JSON, such as {"type": "bar", "data": {"labels": ["Jan", "Feb"], "datasets": [{"label": "Sales", "data": [12, 19]}]}}.'
    args_schema: type[BaseModel] = ChartQueryChartToolSchema

    def _request(self, chart_config: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        try:
            config = json.loads(chart_config) if isinstance(chart_config, str) else chart_config
        except ValueError:
            raise ChartQueryError('chart_config must be valid JSON, such as {"type": "bar", "data": {...}}.') from None
        if not isinstance(config, dict):
            raise ChartQueryError("chart_config must be a JSON object.")
        return "GET", "/v1/chart", {"query": json.dumps(config, separators=(",", ":")), "share": "true"}, None

    def _run(self, chart_config: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(chart_config))))
        except ChartQueryError as exc:
            return str(exc)


class ChartQueryChartFromPromptToolSchema(BaseModel):
    """Input for ChartQueryChartFromPromptTool."""

    prompt: str = Field(default=..., description="What to chart: type, topic, colors, style.")
    data: str | None = Field(
        default=None, description="Optional data as CSV, JSON or text, such as Jan: 12400, Feb: 18900."
    )


class ChartQueryChartFromPromptTool(_ChartQueryTool):
    """Build a chart from a plain-words description through ChartQuery."""

    name: str = "Build a chart from a description with ChartQuery"
    description: str = "Describe a chart in plain words, with optional raw data as CSV, JSON or text; ChartQuery's AI builds it and returns public links to the image. Returns JSON with render_url (the image), view_url (a page to open) and expires_in."
    args_schema: type[BaseModel] = ChartQueryChartFromPromptToolSchema

    def _request(
        self, prompt: str, data: str | None = None
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "GET",
            "/v1/chart/ai",
            clean({"prompt": required(prompt, "prompt"), "raw_data": data, "share": True}),
            None,
        )

    def _run(self, prompt: str, data: str | None = None, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(prompt, data))))
        except ChartQueryError as exc:
            return str(exc)


class ChartQueryDiagramToolSchema(BaseModel):
    """Input for ChartQueryDiagramTool."""

    source: str = Field(default=..., description="The diagram source code, such as graph TD; A[Start] --> B[End].")


class ChartQueryDiagramTool(_ChartQueryTool):
    """Render a diagram from its source through ChartQuery."""

    name: str = "Render a diagram with ChartQuery"
    description: str = "Render a diagram from its source code (Mermaid by default, or D2, PlantUML, Graphviz and more) with ChartQuery and get public links to the image. Returns JSON with render_url (the image), view_url (a page to open) and expires_in. Input is the diagram source."
    args_schema: type[BaseModel] = ChartQueryDiagramToolSchema
    language: str = "mermaid"
    """Diagram language: mermaid, d2, plantuml, graphviz, bpmn, dbml and more."""
    image_format: Literal["png", "svg", "jpeg", "pdf"] = "png"
    """Image format."""

    def _request(self, source: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "GET",
            "/v1/diagram",
            {
                "query": required(source, "source"),
                "type": self.language,
                "format": self.image_format,
                "share": "true",
            },
            None,
        )

    def _run(self, source: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(source))))
        except ChartQueryError as exc:
            return str(exc)


class ChartQueryQRCodeToolSchema(BaseModel):
    """Input for ChartQueryQRCodeTool."""

    data: str = Field(default=..., description="The text to encode: a URL, mailto:, tel:, a WIFI: string or any text.")


class ChartQueryQRCodeTool(_ChartQueryTool):
    """Generate a QR code through ChartQuery."""

    name: str = "Generate a QR code with ChartQuery"
    description: str = "Generate a QR code with ChartQuery for a link, email, phone number, Wi-Fi network or any text, and get public links to the image. Returns JSON with render_url (the image), view_url (a page to open) and expires_in. Input is the text to encode."
    args_schema: type[BaseModel] = ChartQueryQRCodeToolSchema
    size: int = Field(default=300, ge=50, le=2000)
    """Width and height in pixels."""
    image_format: Literal["png", "svg"] = "png"
    """Image format."""

    def _request(self, data: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "GET",
            "/v1/qrcode",
            {"data": required(data, "data"), "size": self.size, "format": self.image_format, "share": "true"},
            None,
        )

    def _run(self, data: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(data))))
        except ChartQueryError as exc:
            return str(exc)
