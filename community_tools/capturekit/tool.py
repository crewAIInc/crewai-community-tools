"""CaptureKit tools for CrewAI agents.

Copy this folder into your project and install `requirements.txt`. The tools read
CAPTUREKIT_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.capturekit.dev"
DEFAULT_TIMEOUT = 120.0
USER_AGENT = "crewai-capturekit"
ENV = "CAPTUREKIT_API_KEY"

_ERROR_DETAIL_LIMIT = 300


class CaptureKitError(Exception):
    """A CaptureKit request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``CAPTUREKIT_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get(ENV) or "").strip()
    if not key:
        raise CaptureKitError(f"Set {ENV} or pass api_key. Get a key at https://app.capturekit.dev")
    return key


def required(value: Any, name: str) -> str:
    """Return ``value`` stripped, or raise if it is empty."""
    text = str(value or "").strip()
    if not text:
        raise CaptureKitError(f"{name} is empty.")
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


def _error(response: httpx.Response, api_key: str) -> CaptureKitError:
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
    # CaptureKit answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "CaptureKit rate limit exceeded"
    elif status == 401:
        label = "CaptureKit rejected the API key"
    elif status == 402:
        label = "CaptureKit account is out of credits"
    else:
        label = "CaptureKit API error"
    return CaptureKitError(f"{label} (HTTP {status})" + (f": {detail}" if detail else ""), status)


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
        raise CaptureKitError("CaptureKit returned a response that is not JSON.") from exc


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
        raise CaptureKitError(f"CaptureKit did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise CaptureKitError(f"Could not reach CaptureKit: {_redact(str(exc), api_key)}") from exc
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
        raise CaptureKitError(f"CaptureKit did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise CaptureKitError(f"Could not reach CaptureKit: {_redact(str(exc), api_key)}") from exc
    return _decode(response, api_key)


def check_url(url: str) -> str:
    """Return ``url`` stripped, or raise if it is not an absolute http(s) URL."""
    url = (url or "").strip()
    if not re.match(r"^https?://[^/\s]+", url, re.IGNORECASE):
        raise CaptureKitError(f"Use an absolute http(s) URL, such as https://example.com; got {url!r}.")
    return url


class _CaptureKitTool(BaseTool):
    """Shared settings and request handling for the CaptureKit tools. Errors come back as text."""

    api_key: str | None = Field(default=None, description="CaptureKit API key. Defaults to CAPTUREKIT_API_KEY.")
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [EnvVar(name="CAPTUREKIT_API_KEY", description="API key for CaptureKit", required=True)]
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


class CaptureKitPageContentToolSchema(BaseModel):
    """Input for CaptureKitPageContentTool."""

    url: str = Field(default=..., description="The absolute http(s) URL of the page.")


class CaptureKitPageContentTool(_CaptureKitTool):
    """Load a web page in a real browser through CaptureKit and return it as Markdown."""

    name: str = "Read a web page with CaptureKit"
    description: str = "Read a web page with CaptureKit and get its content as clean Markdown, with its title. Input is the absolute URL of the page."
    args_schema: type[BaseModel] = CaptureKitPageContentToolSchema
    main_content_only: bool = True
    """Keep the article or product and drop menus, sidebars and footers."""
    max_characters: int | None = Field(default=20_000, ge=1)
    """Cut the Markdown at this many characters; ``None`` keeps it all."""

    def _request(self, url: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "GET",
            "/v1/content",
            clean(
                {
                    "url": check_url(url),
                    "include_markdown": True,
                    "remove_cookie_banners": True,
                    "use_defuddle": True if self.main_content_only else None,
                }
            ),
            None,
        )

    def _output(self, data: Any) -> Any:
        page = data.get("data") if isinstance(data, dict) else None
        if not isinstance(page, dict) or not isinstance(page.get("markdown"), str):
            raise CaptureKitError("CaptureKit returned an unexpected response; expected the page Markdown.")
        metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
        markdown = page["markdown"]
        if self.max_characters is not None:
            markdown = markdown[: self.max_characters]
        title = metadata.get("title") or metadata.get("site") or "Page"
        return f"# {title}\n\n{markdown or '(The page has no readable text.)'}"

    def _run(self, url: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(url))))
        except CaptureKitError as exc:
            return str(exc)


class CaptureKitAnalyzePageToolSchema(BaseModel):
    """Input for CaptureKitAnalyzePageTool."""

    url: str = Field(default=..., description="The absolute http(s) URL of the page.")
    question: str | None = Field(
        default=None, description="Optional question or instructions for the AI about this page."
    )


class CaptureKitAnalyzePageTool(_CaptureKitTool):
    """Get an AI profile of a web page through CaptureKit."""

    name: str = "Analyze a website with CaptureKit"
    description: str = "Read a web page with AI through CaptureKit and get a summary, the value proposition, key features, use cases, target audience, and the emails and phone numbers found. Input is the URL, and optionally a question about the page."
    args_schema: type[BaseModel] = CaptureKitAnalyzePageToolSchema

    def _request(
        self, url: str, question: str | None = None
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return "GET", "/v1/analyze", clean({"url": check_url(url), "custom_prompt": question}), None

    def _output(self, data: Any) -> Any:
        return data.get("data", data) if isinstance(data, dict) else data

    def _run(self, url: str, question: str | None = None, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(url, question))))
        except CaptureKitError as exc:
            return str(exc)
