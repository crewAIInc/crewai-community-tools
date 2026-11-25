"""CrewAI tools for Webcite: claim verification, document parsing and source search.

Webcite checks claims against sources and returns a verdict with citations bound to the
passages that support or contradict them, reads PDFs, scans and office files into Markdown,
and finds sources for a query. The tools call the public REST API at https://api.webcite.co
and read WEBCITE_API_KEY from the environment. Each call spends credits in the account that
owns the key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.webcite.co"
TIMEOUT_SECONDS = 600
MAX_CITATIONS = 5


class WebciteError(RuntimeError):
    """Raised when the Webcite API cannot complete a request."""


def _api_key() -> str:
    key = os.environ.get("WEBCITE_API_KEY", "").strip()
    if not key:
        raise WebciteError(
            "Set WEBCITE_API_KEY to a key from https://webcite.co/api-keys."
        )
    return key


def _request(path: str, **kwargs: Any) -> dict[str, Any]:
    base_url = os.environ.get("WEBCITE_API_URL", DEFAULT_BASE_URL).rstrip("/")
    headers = {"x-api-key": _api_key(), "Accept": "application/json"}
    response = requests.post(
        f"{base_url}{path}", headers=headers, timeout=TIMEOUT_SECONDS, **kwargs
    )
    if response.status_code == 401:
        raise WebciteError("Webcite rejected the API key. Check WEBCITE_API_KEY.")
    if response.status_code >= 400:
        raise WebciteError(
            f"Webcite returned HTTP {response.status_code} for {path}: {response.text[:300]}"
        )
    body: dict[str, Any] = response.json()
    return body


def _clip(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _format_citations(citations: list[dict[str, Any]]) -> list[str]:
    lines = []
    for citation in citations[:MAX_CITATIONS]:
        stance = citation.get("stance")
        prefix = f"[{citation.get('id')}]"
        if stance:
            prefix = f"{prefix} {stance}"
        lines.append(f"{prefix} {citation.get('url')}")
        snippet = (citation.get("snippet") or "").strip()
        if snippet:
            lines.append(f"    {_clip(snippet)}")
    return lines


class WebciteVerifyClaimInput(BaseModel):
    claim: str = Field(
        ..., description="One factual claim to check, as a full sentence."
    )
    source_urls: list[str] | None = Field(
        default=None,
        description="Optional URLs to check the claim against instead of searching the web.",
    )


class WebciteVerifyClaimTool(BaseTool):
    name: str = "webcite_verify_claim"
    description: str = (
        "Check whether a factual claim is supported or contradicted by sources. Returns a "
        "verdict (for example supported, contradicted or partially_supported), a short "
        "summary and the cited sources with the passage each one contains. Use this before "
        "stating a fact, number or quote that the answer depends on."
    )
    args_schema: type[BaseModel] = WebciteVerifyClaimInput

    def _run(self, claim: str, source_urls: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"claim": claim}
        if source_urls:
            payload["source_urls"] = source_urls
        result = _request("/api/v1/verify", json=payload)
        groups = result.get("claim_groups") or []
        if not groups:
            return (
                "Rephrase this as one factual sentence and call webcite_verify_claim "
                f"again: {claim}"
            )
        lines = []
        for group in groups:
            verdict = group.get("verdict") or {}
            lines.append(f"Claim: {group.get('claim', claim)}")
            if verdict.get("result"):
                lines.append(f"Verdict: {verdict['result']}")
            if verdict.get("summary"):
                lines.append(f"Summary: {verdict['summary']}")
            lines.extend(_format_citations(group.get("citations") or []))
        credits = (result.get("usage") or {}).get("credits")
        if credits is not None:
            lines.append(f"Credits used: {credits}")
        return "\n".join(lines)


class WebciteExtractDocumentInput(BaseModel):
    source: str = Field(
        ...,
        description="A local file path or an http(s) URL of a PDF, image, spreadsheet or office file.",
    )


class WebciteExtractDocumentTool(BaseTool):
    name: str = "webcite_extract_document"
    description: str = (
        "Read a document into Markdown, including scanned pages (OCR) and tables. Pass a "
        "local file path or a direct URL to the file. Use this when the task needs the text "
        "or tables inside a PDF, image, spreadsheet or office file."
    )
    args_schema: type[BaseModel] = WebciteExtractDocumentInput

    def _run(self, source: str) -> str:
        if source.startswith(("http://", "https://")):
            payload = {"asset_url": source}
        else:
            path = Path(source).expanduser()
            with path.open("rb") as handle:
                uploaded = _request(
                    "/api/v1/upload", files={"file": (path.name, handle)}
                )
            payload = {"asset_id": uploaded["data"]["asset_id"]}
        result = _request("/api/v1/extract", json=payload)
        markdown = result.get("markdown") or ""
        state = result.get("state")
        if state != "complete":
            reason = f" ({result['reason']})" if result.get("reason") else ""
            return f"Extraction state: {state}{reason}. Text read so far:\n\n{markdown}"
        return markdown


class WebciteSearchSourcesInput(BaseModel):
    query: str = Field(..., description="What to find sources for.")
    limit: int = Field(
        default=5, ge=1, le=20, description="Number of sources to return."
    )


class WebciteSearchSourcesTool(BaseTool):
    name: str = "webcite_search_sources"
    description: str = (
        "Find web sources for a query. Returns titles, URLs and the relevant passage from "
        "each source, without a verdict. Use this to gather candidate sources; use "
        "webcite_verify_claim to judge a specific claim."
    )
    args_schema: type[BaseModel] = WebciteSearchSourcesInput

    def _run(self, query: str, limit: int = 5) -> str:
        result = _request(
            "/api/v1/sources/search", json={"query": query, "limit": limit}
        )
        groups = result.get("claim_groups") or []
        grouped = groups[0].get("citations") if groups else None
        citations = grouped or result.get("citations") or []
        if not citations:
            return (
                f"Call webcite_search_sources again with a broader query than: {query}"
            )
        lines = []
        for citation in citations[:limit]:
            lines.append(f"{citation.get('title')}: {citation.get('url')}")
            snippet = (citation.get("snippet") or "").strip()
            if snippet:
                lines.append(f"    {_clip(snippet)}")
        return "\n".join(lines)
