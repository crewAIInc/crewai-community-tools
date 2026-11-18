"""HuntAPI tools for CrewAI agents.

Copy this folder into your project and install `requirements.txt`. The tools read
HUNTAPI_API_KEY from the environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Literal
from urllib.parse import quote

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.huntapi.com"
DEFAULT_TIMEOUT = 60.0
USER_AGENT = "crewai-huntapi"
ENV = "HUNTAPI_API_KEY"

_ERROR_DETAIL_LIMIT = 300


class HuntAPIError(Exception):
    """A HuntAPI request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``HUNTAPI_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get(ENV) or "").strip()
    if not key:
        raise HuntAPIError(f"Set {ENV} or pass api_key. Get a key at https://app.huntapi.com")
    return key


def required(value: Any, name: str) -> str:
    """Return ``value`` stripped, or raise if it is empty."""
    text = str(value or "").strip()
    if not text:
        raise HuntAPIError(f"{name} is empty.")
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


def _error(response: httpx.Response, api_key: str) -> HuntAPIError:
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
    # HuntAPI answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "HuntAPI rate limit exceeded"
    elif status == 401:
        label = "HuntAPI rejected the API key"
    elif status == 402:
        label = "HuntAPI account is out of credits"
    else:
        label = "HuntAPI API error"
    return HuntAPIError(f"{label} (HTTP {status})" + (f": {detail}" if detail else ""), status)


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
        raise HuntAPIError("HuntAPI returned a response that is not JSON.") from exc


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
        raise HuntAPIError(f"HuntAPI did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise HuntAPIError(f"Could not reach HuntAPI: {_redact(str(exc), api_key)}") from exc
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
        raise HuntAPIError(f"HuntAPI did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise HuntAPIError(f"Could not reach HuntAPI: {_redact(str(exc), api_key)}") from exc
    return _decode(response, api_key)


JOB_WAIT = 140.0
POLL_EVERY = 3.0
_FAILED = ("Error", "NotFound", "BadRequest")


def _job_result(job_id: str, job: Any) -> dict[str, Any]:
    job = job if isinstance(job, dict) else {}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    status = job.get("status")
    out: dict[str, Any] = {"job_id": job_id, "status": status, "progress": job.get("progress")}
    if status == "CompletedJob":
        out["download_url"] = result.get("response")
        out["title"] = metadata.get("title")
    elif status in _FAILED:
        out["error"] = job.get("error") or "private, deleted, age-restricted or unsupported content"
    else:
        out["note"] = "Still running: check it later with the job id."
    return out


def _job_id(start: Any) -> str:
    job_id = start.get("job_id") if isinstance(start, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise HuntAPIError("HuntAPI did not return a job id.")
    return job_id


def download(
    path: str, params: dict[str, Any], *, api_key: str, base_url: str, timeout: float, wait: float = JOB_WAIT
) -> dict[str, Any]:
    """Start a download job and wait up to ``wait`` seconds for its link."""
    job_id = _job_id(request("GET", path, params=params, api_key=api_key, base_url=base_url, timeout=timeout))
    deadline = time.monotonic() + wait
    while True:
        job = request("GET", f"/v1/jobs/{quote(job_id, safe='')}", api_key=api_key, base_url=base_url, timeout=timeout)
        status = job.get("status") if isinstance(job, dict) else None
        if status == "CompletedJob" or status in _FAILED or time.monotonic() >= deadline:
            return _job_result(job_id, job)
        time.sleep(POLL_EVERY)


async def adownload(
    path: str, params: dict[str, Any], *, api_key: str, base_url: str, timeout: float, wait: float = JOB_WAIT
) -> dict[str, Any]:
    """Async version of :func:`download`."""
    start = await arequest("GET", path, params=params, api_key=api_key, base_url=base_url, timeout=timeout)
    job_id = _job_id(start)
    deadline = time.monotonic() + wait
    while True:
        job = await arequest(
            "GET", f"/v1/jobs/{quote(job_id, safe='')}", api_key=api_key, base_url=base_url, timeout=timeout
        )
        status = job.get("status") if isinstance(job, dict) else None
        if status == "CompletedJob" or status in _FAILED or time.monotonic() >= deadline:
            return _job_result(job_id, job)
        await asyncio.sleep(POLL_EVERY)


class _HuntAPITool(BaseTool):
    """Shared settings and request handling for the HuntAPI tools. Errors come back as text."""

    api_key: str | None = Field(default=None, description="HuntAPI API key. Defaults to HUNTAPI_API_KEY.")
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [EnvVar(name="HUNTAPI_API_KEY", description="API key for HuntAPI", required=True)]
    )

    def _output(self, data: Any) -> Any:
        return data

    def _send(self, method: str, path: str, params: dict[str, Any] | None, body: dict[str, Any] | None) -> Any:
        api_key = resolve_api_key(self.api_key)
        if method == "DOWNLOAD":
            return download(path, clean(params or {}), api_key=api_key, base_url=self.base_url, timeout=self.timeout)
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


class HuntAPIMediaMetadataToolSchema(BaseModel):
    """Input for HuntAPIMediaMetadataTool."""

    url: str = Field(default=..., description="The public video or audio URL.")


class HuntAPIMediaMetadataTool(_HuntAPITool):
    """Read the metadata of a public video through HuntAPI."""

    name: str = "Get video metadata with HuntAPI"
    description: str = "Get the metadata of a public video or audio URL (YouTube, TikTok, Instagram, Vimeo, X, Reddit and more) without downloading it: title, channel, duration, views, likes, upload date and thumbnail. Input is the URL."
    args_schema: type[BaseModel] = HuntAPIMediaMetadataToolSchema

    def _request(self, url: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return "GET", "/v1/media/metadata", {"query": required(url, "url")}, None

    def _run(self, url: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(url))))
        except HuntAPIError as exc:
            return str(exc)


class HuntAPIVideoDownloadToolSchema(BaseModel):
    """Input for HuntAPIVideoDownloadTool."""

    url: str = Field(default=..., description="The public video URL.")


class HuntAPIVideoDownloadTool(_HuntAPITool):
    """Download a public video through HuntAPI and return the download link."""

    name: str = "Download a video with HuntAPI"
    description: str = "Download a public video (YouTube, TikTok, Instagram, Vimeo, X, Reddit and more) and get a download link. Waits up to 140 seconds; a longer job returns its job id. Input is the URL."
    args_schema: type[BaseModel] = HuntAPIVideoDownloadToolSchema
    quality: Literal["best", "1080p", "720p", "480p", "360p"] = "best"
    """Video quality."""
    video_format: Literal["mp4", "webm", "mkv"] = "mp4"
    """Output format."""

    def _request(self, url: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "DOWNLOAD",
            "/v1/video/download",
            {"query": required(url, "url"), "video_quality": self.quality, "video_format": self.video_format},
            None,
        )

    def _run(self, url: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(url))))
        except HuntAPIError as exc:
            return str(exc)


class HuntAPIAudioDownloadToolSchema(BaseModel):
    """Input for HuntAPIAudioDownloadTool."""

    url: str = Field(default=..., description="The public video or audio URL.")


class HuntAPIAudioDownloadTool(_HuntAPITool):
    """Extract the audio of a public video through HuntAPI and return the download link."""

    name: str = "Extract audio with HuntAPI"
    description: str = "Extract the audio of a public video or audio URL (a podcast, a talk, a song) and get a download link. Waits up to 140 seconds; a longer job returns its job id. Input is the URL."
    args_schema: type[BaseModel] = HuntAPIAudioDownloadToolSchema
    audio_format: Literal["mp3", "m4a", "wav"] = "mp3"
    """Output format."""
    quality: Literal["best", "320kbps", "256kbps", "128kbps"] = "best"
    """Audio quality."""

    def _request(self, url: str) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        return (
            "DOWNLOAD",
            "/v1/audio/download",
            {"query": required(url, "url"), "audio_format": self.audio_format, "audio_quality": self.quality},
            None,
        )

    def _run(self, url: str, **kwargs: Any) -> str:
        try:
            return self._text(self._output(self._send(*self._request(url))))
        except HuntAPIError as exc:
            return str(exc)
