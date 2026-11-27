"""Harny memory tools for CrewAI: long-term memory of each person that keeps up when facts change.

Harny gives every customer of an app their own AI agent, one that knows that customer and acts for them.
These two tools bring the part that knows each person to a CrewAI agent. One tool instance serves one person.
The key is read from HARNY_API_KEY.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr


class HarnyRecallToolSchema(BaseModel):
    """Input accepted by the Harny recall tool."""

    query: str = Field(
        ...,
        min_length=1,
        description="What to look up, in plain words, e.g. 'the Acme pilot budget'.",
    )


class HarnyRememberToolSchema(BaseModel):
    """Input accepted by the Harny remember tool."""

    content: str = Field(
        ...,
        min_length=1,
        description="One fact worth keeping, written as a full sentence, e.g. 'Nadia prefers async standups.'",
    )
    label: str | None = Field(
        default=None, description="An optional short name for the fact."
    )


def _fact_id(user: str, text: str) -> str:
    """A stable id for a saved fact: saying the same thing twice replaces nothing."""
    return hashlib.sha256(f"{user}\n{text}".encode()).hexdigest()


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _code(exc: Exception) -> str:
    return getattr(exc, "error_code", None) or type(exc).__name__


def _is_unknown_person(exc: Exception) -> bool:
    return str(_code(exc)).replace("_", "-") == "user-not-found-in-app"


class HarnyMemoryToolBase(BaseTool):
    """Shared setup for the Harny memory tools: one tool instance serves one person."""

    user: str = Field(
        ...,
        min_length=1,
        description="Your id for the person this agent is talking to.",
    )
    level: str = Field(
        default="medium",
        description="Evidence size for recall: low, medium, high or extra-high.",
    )
    api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Harny app key. Defaults to the HARNY_API_KEY environment variable.",
    )
    base_url: str | None = Field(
        default=None,
        description="Harny API address. Defaults to https://api.harny.ai.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="HARNY_API_KEY",
                description="Harny app key, unless passed to the constructor.",
                required=False,
            ),
        ]
    )

    _client: Any = PrivateAttr(default=None)
    _error_type: type[Exception] = PrivateAttr(default=Exception)

    def __init__(self, *, client: Any = None, **kwargs: Any) -> None:
        """Accept an optional ready harny.SyncHarnyClient, whose lifetime the caller owns."""
        super().__init__(**kwargs)
        try:
            from harny import HarnyError, SyncHarnyClient
        except ImportError as exc:
            raise ImportError(
                "Install Harny with `pip install -r requirements.txt` (or `pip install harny`)."
            ) from exc
        self._error_type = HarnyError
        self._client = (
            client
            if client is not None
            else SyncHarnyClient(api_key=self.api_key, base_url=self.base_url)
        )

    def _add_person(self) -> None:
        """Create the person in the Harny app (first save for someone Harny has not seen yet)."""
        base = (
            self.base_url or os.environ.get("HARNY_API_URL") or "https://api.harny.ai"
        ).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        key = self.api_key or os.environ.get("HARNY_API_KEY", "")
        response = httpx.post(
            f"{base}/users",
            headers={"Authorization": f"Bearer {key}"},
            json={"identifier": self.user, "identifier_type": "app_id"},
            timeout=30,
        )
        response.raise_for_status()


class HarnyRecallTool(HarnyMemoryToolBase):
    """Look up what is remembered about the person, with the date each fact happened."""

    name: str = "harny_recall"
    description: str = (
        "Recall long-term memory for this person across chat sessions: their preferences, past decisions, plans and "
        "anything they told you before. Use it before answering anything that depends on what they said earlier. "
        "Returns dated facts; where two facts disagree, the newer one wins."
    )
    args_schema: type[BaseModel] = HarnyRecallToolSchema

    def _run(self, query: str) -> str:
        try:
            answer = self._client.recall(query, level=self.level, user=self.user)
        except self._error_type as exc:  # reported to the agent in words, not raised
            if _is_unknown_person(exc):
                return "Nothing is remembered about this yet."
            return f"Memory is unavailable right now ({_code(exc)})."
        lines = []
        for result in answer.results:
            if not result.content.strip():
                continue
            when = str(result.occurredAt)[:10] if result.occurredAt else ""
            lines.append(
                f"- ({when}) {result.content}" if when else f"- {result.content}"
            )
        pending = answer.pending.message if answer.pending else None
        if not lines:
            return "Nothing is remembered about this yet." + (
                f" {pending}" if pending else ""
            )
        return "\n".join(lines) + (f"\n(Note: {pending})" if pending else "")


class HarnyRememberTool(HarnyMemoryToolBase):
    """Save one fact about the person so it is recalled in later conversations."""

    name: str = "harny_remember"
    description: str = (
        "Remember one fact in this person's long-term memory across chat sessions (a preference, decision, plan or "
        "detail they would expect you to remember). Write it as a full sentence. Saved facts are recalled in later "
        "conversations."
    )
    args_schema: type[BaseModel] = HarnyRememberToolSchema

    def _save(self, content: str, label: str | None) -> Any:
        return self._client.ingest_data_point(
            content,
            id=_fact_id(self.user, content),
            label=label,
            timestamp=_now_iso(),
            user=self.user,
        )

    def _run(self, content: str, label: str | None = None) -> str:
        try:
            try:
                receipt = self._save(content, label)
            except self._error_type as exc:
                if not _is_unknown_person(exc):
                    raise
                self._add_person()
                receipt = self._save(content, label)
        except (self._error_type, httpx.HTTPError) as exc:
            return f"Could not save that right now ({_code(exc)})."
        return "Already remembered." if receipt.unchanged else "Remembered."
