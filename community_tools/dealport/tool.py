"""CrewAI tools for the Dealport V4 REST API (https://app.dealport.com).

Reads DP_API_KEY (or DEALPORT_API_KEY) from the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any
import urllib.parse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


DEFAULT_BASE_URL = "https://app.dealport.com"
VALID_NYC_BOROUGHS = frozenset({"MN", "BK", "QN", "BX", "SI"})


def _dealport_request(
    path: str,
    *,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> str:
    key = (
        api_key
        or os.environ.get("DP_API_KEY")
        or os.environ.get("DEALPORT_API_KEY")
        or ""
    ).strip()
    if not key:
        return "Error: Set DP_API_KEY (from https://app.dealport.com/connect?tab=keys)."
    base = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        return "Error: base_url must be a valid http(s) URL."
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "crewai-community-dealport/1.0",
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = requests.request(
            method=method,
            url=f"{base}{path}",
            json=json_body,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code >= 300:
            return f"Dealport API error ({resp.status_code}): {resp.text[:300].strip()}"
        if not resp.text:
            return "{}"
        return json.dumps(resp.json(), ensure_ascii=False)
    except requests.Timeout:
        return "Dealport API request timed out."
    except ValueError:
        return resp.text
    except requests.RequestException as err:
        return f"Dealport API network error: {err}"


def _company_filters(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(params.get("q"), str) and params["q"].strip():
        out["q"] = params["q"].strip()
    if isinstance(params.get("state"), str) and params["state"].strip():
        out["state"] = params["state"].strip().upper()
    if isinstance(params.get("domain"), str) and params["domain"].strip():
        out["domain"] = params["domain"].strip()
    for list_key in ("cities", "counties", "naics"):
        items = params.get(list_key)
        if isinstance(items, list):
            cleaned = [str(x).strip() for x in items if str(x).strip()]
            if cleaned:
                out[list_key] = cleaned
    for num_key in (
        "min_revenue",
        "max_revenue",
        "min_employees",
        "max_employees",
        "limit",
    ):
        val = params.get(num_key)
        if val is not None and not isinstance(val, bool):
            out[num_key] = val
    if isinstance(params.get("cursor"), str) and params["cursor"].strip():
        out["cursor"] = params["cursor"].strip()
    for bool_key in ("independent", "licensed", "owner_contact"):
        if params.get(bool_key) is True:
            out[bool_key] = True
    return out


class DealportCompanyFilterInput(BaseModel):
    q: str | None = Field(
        default=None, description="Company name prefix (at least 2 letters)."
    )
    state: str | None = Field(
        default=None, description="Two-letter US state code, e.g. OH."
    )
    cities: list[str] | None = Field(default=None, description="Cities within state.")
    counties: list[str] | None = Field(
        default=None, description="Counties within state."
    )
    naics: list[str] | None = Field(
        default=None, description="2-to-6 digit NAICS codes, e.g. ['238220']."
    )
    domain: str | None = Field(
        default=None, description="Company website domain, e.g. example.com."
    )
    min_revenue: float | None = Field(
        default=None, ge=0, description="Minimum annual revenue in USD."
    )
    max_revenue: float | None = Field(
        default=None, ge=0, description="Maximum annual revenue in USD."
    )
    min_employees: int | None = Field(
        default=None, ge=0, description="Minimum headcount."
    )
    max_employees: int | None = Field(
        default=None, ge=0, description="Maximum headcount."
    )
    independent: bool | None = Field(
        default=None, description="Only privately held independent companies."
    )
    licensed: bool | None = Field(
        default=None, description="Only companies holding an active state license."
    )
    owner_contact: bool | None = Field(
        default=None, description="Only companies with owner phone or email."
    )


class DealportCompanySearchInput(DealportCompanyFilterInput):
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum company Records to return (1 to 50).",
    )
    cursor: str | None = Field(
        default=None, description="Pagination cursor from a prior response."
    )


class DealportBuildingSearchInput(BaseModel):
    borough: str = Field(..., description="NYC borough code: MN, BK, QN, BX, or SI.")
    min_units: int | None = Field(default=None, ge=1, description="Minimum unit count.")
    q: str | None = Field(
        default=None, description="Building address or owner name filter."
    )
    loan_due_by: int | None = Field(
        default=None, description="Maximum ACRIS mortgage maturity year."
    )
    loan_holder: str | None = Field(
        default=None, description="Mortgage holder category or lender filter."
    )
    signal: str | None = Field(
        default=None, description="Optional building signal filter."
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum building Records to return (1 to 50).",
    )
    cursor: str | None = Field(
        default=None, description="Pagination cursor from a prior response."
    )


class DealportGetRecordInput(BaseModel):
    record_id: str = Field(
        ..., description="Dealport company UUID or 10-digit NYC BBL."
    )


class DealportFindOwnerInput(BaseModel):
    company_id: str = Field(..., description="Dealport company UUID.")
    person: str | None = Field(
        default=None, description="Optional full name of a specific principal."
    )


class DealportPreviewCompaniesTool(BaseTool):
    name: str = "dealport_preview_companies"
    description: str = (
        "Size a US private company buy-box for $0.00 before spending balance. "
        "Returns matching_count and up to 5 sample companies from Dealport."
    )
    args_schema: type[BaseModel] = DealportCompanyFilterInput

    def _run(self, **kwargs: Any) -> str:
        filters = _company_filters(kwargs)
        filters.pop("cursor", None)
        return _dealport_request(
            "/api/v1/companies/preview", method="POST", json_body=filters
        )


class DealportSearchCompaniesTool(BaseTool):
    name: str = "dealport_search_companies"
    description: str = (
        "Search Dealport's 28M+ cited US private company Records across geography, NAICS, "
        "revenue, headcount, and license filters ($0.01/company; $0.00 30-day repeat)."
    )
    args_schema: type[BaseModel] = DealportCompanySearchInput

    def _run(self, **kwargs: Any) -> str:
        filters = _company_filters(kwargs)
        pairs: list[tuple[str, str]] = []
        for k, v in filters.items():
            if v is True:
                pairs.append((k, "1"))
            elif isinstance(v, list):
                pairs.append((k, ",".join(str(x) for x in v)))
            else:
                pairs.append((k, str(v)))
        qs = urllib.parse.urlencode(pairs)
        return _dealport_request(f"/api/v1/companies{f'?{qs}' if qs else ''}")


class DealportSearchBuildingsTool(BaseTool):
    name: str = "dealport_search_buildings"
    description: str = (
        "Search NYC commercial and multifamily building Records by borough (MN, BK, QN, BX, SI), "
        "unit count, and ACRIS mortgage maturity ($0.01/building; $0.00 30-day repeat)."
    )
    args_schema: type[BaseModel] = DealportBuildingSearchInput

    def _run(self, borough: str, **kwargs: Any) -> str:
        clean_borough = borough.strip().upper()
        if clean_borough not in VALID_NYC_BOROUGHS:
            return "Error: borough must be one of MN, BK, QN, BX, or SI."
        pairs: list[tuple[str, str]] = [("borough", clean_borough)]
        for key in (
            "min_units",
            "q",
            "loan_due_by",
            "loan_holder",
            "signal",
            "limit",
            "cursor",
        ):
            val = kwargs.get(key)
            if val is not None and str(val).strip():
                pairs.append((key, str(val).strip()))
        return _dealport_request(f"/api/v1/buildings?{urllib.parse.urlencode(pairs)}")


class DealportGetRecordTool(BaseTool):
    name: str = "dealport_get_record"
    description: str = (
        "Open a cited Dealport company Record (UUID) or NYC building Record (10-digit BBL) "
        "with primary public-filing facts and principals ($0.01; $0.00 30-day repeat)."
    )
    args_schema: type[BaseModel] = DealportGetRecordInput

    def _run(self, record_id: str) -> str:
        cleaned = record_id.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
            return "Error: record_id must be a single company UUID or 10-digit NYC BBL."
        encoded = urllib.parse.quote(cleaned, safe="")
        return _dealport_request(f"/api/v1/records/{encoded}")


class DealportFindOwnerTool(BaseTool):
    name: str = "dealport_find_owner"
    description: str = (
        "Resolve the verified human owner behind a Dealport company UUID with direct "
        "phones and emails ($0.09 hit; $0.00 miss or 30-day repeat)."
    )
    args_schema: type[BaseModel] = DealportFindOwnerInput

    def _run(self, company_id: str, person: str | None = None) -> str:
        clean_id = company_id.strip()
        if not clean_id:
            return "Error: company_id is required."
        body: dict[str, Any] = {"company_id": clean_id}
        if isinstance(person, str) and person.strip():
            body["person"] = person.strip()
        return _dealport_request("/api/v1/owners", method="POST", json_body=body)
