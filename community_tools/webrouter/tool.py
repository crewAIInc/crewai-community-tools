"""CrewAI tools for web search and page extraction through WebRouter.

WebRouter (https://github.com/Nimbleway/webrouter) is an open-source library
that puts one interface in front of several search and extraction providers.
It works with no API key. A provider key set in the environment, such as
TAVILY_API_KEY, EXA_API_KEY or FIRECRAWL_API_KEY, adds that provider to the
routing, and the calls it serves are billed to that key.
"""

from pydantic import BaseModel, Field
from web_router import WebRouter, WebRouterError

from crewai.tools import BaseTool


def _clip(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " [...]"


class WebRouterSearchInput(BaseModel):
    query: str = Field(..., description="What to search the web for.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many results to return, from 1 to 20.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Only return results from these domains, for example ['docs.python.org'].",
    )


class WebRouterSearchTool(BaseTool):
    name: str = "webrouter_search"
    description: str = (
        "Search the web and return ranked results, each with a title, URL and "
        "snippet. Use this to find current information or pages on a topic. "
        "To read a result in full, pass its URL to webrouter_extract."
    )
    args_schema: type[BaseModel] = WebRouterSearchInput
    provider: str | None = Field(
        default=None,
        description="Send every call to this provider, e.g. 'tavily'. None lets WebRouter pick and fall back.",
    )
    max_chars_per_result: int | None = Field(
        default=1000,
        description="Cap on each result's snippet. None for no cap.",
    )

    def _run(
        self,
        query: str,
        max_results: int = 5,
        include_domains: list[str] | None = None,
    ) -> str:
        try:
            with WebRouter() as router:
                response = router.search(
                    query,
                    provider=self.provider,
                    max_results=max_results,
                    include_domains=include_domains,
                )
        except WebRouterError as error:
            return f"Search failed: {error}"

        if not response.results:
            return f"No results for {query!r}."
        lines = []
        for rank, result in enumerate(response.results, start=1):
            snippet = " ".join((result.description or result.content).split())
            lines.append(f"{rank}. {result.title or result.url}\n   {result.url}")
            if snippet:
                lines.append(f"   {_clip(snippet, self.max_chars_per_result)}")
        return "\n".join(lines)


class WebRouterExtractInput(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="One to ten http(s) URLs to read.",
    )


class WebRouterExtractTool(BaseTool):
    name: str = "webrouter_extract"
    description: str = (
        "Read one or more web pages and return their main content as markdown. "
        "Use this when you have a URL and need what the page says. URLs that "
        "could not be read are listed with the reason."
    )
    args_schema: type[BaseModel] = WebRouterExtractInput
    provider: str | None = Field(
        default=None,
        description="Send every call to this provider, e.g. 'firecrawl'. None lets WebRouter pick and fall back.",
    )
    max_chars_per_page: int | None = Field(
        default=20000,
        description="Cap on each page's content. None for no cap.",
    )

    def _run(self, urls: list[str]) -> str:
        try:
            with WebRouter() as router:
                response = router.extract(
                    urls, provider=self.provider, format="markdown"
                )
        except WebRouterError as error:
            return f"Could not read {', '.join(urls)}: {error}"

        sections = []
        for document in response.documents:
            title = document.title or document.url
            content = _clip(document.content, self.max_chars_per_page)
            sections.append(f"# {title}\nURL: {document.url}\n\n{content}")
        if response.failures:
            failed = "\n".join(
                f"- {failure.url}: {failure.error}" for failure in response.failures
            )
            sections.append(f"Could not read:\n{failed}")
        return "\n\n---\n\n".join(sections)
