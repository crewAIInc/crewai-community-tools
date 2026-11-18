"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export PILOTERR_API_KEY="your-api-key"   # get one at https://app.piloterr.com
    python example.py

Environment variables:
- PILOTERR_API_KEY (required): Piloterr API key. Each call uses Piloterr credits.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import PiloterrGoogleSearchTool, PiloterrScrapeWebsiteTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # PiloterrGoogleSearchTool
    print(tools[0].run(search_query="crewai multi-agent framework"))
    print()
    # PiloterrScrapeWebsiteTool
    print(tools[1].run(website_url="https://example.com"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Web researcher",
        goal="Answer questions from Google results and the pages behind them",
        backstory=("You search Google with Piloterr, read the most relevant pages, and cite them."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "Find the official CrewAI documentation page about tools and summarize what it says in three bullet points, with the link."
        ),
        expected_output="Three bullet points and the link of the page you read.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    print("Crew result")
    print(crew.kickoff())


def main() -> None:
    if not os.environ.get("PILOTERR_API_KEY"):
        print("Set PILOTERR_API_KEY first. Get a key at https://app.piloterr.com")
        return

    tools: list[BaseTool] = [PiloterrGoogleSearchTool(), PiloterrScrapeWebsiteTool()]
    direct_call(tools)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(tools)


if __name__ == "__main__":
    main()
