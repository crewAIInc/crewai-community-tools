"""Copy this folder and run the example from inside it (Python 3.11 or later).

    pip install -r requirements.txt
    python example.py

No API key is required: WebRouter searches and reads pages through the
providers' free tiers. Optional provider keys, read from the environment, add
providers to the routing and bill the calls they serve, for example
TAVILY_API_KEY, EXA_API_KEY, FIRECRAWL_API_KEY or NIMBLE_API_KEY. The crew runs
only when OPENAI_API_KEY or MODEL is already set.
"""

import os

from crewai import Agent, Crew, Process, Task

from tool import WebRouterExtractTool, WebRouterSearchTool


def direct_call(search: WebRouterSearchTool, extract: WebRouterExtractTool) -> None:
    print("Direct call: search")
    print(search.run(query="CrewAI multi-agent framework", max_results=3))
    print()
    print("Direct call: extract")
    print(extract.run(urls=["https://docs.crewai.com/en/introduction"])[:1500])


def run_crew(search: WebRouterSearchTool, extract: WebRouterExtractTool) -> None:
    agent = Agent(
        role="Web researcher",
        goal="Answer questions from current web sources and cite the pages you used",
        backstory=(
            "You call webrouter_search to find pages, then webrouter_extract to "
            "read the ones that matter before you answer."
        ),
        tools=[search, extract],
        verbose=True,
    )
    task = Task(
        description=(
            "What are Flows in CrewAI, and how do they differ from Crews? "
            "Answer in a short paragraph and list the URLs you read."
        ),
        expected_output="A short paragraph followed by the source URLs.",
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
    search = WebRouterSearchTool()
    extract = WebRouterExtractTool()
    direct_call(search, extract)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print()
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(search, extract)


if __name__ == "__main__":
    main()
