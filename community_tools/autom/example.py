"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export AUTOM_API_KEY="your-api-key"   # get one at https://app.autom.dev
    python example.py

Environment variables:
- AUTOM_API_KEY (required): Autom API key. Each search uses Autom credits.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import AutomSearchTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # AutomSearchTool
    print(tools[0].run(search_query="best open-source agent frameworks"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Web researcher",
        goal="Answer questions with current web results",
        backstory=("You search the web with Autom before you answer, and you cite the links you used."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "Find three recent articles about open-source agent frameworks and summarize each in one sentence with its link."
        ),
        expected_output="Three one-sentence summaries, each with its link.",
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
    if not os.environ.get("AUTOM_API_KEY"):
        print("Set AUTOM_API_KEY first. Get a key at https://app.autom.dev")
        return

    tools: list[BaseTool] = [AutomSearchTool()]
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
