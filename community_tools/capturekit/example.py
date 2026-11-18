"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export CAPTUREKIT_API_KEY="your-api-key"   # get one at https://app.capturekit.dev
    python example.py

Environment variables:
- CAPTUREKIT_API_KEY (required): CaptureKit API key. Each call uses CaptureKit credits.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import CaptureKitAnalyzePageTool, CaptureKitPageContentTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # CaptureKitPageContentTool
    print(tools[0].run(url="https://example.com"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Web analyst",
        goal="Answer questions about websites from their actual content",
        backstory=("You read pages with CaptureKit as clean Markdown before you answer."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=("Read https://www.crewai.com and describe in three bullet points what the company offers."),
        expected_output="Three bullet points about the offer.",
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
    if not os.environ.get("CAPTUREKIT_API_KEY"):
        print("Set CAPTUREKIT_API_KEY first. Get a key at https://app.capturekit.dev")
        return

    tools: list[BaseTool] = [CaptureKitPageContentTool(), CaptureKitAnalyzePageTool()]
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
