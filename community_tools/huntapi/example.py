"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export HUNTAPI_API_KEY="your-api-key"   # get one at https://app.huntapi.com
    python example.py

Environment variables:
- HUNTAPI_API_KEY (required): HuntAPI API key. Each metadata read or download uses
  HuntAPI credits. Downloads run on HuntAPI's servers and return a link; nothing is
  saved locally.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import HuntAPIAudioDownloadTool, HuntAPIMediaMetadataTool, HuntAPIVideoDownloadTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # HuntAPIMediaMetadataTool
    print(tools[0].run(url="https://www.youtube.com/watch?v=jNQXAC9IVRw"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Media assistant",
        goal="Answer questions about public videos",
        backstory=("You read video details with HuntAPI before you answer, and you only start a download when asked."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "Give the title, channel, duration and view count of https://www.youtube.com/watch?v=jNQXAC9IVRw."
        ),
        expected_output="Title, channel, duration and view count.",
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
    if not os.environ.get("HUNTAPI_API_KEY"):
        print("Set HUNTAPI_API_KEY first. Get a key at https://app.huntapi.com")
        return

    tools: list[BaseTool] = [HuntAPIMediaMetadataTool(), HuntAPIVideoDownloadTool(), HuntAPIAudioDownloadTool()]
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
