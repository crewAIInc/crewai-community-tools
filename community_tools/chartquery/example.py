"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export CHARTQUERY_API_KEY="your-api-key"   # get one at https://app.chartquery.com
    python example.py

Environment variables:
- CHARTQUERY_API_KEY (required): ChartQuery API key. Each image uses ChartQuery credits
  (3 to 10). Images are created with sharing on: anyone with the link can view them, so
  do not chart private data.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import ChartQueryChartFromPromptTool, ChartQueryChartTool, ChartQueryDiagramTool, ChartQueryQRCodeTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # ChartQueryChartFromPromptTool
    print(tools[1].run(prompt="Monthly revenue as a bar chart", data="Jan: 12400, Feb: 18900, Mar: 21300"))
    print()
    # ChartQueryDiagramTool
    print(tools[2].run(source="graph TD; Lead-->Qualified; Qualified-->Customer"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Reporting assistant",
        goal="Turn numbers and processes into images people can share",
        backstory=("You build charts and diagrams with ChartQuery and give the image links."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "Make a bar chart of quarterly signups (Q1: 320, Q2: 410, Q3: 530) and a flowchart of the sign-up flow (visit, sign up, verify email, first project). Give both links."
        ),
        expected_output="Two image links, one chart and one diagram.",
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
    if not os.environ.get("CHARTQUERY_API_KEY"):
        print("Set CHARTQUERY_API_KEY first. Get a key at https://app.chartquery.com")
        return

    tools: list[BaseTool] = [
        ChartQueryChartTool(),
        ChartQueryChartFromPromptTool(),
        ChartQueryDiagramTool(),
        ChartQueryQRCodeTool(),
    ]
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
