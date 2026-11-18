"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export ASIN_API_KEY="your-api-key"   # get one at https://app.asin.dev
    python example.py

Environment variables:
- ASIN_API_KEY (required): Asin API key. Searches and price histories are free; reading
  live offers uses Asin credits.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import AsinAmazonSearchTool, AsinOffersTool, AsinPriceHistoryTool


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # AsinAmazonSearchTool
    print(tools[0].run(query="mechanical keyboard"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Shopping assistant",
        goal="Tell whether an Amazon price is a good deal",
        backstory=(
            "You look up the product with Asin and compare its current price with its history before you answer."
        ),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "Find a popular mechanical keyboard on Amazon and say whether its current price is a good deal compared with its price history."
        ),
        expected_output="The product, its current price, its price history range and a verdict.",
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
    if not os.environ.get("ASIN_API_KEY"):
        print("Set ASIN_API_KEY first. Get a key at https://app.asin.dev")
        return

    tools: list[BaseTool] = [AsinAmazonSearchTool(), AsinPriceHistoryTool(), AsinOffersTool()]
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
