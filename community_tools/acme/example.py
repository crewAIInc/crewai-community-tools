"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    python example.py

No API key is required for the direct calls. The crew runs only when
OPENAI_API_KEY or MODEL is already set. A real provider lists its own
environment variables here, for example ACME_API_KEY.
"""

import os

from crewai import Agent, Crew, Process, Task

from tool import AcmeCatalogLookupTool


def direct_call(tool: AcmeCatalogLookupTool) -> None:
    print("Direct call")
    print(tool.run(query="ACME-100"))
    print()
    print(tool.run(query="gadget"))
    print()
    print(tool.run(query="missing"))


def run_crew(tool: AcmeCatalogLookupTool) -> None:
    agent = Agent(
        role="Catalog assistant",
        goal="Answer product questions using only the Acme catalog tool",
        backstory=(
            "You work a sample catalog. You call acme_catalog_lookup "
            "before you answer, and you say when a product is not in the catalog."
        ),
        tools=[tool],
        verbose=True,
    )
    task = Task(
        description=(
            "A customer asks two things: the price and stock of ACME-100, "
            "and whether you carry a gadget. Reply with both answers."
        ),
        expected_output="Price and stock for ACME-100, plus the gadget result.",
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
    tool = AcmeCatalogLookupTool()
    direct_call(tool)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print()
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(tool)


if __name__ == "__main__":
    main()
