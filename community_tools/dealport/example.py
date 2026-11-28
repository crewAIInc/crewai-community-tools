"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export DP_API_KEY=dp_live_...   # from https://app.dealport.com/connect?tab=keys
    python example.py

Environment variables:
    DP_API_KEY (or DEALPORT_API_KEY): required for Dealport API calls. New accounts start
        with a $5.00 free USD balance, and dealport_preview_companies costs $0.00.
    OPENAI_API_KEY or MODEL: the crew runs only when one of them is set.
"""

import os

from crewai import Agent, Crew, Process, Task

from tool import (
    DealportFindOwnerTool,
    DealportGetRecordTool,
    DealportPreviewCompaniesTool,
    DealportSearchBuildingsTool,
    DealportSearchCompaniesTool,
)


def direct_call(preview: DealportPreviewCompaniesTool) -> None:
    print("Direct call: free ($0.00) US private company market sizing")
    print(
        preview.run(
            state="OH",
            naics=["238220"],
            min_employees=15,
            independent=True,
        )
    )


def run_crew(
    preview: DealportPreviewCompaniesTool,
    search_companies: DealportSearchCompaniesTool,
    search_buildings: DealportSearchBuildingsTool,
    get_record: DealportGetRecordTool,
    find_owner: DealportFindOwnerTool,
) -> None:
    agent = Agent(
        role="Private market research analyst",
        goal="Size target US SMB markets and summarize sample companies with public-filing citations",
        backstory=(
            "You start every search with dealport_preview_companies ($0.00) to confirm "
            "market count before pulling paid company or building Records."
        ),
        tools=[preview, search_companies, search_buildings, get_record, find_owner],
        verbose=True,
    )
    task = Task(
        description=(
            "Size the independent Ohio HVAC and plumbing market (NAICS 238220) with at least "
            "15 employees using dealport_preview_companies, and summarize the total count and "
            "sample companies returned."
        ),
        expected_output="Matching company count and a concise summary of the sample companies.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    print("Crew result")
    print(crew.kickoff())


def main() -> None:
    if not os.environ.get("DP_API_KEY") and not os.environ.get("DEALPORT_API_KEY"):
        print(
            "Set DP_API_KEY (from https://app.dealport.com/connect?tab=keys) and run this file again."
        )
        return

    preview = DealportPreviewCompaniesTool()
    search_companies = DealportSearchCompaniesTool()
    search_buildings = DealportSearchBuildingsTool()
    get_record = DealportGetRecordTool()
    find_owner = DealportFindOwnerTool()
    direct_call(preview)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print()
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(preview, search_companies, search_buildings, get_record, find_owner)


if __name__ == "__main__":
    main()
