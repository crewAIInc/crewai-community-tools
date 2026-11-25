"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export WEBCITE_API_KEY=...   # from https://webcite.co/api-keys
    python example.py

Environment variables:
    WEBCITE_API_KEY  required for the direct calls; each call spends credits.
    OPENAI_API_KEY or MODEL  the crew runs only when one of them is set.
"""

import os

from crewai import Agent, Crew, Process, Task

from tool import (
    WebciteExtractDocumentTool,
    WebciteSearchSourcesTool,
    WebciteVerifyClaimTool,
)

CLAIM = "The Great Wall of China is visible from the Moon with the naked eye."
DOCUMENT = "https://www.irs.gov/pub/irs-pdf/fw9.pdf"


def direct_call(
    verify: WebciteVerifyClaimTool,
    extract: WebciteExtractDocumentTool,
    search: WebciteSearchSourcesTool,
) -> None:
    print("Verify a claim")
    print(verify.run(claim=CLAIM))
    print()
    print("Read a document into Markdown (first 500 characters)")
    print(extract.run(source=DOCUMENT)[:500])
    print()
    print("Search sources")
    print(search.run(query="Great Wall of China visibility from space", limit=3))


def run_crew(verify: WebciteVerifyClaimTool, search: WebciteSearchSourcesTool) -> None:
    agent = Agent(
        role="Fact checker",
        goal="Say whether each claim holds, citing the sources Webcite returns",
        backstory=(
            "You check claims before they are published. You call webcite_verify_claim "
            "for each claim and report the verdict with the cited URLs."
        ),
        tools=[verify, search],
        verbose=True,
    )
    task = Task(
        description=f"Check this claim and report the verdict with sources: {CLAIM}",
        expected_output="The verdict, a one-sentence reason and the cited URLs.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    print("Crew result")
    print(crew.kickoff())


def main() -> None:
    if not os.environ.get("WEBCITE_API_KEY"):
        print(
            "Set WEBCITE_API_KEY (from https://webcite.co/api-keys) and run this file again."
        )
        return

    verify = WebciteVerifyClaimTool()
    extract = WebciteExtractDocumentTool()
    search = WebciteSearchSourcesTool()
    direct_call(verify, extract, search)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print()
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(verify, search)


if __name__ == "__main__":
    main()
