"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    export VEILLE_API_KEY="your-api-key"   # get one at https://app.veille.io
    python example.py

Environment variables:
- VEILLE_API_KEY (required): Veille API key. Each check uses Veille credits.
- OPENAI_API_KEY, or MODEL for another provider: only for the crew part.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from tool import (
    VeilleDomainCheckTool,
    VeilleEmailCheckTool,
    VeilleIBANCheckTool,
    VeilleIPCheckTool,
    VeillePhoneCheckTool,
    VeilleVATCheckTool,
)


def direct_call(tools: list[BaseTool]) -> None:
    print("Direct call")
    # VeilleEmailCheckTool
    print(tools[0].run(email="test@mailinator.com"))
    print()
    # VeilleDomainCheckTool
    print(tools[1].run(domain="example.com"))
    print()


def run_crew(tools: list[BaseTool]) -> None:
    agent = Agent(
        role="Sign-up reviewer",
        goal="Flag fake or risky sign-ups",
        backstory=("You check every email, domain and phone number with Veille before you decide."),
        tools=tools,
        verbose=True,
    )
    task = Task(
        description=(
            "A new account signed up with the email test@mailinator.com. Check it and say whether to accept, review or block the sign-up, and why."
        ),
        expected_output="Accept, review or block, with the reasons from the checks.",
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
    if not os.environ.get("VEILLE_API_KEY"):
        print("Set VEILLE_API_KEY first. Get a key at https://app.veille.io")
        return

    tools: list[BaseTool] = [
        VeilleEmailCheckTool(),
        VeilleDomainCheckTool(),
        VeilleIPCheckTool(),
        VeillePhoneCheckTool(),
        VeilleVATCheckTool(),
        VeilleIBANCheckTool(),
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
