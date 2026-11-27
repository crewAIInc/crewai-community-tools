"""Copy this folder and run the example from inside it.

    pip install -r requirements.txt
    python example.py

Environment variables:
    HARNY_API_KEY   Harny app key (starts with fsa_). No account yet: `pip install harny`, then
                    `harny init`; it creates an app and saves the key to .env without printing it.
    OPENAI_API_KEY  or MODEL, only for the crew at the end; the direct calls need no model key.
"""

import os
import time
import uuid

from crewai import Agent, Crew, Process, Task

from tool import HarnyRecallTool, HarnyRememberTool

# Your own id for the person. Each person's memory is kept apart from everyone else's.
PERSON = os.environ.get("HARNY_EXAMPLE_USER") or f"example-{uuid.uuid4().hex[:8]}"


def direct_call(remember: HarnyRememberTool, recall: HarnyRecallTool) -> None:
    print("Direct call")
    print(remember.run(content="Nadia is allergic to peanuts.", label="allergy"))
    print(remember.run(content="Nadia's dog is called Biscuit."))
    print(
        remember.run(content="Nadia is allergic to peanuts.", label="allergy")
    )  # same words: one fact
    for _ in range(15):  # a new fact can take a few seconds to be recallable
        answer = recall.run(query="What should Nadia avoid eating?")
        if not answer.startswith("Nothing is remembered"):
            break
        time.sleep(2)
    print(answer)


def run_crew(remember: HarnyRememberTool, recall: HarnyRecallTool) -> None:
    agent = Agent(
        role="Personal assistant",
        goal="Help one person, using what they told you in earlier conversations",
        backstory=(
            "You call harny_recall before you answer anything that depends on what the person said "
            "before, and harny_remember when they tell you something worth keeping."
        ),
        tools=[recall, remember],
        verbose=True,
    )
    task = Task(
        description=(
            "Nadia says: 'I'm packing snacks for a walk with my dog. What should I avoid, "
            "and what was my dog's name again?' Reply to her."
        ),
        expected_output="A short reply that names what to avoid and the dog's name.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    print("Crew result")
    print(crew.kickoff())


def main() -> None:
    recall = HarnyRecallTool(user=PERSON)
    remember = HarnyRememberTool(user=PERSON)
    direct_call(remember, recall)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MODEL"):
        print()
        print(
            "Crew wiring is in run_crew(). Set OPENAI_API_KEY, or MODEL for "
            "another provider, then run this file again to kick off the crew."
        )
        return

    print()
    run_crew(remember, recall)


if __name__ == "__main__":
    main()
