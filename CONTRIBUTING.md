# Contributing

Open one pull request per provider folder. CrewAI maintainers merge into this repository. Merging here does not publish the tool or add it to CrewAI itself.

## What to include

```text
community_tools/<provider>/
  tool.py
  requirements.txt
  example.py
```

- Subclass `crewai.tools.BaseTool` in `tool.py`.
- Set `name`, `description`, and `args_schema`.
- Implement `_run`.
- Write `description` so an agent can tell when to call the tool.
- List dependencies in `requirements.txt`.
- In `example.py`, call the tool directly, then pass that same instance to an agent.
- Read secrets from environment variables. Name them in a comment at the top of `example.py`. Do not commit keys, tokens, or `.env` files.

## What to leave out

- Extra files beyond `tool.py`, `requirements.txt`, and `example.py`.
- Changes to the main CrewAI repository.
- A shared registry or auto-discovery mechanism.
- Live API calls in CI.
- Vendored copies of `crewai` or `crewai-tools`.
- Package metadata, a workspace, or a published install path for the tool.

Use `community_tools/acme/` as the pattern to copy.
