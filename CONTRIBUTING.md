# Contributing

Open one pull request per provider folder. CrewAI maintainers merge into this repository. Merging here does not publish the tool or add it to CrewAI itself.

## What to include

```text
community_tools/<provider>/
  README.md
  pyproject.toml
  src/<provider>_crewai_tool/__init__.py
  src/<provider>_crewai_tool/tool.py
  examples/basic.py
```

- Subclass `crewai.tools.BaseTool`.
- Set `name`, `description`, and `args_schema`.
- Implement `_run`.
- Write `description` so an agent can tell when to call the tool.
- Read secrets from environment variables. Do not commit keys, tokens, or `.env` files.
- Document required environment variables and a local install command in the provider README.
- Keep `examples/basic.py` small enough to show construction and one call.

## What to leave out

- Changes to the main CrewAI repository.
- A shared registry or auto-discovery mechanism.
- Live API calls in CI.
- Vendored copies of `crewai` or `crewai-tools`.

Use `community_tools/acme/` as the pattern to copy.
