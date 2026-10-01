# CrewAI community tools

Examples of third-party tools that plug into [CrewAI](https://github.com/crewAIInc/crewAI).

CrewAI maintainers review and merge changes here. This repository is not part of the `crewai` or `crewai-tools` packages. Nothing in this repo is published to PyPI from these workflows, listed in the main CrewAI docs, or loaded automatically by the framework.

## How a tool is added

Each provider gets one folder under `community_tools/`:

```text
community_tools/
  <provider>/
    README.md
    pyproject.toml
    src/<provider>_crewai_tool/
      __init__.py
      tool.py
    examples/
      basic.py
```

`tool.py` subclasses `crewai.tools.BaseTool`. `examples/basic.py` shows the smallest way to call that tool from outside the provider package.

See `community_tools/acme/` for a complete reference. It uses a local sample catalog, so it runs without API keys.

## Run the reference example

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "community_tools/acme[examples]"
python community_tools/acme/examples/basic.py
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
