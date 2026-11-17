# CrewAI community tools

Examples of third-party tools that plug into [CrewAI](https://github.com/crewAIInc/crewAI).

CrewAI maintainers review and merge changes here. This repository is not part of the `crewai` or `crewai-tools` packages. Nothing in this repo is published to PyPI from these workflows, listed in the main CrewAI docs, or loaded automatically by the framework.

## How a tool is added

Each provider gets one folder under `community_tools/`:

```text
community_tools/
  <provider>/
    tool.py
    requirements.txt
    example.py
```

`tool.py` subclasses `crewai.tools.BaseTool`. `requirements.txt` lists what that tool needs. `example.py` shows a direct call, then the same tool passed to an agent.

See `community_tools/acme/` for the reference. Acme is a stand-in provider with a local catalog. Copy the folder into your project and install from its `requirements.txt`. This repository does not publish the tool.

## Run the reference example

```bash
cd community_tools/acme
pip install -r requirements.txt
python example.py
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
