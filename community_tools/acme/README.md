# Acme catalog lookup

Reference community tool. It looks up a product in a built-in sample catalog. No API key is required.

Use this folder as the pattern for a real provider tool. Replace the in-memory catalog with the provider SDK, and read credentials from the environment.

## Install

From the repository root:

```bash
pip install -e "community_tools/acme"
```

## Use the tool directly

```python
from acme_crewai_tool import AcmeCatalogLookupTool

tool = AcmeCatalogLookupTool()
print(tool.run(query="ACME-100"))
```

## Pass it to an agent

```python
from crewai import Agent
from acme_crewai_tool import AcmeCatalogLookupTool

agent = Agent(
    role="Catalog assistant",
    goal="Answer questions about Acme sample products",
    backstory="You look up products with the Acme catalog tool.",
    tools=[AcmeCatalogLookupTool()],
    verbose=True,
)
```

Running an agent also needs a configured LLM. `examples/basic.py` only calls the tool, so it does not need one.

## Environment variables

None. A real provider tool should list every variable it reads, for example `ACME_API_KEY`.
