"""Smallest usage example for the Acme community tool."""

from acme_crewai_tool import AcmeCatalogLookupTool


def main() -> None:
    tool = AcmeCatalogLookupTool()
    print(tool.run(query="ACME-100"))
    print()
    print(tool.run(query="gadget"))
    print()
    print(tool.run(query="missing"))


if __name__ == "__main__":
    main()
