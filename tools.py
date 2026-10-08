"""Expose University of Hamburg knowledge and application lookup tools."""

import json
from pathlib import Path

from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from Rag import DATA_DIR, build_university_index


def create_university_knowledge_tool(data_dir: Path = DATA_DIR) -> QueryEngineTool:
    """Build a tool that answers questions using University of Hamburg documents."""
    index = build_university_index(data_dir)
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            llm=OpenAI(model="gpt-4o-mini", temperature=0),
            similarity_top_k=3,
        ),
        name="university_knowledge",
        description=(
            "Search the provided University of Hamburg documents for study rules, "
            "modules, administration, support, and program information."
        ),
    )


def create_application_lookup_tool(data_dir: Path = DATA_DIR) -> FunctionTool:
    def lookup_application(application_id: str) -> dict:
        """Look up a university application by its exact application ID."""
        with (data_dir / "applications.json").open(encoding="utf-8") as file:
            applications = json.load(file)
        return applications.get(
            application_id.upper(),
            {"error": "Application not found"},
        )

    return FunctionTool.from_defaults(
        fn=lookup_application,
        name="lookup_application",
        description=(
            "Look up an applicant's current application status using an exact "
            "application ID."
        ),
    )


if __name__ == "__main__":
    knowledge_tool = create_university_knowledge_tool()
    application_tool = create_application_lookup_tool()
    print(knowledge_tool.call("When is the application deadline?"))
    print(application_tool.call(application_id="APP1001"))
