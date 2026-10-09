"""Expose University of Hamburg knowledge and live web search tools."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI as OpenAIClient

from rag import DATA_DIR, build_university_index


def create_university_knowledge_tool(data_dir: Path = DATA_DIR) -> QueryEngineTool:
    """Build a tool that answers questions using University of Hamburg documents."""
    index = build_university_index(data_dir)
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            llm=LlamaOpenAI(model="gpt-4o-mini", temperature=0),
            similarity_top_k=3,
        ),
        name="university_knowledge",
        description=(
            "Search the provided University of Hamburg documents for study rules, "
            "modules, administration, course registration dates, support, and "
            "program information."
        ),
    )


def create_university_web_search_tool() -> FunctionTool:
    def search_university_web(query: str) -> dict:
        """Search live official University of Hamburg webpages for current information."""
        load_dotenv(DATA_DIR.parent / ".env")
        client = OpenAIClient()
        current_date = datetime.now(ZoneInfo("Europe/Berlin")).date().isoformat()

        try:
            response = client.responses.create(
                model="gpt-5.5",
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": ["uni-hamburg.de"]},
                        "search_context_size": "medium",
                    }
                ],
                tool_choice="required",
                include=["web_search_call.action.sources"],
                input=(
                    f"Today is {current_date}. Search only official University of "
                    "Hamburg websites for the following question. Give a concise "
                    "answer and cite the relevant URLs. If the official sources do "
                    f"not answer it, say so. Question: {query}"
                ),
            )
        except Exception as error:
            return {
                "error": "University web search failed",
                "details": str(error),
            }

        response_data = response.model_dump()
        sources = []
        for item in response_data.get("output", []):
            if item.get("type") != "web_search_call":
                continue
            for source in (item.get("action") or {}).get("sources", []):
                url = source.get("url")
                if url and url not in sources:
                    sources.append(url)

        return {
            "answer": response.output_text,
            "sources": sources,
            "searched_on": current_date,
        }

    return FunctionTool.from_defaults(
        fn=search_university_web,
        name="search_university_web",
        description=(
            "Search live official University of Hamburg webpages for current or "
            "time-sensitive information. Use this when saved university documents "
            "may be outdated or do not contain the answer."
        ),
    )


if __name__ == "__main__":
    knowledge_tool = create_university_knowledge_tool()
    web_search_tool = create_university_web_search_tool()
    print(knowledge_tool.call("How many credits does the IAS thesis have?"))
    print(web_search_tool.call(query="What are the current registration periods?"))
