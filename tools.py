"""Expose policy retrieval and order lookup tools for the shop."""

import json
from pathlib import Path

from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI
from Rag import DATA_DIR, build_policy_index


def create_shop_policy_tool(data_dir: Path = DATA_DIR) -> QueryEngineTool:
    """Build a tool that answers questions about the shop's policies."""
    index = build_policy_index(data_dir)
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            llm=OpenAI(model="gpt-4o-mini", temperature=0),
            similarity_top_k=3,
        ),
        name="query_engine",
        description="A query engine for retrieving information about the company's policies.",
    )

def create_order_lookup_tool(data_dir: Path = DATA_DIR) -> FunctionTool:
    def lookup_order(order_id: str) -> dict:
        """Look up a shop order by its exact order ID."""
        with (DATA_DIR / "orders.json").open(encoding="utf-8") as file:
            orders_data = json.load(file)
        return orders_data.get(order_id, {"error": "Order not found"})

    return FunctionTool.from_defaults(
        fn=lookup_order,
        name="lookup_order",
        description="A function for looking up a shop order by its exact order ID.",
    )


if __name__ == "__main__":
    shop_policy_tool = create_shop_policy_tool()
    order_lookup_tool = create_order_lookup_tool()
    print(shop_policy_tool.call("What is the return window?"))
    print(order_lookup_tool.call(order_id="A101"))
