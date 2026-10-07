import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from tools import create_order_lookup_tool, create_shop_policy_tool


async def main():
    policy_tool = create_shop_policy_tool()
    order_tool = create_order_lookup_tool()

    agent = FunctionAgent(
        tools=[policy_tool, order_tool],
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
        system_prompt=(
            "You are a shop support assistant. "
            "Use the policy tool for questions about shop policies. "
            "Use lookup_order for information about a specific order. "
            "If an order question has no order ID, ask for it. "
            "Do not invent policies or order details."
        ),
    )

    response = await agent.run(
        user_msg="What is the status of order A101?"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())