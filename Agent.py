import asyncio
import json

from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
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

    # Reuse the context so the agent remembers previous turns in this session.
    context = Context(agent)
    print("Shop assistant ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        handler = agent.run(user_msg=question, ctx=context)
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                print(f"\nTool called: {event.tool_name}", flush=True)
                print(
                    f"Arguments: {json.dumps(event.tool_kwargs, ensure_ascii=False)}",
                    flush=True,
                )
            elif isinstance(event, ToolCallResult):
                label = "Tool error" if event.tool_output.is_error else "Tool result"
                print(f"\n{label} ({event.tool_name}):", flush=True)
                print(event.tool_output.content, flush=True)

        response = await handler
        print(f"\nAgent answer:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
