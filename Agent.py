import asyncio
import json

from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from tools import create_application_lookup_tool, create_university_knowledge_tool


async def main():
    knowledge_tool = create_university_knowledge_tool()
    application_tool = create_application_lookup_tool()

    agent = FunctionAgent(
        tools=[knowledge_tool, application_tool],
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
        system_prompt=(
            "You are a University of Hamburg study assistant. "
            "Use university_knowledge for questions about IAS study rules, modules, "
            "study administration, and student support. "
            "Use lookup_application for the status of a specific application. "
            "If an application-status question has no application ID, ask for it. "
            "Do not invent university information or application details."
        ),
    )

    # Reuse the context so the agent remembers previous turns in this session.
    context = Context(agent)
    print("University assistant ready. Type 'exit' or 'quit' to stop.")

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
