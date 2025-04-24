# examples/ollama_with_tools_langgraphagent.py

import asyncio
import logging

from langchain_core.messages import HumanMessage

from app.core.langgraph.graph import LangGraphAgent
from app.core.logging import logger


async def main():
    logger.setLevel(logging.INFO)
    agent = LangGraphAgent()
    session_id = "demo-session-1"
    queries = [
        HumanMessage(content="Search: what’s the weather in San Francisco?"),
        HumanMessage(content="What games are on today?")
    ]

    print("=== Streaming response ===")
    async for token in agent.get_stream_response(queries, session_id):
        print(token, end="", flush=True)
    print("\n=== End of stream ===")

    history = await agent.get_chat_history(session_id)
    print("\n=== Chat history ===")
    for msg in history:
        print(f"{msg.role}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())

