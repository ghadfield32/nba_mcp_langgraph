# examples/ollama_with_tools_langgraphagent.py

import asyncio
import logging
import time

from langchain_core.messages import HumanMessage

from app.core.langgraph.graph import LangGraphAgent
from app.core.logging import logger


async def main():
    # Set DEBUG level for both app logger and root logger for maximum visibility
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create the agent
    logger.info("Initializing LangGraphAgent")
    agent = LangGraphAgent()
    
    # Give some time for MCP tools to load asynchronously
    logger.info("Waiting for MCP tools to initialize...")
    await asyncio.sleep(2)  # Wait 2 seconds for tool initialization
    
    session_id = "demo-session-1"
    queries = [
        HumanMessage(content="Search: what's the weather in San Francisco?"),
        HumanMessage(content="What were the top 5 players in the points per game in the NBA in 2024-25?")  # More specific NBA query
    ]

    # Ensure tools are loaded
    if not agent._mcp_tools:
        logger.warning("MCP tools not loaded yet, waiting longer...")
        for _ in range(5):  # Try for up to 5 more seconds
            await asyncio.sleep(1)
            if agent._mcp_tools:
                logger.info(f"MCP tools loaded: {list(agent._mcp_tools.keys())}")
                break
        else:
            logger.error("Failed to load MCP tools after waiting")
    else:
        logger.info(f"MCP tools already loaded: {list(agent._mcp_tools.keys())}")
    
    # Create graph if needed
    if agent._graph is None:
        logger.info("Creating LangGraph")
        await agent.create_graph()
    
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

