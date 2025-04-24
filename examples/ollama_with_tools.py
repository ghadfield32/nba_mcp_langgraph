# examples/ollama_with_tools.py

import asyncio
import logging

from langchain_core.messages import HumanMessage

from app.core.langgraph.graph import configure_graph
from app.core.logging import logger


async def main():
    # Set up logging to see tool execution details
    logger.setLevel(logging.INFO)
    
    # Configure the graph with MCP tools
    graph = await configure_graph()
    
    # Create an initial state with a test question
    initial_state = {
        "messages": [HumanMessage(content="what was the play by play of the heat game 04/23/2025?")]
    }
    
    # Stream the response
    try:
        logger.info("Starting graph execution")
        async for event in graph.astream(initial_state):
            # Print the event type and relevant details
            event_type = list(event.keys())[0] if event else "Unknown"
            if event_type == "chat":
                messages = event["chat"]["messages"]
                for msg in messages:
                    logger.info(f"Received chat message: {msg}")
                    print(f"Chat: {msg.content}")
                    
                    # Check for tool calls in the message
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            logger.info(f"Tool call detected: {tool_call}")
            
            elif event_type == "tool_call":
                messages = event["tool_call"]["messages"]
                for msg in messages:
                    logger.info(f"Tool result: {msg}")
                    print(f"Tool result: {msg.content[:100]}...")
            else:
                logger.info(f"Event: {event}")
                print(event)
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
