# examples/ollama_with_tools.py

import asyncio

from langchain_core.messages import HumanMessage

from app.core.langgraph.graph import configure_graph


async def main():
    graph = await configure_graph()
    initial_state = {
        "messages": [HumanMessage(content="Who were the top 5 scorers in the NBA last season?")]
    }
    async for event in graph.astream(initial_state):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
