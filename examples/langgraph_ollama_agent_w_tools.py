"""
location: examples/langgraph_ollama_agent_w_tools.py
"""
import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)

from app.core.config import settings

# Import the LLM provider from our app
from app.core.langgraph.llm_provider import get_llm

# ── 1) Simplified port configuration - always use port 8000 ───────────────────
MCP_PORT = int(os.getenv("NBA_MCP_PORT", "8000"))
MCP_URL  = f"http://localhost:{MCP_PORT}/mcp/messages"


# Load environment variables
load_dotenv()

# Configure the provider in settings if needed from environment variables
if "OLLAMA_HOST" in os.environ:
    settings.ollama_base_url = os.environ["OLLAMA_HOST"]
    
# Use our modular LLM provider
llm = get_llm()

def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False



def get_system_prompt() -> str:
    return (
        f"Today is {datetime.now():%Y-%m-%d}.\n"
        "Tools you can call:\n"
        "- get_league_leaders_info(season, stat_category, per_mode)\n"
        "    • per_mode must be one of: 'Totals', 'PerGame', 'Per48'\n"
        "    • e.g.: get_league_leaders_info('2024-25','AST','PerGame')\n"
        "- get_player_career_information(player_name, season)\n"
        "- get_live_scores(target_date)\n"
        "- play_by_play_info_for_current_games()\n"
        "When you want data, emit a tool call. Otherwise, answer directly."
    )


def create_chatbot_node(llm_instance, tools):
    """
    llm_instance: ChatOllama bound to your MCP tools
    tools: list of tool objects
    """
    async def chatbot(state: MessagesState):
        msgs = state["messages"]
        full = [AIMessage(content=get_system_prompt())] + msgs
        response = await llm_instance.ainvoke(full)
        # DEBUG: check that we got structured tool_calls
        print("DEBUG tool_calls:", getattr(response, "tool_calls", None), file=sys.stderr)
        return {"messages": msgs + [response]}
    return chatbot

async def async_tool_executor(state):
    messages = state["messages"]
    last     = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {"messages": messages}

    new_msgs = messages.copy()
    for tc in tool_calls:
        # 1) Normalize call into (name, args, call_id)
        if isinstance(tc, dict):
            name    = tc.get("name")
            args    = tc.get("args", {}) or {}
            call_id = tc.get("id")
        else:
            name    = tc.name
            args    = tc.args or {}
            call_id = tc.id

        # 2) Lookup the tool by name
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            new_msgs.append(
                AIMessage(content=f"Unknown tool {name}, available: {[t.name for t in tools]}")
            )
            continue

        # 3) Execute the tool, sync or async
        try:
            if call_id and hasattr(tool, "coroutine") and asyncio.iscoroutinefunction(tool.coroutine):
                result = await tool.coroutine(**args)
            else:
                result = tool.func(**args) if hasattr(tool, "func") else tool(**args)

            new_msgs.append(
                ToolMessage(content=str(result), tool_call_id=call_id, name=name)
            )
        except Exception as e:
            new_msgs.append(
                AIMessage(content=f"Error running {name}: {e}")
            )

    return {"messages": new_msgs}


async def main():
    try:
        async with MultiServerMCPClient({
            "nba": {"url": MCP_URL, "transport": "sse", "timeout": 30}
        }) as client:
            # — DEBUG: did we actually connect?
            print("DEBUG: MultiServerMCPClient connected, listing resources…", file=sys.stderr)
            try:
                res = await client.list_resources()
                print("DEBUG: Available resources:", res, file=sys.stderr)
            except Exception:
                print("DEBUG: list_resources() failed:", file=sys.stderr)
                traceback.print_exc()

            global tools
            tools = client.get_tools()

            # 1) bind LLM to the tools
            llm_with_tools = llm.bind_tools(tools)

            print("Loaded tools:", [t.name for t in tools])

            # 2) wire up the graph, passing in the bound LLM
            builder = StateGraph(MessagesState)
            builder.add_node("chatbot", create_chatbot_node(llm_with_tools, tools))
            builder.add_node("tools",  async_tool_executor)

            def router(state):
                last = state["messages"][-1]
                return "tools" if getattr(last, "tool_calls", []) else END

            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", router, {"tools": "tools", END: END})
            builder.add_edge("tools", "chatbot")
            graph = builder.compile()

            print("Enter a question:")
            state = {"messages": [HumanMessage(content=input("> "))]}
            result = await graph.ainvoke(state)
            for msg in result["messages"]:
                print(f"{msg.__class__.__name__}: {msg.content}")

    except Exception as e:
        print("❌ Agent startup failed with traceback:", file=sys.stderr)
        traceback.print_exc()
        return

if __name__ == "__main__":
    # 3) Launch the MCP with the standardized endpoint 
    print(f"✓ Assuming an existing server on port {MCP_PORT}")
    # No in-script Uvicorn spawn: skip straight to the agent
    server_proc = None

    try:
        print("Langgraph agent starting…")
        asyncio.run(main())
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()