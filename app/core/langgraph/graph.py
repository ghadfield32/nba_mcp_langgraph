"""This file contains the LangGraph Agent/workflow and interactions with the LLM.

location: app\core\langgraph\graph.py
"""


import ast
import asyncio
import inspect
import json
import sys
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.llm_provider import get_llm
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.prompts import SYSTEM_PROMPT
from app.schemas import (
    GraphState,
    Message,
)
from app.services.mcp.nba_mcp.nba_server import mcp_server
from app.utils import (
    dump_messages,
    prepare_messages,
)

# Ensure psycopg compatibility on Windows for any late pool creation
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
async def get_mcp_tools():
    """Get tools from the MCP NBA server.
    
    Returns:
        List of MCP tools for the LangGraph agent.
    """
    return await mcp_server.get_tools()


async def configure_graph():
    """Configure a LangGraph workflow with NBA MCP tools."""

    # 1) Load the LLM
    llm = get_llm()
    
    # 2) Fetch and normalize MCP tools
    raw_mcp = await get_mcp_tools()
    if isinstance(raw_mcp, dict):
        mcp_tools = list(raw_mcp.values())
    else:
        mcp_tools = raw_mcp
    logger.info(f"Loaded {len(mcp_tools)} MCP tools for standalone graph")
    
    # 3) Convert MCP tools into OpenAI‐compatible function definitions
    mcp_fn_defs = []
    for t in mcp_tools:
        schema = getattr(t, "parameters", {})  # should be a dict with 'properties' & 'required'
        
        # Debug: Log the original tool schema
        tool_name = getattr(t, "name", "unknown")
        logger.info(f"Processing tool schema for: {tool_name}")
        logger.info(f"Original schema: {schema}")
        
        # Special handling for get_league_leaders_info tool
        if tool_name == "get_league_leaders_info":
            logger.info("Applying special schema fix for get_league_leaders_info")
            # Check if params is in the properties
            if "params" in schema.get("properties", {}):
                # This is correct, use as is
                logger.info("Schema already has 'params' property, keeping as is")
            else:
                # Need to restructure schema to expect a "params" object
                # that contains all the parameters for LeagueLeadersParams
                logger.info("Restructuring schema to use 'params' wrapper")
                schema = {
                    "properties": {
                        "params": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", [])
                        }
                    },
                    "required": ["params"]
                }
                logger.info(f"Updated schema: {schema}")
        
        mcp_fn_defs.append({
            "name": tool_name,
            "description": getattr(t, "description", "") or "",
            "parameters": {
                "title": tool_name,
                "description": getattr(t, "description", "") or "",
                "type": "object",
                **schema
            }
        })

    # 4) Combine your built‐in tools (already OpenAI‐compatible) + MCP function defs
    fn_defs = tools + mcp_fn_defs
    bound_llm = llm.bind_tools(fn_defs)
    
    # 5) Build lookup by name for *actual* tool objects
    tools_by_name = {tool.name: tool for tool in tools}
    for t in mcp_tools:
        tools_by_name[t.name] = t
    
    # 6) Define chat node
    async def chat(state):
        messages = prepare_messages(state["messages"], bound_llm, SYSTEM_PROMPT)
        try:
            return {"messages": [await bound_llm.ainvoke(dump_messages(messages))]}
        except Exception as e:
            logger.error(f"Error in chat node: {e}")
            raise

    # 7) Define tool call node
    async def tool_call(state):
        """
        Process the last message's tool_calls, introspect each tool's signature,
        wrap args into Pydantic models when needed, or pass as keyword args.
        """
        outputs = []
        for call in state["messages"][-1].tool_calls:
            tool_name = call["name"]
            args      = call["args"]  # e.g. {"params": {...}}
            tool      = tools_by_name.get(tool_name)

            logger.info(f"[DEBUG] Calling tool '{tool_name}' with args keys: {list(args.keys())}")

            # Special case: If args has a 'params' key, parse it if it's a string
            if "params" in args and isinstance(args["params"], str):
                raw_params = args["params"]
                logger.info("[DEBUG] Found 'params' key with string value, attempting to parse")
                try:
                    # First try JSON parsing
                    args["params"] = json.loads(raw_params)
                    logger.info("[DEBUG] Parsed JSON string to dict")
                except json.JSONDecodeError:
                    try:
                        # If JSON fails, try Python literal_eval for Python dict-like strings
                        args["params"] = ast.literal_eval(raw_params)
                        logger.info("[DEBUG] Parsed Python dict-like string using ast.literal_eval")
                    except (SyntaxError, ValueError):
                        logger.error(f"[DEBUG] Failed to parse params string: {raw_params}")
                        # Keep as string - will likely fail but not our fault

            # Special case for get_league_leaders_info: adjust field names if needed
            if tool_name == "get_league_leaders_info" and "params" in args and isinstance(args["params"], dict):
                params = args["params"]
                # Map LLM generated fields to expected field names
                field_mapping = {
                    "stat": "stat_category",
                    "mode": "per_mode",
                    "points": "PTS",
                    "per_game": "PerGame",
                    "totals": "Totals",
                    "per48": "Per48"
                }
                
                # Create a new params dict with correctly mapped field names
                fixed_params = {}
                
                # Handle stat_category (required)
                if "stat_category" in params:
                    fixed_params["stat_category"] = params["stat_category"]
                elif "stat" in params:
                    stat_value = params["stat"].upper() if params["stat"].lower() != "points" else "PTS"
                    fixed_params["stat_category"] = stat_value
                
                # Handle per_mode (required)
                if "per_mode" in params:
                    fixed_params["per_mode"] = params["per_mode"]
                elif "mode" in params:
                    mode_value = params["mode"]
                    if mode_value.lower() == "per_game":
                        fixed_params["per_mode"] = "PerGame"
                    elif mode_value.lower() == "totals":
                        fixed_params["per_mode"] = "Totals"
                    elif mode_value.lower() == "per48":
                        fixed_params["per_mode"] = "Per48"
                
                # Handle season (optional)
                if "season" in params:
                    season_value = params["season"]
                    if season_value.lower() == "last" or season_value.lower() == "latest":
                        # Current NBA season is 2024-25
                        fixed_params["season"] = "2024-25"
                    else:
                        fixed_params["season"] = season_value
                
                logger.info(f"[DEBUG] Mapped params from {params} to {fixed_params}")
                args["params"] = fixed_params

            # Inspect the run() signature
            sig = inspect.signature(tool.run)
            # Skip the 'self' parameter
            param_items = list(sig.parameters.items())[1:]
            logger.info(f"[DEBUG] tool.run signature parameters: {[n for n,_ in param_items]}")

            # If exactly one parameter: either a Pydantic model or a JSON payload
            if len(param_items) == 1:
                name, param = param_items[0]
                ann = param.annotation

                # 1) Pydantic-model branch
                if inspect.isclass(ann) and issubclass(ann, BaseModel):
                    logger.info(f"[DEBUG] Wrapping args into Pydantic model {ann.__name__}")
                    # The LLM payload often nests all fields under "params"
                    payload = args.get(name, args)
                    if isinstance(payload, dict):
                        model = ann(**payload)
                    else:
                        # If it's a JSON string, let Pydantic parse it
                        model = ann.parse_raw(payload)
                    result = await tool.run(model)

                # 2) Single-arg MCP tool: For tools expecting 'params' object, pass that directly
                elif name == "context" and "params" in args:
                    logger.info("[DEBUG] Single-arg MCP tool with 'context' parameter, wrapping params in dict")
                    # Wrap params in a dict with 'params' key to match expected model structure
                    result = await tool.run({"params": args["params"]})
                # 3) Other single-arg tool
                else:
                    logger.info("[DEBUG] Single-arg MCP tool, passing full args dict")
                    result = await tool.run(args)

            # Multi-arg: expand as keywords
            else:
                logger.info("[DEBUG] Multi-arg tool, calling run(**args)")
                result = await tool.run(**args)

            outputs.append(
                ToolMessage(content=result, name=tool_name, tool_call_id=call["id"])
            )

        # Preserve entire conversation history so the next chat node has full context
        return {"messages": state["messages"] + outputs}


    # 8) Termination logic
    def should_continue(state):
        return "end" if not state["messages"][-1].tool_calls else "continue"
    
    # 9) Build the StateGraph
    graph = StateGraph(dict)
    graph.add_node("chat", chat)
    graph.add_node("tool_call", tool_call)
    graph.add_conditional_edges("chat", should_continue, {"continue": "tool_call", "end": END})
    graph.add_edge("tool_call", "chat")
    graph.set_entry_point("chat")
    graph.set_finish_point("chat")

    # 10) Compile & return
    return graph.compile(name=f"{settings.PROJECT_NAME} Example Agent")


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the modular LLM provider
        self.llm = get_llm().bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self._mcp_tools = None

        logger.info("llm_initialized", model=settings.model_name, environment=settings.app_env)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.app_env == Environment.DEVELOPMENT.value:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.app_env == Environment.PRODUCTION.value:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                self._connection_pool = AsyncConnectionPool(
                    settings.POSTGRES_URL,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.app_env)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.app_env)
                # In production, we might want to degrade gracefully
                if settings.app_env == Environment.PRODUCTION.value:
                    logger.warning("continuing_without_connection_pool", environment=settings.app_env)
                    return None
                raise e
        return self._connection_pool

    async def _chat(self, state: GraphState) -> dict:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            dict: Updated state with new messages.
        """
        messages = prepare_messages(state.messages, self.llm, SYSTEM_PROMPT)

        llm_calls_num = 0

        # Configure retry attempts based on environment
        max_retries = settings.max_llm_call_retries

        for attempt in range(max_retries):
            try:
                generated_state = {"messages": [await self.llm.ainvoke(dump_messages(messages))]}
                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.model_name,
                    environment=settings.app_env,
                )
                return generated_state
            except OpenAIError as e:
                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.app_env,
                )
                llm_calls_num += 1

                # In production, we might want to fall back to a more reliable model
                if settings.app_env == Environment.PRODUCTION.value and attempt == max_retries - 2:
                    # Only apply this fallback for OpenAI models
                    if settings.llm_provider.lower() == "openai":
                        fallback_model = "gpt-4o"
                        logger.warning(
                            "using_fallback_model", model=fallback_model, environment=settings.app_env
                        )
                        # Re-initialize the LLM with the fallback model
                        temp_settings = settings.model_copy(update={"model_name": fallback_model})
                        with settings._set_temporary(temp_settings):
                            self.llm = get_llm().bind_tools(tools)

                continue

        raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts")

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> GraphState:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Dict with updated messages containing tool responses.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            
            if tool_name in self.tools_by_name:
                tool = self.tools_by_name[tool_name]
            elif self._mcp_tools and tool_name in self._mcp_tools:
                tool = self._mcp_tools[tool_name]
            else:
                tool = None
                
            if tool is None:
                tool_result = f"Error: Tool '{tool_name}' not found"
                logger.error(f"Tool not found: {tool_name}")
            else:
                try:
                    # Handle different tool invocation patterns
                    if hasattr(tool, "ainvoke"):
                        tool_result = await tool.ainvoke(args)
                    elif hasattr(tool, "arun"):
                        tool_result = await tool.arun(**args)
                    elif hasattr(tool, "run_async"):
                        tool_result = await tool.run_async(args)
                    else:
                        # Fallback to synchronous run
                        tool_result = tool.run(**args)
                except Exception as e:
                    tool_result = f"Error executing tool '{tool_name}': {str(e)}"
                    logger.error(f"Tool execution error: {e}")
                
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def _should_continue(self, state: GraphState) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
        """
        messages = state.messages
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow with MCP tools."""

        if self._graph is None:
            # 1) Fetch and normalize MCP tools
            raw_mcp = await get_mcp_tools()
            mcp_tools = list(raw_mcp.values()) if isinstance(raw_mcp, dict) else raw_mcp
            self._mcp_tools = {t.name: t for t in mcp_tools}
            logger.info(f"Loaded {len(mcp_tools)} MCP tools")

            # 2) Build OpenAI‐compatible function defs for binding
            mcp_fn_defs = []
            for t in mcp_tools:
                schema = getattr(t, "parameters", {})
                mcp_fn_defs.append({
                    "name": t.name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": {
                        "title": t.name,
                        "description": getattr(t, "description", "") or "",
                        "type": "object",
                        **schema
                    }
                })

            # 3) Bind only function defs, not raw tool objects
            fn_defs = tools + mcp_fn_defs
            self.llm = get_llm().bind_tools(fn_defs)

            # 4) Update tools_by_name for actual execution
            self.tools_by_name = {tool.name: tool for tool in tools}
            self.tools_by_name.update(self._mcp_tools)

            # Define a wrapped tool call function that properly invokes tools
            async def wrapped_tool_call(state: GraphState) -> dict:
                outputs = []
                for tool_call in state.messages[-1].tool_calls:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    
                    if tool_name in self.tools_by_name:
                        tool = self.tools_by_name[tool_name]
                    elif self._mcp_tools and tool_name in self._mcp_tools:
                        tool = self._mcp_tools[tool_name]
                    else:
                        tool = None
                        
                    if tool is None:
                        tool_result = f"Error: Tool '{tool_name}' not found"
                        logger.error(f"Tool not found: {tool_name}")
                    else:
                        try:
                            # Handle different tool invocation patterns
                            if hasattr(tool, "ainvoke"):
                                tool_result = await tool.ainvoke(args)
                            elif hasattr(tool, "arun"):
                                tool_result = await tool.arun(**args)
                            elif hasattr(tool, "run_async"):
                                tool_result = await tool.run_async(args)
                            else:
                                # Fallback to synchronous run
                                tool_result = tool.run(**args)
                        except Exception as e:
                            tool_result = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(f"Tool execution error: {e}")
                        
                    outputs.append(
                        ToolMessage(
                            content=tool_result,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                # Return the *full* history plus tool outputs
                return {"messages": state.messages + outputs}

            # … rest of your graph creation (unchanged) …
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("chat", self._chat)
            graph_builder.add_node("tool_call", wrapped_tool_call)
            graph_builder.add_conditional_edges(
                "chat", self._should_continue, {"continue": "tool_call", "end": END}
            )
            graph_builder.add_edge("tool_call", "chat")
            graph_builder.set_entry_point("chat")
            graph_builder.set_finish_point("chat")

            # 5) (Optional) Checkpointer setup...
            connection_pool = await self._get_connection_pool()
            if connection_pool:
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()
            else:
                checkpointer = None
                if settings.app_env != Environment.PRODUCTION.value:
                    raise Exception("Connection pool initialization failed")

            self._graph = graph_builder.compile(
                checkpointer=checkpointer,
                name=f"{settings.PROJECT_NAME} Agent ({settings.app_env})"
            )
            logger.info("graph_created", graph_name=self._graph.name)

        return self._graph


    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.app_env,
                    debug=False,
                    user_id=user_id,
                    session_id=session_id,
                )
            ],
        }
        try:
            response = await self._graph.ainvoke(
                {"messages": dump_messages(messages), "session_id": session_id}, config
            )
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise e

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.app_env, debug=False, user_id=user_id, session_id=session_id
                )
            ],
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "session_id": session_id}, config, stream_mode="messages"
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
