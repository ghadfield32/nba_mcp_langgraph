"""This file contains the LangGraph Agent/workflow and interactions with the LLM.

location: app\core\langgraph\graph.py
"""


import ast
import asyncio
import inspect
import json
import socket
import sys
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Literal,
    Optional,
)
from urllib.parse import (
    urlparse,
    urlunparse,
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
    fix_messages_for_ollama,
    prepare_messages,
)


def resolve_mcp_tool(entry: Any) -> Optional[Any]:
    """
    If entry is a Tool instance, return it.
    If entry is a string, try to look it up in mcp_server._tool_manager._tools.
    Otherwise return None.
    """
    if hasattr(entry, "name"):
        return entry
    if isinstance(entry, str):
        # fastmcp stores tools in _tool_manager._tools: name→Tool
        tools_map = getattr(mcp_server, "_tool_manager", None)
        if tools_map:
            return tools_map._tools.get(entry)
    return None

# -----------------------------------------------------------------------------
# Unified tool invoker: maps any tool interface (ainvoke/arun/run/run_async)
# into a single `await _invoke(tool, args: dict)` call.
# -----------------------------------------------------------------------------


async def _invoke_tool_sync(tool: Any, args: dict) -> Any:
    """
    Call `tool.run` *safely* in every signature permutation LangChain supports.

    Resolution order (first one that works wins):
      1.   run(arguments=<dict>)
      2.   run(**args)          – including zero-arg `run()`
    """
    logger.debug(f">>> _invoke_tool_sync: calling {getattr(tool, 'name', type(tool).__name__)!r} with args={args!r}")
    
    run_fn: Callable = getattr(tool, "run")
    logger.debug(f"Tool run function: {run_fn}")
    
    try:                                            # ① run(arguments=...)
        logger.debug("Attempting run(arguments=...)")
        result = run_fn(arguments=args)
        logger.debug(f"run(arguments=...) succeeded")
    except TypeError as e:
        logger.debug(f"run(arguments=...) failed: {e}")
        try:                                        # ② run(**args)  / run()
            logger.debug(f"Attempting run(**args) with {args!r}")
            result = run_fn(**args)
            logger.debug(f"run(**args) succeeded")
        except TypeError as e2:
            logger.debug(f"run(**args) failed: {e2}, trying run() with no args")
            # Last-ditch effort: maybe the tool takes *no* args at all.
            result = run_fn()
            logger.debug(f"run() with no args succeeded")

    if inspect.isawaitable(result):
        logger.debug(f"Result is awaitable, awaiting it")
        result = await result                      # Handle async def run(...)
    
    logger.debug(f"<<< {getattr(tool, 'name', type(tool).__name__)!r} returned: {result!r}")
    return result


async def _invoke_tool_async(tool: Any, args: dict) -> Any:
    """
    Prefer an async entry-point if the tool exposes one; otherwise delegate
    to `_invoke_tool_sync`.

    Resolution order:
      1.  ainvoke(input=<str>)         – built-ins like DuckDuckGo
      2.  ainvoke(arguments=<dict>)    – custom tools following LC function-call
      3.  ainvoke(**args) / ainvoke()  – rare, but handle it
      4.  fall back to _invoke_tool_sync
    """
    tool_name = getattr(tool, "name", type(tool).__name__)
    logger.debug(f">>> _invoke_tool_async: calling {tool_name!r} with args={json.dumps(args, default=str)}")
    logger.debug(f"Tool type: {type(tool)}, dir: {dir(tool)}")
    
    # ---- 1) Fast path: ainvoke ------------------------------------------------
    if hasattr(tool, "ainvoke"):
        logger.debug(f"Tool has ainvoke method")
        async_fn: Callable[..., Awaitable] = getattr(tool, "ainvoke")
        sig = inspect.signature(async_fn)
        logger.debug(f"ainvoke signature: {sig}")
        try:
            if "input" in sig.parameters:           # DuckDuckGo etc.
                logger.debug(f"Calling ainvoke(input=...)")
                result = await async_fn(input=args.get("query") or args or "")
                logger.debug(f"ainvoke(input=...) succeeded")
                logger.debug(f"<<< {tool_name!r} returned: {result!r}")
                return result
            elif "arguments" in sig.parameters:     # LangGraph style
                logger.debug(f"Calling ainvoke(arguments=...)")
                result = await async_fn(arguments=args)
                logger.debug(f"ainvoke(arguments=...) succeeded")
                logger.debug(f"<<< {tool_name!r} returned: {result!r}")
                return result
            else:                                   # kwargs / no-arg style
                logger.debug(f"Calling ainvoke(**args)")
                result = await async_fn(**args)
                logger.debug(f"ainvoke(**args) succeeded")
                logger.debug(f"<<< {tool_name!r} returned: {result!r}")
                return result
        except TypeError as e:
            logger.debug(f"ainvoke failed: {e}")
            # Fall through – maybe wrong style; we'll try sync version
            pass

    # ---- 2) Other async variants ---------------------------------------------
    for meth in ("arun", "run_async"):
        if hasattr(tool, meth):
            logger.debug(f"Tool has {meth} method, trying it")
            result = getattr(tool, meth)(arguments=args)
            if inspect.isawaitable(result):
                logger.debug(f"{meth} returned awaitable, awaiting it")
                result = await result
            logger.debug(f"<<< {tool_name!r} returned from {meth}: {result!r}")
            return result

    # ---- 3) Fallback to sync --------------------------------------------------
    logger.debug(f"No async methods found, falling back to _invoke_tool_sync")
    return await _invoke_tool_sync(tool, args)


def normalize_db_url(url: str) -> str:
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port
    try:
        socket.getaddrinfo(host, port)
        return url
    except socket.gaierror:
        if host == "db":
            new = parsed._replace(netloc=f"{parsed.username}:{parsed.password}@localhost:{port}")
            return urlunparse(new)
        raise


# Ensure psycopg compatibility on Windows for any late pool creation
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
async def get_mcp_tools():
    """Get tools from the MCP NBA server, always resolving to Tool objects."""
    raw = await mcp_server.get_tools()
    entries = raw.values() if isinstance(raw, dict) else raw
    resolved = []
    for e in entries:
        t = resolve_mcp_tool(e)
        if t is not None:
            resolved.append(t)
        else:
            logger.warning(f"Could not resolve MCP tool entry: {e!r}")
    return resolved


async def configure_graph():
    """Configure a LangGraph workflow with NBA MCP tools."""

    # 1) Load the LLM
    llm = get_llm()
    
    # 2) Fetch and resolve MCP tools
    mcp_tools = await get_mcp_tools()  # already resolved to Tool objects
    logger.info(f"Loaded {len(mcp_tools)} MCP tools for standalone graph")
    
    # 3) Convert MCP tools into OpenAI‐compatible function definitions
    mcp_fn_defs = []
    for t in mcp_tools:
        schema = getattr(t, "parameters", {}) or {}  # Ensure we have a dict even if None
        
        # Debug: Log the original tool schema
        tool_name = getattr(t, "name", "unknown")
        logger.info(f"Processing tool schema for: {tool_name}")
        logger.debug(f"Original schema: {schema}")
        
        # Special case for league leaders - flatten parameters instead of nesting
        if tool_name == "get_league_leaders_info":
            logger.info("Applying special schema fix for get_league_leaders_info")
            # Extract the properties from within params and use them directly at top level
            if "properties" in schema and "params" in schema["properties"]:
                params_schema = schema["properties"]["params"]
                if "properties" in params_schema:
                    # Use the inner properties directly
                    mcp_fn_defs.append({
                        "name": tool_name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": {
                            "type": "object",
                            "properties": params_schema["properties"],
                            "required": params_schema.get("required", [])
                        }
                    })
                else:
                    # Fallback if structure not as expected
                    mcp_fn_defs.append({
                        "name": tool_name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": schema
                    })
            else:
                # Fallback to original schema
                mcp_fn_defs.append({
                    "name": tool_name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": schema
                })
        else:
            # Default handling for other tools
            mcp_fn_defs.append({
                "name": tool_name,
                "description": getattr(t, "description", "") or "",
                "parameters": {
                    "type": "object",
                    **schema
                }
            })

    # 4) Combine built‐in tools + MCP function defs
    fn_defs = tools + mcp_fn_defs
    logger.info(f"Binding {len(fn_defs)} total tools/functions to LLM")
    bound_llm = llm.bind_tools(fn_defs)
    
    # 5) Build lookup by name for *actual* tool objects
    tools_by_name = {tool.name: tool for tool in tools}
    for t in mcp_tools:
        tools_by_name[t.name] = t
    
    # 6) Define chat node
    async def chat(state):
        """Generate a response from the LLM."""
        logger.debug(f"chat node: Processing state with {len(state['messages'])} messages")
        
        messages = prepare_messages(state["messages"], bound_llm, SYSTEM_PROMPT)
        logger.debug(f"Prepared {len(messages)} messages for LLM")
        
        try:
            # Apply Ollama-specific message fixes if using Ollama
            if (hasattr(bound_llm, "_llm_type") and "ollama" in bound_llm._llm_type.lower()) or \
               settings.llm_provider.lower() == "ollama":
                logger.debug("Detected Ollama LLM in standalone chat node, applying message fixes")
                
                # Skip fix_messages_for_ollama and just dump the messages directly
                message_dicts = []
                for msg in messages:
                    if hasattr(msg, "model_dump"):
                        message_dicts.append(msg.model_dump())
                    elif hasattr(msg, "content"):  # For BaseMessage types
                        msg_dict = {"role": msg.type, "content": msg.content}
                        if hasattr(msg, "additional_kwargs"):
                            for k, v in msg.additional_kwargs.items():
                                msg_dict[k] = v
                        message_dicts.append(msg_dict)
                    else:  # Already a dict
                        message_dicts.append(msg)
                
                logger.debug(f"Invoking Ollama LLM with {len(message_dicts)} messages")
                
                # Show sample of the first few messages
                for i, msg in enumerate(message_dicts[:2]):
                    logger.debug(f"Message {i}: role={msg.get('role')}, content={msg.get('content')[:50]}...")
                
                response = await bound_llm.ainvoke(message_dicts)
                
                # Debug the response and tool calls
                logger.debug(f"Ollama response type: {type(response)}")
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.debug(f"LLM response contains {len(response.tool_calls)} tool_calls:")
                    for i, tc in enumerate(response.tool_calls):
                        logger.debug(f"Tool call {i}: name={tc.get('name')}, args={json.dumps(tc.get('args'), default=str)}")
                else:
                    logger.debug("LLM response contains NO tool_calls")
                
                return {"messages": state["messages"] + [response]}
            else:
                logger.debug(f"Using standard message format for LLM type: {getattr(bound_llm, '_llm_type', 'unknown')}")
                
                # Prepare OpenAI format messages
                openai_messages = convert_to_openai_messages(messages)
                logger.debug(f"Converted to {len(openai_messages)} OpenAI format messages")
                
                response = await bound_llm.ainvoke(openai_messages)
                
                # Debug the response and tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    logger.debug(f"LLM response contains {len(response.tool_calls)} tool_calls:")
                    for i, tc in enumerate(response.tool_calls):
                        logger.debug(f"Tool call {i}: name={tc.get('name')}, args={json.dumps(tc.get('args'), default=str)}")
                else:
                    logger.debug("LLM response contains NO tool_calls")
                    
                return {"messages": state["messages"] + [response]}
        except Exception as e:
            logger.error(f"Error in chat node: {e}", exc_info=True)
            return {
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Error: Failed to get response from LLM. {str(e)}"}
                ]
            }

    # 7) Define tool call node
    async def tool_call(state):
        """Process tool calls from the last message."""
        outputs = []
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return {"messages": state["messages"]}

        for call in last.tool_calls:
            name = call["name"]
            args = call["args"]
            tool = tools_by_name.get(name)

            if tool is None:
                logger.error(f"Tool '{name}' not found")
                tool_result = f"Error: Tool '{name}' not found"
            else:
                try:
                    # special handling for league leaders - handle both direct args and params-wrapped args
                    if name == "get_league_leaders_info":
                        from app.services.mcp.nba_mcp.nba_server import LeagueLeadersParams
                        if "params" in args and isinstance(args["params"], str):
                            logger.warning(f"Received params as string: {args['params']}")
                            # Try to parse the string as a dict
                            try:
                                import ast
                                params_dict = ast.literal_eval(args["params"])
                                # Map common field names if needed
                                if "stat" in params_dict and "stat_category" not in params_dict:
                                    params_dict["stat_category"] = params_dict.pop("stat")
                                if "mode" in params_dict and "per_mode" not in params_dict:
                                    params_dict["per_mode"] = params_dict.pop("mode")
                                params = LeagueLeadersParams(**params_dict)
                            except Exception as e:
                                logger.error(f"Failed to parse params string: {e}")
                                # Use original args as fallback
                                params = LeagueLeadersParams(**args)
                        elif "params" in args and isinstance(args["params"], dict):
                            # Already properly structured
                            params = LeagueLeadersParams(**args["params"])
                        else:
                            # Direct args format - create params from the args directly
                            params = LeagueLeadersParams(**args)
                        
                        tool_result = await _invoke_tool_async(tool, {"params": params})
                    else:
                        tool_result = await _invoke_tool_async(tool, args)
                except Exception as e:
                    logger.error(f"Error executing '{name}': {e}", exc_info=True)
                    tool_result = f"Error executing '{name}': {e}"

                outputs.append(ToolMessage(
                    content=tool_result,
                    name=name,
                    tool_call_id=call["id"],
                ))

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
        # Initialize attributes
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self._mcp_tools = None
        
        # Immediately load MCP tools
        asyncio.create_task(self._load_mcp_tools())
        
        # Use the modular LLM provider but don't bind tools yet until MCP tools are loaded
        self.llm = get_llm()
        self.tools_by_name = {tool.name: tool for tool in tools}

        logger.info("llm_initialized", model=settings.model_name, environment=settings.app_env)

    async def _load_mcp_tools(self):
        """Load MCP tools asynchronously."""
        try:
            # Fetch and normalize MCP tools
            mcp_tools = await get_mcp_tools()  # already resolved to Tool objects
            self._mcp_tools = {t.name: t for t in mcp_tools}
            
            # Extract schemas for binding to LLM
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
            
            # Combine built-in tools with MCP tools and bind to LLM
            fn_defs = tools + mcp_fn_defs
            self.llm = get_llm().bind_tools(fn_defs)
            
            # Update tools_by_name with MCP tools
            self.tools_by_name.update(self._mcp_tools)
            
            logger.info(f"MCP tools loaded and bound to LLM: {len(self._mcp_tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            # Don't raise - we'll try again on demand

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
        """Get a PostgreSQL connection pool using environment‐specific settings."""
        if self._connection_pool is None:

            # 1) Parse the URL
            raw_url = settings.POSTGRES_URL
            logger.debug(f"Raw POSTGRES_URL = {raw_url}")
            parsed = urlparse(raw_url)
            host, port = parsed.hostname, parsed.port
            logger.debug(f"Parsed DB host={host!r}, port={port!r}")

            # 2) Try DNS resolution, fallback if host == "db"
            try:
                socket.getaddrinfo(host, port)
                logger.debug("DNS resolution succeeded for host={!r}".format(host))
                db_url = raw_url
            except socket.gaierror as dns_err:
                logger.error(f"DNS resolution failed for host={host!r}: {dns_err}")
                if host == "db":
                    logger.info("Falling back to localhost for database host")
                    user = parsed.username or ""
                    pwd  = parsed.password or ""
                    creds = f"{user}:{pwd}@" if user and pwd else ""
                    new_netloc = f"{creds}localhost:{port}"
                    fixed = parsed._replace(netloc=new_netloc)
                    db_url = urlunparse(fixed)
                    logger.debug(f"Using fallback DB URL: {db_url}")
                else:
                    # If it's some other host, re-raise
                    raise

            # 3) Build and open the async pool
            try:
                max_size = settings.POSTGRES_POOL_SIZE
                self._connection_pool = AsyncConnectionPool(
                    db_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info(
                    "connection_pool_created",
                    max_size=settings.POSTGRES_POOL_SIZE,
                    environment=settings.app_env,
                )
            except Exception as e:
                logger.error(
                    "connection_pool_creation_failed",
                    error=str(e),
                    environment=settings.app_env,
                )
                if settings.app_env == Environment.PRODUCTION.value:
                    logger.warning(
                        "continuing_without_connection_pool",
                        environment=settings.app_env,
                    )
                    return None
                # In dev, surface the exception so you see what's wrong
                raise

        return self._connection_pool


    async def _chat(self, state: GraphState) -> dict:
        """Process user messages and generate LLM responses.

        Args:
            state: The agent state including messages history

        Returns:
            Updated state with LLM response

        Raises:
            Exception: If the LLM fails to generate a response after max retries
        """
        logger.debug(f"Chat node entered with state: {state.model_dump()}")
        
        # Get the messages from the state
        messages = prepare_messages(state.messages, self.llm, SYSTEM_PROMPT)
        logger.debug(f"Prepared messages for LLM")
        
        # Call the LLM to generate a response
        llm_calls_num = 0
        max_retries = settings.max_llm_call_retries

        for attempt in range(max_retries):
            try:
                # Apply Ollama-specific fixes if using Ollama
                if (hasattr(self.llm, "_llm_type") and "ollama" in self.llm._llm_type.lower()) or \
                   settings.llm_provider.lower() == "ollama":
                    logger.debug(f"Detected Ollama LLM, applying message fixes for attempt {attempt+1}")
                    fixed_messages = fix_messages_for_ollama(messages)
                    
                    # Dump messages to dict format for LLM
                    message_dicts = []
                    for msg in fixed_messages:
                        if hasattr(msg, "model_dump"):
                            message_dicts.append(msg.model_dump())
                        else:
                            # Handle LangChain BaseMessage types
                            msg_dict = {"role": msg.type, "content": msg.content}
                            if hasattr(msg, "additional_kwargs"):
                                for k, v in msg.additional_kwargs.items():
                                    msg_dict[k] = v
                            message_dicts.append(msg_dict)
                    
                    logger.debug(f"Invoking Ollama LLM with {len(message_dicts)} fixed messages")
                    logger.debug(f"Message samples: {message_dicts[0]}")
                    generated_state = {"messages": [await self.llm.ainvoke(message_dicts)]}
                else:
                    logger.debug(f"Using standard message format for LLM type: {getattr(self.llm, '_llm_type', 'unknown')}")
                    generated_state = {"messages": [await self.llm.ainvoke(dump_messages(messages))]}
                    
                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.model_name,
                    environment=settings.app_env,
                )
                return generated_state
            except Exception as e:
                # Check for authentication errors (401)
                if hasattr(e, "status_code") and e.status_code == 401:
                    logger.error(
                        "authentication_failed",
                        error=str(e),
                        provider=settings.llm_provider,
                        model=settings.model_name,
                        environment=settings.app_env,
                    )
                    raise RuntimeError(f"LLM authentication failed (401). Check your API key for {settings.llm_provider}.") from e
                
                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.app_env,
                )
                # Log more detailed error info
                logger.debug(f"Error type: {type(e).__name__}")
                logger.debug(f"Error details: {e}")
                
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
                            self.llm = get_llm()

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
        logger.debug(f"Processing tool calls from last message. Message content: {state.messages[-1].content}")
        
        # Check if we have any tool calls
        if not hasattr(state.messages[-1], 'tool_calls') or not state.messages[-1].tool_calls:
            logger.warning("No tool_calls attribute found in the last message or it's empty")
            # Return unchanged state if no tool calls
            return {"messages": state.messages}
        
        logger.debug(f"Found {len(state.messages[-1].tool_calls)} tool calls to process")
        
        for tool_call in state.messages[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            logger.debug(f"Processing tool call: {tool_name} with args: {args}")
            
            # Check if mcp_tools is initialized
            if self._mcp_tools is None:
                logger.warning("MCP tools are not initialized. Initializing now...")
                # Lazy-load MCP tools if not already loaded
                raw_mcp = await get_mcp_tools()
                mcp_tools = list(raw_mcp.values()) if isinstance(raw_mcp, dict) else raw_mcp
                self._mcp_tools = {t.name: t for t in mcp_tools}
                logger.info(f"Loaded {len(self._mcp_tools)} MCP tools")
            
            # Find the tool
            if tool_name in self.tools_by_name:
                tool = self.tools_by_name[tool_name]
                logger.debug(f"Found tool '{tool_name}' in built-in tools")
            elif self._mcp_tools and tool_name in self._mcp_tools:
                tool = self._mcp_tools[tool_name]
                logger.debug(f"Found tool '{tool_name}' in MCP tools")
            else:
                tool = None
                logger.error(f"Tool not found: {tool_name}")
                logger.debug(f"Available tools: {list(self.tools_by_name.keys())}")
                if self._mcp_tools:
                    logger.debug(f"Available MCP tools: {list(self._mcp_tools.keys())}")
            
            if tool is None:
                tool_result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    # Special handling for tools that need structured input
                    if tool_name == "get_league_leaders_info":
                        logger.debug(f"Special handling for get_league_leaders_info with args: {args}")
                        from app.services.mcp.nba_mcp.nba_server import LeagueLeadersParams

                        # Handle params as a string (from Ollama)
                        if "params" in args and isinstance(args["params"], str):
                            logger.warning(f"Received params as string: {args['params']}")
                            try:
                                import ast
                                params_dict = ast.literal_eval(args["params"])
                                # Map common field names if needed
                                if "stat" in params_dict and "stat_category" not in params_dict:
                                    params_dict["stat_category"] = params_dict.pop("stat")
                                if "mode" in params_dict and "per_mode" not in params_dict:
                                    params_dict["per_mode"] = params_dict.pop("mode")
                                params = LeagueLeadersParams(**params_dict)
                                tool_result = await _invoke_tool_async(tool, {"params": params})
                            except Exception as e:
                                logger.error(f"Failed to parse params string: {e}")
                                tool_result = f"Error with parameters for '{tool_name}': {str(e)}"
                        # Handle params as a dict
                        elif "params" in args and isinstance(args["params"], dict):
                            params = LeagueLeadersParams(**args["params"])
                            tool_result = await _invoke_tool_async(tool, {"params": params})
                        # Handle direct args (flattened schema)
                        else:
                            params = LeagueLeadersParams(**args)
                            tool_result = await _invoke_tool_async(tool, {"params": params})
                    else:
                        logger.debug(f"Standard invocation for {tool_name} with args: {args}")
                        tool_result = await _invoke_tool_async(tool, args)
                    
                    logger.info(f"Tool '{tool_name}' execution successful")
                except Exception as e:
                    logger.error(f"Tool execution error: {e}", exc_info=True)
                    tool_result = f"Error executing tool '{tool_name}': {str(e)}"
                
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        
        logger.debug(f"Tool calls processed, returning {len(outputs)} tool responses")
        return {"messages": state.messages + outputs}

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
            mcp_tools = await get_mcp_tools()  # already resolved to Tool objects
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
                            # Use the helper function to invoke the tool
                            tool_result = await _invoke_tool_async(tool, args)
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
            # Process messages based on LLM provider
            message_dicts = dump_messages(messages)
            if settings.llm_provider.lower() == "ollama":
                # Apply Ollama-specific fixes
                fixed_lc_messages = fix_messages_for_ollama(messages)
                message_dicts = dump_messages(fixed_lc_messages)
                
            response = await self._graph.ainvoke(
                {"messages": message_dicts, "session_id": session_id}, config
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
        
        # Ensure MCP tools are loaded
        if self._mcp_tools is None:
            logger.info("MCP tools not loaded yet, loading now...")
            # Fetch and normalize MCP tools
            mcp_tools = await get_mcp_tools()  # already resolved to Tool objects
            self._mcp_tools = {t.name: t for t in mcp_tools}
            
            # Update tools_by_name with MCP tools
            self.tools_by_name.update(self._mcp_tools)
            logger.info(f"Loaded {len(self._mcp_tools)} MCP tools")
            
            # Extract schemas for binding to LLM
            mcp_fn_defs = []
            for t in mcp_tools:
                schema = getattr(t, "parameters", {}) or {}
                mcp_fn_defs.append({
                    "name": t.name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": schema if t.name == "get_league_leaders_info" else {
                        "type": "object",
                        **schema
                    }
                })
            
            # Combine built-in tools with MCP tools and bind to LLM
            fn_defs = tools + mcp_fn_defs
            self.llm = get_llm().bind_tools(fn_defs)
            logger.info("Tools bound to LLM")
        
        if self._graph is None:
            self._graph = await self.create_graph()

        try:
            # Process messages based on LLM provider
            message_dicts = dump_messages(messages)
            if settings.llm_provider.lower() == "ollama":
                # Apply Ollama-specific fixes
                fixed_lc_messages = fix_messages_for_ollama(messages)
                message_dicts = dump_messages(fixed_lc_messages)
                
            logger.debug(f"Starting stream with {len(message_dicts)} messages")
            async for token, _ in self._graph.astream(
                {"messages": message_dicts, "session_id": session_id}, config, stream_mode="messages"
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
