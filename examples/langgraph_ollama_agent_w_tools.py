# examples/langgraph_ollama_agent_w_tools.py

import argparse
import asyncio
import http.client
import json
import os
import pprint  # debug imports
import socket
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

load_dotenv()

print("DEBUG: interpreter =", sys.executable, file=sys.stderr)
print("DEBUG: sys.path =", pprint.pformat(sys.path), file=sys.stderr)

# Verify project structure
project_root = Path(__file__).resolve().parent.parent
print(f"DEBUG: Project root appears to be: {project_root}", file=sys.stderr)
print(f"DEBUG: Checking if 'app' module exists at {project_root/'app'}", file=sys.stderr)
if not (project_root/'app').exists():
    print(f"ERROR: 'app' directory not found at {project_root/'app'}", file=sys.stderr)
    print("HINT: Make sure you're running this script from the correct directory", file=sys.stderr)
    sys.exit(1)

# Add project root to path if not already there
if str(project_root) not in sys.path:
    print(f"DEBUG: Adding {project_root} to sys.path", file=sys.stderr)
    sys.path.insert(0, str(project_root))

# Debug MCP server configuration
try:
    from app.services.mcp.nba_mcp.nba_server import mcp_server
    print("DEBUG: NBA MCP server configuration:", file=sys.stderr)
    print(f"  message_path: {mcp_server.settings.message_path}", file=sys.stderr)
    print(f"  sse_path: {mcp_server.settings.sse_path}", file=sys.stderr)
    print(f"  host: {mcp_server.settings.host}", file=sys.stderr)
    print(f"  port: {mcp_server.settings.port}", file=sys.stderr)
except ImportError as e:
    print(f"DEBUG: Could not import NBA MCP server module: {e}", file=sys.stderr)
    print("HINT: This is expected if the module is not installed", file=sys.stderr)
except Exception as e:
    print(f"DEBUG: Error accessing NBA MCP server settings: {e}", file=sys.stderr)

# Wrap the import in try/except to catch missing module errors
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ModuleNotFoundError as e:
    print("DEBUG: failed to import langchain_mcp_adapters.client:", e, file=sys.stderr)
    print("HINT: Try installing with 'pip install -e \".[examples]\"'", file=sys.stderr)
    raise

try:
    from langchain_ollama import ChatOllama
except ModuleNotFoundError as e:
    print("DEBUG: failed to import langchain_ollama:", e, file=sys.stderr)
    print("HINT: Try installing with 'pip install -e \".[examples]\"'", file=sys.stderr)
    raise

from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)

# ── 1) Arg parsing: pick a mode, default "local" ────────────────────────
parser = argparse.ArgumentParser(
    description="Run NBA MCP in local/claude mode + Langgraph agent"
)
parser.add_argument(
    "--mode",
    choices=["claude", "local"],
    default=os.getenv("NBA_MCP_MODE", "local"),
    help="Which NBA‑MCP mode to run (and which port to bind)"
)
args = parser.parse_args()
MODE = args.mode

# ----------  REPLACE COMPLETELY  ----------
# ---------------------------------------------------------------------------
def get_mcp_endpoints() -> tuple[str, str]:
    """
    Get the message and SSE endpoints for the NBA MCP server.
    Uses two separate ports:
    - NBA_MCP_PORT (default 8000) for the main API 
    - NBA_MCP_SSE_PORT (default 8001) for the SSE server
    """
    import os
    import sys

    from app.services.mcp.nba_mcp.nba_server import mcp_server

    # Get configuration from environment or use defaults
    bind_host = mcp_server.settings.host or ""
    client_host = "localhost" if bind_host in ("", "0.0.0.0", "127.0.0.1") else bind_host
    
    # Get both ports
    api_port = int(os.getenv("NBA_MCP_PORT", "8000"))
    sse_port = int(os.getenv("NBA_MCP_SSE_PORT", "8001"))
    
    # Get paths from MCP server settings
    msg_path = mcp_server.settings.message_path.rstrip("/")
    sse_path = mcp_server.settings.sse_path.rstrip("/")

    # Build URLs using the appropriate ports
    # API endpoint uses the main API port
    message_url = f"http://{client_host}:{api_port}{msg_path}"
    # SSE endpoint uses the dedicated SSE port
    sse_url = f"http://{client_host}:{sse_port}{sse_path}"

    # Debug log
    print(f"\n=== RESOLVED ENDPOINTS ===", file=sys.stderr)
    print(f"message_url={message_url}", file=sys.stderr)
    print(f"sse_url    ={sse_url}", file=sys.stderr)
    print(f"Main API port: {api_port}", file=sys.stderr)
    print(f"SSE server port: {sse_port}", file=sys.stderr)

    return message_url, sse_url




# ---------------------------------------------------------------------------

def discover_sse_url(url: str, timeout: int = 4) -> str:
    """
    Confirm that *url* is live; otherwise crawl all FastAPI routes until
    we find one that serves text/event-stream.
    """
    import http.client
    import json
    import urllib.parse

    def _is_sse(u: str) -> bool:
        try:
            pr = urllib.parse.urlparse(u)
            conn = http.client.HTTPConnection(pr.hostname, pr.port or 80, timeout=timeout)
            conn.request("GET", pr.path, headers={"Accept": "text/event-stream"})
            r = conn.getresponse()
            ok = r.status == 200 and "text/event-stream" in (r.getheader("Content-Type") or "")
            conn.close()
            return ok
        except Exception as e:
            print(f"DEBUG: Error checking SSE URL {u}: {e}", file=sys.stderr)
            return False

    if _is_sse(url):
        return url  # happy path -------------------------------------------------

    # Define potential SSE paths to try
    pr = urllib.parse.urlparse(url)
    base = f"{pr.scheme}://{pr.netloc}"
    
    # Import the server settings if possible
    try:
        from app.services.mcp.nba_mcp.nba_server import mcp_server
        msg_path = mcp_server.settings.message_path.rstrip("/")
        sse_path = mcp_server.settings.sse_path
    except ImportError:
        msg_path = "/messages"
        sse_path = "/sse"
    
    # Try these paths in order of likelihood
    candidates = [
        f"{base}/sse-transport{sse_path}",     # /sse-transport/sse (explicit mount)
        f"{base}{msg_path}",                    # /messages/
        f"{base}{sse_path}",                    # /sse
        f"{base}/sse-transport{msg_path}",      # /sse-transport/messages/
        f"{base}/events",                       # Common alternative
        f"{base}/stream",                       # Another alternative
    ]
    
    print(f"DEBUG: Original URL {url} not working, trying alternatives...", file=sys.stderr)
    
    for candidate in candidates:
        if _is_sse(candidate):
            print(f"DEBUG: Found working SSE endpoint at {candidate}", file=sys.stderr)
            return candidate

    # otherwise fetch the OpenAPI spec and look for a better candidate
    try:
        conn = http.client.HTTPConnection(pr.hostname, pr.port or 80, timeout=timeout)
        conn.request("GET", "/openapi.json")
        resp = conn.getresponse()
        if resp.status == 200:
            spec = json.loads(resp.read().decode())
            conn.close()
            
            # Look for SSE endpoints in the OpenAPI spec
            for path, desc in spec.get("paths", {}).items():
                if "get" in desc:
                    get_op = desc["get"]
                    responses = get_op.get("responses", {})
                    if "200" in responses:
                        content_types = responses["200"].get("content", {})
                        if "text/event-stream" in content_types:
                            full_url = f"{base}{path}"
                            if _is_sse(full_url):
                                print(f"DEBUG: Found SSE endpoint in OpenAPI spec: {full_url}", file=sys.stderr)
                                return full_url
    except Exception as e:
        print(f"DEBUG: Error parsing OpenAPI spec: {e}", file=sys.stderr)

    raise RuntimeError("❌ No live SSE endpoint discovered")

# ------------------------------------------

def test_mcp_endpoints(base_url: str = "http://localhost:8000", timeout: int = 2) -> dict:
    """
    Test multiple potential MCP endpoints and return detailed status for each.
    This helps diagnose where the problem might be.
    """
    import http.client
    import urllib.parse
    
    results = {}
    
    # Define all potential endpoints to test
    test_paths = [
        "/",                            # Root path (health check)
        "/sse",                         # Direct SSE endpoint
        "/messages",                    # Direct message endpoint
        "/messages/",                   # With trailing slash
        "/sse-transport",               # Primary mount
        "/sse-transport/sse",           # Mounted SSE
        "/sse-transport/messages",      # Mounted messages
        "/api/v1/sse",                  # API versioned
        "/events",                      # Alternative name
        "/stream",                      # Alternative name
        "/health",                      # Health endpoint
    ]
    
    # Test headers to try
    header_variants = [
        {},  # No headers
        {"Accept": "text/event-stream"},  # SSE header
        {"Accept": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}  # Full SSE headers
    ]
    
    print("\n=== TESTING ALL POTENTIAL MCP ENDPOINTS ===", file=sys.stderr)
    
    for path in test_paths:
        url = f"{base_url}{path}"
        path_results = []
        
        for headers in header_variants:
            try:
                pr = urllib.parse.urlparse(url)
                conn = http.client.HTTPConnection(pr.netloc, timeout=timeout)
                conn.request("GET", pr.path, headers=headers)
                resp = conn.getresponse()
                
                # Read a small part of the body to check content type
                body_start = resp.read(100)
                ctype = resp.getheader("Content-Type", "")
                
                # Check if it looks like an SSE stream
                is_sse = "text/event-stream" in ctype or b"data:" in body_start
                
                status_desc = f"{resp.status} {resp.reason}"
                
                # Add results for this header variant
                path_results.append({
                    "status": resp.status,
                    "content_type": ctype,
                    "is_sse": is_sse,
                    "headers_used": headers
                })
                
                conn.close()
            except Exception as e:
                path_results.append({
                    "error": str(e),
                    "headers_used": headers
                })
        
        # Store results for this path
        results[path] = path_results
        
        # Print immediate results for debugging
        status_summary = "✅" if any(r.get("is_sse", False) for r in path_results) else "❌"
        print(f"{status_summary} {url}: {[r.get('status', 'ERR') for r in path_results]}", file=sys.stderr)
    
    return results

# ------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Fix parameters to use correct Ollama arguments - removed timeout param
llm = ChatOllama(
    base_url=OLLAMA_HOST,
    model="llama3.2:3b"
)

print(f"DEBUG: Configured ChatOllama with base_url={OLLAMA_HOST}, model=llama3.2:3b", file=sys.stderr)

def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait for a port to be available within the timeout period."""
    print(f"DEBUG: Waiting for port {host}:{port} to be available...", file=sys.stderr)
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                print(f"DEBUG: Successfully connected to {host}:{port}", file=sys.stderr)
                return True
            except OSError as e:
                print(f"DEBUG: Failed to connect to {host}:{port}: {e}", file=sys.stderr)
                time.sleep(0.1)
    print(f"DEBUG: Timeout waiting for {host}:{port} after {timeout} seconds", file=sys.stderr)
    return False

import http.client
import sys
import time
import urllib.parse


def check_server_health(
    url: str,
    max_retries: int = 10,
    retry_delay: float = 2.0,
    headers: dict | None = None
) -> bool:
    """
    Ping `url` up to max_retries times.
    If `headers` is provided, include them in the GET request.
    """
    print(f"DEBUG: Checking server health at {url}", file=sys.stderr)
    parsed = urllib.parse.urlparse(url)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme=="https" else 80)
    path = parsed.path or "/"
    hdrs = headers or {}

    for attempt in range(1, max_retries+1):
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            print(f"DEBUG: Health check attempt {attempt}/{max_retries}: GET {path}", file=sys.stderr)
            conn.request("GET", path, headers=hdrs)
            resp = conn.getresponse()
            print(f"DEBUG: Received status {resp.status}", file=sys.stderr)
            conn.close()
            if resp.status < 500:
                print(f"DEBUG: Server is healthy at {url}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"DEBUG: Health check attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)

        if attempt < max_retries:
            print(f"DEBUG: Waiting {retry_delay}s before next attempt...", file=sys.stderr)
            time.sleep(retry_delay)

    print(f"DEBUG: Server health check failed after {max_retries} attempts", file=sys.stderr)
    return False




def check_fastapi_middleware():
    """Inspect the FastAPI middleware to see if SSE paths are correctly handled."""
    try:
        from app.main import app
        print("\n=== FASTAPI MIDDLEWARE INSPECTION ===", file=sys.stderr)
        
        # Get all middleware
        middleware_stack = getattr(app, "middleware_stack", None)
        
        # Print middleware info
        print(f"Middleware stack: {middleware_stack}", file=sys.stderr)
        
        # Since middleware might be a method rather than a list, let's check the app._middleware
        # This is where FastAPI stores middleware in some versions
        _middleware = getattr(app, "_middleware", [])
        if hasattr(_middleware, "__len__"):
            print(f"_middleware count: {len(_middleware)}", file=sys.stderr)
            for i, mw in enumerate(_middleware):
                print(f"Middleware {i}: {mw}", file=sys.stderr)
            
            # Check for SSE-related middleware
            found_sse_middleware = False
            for mw in _middleware:
                mw_str = str(mw)
                if "SSE" in mw_str or "Event" in mw_str or "Stream" in mw_str:
                    found_sse_middleware = True
                    print(f"Found potential SSE middleware: {mw}", file=sys.stderr)
            
            if not found_sse_middleware:
                print("Warning: No explicit SSE middleware found in _middleware", file=sys.stderr)
        else:
            print("_middleware is not an iterable", file=sys.stderr)
            
        # Try to dig deeper into the app structure to find middleware
        print("\nChecking app structure for middleware...", file=sys.stderr)
        for attr_name in dir(app):
            if "middleware" in attr_name.lower():
                attr = getattr(app, attr_name)
                print(f"Found: app.{attr_name} = {type(attr).__name__}", file=sys.stderr)
                
        # Try to directly inspect the Starlette app mounted at /sse-transport
        for route in app.routes:
            if hasattr(route, "app") and hasattr(route, "path") and route.path == "/sse-transport":
                sse_app = route.app
                print(f"\nInspecting SSE app mounted at /sse-transport:", file=sys.stderr)
                print(f"Type: {type(sse_app).__name__}", file=sys.stderr)
                
                # Try to access middleware on this app
                sse_middleware = getattr(sse_app, "middleware", [])
                if hasattr(sse_middleware, "__len__"):
                    print(f"SSE app middleware count: {len(sse_middleware)}", file=sys.stderr)
                    for i, mw in enumerate(sse_middleware):
                        print(f"SSE Middleware {i}: {mw}", file=sys.stderr)
                else:
                    print("SSE app middleware is not iterable", file=sys.stderr)
                    
                # Check for specific handlers
                if hasattr(sse_app, "handlers"):
                    print(f"SSE app has {len(sse_app.handlers)} handlers", file=sys.stderr)
                
                # Try to access routes on this app
                sse_routes = getattr(sse_app, "routes", [])
                if hasattr(sse_routes, "__len__"):
                    print(f"SSE app routes count: {len(sse_routes)}", file=sys.stderr)
                    for i, route in enumerate(sse_routes):
                        print(f"SSE Route {i}: {route}", file=sys.stderr)
                else:
                    print("SSE app routes is not iterable", file=sys.stderr)
                
                break
                
        return True
    except Exception as e:
        print(f"Error inspecting FastAPI middleware: {e}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return False

def check_nba_mcp_installed():
    """Check if the app.services.mcp.nba_mcp.nba_server module is installed and accessible."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("app.services.mcp.nba_mcp.nba_server")
        if spec is None:
            print("ERROR: app.services.mcp.nba_mcp.nba_server module not found", file=sys.stderr)
            print("Checking for nba_server.py file directly...", file=sys.stderr)
            
            # Look for the nba_server.py file directly
            potential_paths = [
                project_root / "app" / "services" / "mcp" / "nba_mcp" / "nba_server.py",
                project_root.parent / "app" / "services" / "mcp" / "nba_mcp" / "nba_server.py",
            ]
            
            for path in potential_paths:
                if path.exists():
                    print(f"Found nba_server.py at {path}", file=sys.stderr)
                    return True
            
            print("HINT: Make sure you've installed the package or are running from the correct directory", file=sys.stderr)
            return False
        
        print(f"DEBUG: Found NBA MCP server module at {spec.origin}", file=sys.stderr)
        return True
    except ImportError as e:
        print(f"ERROR: Failed to check for NBA MCP server module: {e}", file=sys.stderr)
        return False

def get_system_prompt() -> str:
    return (
        f"Today is {datetime.now():%Y-%m-%d}.\n"
        "Tools you can call:\n"
        "- get_league_leaders_info(season, stat_category, per_mode)\n"
        "    • per_mode must be one of: 'Totals', 'PerGame', 'Per48'\n"
        "    • stat_category must be one of: 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT'\n"
        "    • e.g.: get_league_leaders_info('2024-25','AST','PerGame')\n"
        "- get_player_career_information(player_name, season)\n"
        "    • e.g.: get_player_career_information('LeBron James', '2023-24')\n"
        "- get_live_scores(target_date)\n"
        "    • e.g.: get_live_scores('2024-05-15')\n"
        "- get_date_range_game_log_or_team_game_log(season, team, date_from, date_to)\n"
        "    • e.g.: get_date_range_game_log_or_team_game_log('2023-24', 'Lakers', '2024-01-01', '2024-01-31')\n"
        "- play_by_play(game_date, team, start_period, end_period, start_clock, recent_n, max_lines)\n"
        "    • e.g.: play_by_play('2024-05-15', 'Lakers', 1, 4, None, 5, 200)\n"
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

# Function that resolves file paths in the codebase
def get_file_content(file_path):
    """Get the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

async def main():
    max_retries = 5
    retry_delay = 2.0
    
    # Declare global variable at the beginning of the function
    global tools
    
    # Check if the NBA MCP module is installed
    if not check_nba_mcp_installed():
        print("ERROR: NBA MCP server module not found. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    
    # Get the configured ports
    api_port = int(os.getenv("NBA_MCP_PORT", "8000"))
    sse_port = int(os.getenv("NBA_MCP_SSE_PORT", "8001"))
    
    # NEW: debug–inspect port 8000
    print("\nDEBUG: Running test_mcp_endpoints on port 8000…", file=sys.stderr)
    ep_status = test_mcp_endpoints(f"http://localhost:{api_port}")
    for path, results in ep_status.items():
        ok = any(r.get("status", 0) < 500 for r in results)
        mark = "✅" if ok else "❌"
        print(f"{mark} {path} → {[r.get('status','ERR') for r in results]}", file=sys.stderr)
    print("DEBUG: Done testing port 8000\n", file=sys.stderr)
    # If "/" isn’t responding, bail:
    if not any(r.get("status", 0) < 500 for r in ep_status.get("/", [])):
        print(f"ERROR: port {api_port} not responding at '/'. Did you start uvicorn app.main:app?", file=sys.stderr)
        sys.exit(1)
        
    # Resolve both the HTTP‐API and SSE URLs
    message_url, sse_url = get_mcp_endpoints()
    
    # First check if the FastAPI main API server is running
    server_running = check_server_health(f"http://localhost:{api_port}", max_retries=3, retry_delay=1.0)
    if not server_running:
        print(f"ERROR: FastAPI server does not appear to be running on port {api_port}.", file=sys.stderr)
        print("Try starting the main server with:", file=sys.stderr)
        print(f"    uvicorn app.main:app --host 0.0.0.0 --port {api_port} --reload", file=sys.stderr)
        sys.exit(1)
    
    # Check if the SSE server is running (must send SSE headers)
    sse_running = check_server_health(
        sse_url,
        max_retries=3,
        retry_delay=1.0,
        headers={
            "Accept":        "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
        }
    )
    if not sse_running:
        print(f"ERROR: SSE server does not appear to be running at {sse_url}.", file=sys.stderr)
        print("Try starting the SSE server with:", file=sys.stderr)
        print("    python run_sse.py --mode local", file=sys.stderr)
        print("\nBoth servers must be running for this example to work:", file=sys.stderr)
        print(f"1. Main API: uvicorn app.main:app --host 0.0.0.0 --port {api_port}", file=sys.stderr)
        print(f"2. SSE Server: python run_sse.py --mode local", file=sys.stderr)
        sys.exit(1)
    
    # Use the verified SSE URL for the connection
    connection_config = {
        "nba": {
            "url":       sse_url,
            "transport": "sse",
            "timeout":   30,
            "headers": {
                "Accept":        "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection":    "keep-alive",
            }
        }
    }
    print(f"DEBUG: Connection config: {connection_config}", file=sys.stderr)
    
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: Attempt {attempt+1}/{max_retries} connecting to NBA MCP server at {sse_url}", file=sys.stderr)
            
            # Try with standard MultiServerMCPClient
            print("DEBUG: Trying MultiServerMCPClient...", file=sys.stderr)
            async with MultiServerMCPClient(connection_config) as client:
                print("DEBUG: Successfully connected to NBA MCP server", file=sys.stderr)
                tools = client.get_tools()
                
                # Print the available tools for debugging
                tool_names = [t.name for t in tools]
                print(f"DEBUG: Loaded {len(tools)} tools: {tool_names}", file=sys.stderr)
                
                # Proceed with the rest of the function as before
                return await run_agent_with_tools(tools)
                
        except Exception as e:
            print(f"DEBUG: Attempt {attempt+1}/{max_retries} failed: {e}", file=sys.stderr)
            print(f"Exception details: {type(e).__name__} - {str(e)}", file=sys.stderr)
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...", file=sys.stderr)
                await asyncio.sleep(retry_delay)
    
    # If we've exhausted all retries, report failure
    print(f"❌ Could not run agent after multiple retries", file=sys.stderr)
    
    # Provide detailed troubleshooting advice
    print("\n=== TROUBLESHOOTING SUGGESTIONS ===", file=sys.stderr)
    print("1. Make sure both servers are running:", file=sys.stderr)
    print(f"   - Main API: uvicorn app.main:app --host 0.0.0.0 --port {api_port}", file=sys.stderr)
    print(f"   - SSE Server: python run_sse.py --mode local", file=sys.stderr)
    print("2. Check that both ports are available and not blocked by a firewall", file=sys.stderr)
    print("3. Make sure the NBA_MCP_PORT and NBA_MCP_SSE_PORT environment variables are set correctly", file=sys.stderr)
    print("4. Try manually accessing the SSE endpoint in a browser:", file=sys.stderr) 
    print(f"   - http://localhost:{sse_port}/sse", file=sys.stderr)

async def run_agent_with_tools(tools):
    """Extract the agent running logic into a separate function for better organization"""
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

def discover_mounted_app_paths(base_url="http://localhost:8000", mount_path="/sse-transport", depth=1):
    """
    More thoroughly check what paths are available under a specific mount point.
    This function uses a recursive approach to discover available endpoints.
    
    Args:
        base_url: The base URL of the server
        mount_path: The mount point to explore
        depth: How deep to explore in the URL tree
    
    Returns:
        A list of discovered endpoints
    """
    import http.client
    import json
    import urllib.parse
    
    print(f"\n=== EXPLORING PATHS UNDER {base_url}{mount_path} (depth={depth}) ===", file=sys.stderr)
    
    # Common path patterns to try
    patterns = [
        "",  # The mount point itself
        "/",
        "/sse",
        "/messages",
        "/events",
        "/stream",
        "/api",
        "/mcp"
    ]
    
    # If we have OpenAPI docs, try to extract paths
    try:
        openapi_url = f"{base_url}/openapi.json"
        parsed_url = urllib.parse.urlparse(openapi_url)
        conn = http.client.HTTPConnection(parsed_url.netloc, timeout=5)
        conn.request("GET", parsed_url.path)
        response = conn.getresponse()
        
        if response.status == 200:
            openapi_data = json.loads(response.read().decode('utf-8'))
            api_paths = list(openapi_data.get("paths", {}).keys())
            print(f"Found {len(api_paths)} paths in OpenAPI schema", file=sys.stderr)
            
            # Add relevant paths to patterns
            for path in api_paths:
                if mount_path.rstrip('/') in path:
                    # Extract just the part after the mount point
                    subpath = path.replace(mount_path.rstrip('/'), '')
                    if subpath and subpath not in patterns:
                        patterns.append(subpath)
        
        conn.close()
    except Exception as e:
        print(f"Error fetching OpenAPI schema: {e}", file=sys.stderr)
    
    full_url = f"{base_url}{mount_path}"
    discovered = []
    
    # First, check the mount point itself
    try:
        parsed_url = urllib.parse.urlparse(full_url)
        conn = http.client.HTTPConnection(parsed_url.netloc, timeout=5)
        conn.request("GET", parsed_url.path, headers={"Accept": "text/event-stream"})
        response = conn.getresponse()
        
        status = response.status
        content_type = response.getheader("Content-Type", "")
        data = response.read(100)
        
        print(f"Base mount point: {full_url}", file=sys.stderr)
        print(f"  Status: {status}, Content-Type: {content_type}", file=sys.stderr)
        print(f"  Data preview: {data[:50]}", file=sys.stderr)
        
        if status == 200:
            discovered.append(mount_path)
            if "text/event-stream" in content_type:
                print(f"  ✅ This is an SSE endpoint!", file=sys.stderr)
            
        conn.close()
    except Exception as e:
        print(f"Error checking base mount point: {e}", file=sys.stderr)
    
    # Now try each pattern
    for pattern in patterns:
        if pattern == "":
            continue  # Already checked the base path
            
        test_path = f"{mount_path}{pattern}"
        test_url = f"{base_url}{test_path}"
        
        try:
            parsed_url = urllib.parse.urlparse(test_url)
            conn = http.client.HTTPConnection(parsed_url.netloc, timeout=5)
            conn.request("GET", parsed_url.path, headers={"Accept": "text/event-stream"})
            response = conn.getresponse()
            
            status = response.status
            content_type = response.getheader("Content-Type", "")
            
            print(f"Test path: {test_url}", file=sys.stderr)
            print(f"  Status: {status}, Content-Type: {content_type}", file=sys.stderr)
            
            if status == 200:
                discovered.append(test_path)
                data = response.read(100)
                print(f"  Data preview: {data[:50]}", file=sys.stderr)
                
                if "text/event-stream" in content_type:
                    print(f"  ✅ This is an SSE endpoint!", file=sys.stderr)
                    
                # If we should go deeper and haven't reached max depth
                if depth > 1 and pattern:
                    sub_paths = discover_mounted_app_paths(base_url, test_path, depth - 1)
                    discovered.extend(sub_paths)
            else:
                # Just consume and discard the response body
                response.read()
            
            conn.close()
        except Exception as e:
            print(f"Error checking {test_url}: {e}", file=sys.stderr)
    
    return discovered

def troubleshoot_fastmcp_server():
    """
    Check the FastMCP server configuration and try to fix any issues.
    This function:
    1. Inspects the FastMCP server settings
    2. Tries to update the settings if needed
    3. Returns information about what was changed
    """
    print("\n=== TROUBLESHOOTING FASTMCP SERVER ===", file=sys.stderr)
    
    try:
        from app.services.mcp.nba_mcp.nba_server import mcp_server

        # Check current settings
        print("Current MCP Server settings:", file=sys.stderr)
        print(f"  host: {mcp_server.settings.host}", file=sys.stderr)
        print(f"  port: {mcp_server.settings.port}", file=sys.stderr)
        print(f"  message_path: {mcp_server.settings.message_path}", file=sys.stderr)
        print(f"  sse_path: {mcp_server.settings.sse_path}", file=sys.stderr)
        print(f"  debug: {mcp_server.settings.debug}", file=sys.stderr)
        
        # Make a backup of current settings
        original_settings = {
            "host": mcp_server.settings.host,
            "port": mcp_server.settings.port,
            "message_path": mcp_server.settings.message_path,
            "sse_path": mcp_server.settings.sse_path,
            "debug": mcp_server.settings.debug
        }
        
        # Let's try to modify the settings to make it more likely to work
        changes_made = []
        
        # 1. Enable debug mode
        if not mcp_server.settings.debug:
            mcp_server.settings.debug = True
            changes_made.append("Enabled debug mode")
        
        # 2. Ensure paths have leading slashes
        if not mcp_server.settings.message_path.startswith('/'):
            mcp_server.settings.message_path = '/' + mcp_server.settings.message_path
            changes_made.append(f"Added leading slash to message_path: {mcp_server.settings.message_path}")
        
        if not mcp_server.settings.sse_path.startswith('/'):
            mcp_server.settings.sse_path = '/' + mcp_server.settings.sse_path
            changes_made.append(f"Added leading slash to sse_path: {mcp_server.settings.sse_path}")
        
        # 3. Make sure host is set to 0.0.0.0
        if mcp_server.settings.host != "0.0.0.0":
            mcp_server.settings.host = "0.0.0.0"
            changes_made.append(f"Changed host to 0.0.0.0")
        
        # 4. Try to restart the SSE app with new settings
        if changes_made:
            print("\nChanges made to MCP server settings:", file=sys.stderr)
            for change in changes_made:
                print(f"  - {change}", file=sys.stderr)
            
            print("\nAttempting to rebuild SSE app with new settings...", file=sys.stderr)
            try:
                # Force a rebuild of the SSE app
                new_sse_app = mcp_server.sse_app()
                print("SSE app rebuilt successfully", file=sys.stderr)
                
                # Check if it has routes
                if hasattr(new_sse_app, "routes"):
                    print(f"New SSE app has {len(new_sse_app.routes)} routes", file=sys.stderr)
                
                return True, changes_made
            except Exception as e:
                print(f"Error rebuilding SSE app: {e}", file=sys.stderr)
                
                # Restore original settings
                print("Restoring original settings...", file=sys.stderr)
                mcp_server.settings.host = original_settings["host"]
                mcp_server.settings.port = original_settings["port"]
                mcp_server.settings.message_path = original_settings["message_path"]
                mcp_server.settings.sse_path = original_settings["sse_path"]
                mcp_server.settings.debug = original_settings["debug"]
                
                return False, []
        else:
            print("No changes needed to MCP server settings", file=sys.stderr)
            return True, []
    
    except Exception as e:
        print(f"Error troubleshooting FastMCP server: {e}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return False, []

if __name__ == "__main__":
    try:
        print("Langgraph agent starting…")
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Error running agent: {e}", file=sys.stderr)
        sys.exit(1)
        
