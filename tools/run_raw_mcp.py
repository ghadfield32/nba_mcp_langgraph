#!/usr/bin/env python
"""
Run the MCP server directly without FastAPI to test if it works in standalone mode.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in the path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_mcp_standalone():
    """Run the MCP server in standalone mode using its direct run method."""
    from app.services.mcp.nba_mcp.nba_server import mcp_server
    
    print(f"Starting MCP server: {mcp_server.name}")
    print(f"Host: {mcp_server.settings.host}")
    print(f"Port: {mcp_server.settings.port}")
    print(f"SSE path: {mcp_server.settings.sse_path}")
    print(f"Message path: {mcp_server.settings.message_path}")
    
    # This will block until killed
    try:
        print("\nRunning MCP server in SSE mode...")
        mcp_server.run(transport="sse")
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")
    except Exception as e:
        print(f"\nError running MCP server: {e}")

def print_urls_manual():
    """Print the URLs to manually test in a browser or with curl."""
    from app.services.mcp.nba_mcp.nba_server import mcp_server
    
    host = mcp_server.settings.host
    # Use localhost if host is 0.0.0.0
    client_host = "localhost" if host in ("", "0.0.0.0") else host
    port = mcp_server.settings.port
    sse_path = mcp_server.settings.sse_path
    msg_path = mcp_server.settings.message_path
    
    base_url = f"http://{client_host}:{port}"
    
    print("\nAfter the server starts, you can test these URLs:")
    print(f"SSE stream URL (GET): {base_url}{sse_path}")
    print(f"Message post URL (POST): {base_url}{msg_path}")
    print("\nTest SSE in browser: Open the SSE URL in a browser with dev tools open to see the stream")
    print("\nTest with curl:")
    print(f"  curl -N -H \"Accept: text/event-stream\" {base_url}{sse_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Run MCP server in standalone mode")
    parser.add_argument("--transport", choices=["sse", "websocket", "stdio"], 
                        default="sse", help="Transport to use")
    parser.add_argument("--port", type=int, help="Override port to use")
    args = parser.parse_args()
    
    # Override port if specified
    if args.port:
        from app.services.mcp.nba_mcp.nba_server import mcp_server
        mcp_server.settings.port = args.port
        print(f"Port overridden to {args.port}")
    
    # Print instructions for manual testing
    print_urls_manual()
    
    # Run the server
    run_mcp_standalone()

if __name__ == "__main__":
    main() 