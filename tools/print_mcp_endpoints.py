#!/usr/bin/env python
"""
Print exactly where the MCP SSE and message‐post endpoints live,
by importing the server's own `mcp_server` settings.
"""
import importlib
import inspect
import os
import sys
import urllib.parse

# Make sure your project root is on PYTHONPATH
from app.services.mcp.nba_mcp.nba_server import mcp_server


def main():
    host = os.getenv("NBA_MCP_BASE_URL", "http://localhost:8000")
    # ensure no trailing slash
    host = host.rstrip("/")

    # Get the configured paths from the mcp_server
    mp = mcp_server.settings.message_path.rstrip("/")
    sp = mcp_server.settings.sse_path
    
    # Try to determine the actual mount point from the FastAPI app
    print("\n--- MCP Server Configuration ---")
    print(f"Host: {mcp_server.settings.host}")
    print(f"Port: {mcp_server.settings.port}")
    print(f"Message path: {mp}")
    print(f"SSE path: {sp}")
    
    # Try to detect the actual mount point
    mount_point = ""
    try:
        # Import the main app and check if we can find the mount point
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from app.main import app

        # Inspect the main FastAPI app to find where the MCP server is mounted
        print("\n--- Inspecting FastAPI mounts ---")
        for route in app.routes:
            if hasattr(route, "app") and getattr(route, "path", None):
                app_obj = route.app
                path = route.path
                print(f"Mount found: {path} → {app_obj.__class__.__name__}")
                # Check if this is our mcp_server's ASGI app
                if str(app_obj) == str(mcp_server.sse_app()):
                    mount_point = path
                    print(f"✅ Found MCP server mount point: {mount_point}")
    except Exception as e:
        print(f"Error inspecting mounts: {e}")

    # Print the "raw" configured endpoints (without mount point)
    print("\n--- Raw Configured Endpoints (from MCP server settings) ---")
    print("MCP message POST endpoint (raw configuration):")
    print(f"  {host}{mp}    [POST]")
    print()
    print("MCP SSE stream endpoint (raw configuration):")
    print(f"  {host}{sp}    [GET  Accept: text/event-stream]")
    
    # Print with detected mount point (if found)
    if mount_point:
        print("\n--- Actual Endpoints (with FastAPI mount point) ---")
        print("MCP message POST endpoint (actual):")
        print(f"  {host}{mount_point}{mp}    [POST]")
        print()
        print("MCP SSE stream endpoint (actual):")
        print(f"  {host}{mount_point}{sp}    [GET  Accept: text/event-stream]")

if __name__ == "__main__":
    main()
