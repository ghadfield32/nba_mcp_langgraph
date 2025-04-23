#!/usr/bin/env python
"""
A diagnostic script to deeply inspect the FastMCP server and ASGI routes.
"""
import inspect
import os
import sys
from pprint import pprint
from typing import (
    Any,
    Dict,
)

# Add the project root to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def inspect_sse_app():
    """Inspect the ASGI app returned by mcp_server.sse_app()."""
    from app.services.mcp.nba_mcp.nba_server import mcp_server

    # Get the SSE app
    sse_app = mcp_server.sse_app()
    print(f"\n=== SSE App Type: {type(sse_app)} ===")
    
    # Print out basic attributes and methods
    print("\n=== SSE App Attributes ===")
    for attr in dir(sse_app):
        if not attr.startswith('_'):
            try:
                value = getattr(sse_app, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")
    
    # Print out routing information if available
    print("\n=== SSE App Routes (if any) ===")
    try:
        if hasattr(sse_app, 'routes'):
            for route in sse_app.routes:
                print(f"  Route: {route}")
        else:
            print("  No 'routes' attribute found")
    except Exception as e:
        print(f"  Error inspecting routes: {e}")

def inspect_fastmcp_internals():
    """Inspect FastMCP internals."""
    try:
        import fastmcp
        print(f"\n=== FastMCP Version: {fastmcp.__version__} ===")
        
        # Print out the ServerSettings class definition
        from fastmcp.server import ServerSettings
        print("\n=== ServerSettings Class ===")
        print(inspect.getsource(ServerSettings))
        
        # Print out the SSE transport implementation
        from fastmcp.transport.sse import SseServerTransport
        print("\n=== SseServerTransport Class ===")
        print(inspect.getsource(SseServerTransport))
        
    except Exception as e:
        print(f"Error inspecting FastMCP internals: {e}")

def make_direct_request():
    """Make direct ASGI requests to the SSE app to see how it handles different paths."""
    import asyncio

    from app.services.mcp.nba_mcp.nba_server import mcp_server
    
    async def _make_asgi_request(path: str):
        sse_app = mcp_server.sse_app()
        
        # Create a simple ASGI scope
        scope = {
            'type': 'http',
            'method': 'GET',
            'path': path,
            'headers': [
                (b'accept', b'text/event-stream'),
                (b'cache-control', b'no-cache'),
            ],
            'query_string': b'',
            'client': ('127.0.0.1', 12345),
            'server': ('localhost', 8000),
        }
        
        # Simple receive function that just returns body chunks
        async def receive():
            return {'type': 'http.request', 'body': b''}
        
        # A simple send function that just captures the responses
        responses = []
        async def send(message):
            responses.append(message)
        
        print(f"\n=== Making ASGI request to path: {path} ===")
        
        try:
            # Call the ASGI app directly
            await sse_app(scope, receive, send)
            
            # Display the responses
            print(f"Received {len(responses)} response messages:")
            for i, msg in enumerate(responses):
                print(f"  Message {i+1}: {msg.get('type')}")
                if msg.get('type') == 'http.response.start':
                    print(f"    Status: {msg.get('status')}")
                    print(f"    Headers: {msg.get('headers')}")
                elif msg.get('type') == 'http.response.body':
                    print(f"    Body: {msg.get('body')[:100]}")
        except Exception as e:
            print(f"Error during ASGI request: {e}")
    
    # List of paths to test
    paths_to_test = [
        '/',
        '/sse',
        '/messages',
        '/messages/',
    ]
    
    # Run the ASGI requests
    for path in paths_to_test:
        asyncio.run(_make_asgi_request(path))

def inspect_mcp_server_config():
    """Inspect the MCP server configuration."""
    from app.services.mcp.nba_mcp.nba_server import mcp_server
    
    print("\n=== MCP Server Configuration ===")
    print(f"Name: {mcp_server.name}")
    
    settings = mcp_server.settings
    print("\nSettings:")
    for attr in dir(settings):
        if not attr.startswith('_'):
            try:
                value = getattr(settings, attr)
                print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")
    
    # Check if the SSE app has a root_path set
    sse_app = mcp_server.sse_app()
    if hasattr(sse_app, 'root_path'):
        print(f"\nSSE App root_path: {sse_app.root_path}")
    else:
        print("\nSSE App has no root_path attribute")

def main():
    """Run all inspection functions."""
    print("=== Starting FastMCP Inspection ===")
    
    # Inspect the MCP server configuration
    inspect_mcp_server_config()
    
    # Inspect the SSE app
    inspect_sse_app()
    
    # Make direct ASGI requests
    make_direct_request()
    
    # Inspect FastMCP internals
    inspect_fastmcp_internals()
    
    print("\n=== Inspection Complete ===")

if __name__ == "__main__":
    main() 