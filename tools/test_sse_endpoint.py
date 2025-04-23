#!/usr/bin/env python
"""
Test SSE endpoints directly, specifically accounting for mount points.
This helps diagnose issues when the server's configured SSE paths don't
match where they're actually mounted in the FastAPI app.
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse
from typing import (
    Dict,
    List,
    Optional,
)


def test_endpoint(url: str, headers: Optional[Dict[str, str]] = None) -> bool:
    """
    Test if a URL is a valid SSE endpoint by making a request and checking
    the response.
    
    Returns: True if it appears to be an SSE endpoint, False otherwise
    """
    try:
        pr = urllib.parse.urlparse(url)
        host = pr.hostname or "localhost"
        port = pr.port or (443 if pr.scheme == "https" else 80)
        
        print(f"Testing {url}...")
        
        # Use complete headers if none provided
        complete_headers = headers or {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-SSE-Debug": "true"  # Add custom header for debugging
        }
        
        print(f"  Using headers: {complete_headers}")
        
        conn = http.client.HTTPConnection(host, port, timeout=10)  # Longer timeout
        conn.request("GET", pr.path, headers=complete_headers)
        
        # Log the exact request being made
        print(f"  Sent request: GET {pr.path}")
        
        resp = conn.getresponse()
        
        # Read a small amount to check for SSE signatures
        body = resp.read(512)
        ctype = resp.getheader("Content-Type", "")
        status = resp.status
        
        # Print all response headers for debugging
        print(f"  Response headers:")
        for header, value in resp.getheaders():
            print(f"    {header}: {value}")
        
        # Check for SSE indicators
        is_sse = (
            status == 200
            and (
                "text/event-stream" in ctype.lower()
                or b"data:" in body
                or b"event:" in body
            )
        )
        
        print(f"  Status: {status}")
        print(f"  Content-Type: {ctype}")
        print(f"  Body starts with: {body[:100]!r}")
        
        # Additional logging for more insight
        if status == 200 and not is_sse:
            print(f"  Note: Received 200 OK but response doesn't look like SSE")
            print(f"  Full body: {body!r}")
        
        print(f"  Looks like SSE: {'✅ YES' if is_sse else '❌ NO'}")
        
        conn.close()
        return is_sse
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SSE endpoints with various mount points")
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL of the server")
    parser.add_argument("--port", type=int, help="Override port in base URL")
    args = parser.parse_args()
    
    # Normalize base URL
    base = args.base.rstrip("/")
    if args.port:
        pr = urllib.parse.urlparse(base)
        base = f"{pr.scheme}://{pr.hostname}:{args.port}"
    
    # Get information from the server if possible
    try:
        # Attempt to import the server module
        print("Importing server configuration...")
        from app.services.mcp.nba_mcp.nba_server import mcp_server
        msg_path = mcp_server.settings.message_path.rstrip("/")
        sse_path = mcp_server.settings.sse_path
        print(f"Server configuration: message_path={msg_path}, sse_path={sse_path}")
        
        # Try to get FastAPI app and find mount points
        try:
            from app.main import app
            print("\nInspecting FastAPI mounts:")
            mounts = []
            for route in app.routes:
                if hasattr(route, "app") and getattr(route, "path", ""):
                    app_obj = route.app
                    path = route.path
                    print(f"Mount found: {path} → {app_obj.__class__.__name__}")
                    # Check if this is our mcp_server's ASGI app
                    if str(app_obj) == str(mcp_server.sse_app()):
                        mounts.append(path)
                        print(f"✅ This appears to be our MCP server")
            
            if mounts:
                print(f"Found {len(mounts)} potential MCP mount points: {mounts}")
            else:
                print("No MCP server mounts found in FastAPI app")
        except Exception as e:
            print(f"Error inspecting FastAPI app: {e}")
            mounts = []
    except Exception as e:
        print(f"Error importing server configuration: {e}")
        msg_path = "/messages"
        sse_path = "/sse"
        mounts = []
    
    # Define test configurations
    sse_headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    
    # Test direct paths (as configured in mcp_server)
    print("\n--- Testing Direct Paths (from server configuration) ---")
    test_endpoint(f"{base}{sse_path}", sse_headers)
    test_endpoint(f"{base}{msg_path}", sse_headers)
    
    # Test with mount points from FastAPI app
    if mounts:
        print("\n--- Testing with Detected Mount Points ---")
        for mount in mounts:
            test_endpoint(f"{base}{mount}{sse_path}", sse_headers)
            test_endpoint(f"{base}{mount}{msg_path}", sse_headers)
            # Also test the mount point itself
            test_endpoint(f"{base}{mount}", sse_headers)
    
    # Test common mount points
    print("\n--- Testing Common Mount Points ---")
    test_endpoint(f"{base}/sse-transport{sse_path}", sse_headers)
    test_endpoint(f"{base}/sse-transport{msg_path}", sse_headers)
    test_endpoint(f"{base}/sse-transport", sse_headers)
    
    # Test alternate common endpoints
    print("\n--- Testing Alternate Common Endpoints ---")
    for endpoint in ["/events", "/stream", "/api/sse"]:
        test_endpoint(f"{base}{endpoint}", sse_headers)


if __name__ == "__main__":
    main() 