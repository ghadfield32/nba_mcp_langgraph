#!/usr/bin/env python
"""
tools/test_sse_connection.py - A simple tool to test SSE connections directly.

This script attempts to connect to various potential SSE endpoints and
displays detailed information about the responses.
"""

import argparse
import asyncio
import http.client
import json
import sys
import time
import urllib.parse
from pprint import pformat


def test_endpoint(url, headers=None, timeout=5):
    """Test a single endpoint and return detailed information about the response."""
    if headers is None:
        headers = {"Accept": "text/event-stream"}
    
    parsed_url = urllib.parse.urlparse(url)
    host = parsed_url.netloc.split(':')[0]
    port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
    path = parsed_url.path
    
    print(f"\nTesting endpoint: {url}")
    print(f"Host: {host}, Port: {port}, Path: {path}")
    print(f"Headers: {headers}")
    
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        
        # Read a small portion of the response to analyze
        body_start = response.read(1024)
        
        result = {
            "status": response.status,
            "reason": response.reason,
            "headers": dict(response.getheaders()),
            "body_preview": body_start.decode('utf-8', errors='replace')[:200],
            "is_sse": "text/event-stream" in response.getheader("Content-Type", ""),
            "body_starts_with_data": body_start.startswith(b"data:"),
        }
        
        conn.close()
        return result
    
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }


async def test_sse_streaming(url, timeout=30):
    """
    Test if the endpoint actually streams SSE events by connecting
    and listening for a short time.
    """
    import aiohttp
    
    print(f"\nTesting SSE streaming on {url}...")
    events_received = 0
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    print(f"Connection failed with status {response.status}")
                    return False
                
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    print(f"Warning: Content-Type is {content_type}, not text/event-stream")
                
                print("Connection established, listening for events...")
                start_time = time.time()
                max_wait = 5  # Wait up to 5 seconds for events
                
                # Listen for events with a timeout
                while time.time() - start_time < max_wait:
                    try:
                        # Set a timeout for each line read
                        line = await asyncio.wait_for(response.content.readline(), 1.0)
                        if not line:
                            continue
                            
                        line_str = line.decode('utf-8', errors='replace').strip()
                        if line_str:
                            events_received += 1
                            print(f"Event received: {line_str[:50]}...")
                            
                            # If we received a few events, we can consider it working
                            if events_received >= 3:
                                print(f"Successfully received {events_received} events!")
                                return True
                    except asyncio.TimeoutError:
                        # Timeout on this line read, continue
                        continue
                
                if events_received > 0:
                    print(f"Received {events_received} events, but not enough to confirm full functionality")
                    return True
                else:
                    print("No events received within the timeout period")
                    return False
                    
    except Exception as e:
        print(f"Error testing SSE streaming: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SSE connections")
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--url", help="Specific URL to test (overrides base)")
    parser.add_argument("--timeout", type=int, default=5, help="Connection timeout in seconds")
    parser.add_argument("--stream", action="store_true", help="Test actual SSE streaming")
    args = parser.parse_args()
    
    # Define endpoints to test
    if args.url:
        endpoints = [args.url]
    else:
        base = args.base.rstrip("/")
        endpoints = [
            f"{base}/sse",
            f"{base}/messages",
            f"{base}/sse-transport/sse",
            f"{base}/sse-transport/messages",
            f"{base}/sse-transport",
            f"{base}/api/v1/sse",
            f"{base}/events",
            f"{base}/stream",
        ]
    
    # Test with different header combinations
    header_variants = [
        {"Accept": "text/event-stream"},
        {"Accept": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        {},  # No headers
    ]
    
    results = {}
    working_endpoints = []
    
    # First, test if the server is running at all
    root_result = test_endpoint(args.base if args.url is None else args.url, headers={})
    if "error" in root_result:
        print(f"Error: Could not connect to the server at {args.base}")
        print(f"Error details: {root_result['error']}")
        sys.exit(1)
    
    print(f"\nServer root endpoint test: Status {root_result.get('status')}")
    
    # Test each endpoint with each header variant
    for endpoint in endpoints:
        endpoint_results = []
        for headers in header_variants:
            result = test_endpoint(endpoint, headers, args.timeout)
            endpoint_results.append(result)
            
            # Check if this looks like a working SSE endpoint
            if result.get("is_sse", False) or result.get("body_starts_with_data", False):
                if endpoint not in working_endpoints:
                    working_endpoints.append(endpoint)
        
        results[endpoint] = endpoint_results
    
    # Display summary
    print("\n=== SUMMARY ===")
    for endpoint, endpoint_results in results.items():
        statuses = [r.get("status", "ERR") for r in endpoint_results]
        is_sse = any(r.get("is_sse", False) for r in endpoint_results)
        has_data = any(r.get("body_starts_with_data", False) for r in endpoint_results)
        
        status_str = f"{'✅' if is_sse or has_data else '❌'} {endpoint}: Statuses {statuses}"
        if is_sse:
            status_str += " (text/event-stream)"
        if has_data:
            status_str += " (data: prefix)"
            
        print(status_str)
    
    # Test streaming if requested and we found working endpoints
    if args.stream and working_endpoints:
        print("\n=== TESTING SSE STREAMING ===")
        for endpoint in working_endpoints:
            if asyncio.run(test_sse_streaming(endpoint)):
                print(f"✅ Confirmed working SSE streaming on {endpoint}")
            else:
                print(f"❌ No streaming data detected on {endpoint}")
    
    # Provide conclusion and next steps
    if working_endpoints:
        print("\n=== CONCLUSION ===")
        print(f"Found {len(working_endpoints)} potential working SSE endpoints:")
        for endpoint in working_endpoints:
            print(f"  {endpoint}")
        
        if not args.stream:
            print("\nTo test if these endpoints actually stream SSE events, run:")
            print(f"  python {sys.argv[0]} --url={working_endpoints[0]} --stream")
    else:
        print("\n=== CONCLUSION ===")
        print("No working SSE endpoints found.")
        print("\nTROUBLESHOOTING SUGGESTIONS:")
        print("1. Make sure the FastAPI server is running: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        print("2. Check if the SSEPathFixMiddleware is correctly applied in app/main.py")
        print("3. Try running the fix_sse_middleware.py tool: python tools/fix_sse_middleware.py")
        print("4. Restart the server after making changes")


if __name__ == "__main__":
    main() 