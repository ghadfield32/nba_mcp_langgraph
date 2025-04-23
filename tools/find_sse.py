#!/usr/bin/env python
"""
tools/find_sse.py – locate the live SSE stream for the NBA-MCP server.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.parse
from typing import (
    Iterable,
    Optional,
)


def _get(path: str, headers: dict[str, str] | None = None, timeout: int = 4):
    pr = urllib.parse.urlparse(path)
    host, port = pr.hostname, pr.port or (443 if pr.scheme == "https" else 80)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", pr.path or "/", headers=headers or {})
    resp = conn.getresponse()
    body = resp.read(512)
    ctype = resp.getheader("Content-Type", "")
    conn.close()
    return resp.status, ctype, body

def _looks_like_sse(status: int, ctype: str, body: bytes) -> bool:
    """
    Enhanced detection of SSE endpoints, checking for various signatures.
    """
    # First, check standard indicators
    standard_check = (
        status == 200
        and (
            "text/event-stream" in ctype.lower()
            or b"data:" in body[:100]
            or b"event:" in body[:100]
        )
    )
    
    if standard_check:
        return True
        
    # Additional checks for non-standard but likely SSE endpoints
    # Some SSE endpoints might not identify themselves properly in headers
    if status == 200:
        # Check for common SSE signatures in the response body
        sse_signatures = [
            b"data:", b"event:", b"id:", b"retry:",  # Standard SSE fields
            b"text/event-stream",                     # Content type in body
            b"event-stream",                          # Partial content type
            b"EventSource",                           # JavaScript EventSource ref
        ]
        
        for sig in sse_signatures:
            if sig in body:
                print(f"DEBUG: Detected potential SSE signature: {sig}", file=sys.stderr)
                return True
                
    return False

def _probe(urls: Iterable[str]) -> str | None:
    for url in urls:
        for hdrs in (
            {"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            {},
        ):
            try:
                s, ct, data = _get(url, hdrs)
                print(f"CHECK {url:<40} → {s} {ct}", file=sys.stderr)
                if _looks_like_sse(s, ct, data):
                    return url
            except Exception as e:
                print(f"ERR   {url} : {e}", file=sys.stderr)
    return None

def _detect_fastapi_mounts(base_url: str) -> Optional[str]:
    """
    Try to detect the MCP server mount point by inspecting the FastAPI app.
    Returns the mount point if found, or None.
    """
    print("\n--- Detecting FastAPI mount points ---", file=sys.stderr)
    try:
        # First try to import the app directly
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from app.main import app
        from app.services.mcp.nba_mcp.nba_server import mcp_server
        
        print("✅ Successfully imported FastAPI app", file=sys.stderr)
        for route in app.routes:
            if hasattr(route, "app") and getattr(route, "path", None):
                app_obj = route.app
                path = route.path
                print(f"Mount found: {path} → {app_obj.__class__.__name__}", file=sys.stderr)
                # Check if this is our mcp_server's ASGI app
                if str(app_obj) == str(mcp_server.sse_app()):
                    mount_point = path
                    print(f"✅ Found MCP server mount point: {mount_point}", file=sys.stderr)
                    return mount_point
    except Exception as e:
        print(f"Error inspecting FastAPI mounts: {e}", file=sys.stderr)
    
    # If we can't import directly, try to infer from HTML
    try:
        s, ct, body = _get(f"{base_url}/docs")
        if s == 200 and b"SwaggerUIBundle" in body:
            print("✅ Found FastAPI Swagger UI - scanning for mount points", file=sys.stderr)
            mount_candidates = []
            content = body.decode('utf-8', errors='ignore')
            
            # Look for potential mount paths in the swagger HTML
            if "/sse-transport/" in content:
                print("Found mount: /sse-transport/", file=sys.stderr)
                mount_candidates.append("/sse-transport")
            if "/api/v1/" in content:
                print("Found mount: /api/v1/", file=sys.stderr)
                mount_candidates.append("/api/v1")
            
            return mount_candidates[0] if mount_candidates else None
    except Exception as e:
        print(f"Error inspecting FastAPI docs: {e}", file=sys.stderr)
    
    return None

def main() -> None:
    """
    1. Build candidate URLs from mcp_server introspection + common fall-backs.
    2. Probe them; if nothing works, scan /openapi.json for GET 200 endpoints
       that advertise 'text/event-stream'.
    3. Try additional detection methods to find SSE endpoints in non-standard locations.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000",
                    help="Base URL where FastAPI is listening")
    ap.add_argument("--port", type=int,
                    help="Shortcut to change port (overrides --base)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print more detailed debugging information")
    ap.add_argument("--mountpoint", default=None,
                    help="Specify a mount point to check (e.g., /sse-transport)")
    ap.add_argument("--direct", action="store_true",
                    help="Skip auto-detection and directly try endpoints from mcp_server settings")
    args = ap.parse_args()

    # ── normalise base URL ────────────────────────────────────────────────────
    base = args.base.rstrip("/")
    if args.port:
        pr = urllib.parse.urlparse(base)
        base = f"{pr.scheme}://{pr.hostname}:{args.port}"

    # ── obtain paths from the running server (if importable) ──────────────────
    default_msg, default_sse = "/messages", "/sse"
    try:
        from app.services.mcp.nba_mcp.nba_server import mcp_server  # type: ignore
        prefix   = getattr(mcp_server.sse_app(), "root_path", "").rstrip("/")
        msg_path = (mcp_server.settings.message_path or default_msg).rstrip("/")
        sse_path = mcp_server.settings.sse_path or default_sse
        print(
            f"DEBUG: Derived prefix='{prefix}', message_path='{msg_path}', "
            f"sse_path='{sse_path}'",
            file=sys.stderr,
        )
        
        # Also log the actual ASGI app mount path if verbose mode
        if args.verbose:
            print(f"DEBUG: mcp_server.settings.host = {mcp_server.settings.host}", file=sys.stderr)
            print(f"DEBUG: mcp_server.settings.port = {mcp_server.settings.port}", file=sys.stderr)
            asgi_app = mcp_server.sse_app()
            print(f"DEBUG: sse_app() type = {type(asgi_app)}", file=sys.stderr)
            
    except Exception as e:                                     # pragma: no cover
        prefix, msg_path, sse_path = "", default_msg, default_sse
        print(f"DEBUG: Could not import mcp_server: {e}", file=sys.stderr)

    # ── helper to yield both '/path' and '/path/' without duplicates ──────────
    def _variants(p: str) -> list[str]:
        p = p.rstrip("/")
        return [p, f"{p}/"] if p else [""]

    # ── Detect FastAPI mount point ───────────────────────────────────────────
    mount_point = args.mountpoint
    
    # Try to detect the mount point if not specified and not in direct mode
    if not mount_point and not args.direct:
        detected_mount = _detect_fastapi_mounts(base)
        if detected_mount:
            mount_point = detected_mount
            print(f"DEBUG: Using detected mount point: {mount_point}", file=sys.stderr)
        elif args.verbose:
            print("DEBUG: No mount point detected", file=sys.stderr)

    # ── 1) CANDIDATES ---------------------------------------------------------
    candidates: list[str] = []

    # If in direct mode, only try the paths as configured directly in mcp_server
    if args.direct:
        print("DEBUG: Using direct mode - only trying endpoints from mcp_server settings", file=sys.stderr)
        for path in (sse_path, msg_path):
            for v in _variants(f"{base}{path}"):
                candidates.append(v)
        candidates.append(f"{base}/")
    else:
        # 1-a primary prefix-aware endpoints
        for path in (sse_path, msg_path):
            for v in _variants(f"{base}{prefix}{path}"):
                candidates.append(v)

        # 1-b If we have a detected or specified mount point, prioritize those
        if mount_point:
            # Make sure the mount point is properly formatted
            mount_point = mount_point.rstrip("/")
            # Try both the detected mount point with the configured paths
            for path in (sse_path, msg_path):
                for v in _variants(f"{base}{mount_point}{path}"):
                    candidates.insert(0, v)  # Put these at the beginning of the list
            # Also try the mount point itself
            for v in _variants(f"{base}{mount_point}"):
                candidates.insert(0, v)

        # 1-c common "/sse-transport" mounts if not already added
        if not mount_point or mount_point != "/sse-transport":
            for path in (sse_path, msg_path):
                for v in _variants(f"{base}/sse-transport{path}"):
                    if v not in candidates:
                        candidates.append(v)
                
        # 1-d Try nested paths (sse-transport might have its own paths)
        nested_paths = [
            f"/sse-transport/api/v1{sse_path}",
            f"/sse-transport/api/v1{msg_path}",
            "/sse-transport/events",
            "/sse-transport/stream",
            "/sse-transport/",  # The mount point itself might be the SSE endpoint
            "/api/v1/sse",
            "/api/sse",
            "/events",
            "/stream",
        ]
        for p in nested_paths:
            for v in _variants(f"{base}{p}"):
                if v not in candidates:
                    candidates.append(v)

        # 1-e root (just in case someone mounted stream there)
        candidates.append(f"{base}/")

    # de-duplicate but keep order
    seen: set[str] = set()
    candidates = [u for u in candidates if not (u in seen or seen.add(u))]

    # ── 2) PROBE --------------------------------------------------------------
    print("\nProbing for a live SSE stream…", file=sys.stderr)
    winner = _probe(candidates)

    # ── 3) FALL-BACK: parse /openapi.json if still nothing --------------------
    if winner is None and not args.direct:
        try:
            s, _, body = _get(f"{base}/openapi.json")
            if s == 200:
                spec = json.loads(body.decode() or "{}")
                extra = [
                    f"{base}{p}"
                    for p, ops in spec.get("paths", {}).items()
                    if "get" in ops
                    and any(
                        "text/event-stream" in c
                        for c in ops["get"].get("responses", {})
                                   .get("200", {})
                                   .get("content", {})
                                   .keys()
                    )
                ]
                extra = [u for u in extra if u not in candidates]
                if extra:
                    print("\nTrying SSE paths discovered in OpenAPI…",
                          file=sys.stderr)
                    winner = _probe(extra)
                    candidates.extend(extra)
        except Exception as e:                         # pragma: no cover
            print(f"DEBUG: Could not parse /openapi.json: {e}", file=sys.stderr)
            
    # ── 3.5) TRY ALTERNATE DETECTION METHODS ---------------------------------
    if winner is None and not args.direct:
        # Try the FastAPI /docs path to locate mounted apps
        try:
            s, ct, body = _get(f"{base}/docs")
            if s == 200 and b"SwaggerUIBundle" in body:
                print("\nChecking FastAPI Swagger docs for mounted apps...", file=sys.stderr)
                # Look for potential mount points in the swagger UI HTML
                mounts = []
                if b"/sse-transport/" in body:
                    mounts.append("/sse-transport/")
                if b"/api/v1/" in body:
                    mounts.append("/api/v1/")
                
                # Try these mount points with known SSE path patterns
                extra = []
                for mount in mounts:
                    for endpoint in ["", "sse", "stream", "events", "messages"]:
                        path = f"{mount}{endpoint}"
                        if path.endswith("/"):
                            extra.append(f"{base}{path}")
                        else:
                            extra.extend([f"{base}{path}", f"{base}{path}/"])
                
                extra = [u for u in extra if u not in candidates]
                if extra:
                    print("\nTrying mount points discovered in FastAPI docs:", file=sys.stderr)
                    for e in extra:
                        print(f"  {e}", file=sys.stderr)
                    winner = _probe(extra)
                    candidates.extend(extra)
        except Exception as e:
            print(f"DEBUG: Error checking FastAPI docs: {e}", file=sys.stderr)

    # ── 4) RESULT -------------------------------------------------------------
    if winner:
        print("\n✅  SSE endpoint:", winner)
    else:
        print("\n❌  Could not find any working SSE endpoint. Tried:")
        for c in candidates:
            print("   ", c)
        # Provide additional useful information
        print("\nTROUBLESHOOTING SUGGESTIONS:")
        print("1. Ensure the server is running and accessible")
        print("2. Check if the SSE endpoint is correctly mounted in app/main.py")
        print("3. Try running with --verbose for more detailed information")
        print("4. Try the server health check: curl -v http://localhost:8000/health")
        print("5. Check the app mount configuration in app/main.py")
        print("6. Manually try a specific mount point: --mountpoint /sse-transport")
        print("7. Try direct mode (skip auto-detection): --direct")
        print("\nYou can also print the endpoints directly using:")
        print("   python tools/print_mcp_endpoints.py")
        sys.exit(1)


if __name__ == "__main__":
    main()