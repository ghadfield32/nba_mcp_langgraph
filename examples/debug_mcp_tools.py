#!/usr/bin/env python
"""
Debug MCP tools in isolation.

It performs **two** separate checks for every resolved tool:

1.  Direct-Python call via `_invoke_tool_async` (bypasses FastAPI)
2.  FastAPI HTTP request to the tool's route (if the server exposes one)

Results are printed in a compact summary table showing ✔ / ✖, return-type,
error message, and latency.

Usage
-----
  uv run python examples/debug_mcp_tools.py [--only NAME] [--skip-http] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import time
from pprint import pformat
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import httpx
from fastapi.testclient import TestClient

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.langgraph.graph import _invoke_tool_async
from app.main import app as fastapi_app  # FastAPI instance  # FastAPI instance
from app.services.mcp.nba_mcp.nba_server import mcp_server

# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("debug_mcp_tools")

# ------------------------------------------------------------------------------
# Helper: fabricate minimal argument-dict from JSON schema
# ------------------------------------------------------------------------------


def _fake_args(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return dummy args that satisfy *required* fields in the schema."""
    if not schema:
        return {}
    props = schema.get("properties", {})
    required = schema.get("required", [])
    out: Dict[str, Any] = {}
    for name in required:
        spec = props.get(name, {})
        typ = spec.get("type", "string")
        if typ == "integer":
            out[name] = 1
        elif typ == "number":
            out[name] = 1.0
        elif typ == "boolean":
            out[name] = True
        elif typ == "array":
            out[name] = []
        elif typ == "object":
            out[name] = {}
        else:  # string / unknown
            out[name] = "dummy"
    return out


# ------------------------------------------------------------------------------
async def _resolve_tools() -> Dict[str, Any]:
    """Return name→tool dict, resolving string entries."""
    raw = await mcp_server.get_tools()
    mgr = getattr(mcp_server, "_tool_manager", None)
    tool_map = (
        mgr.tools
        if mgr and hasattr(mgr, "tools")
        else getattr(mgr, "_tools", {}) if mgr else {}
    )
    resolved = {}
    for entry in raw:
        obj = entry if hasattr(entry, "name") else tool_map.get(entry)
        if obj:
            resolved[obj.name] = obj
        else:
            log.warning("Could not resolve %r", entry)
    return resolved


# ------------------------------------------------------------------------------
async def _python_check(name: str, tool, schema) -> Tuple[bool, str, float]:
    """Call tool via Python; return (ok, message, seconds)."""
    args = _fake_args(schema)
    t0 = time.perf_counter()
    try:
        res = await _invoke_tool_async(tool, args)
        dt = time.perf_counter() - t0
        return True, f"OK <{type(res).__name__}>", dt
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, f"{e.__class__.__name__}: {e}", dt


def _http_route_for_tool(name: str) -> str | None:
    """Infer HTTP route from MCP documentation, else None."""
    # convention in nba_mcp: /api/v1/mcp/nba/{tool_name}
    # override special cases here if needed
    return f"/api/v1/mcp/nba/{name}"


async def _http_check(
    client: TestClient, name: str, schema
) -> Tuple[bool, str, float]:
    route = _http_route_for_tool(name)
    if route is None:
        return False, "no-route", 0.0
    payload = _fake_args(schema)
    t0 = time.perf_counter()
    try:
        resp = client.get(route, params=payload)
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            return True, "200 OK", dt
        return False, f"{resp.status_code}: {resp.text[:60]}", dt
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, f"{e.__class__.__name__}: {e}", dt


# ------------------------------------------------------------------------------
async def run_checks(only: List[str], skip_http: bool, as_json: bool):
    tools = await _resolve_tools()
    if only:
        tools = {k: v for k, v in tools.items() if k in only}
    if not tools:
        log.error("No tools matched selection")
        return

    client = TestClient(fastapi_app)

    results: List[Dict[str, Any]] = []
    for name, tool in tools.items():
        schema = getattr(tool, "parameters", {}) or {}
        ok_py, msg_py, dt_py = await _python_check(name, tool, schema)
        if skip_http:
            ok_http, msg_http, dt_http = None, "skipped", 0.0
        else:
            ok_http, msg_http, dt_http = await _http_check(client, name, schema)

        results.append(
            dict(
                name=name,
                python_ok=ok_py,
                python_msg=msg_py,
                python_ms=round(dt_py * 1000, 1),
                http_ok=ok_http,
                http_msg=msg_http,
                http_ms=round(dt_http * 1000, 1),
            )
        )

    # ── Pretty print summary ────────────────────────────────────────────────
    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return

    hdr = f"{'TOOL':30} | PY  | HTTP | PY-ms | HTTP-ms | DETAIL"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        py = "✔" if r["python_ok"] else "✖"
        http = (
            "—"
            if r["http_ok"] is None
            else "✔" if r["http_ok"] else "✖"
        )
        print(
            f"{r['name'][:30]:30} | {py:^3} | {http:^4} | "
            f"{r['python_ms']:>5} | {r['http_ms']:>7} | {r['python_msg']}"
        )
        if not r["python_ok"]:
            print(f"{'':30} |     |      |       |         | ↳ {r['python_msg']}")
        if r["http_ok"] is False:
            print(f"{'':30} |     |      |       |         | ↳ {r['http_msg']}")


# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Debug MCP tools.")
    p.add_argument("--only", nargs="*", help="Only test these tool names")
    p.add_argument("--skip-http", action="store_true", help="Skip HTTP checks")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    return p.parse_args()


# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    asyncio.run(run_checks(args.only or [], args.skip_http, args.json))


if __name__ == "__main__":
    main()
