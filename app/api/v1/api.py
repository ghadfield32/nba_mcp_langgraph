"""API v1 router configuration.
location: app/api/v1/api.py
This module sets up the main API router and includes all sub-routers.
It also mounts the MCP SSE ASGI app for streaming endpoints.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chatbot import router as chatbot_router

# Keep this import for diagnostic purposes even if we don't include the router
from app.api.v1.mcp_router import router as mcp_router
from app.core.logging import logger
from app.services.mcp.nba_mcp.nba_server import mcp_server  # <<-- import your SSE app

api_router = APIRouter()

# Include your existing REST routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
api_router.include_router(mcp_router, prefix="/mcp", tags=["mcp"])

# Use API router's mount method to add the MCP SSE app 
# IMPORTANT: The ASGI app expects paths like /messages/{resource}, 
# so we mount at /mcp-sse to allow access via /api/v1/mcp-sse/messages/{resource}
api_router.mount("/mcp-sse", mcp_server.sse_app(), name="mcp_sse")

# Add convenience redirect/instruction for the correct usage
@api_router.get("/mcp-sse")
async def mcp_sse_help():
    """Help endpoint for MCP SSE usage."""
    logger.info("mcp_sse_help_called")
    return {
        "message": "MCP SSE server is mounted here",
        "usage": "To access resources, use the path /api/v1/mcp-sse/messages/{resource_path}",
        "example": "/api/v1/mcp-sse/messages/api-docs://openapi.json?session_id=your_session_id",
        "note": "Most MCP resources require a session_id query parameter"
    }

# Add a diagnostic endpoint to check MCP resources and tools
@api_router.get("/mcp-sse/diagnostic", tags=["mcp"])
async def mcp_diagnostic():
    """Diagnostic endpoint to check MCP resources and tools."""
    logger.info("mcp_diagnostic_called")
    try:
        # 1) Fetch
        resources = await mcp_server.get_resources()
        # DEBUG: inspect types
        logger.debug("Diagnostic raw resources: %r", [(r, type(r)) for r in resources])

        # 2) Convert to JSON-safe strings
        serializable_resources = []
        for r in resources:
            if hasattr(r, "pattern"):
                serializable_resources.append(r.pattern)
            else:
                serializable_resources.append(str(r))

        # 3) Fetch tools, convert to names
        tools = await mcp_server.get_tools()
        tool_names = []
        for t in tools:
            if isinstance(t, str):
                tool_names.append(t)
            else:
                tool_names.append(getattr(t, "name", str(t)))

        # 4) Return clean payload
        return {
            "resources": serializable_resources,
            "tools": tool_names,
            "status": "healthy"
        }

    except Exception as e:
        logger.error(f"Error in mcp_diagnostic: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("health_check_called")
    return {"status": "healthy", "version": "1.0.0"}



@api_router.get("/usage-guide")
async def usage_guide():
    """Guide for using the API correctly, especially for authentication.
    
    Returns:
        dict: Step-by-step guide on how to use the API
    """
    logger.info("usage_guide_called")
    return {
        "title": "API Usage Guide",
        "description": "Follow these steps to use the chat endpoint correctly",
        "steps": [
            {
                "step": 1,
                "title": "Register a user",
                "endpoint": "/api/v1/auth/register",
                "method": "POST",
                "description": "Register a new user to get started",
                "sample_request": {
                    "email": "user@example.com",
                    "password": "StrongPassword123!"
                }
            },
            {
                "step": 2,
                "title": "Login",
                "endpoint": "/api/v1/auth/login",
                "method": "POST",
                "description": "Login to get a user token (form data)",
                "sample_request": {
                    "username": "user@example.com",
                    "password": "StrongPassword123!",
                    "grant_type": "password"
                }
            },
            {
                "step": 3,
                "title": "Create a session",
                "endpoint": "/api/v1/auth/session",
                "method": "POST",
                "description": "Create a chat session (requires user token from step 2)",
                "auth": "Bearer token from step 2"
            },
            {
                "step": 4,
                "title": "Use chat endpoint",
                "endpoint": "/api/v1/chatbot/chat",
                "method": "POST",
                "description": "Send messages (requires session token from step 3)",
                "auth": "Bearer token from step 3 (session token)",
                "sample_request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, how are you?"
                        }
                    ]
                }
            }
        ],
        "common_errors": [
            {
                "status_code": 403,
                "reason": "Using Swagger UI without proper authorization. Make sure to click the 'Authorize' button and enter the correct token.",
                "solution": "After getting a token from login or session creation, click 'Authorize' in Swagger UI and enter 'Bearer your_token_here'"
            },
            {
                "status_code": 401,
                "reason": "Invalid token or expired token",
                "solution": "Generate a new token by logging in again"
            },
            {
                "status_code": 404,
                "reason": "Session not found",
                "solution": "Create a new session using the /api/v1/auth/session endpoint"
            }
        ]
    }

@api_router.get("/mcp-architecture", tags=["docs"])
async def mcp_architecture():
    """Documentation for MCP architecture and integration with LangGraph."""
    logger.info("mcp_architecture_docs_called")
    return {
        "title": "MCP Architecture & Integration",
        "description": "How MCP is integrated with LangGraph and FastAPI",
        "architecture": [
            {
                "component": "FastMCP Server (nba_server.py)",
                "description": "Provides NBA data via MCP resources and tools",
                "features": [
                    "Exposes NBA stats as resources like player stats, league leaders",
                    "Provides tools callable by LangChain/LangGraph",
                    "Serves on port set by NBA_MCP_PORT (default 8000)"
                ]
            },
            {
                "component": "FastAPI Integration",
                "description": "Two integration methods:",
                "methods": [
                    {
                        "type": "SSE Integration",
                        "endpoint": "/api/v1/mcp-sse/messages/{resource_path}",
                        "description": "Used by LangGraph for streaming tool calls",
                        "note": "Primary method for agent interaction"
                    },
                    {
                        "type": "REST Integration (optional)",
                        "endpoint": "/api/v1/mcp/* (commented out)",
                        "description": "Direct REST calls to MCP functions",
                        "note": "Not needed when using LangGraph, but useful for testing"
                    }
                ]
            },
            {
                "component": "LangGraph Agent",
                "description": "Uses LangChain/LangGraph with the LLM provider and MCP tools",
                "details": [
                    "Configured in app/core/langgraph/graph.py",
                    "Uses tools defined in the MCP server",
                    "Accesses MCP via SSE transport at /mcp-sse/messages",
                    "Example in examples/langgraph_ollama_agent_w_tools.py"
                ]
            }
        ],
        "recommended_usage": "For AI agent applications, use the LangGraph integration pattern rather than calling the REST endpoints directly"
    }
