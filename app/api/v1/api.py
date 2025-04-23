"""API v1 router configuration.

This module sets up the main API router and includes all sub-routers for different
endpoints like authentication and chatbot functionality.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chatbot import router as chatbot_router
from app.core.logging import logger

api_router = APIRouter()

from app.services.mcp.nba_mcp.nba_server import router as mcp_nba_router

api_router.include_router(mcp_nba_router)

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])


@api_router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status information.
    """
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
