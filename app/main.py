"""This file contains the main application entry point.

location: app/main.py

"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pprint import pformat
from typing import (
    Any,
    Dict,
)

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    Response,
)
from langfuse import Langfuse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import setup_metrics
from app.core.middleware import MetricsMiddleware
from app.services.database import database_service

# Import the mcp_server early so it's available in the lifespan
from app.services.mcp.nba_mcp.nba_server import mcp_server

# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup",
                project_name=settings.PROJECT_NAME,
                version=settings.VERSION,
                api_prefix=settings.API_V1_STR)

    # Import MCP server resources and tools
    logger.info("Importing MCP server resources and tools")
    await mcp_server.import_server(prefix="mcp-sse", server=mcp_server)

    # Verify MCP server is properly configured
    resources = await mcp_server.get_resources()
    tools = await mcp_server.get_tools()
    
    # Log detailed information about MCP setup
    logger.info(
        "MCP server configuration", 
        resource_count=len(resources),
        tool_count=len(tools),
        mcp_base_port=os.getenv("NBA_MCP_PORT", "8000"),
        has_sse_app=hasattr(mcp_server, 'sse_app'),
    )
    
    logger.debug("Registered resource patterns: %r", resources)
    logger.debug("Registered tool names:        %r", tools)
    
    # Verify router mounts
    from app.api.v1.api import api_router
    mount_paths = [route.path for route in api_router.routes if getattr(route, 'path', None)]
    logger.info(
        "API router mounts", 
        paths=mount_paths,
        mcp_sse_mount="/mcp-sse" in mount_paths,
    )
    
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Note: The main API router includes both handlers for MCP:
# 1. A regular router with explicit handlers at /api/v1/mcp/messages/{path} 
# 2. A direct mount for the SSE app at /api/v1/mcp-sse
# This separation prevents path conflicts

# Set up Prometheus metrics
setup_metrics(app)

# Add custom metrics middleware
app.add_middleware(MetricsMiddleware)

# Set up rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: _rate_limit_exceeded_handler(req, exc))


# Add validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data.

    Args:
        request: The request that caused the validation error
        exc: The validation error

    Returns:
        JSONResponse: A formatted error response
    """
    # Log the validation error
    logger.error(
        "validation_error",
        client_host=request.client.host if request.client else "unknown",
        path=request.url.path,
        errors=str(exc.errors()),
    )

    # Format the errors to be more user-friendly
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint with environment-specific information.

    Returns:
        Dict[str, Any]: Health status information
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = await database_service.health_check()

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return response if db_healthy else JSONResponse(content=response, status_code=status_code)