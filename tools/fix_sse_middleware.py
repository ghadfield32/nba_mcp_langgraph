#!/usr/bin/env python
"""
tools/fix_sse_middleware.py - A tool to fix the SSE middleware configuration in app/main.py.

This script:
1. Makes a backup of app/main.py
2. Updates the SSEPathFixMiddleware to properly handle Accept headers
3. Ensures all SSE mount points have the middleware applied
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main():
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    main_py_path = project_root / "app" / "main.py"
    
    # Check if app/main.py exists
    if not main_py_path.exists():
        print(f"Error: {main_py_path} not found. Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Make a backup of app/main.py
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = main_py_path.with_suffix(f".py.bak_{backup_time}")
    shutil.copy2(main_py_path, backup_path)
    print(f"Created backup of app/main.py at {backup_path}")
    
    # Read the current content of app/main.py
    with open(main_py_path, "r") as f:
        content = f.read()
    
    # Check if SSEPathFixMiddleware exists
    if "SSEPathFixMiddleware" in content:
        print("Found existing SSEPathFixMiddleware, updating it...")
        # Update the existing middleware
        updated_content = update_middleware(content)
    else:
        print("SSEPathFixMiddleware not found, adding it...")
        # Add new middleware
        updated_content = add_middleware(content)
    
    # Write the updated content back to app/main.py
    with open(main_py_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {main_py_path}")
    print("Next steps:")
    print("1. Restart your FastAPI server: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("2. Try running the langgraph example again")

def update_middleware(content):
    """Update the existing SSEPathFixMiddleware implementation."""
    # Find the middleware class definition
    middleware_pattern = r"class SSEPathFixMiddleware\(.*?\):.*?def dispatch\(.*?\):.*?return response"
    middleware_match = re.search(middleware_pattern, content, re.DOTALL)
    
    if not middleware_match:
        print("Warning: Could not find the complete SSEPathFixMiddleware class, adding a new one...")
        return add_middleware(content)
    
    old_middleware = middleware_match.group(0)
    
    # Create the updated middleware implementation
    new_middleware = """class SSEPathFixMiddleware(BaseHTTPMiddleware):
    \"\"\"Middleware to fix path routing for SSE endpoints.\"\"\"
    
    async def dispatch(self, request, call_next):
        # Log the request path for debugging
        logger.debug(f"SSEPathFixMiddleware: Request path: {request.url.path}")
        
        # Check if this is a potential SSE endpoint
        is_sse_path = False
        sse_paths = ['/sse', '/messages', '/sse-transport', '/events', '/stream']
        
        # Test if the path contains any of the SSE path patterns
        for sse_path in sse_paths:
            if sse_path in request.url.path:
                is_sse_path = True
                break
        
        if is_sse_path:
            # Create a modified list of headers with text/event-stream
            headers = [(k, v) for k, v in request.headers.items()]
            has_accept = False
            
            # Check if Accept header already exists
            for i, (name, value) in enumerate(headers):
                if name.lower() == b'accept':
                    # Update existing Accept header to include text/event-stream
                    if b'text/event-stream' not in value:
                        headers[i] = (name, value + b', text/event-stream')
                    has_accept = True
                    break
            
            # Add Accept header if it doesn't exist
            if not has_accept:
                headers.append((b'accept', b'text/event-stream'))
            
            # Update the request headers
            request.headers._list = headers
            logger.debug("SSEPathFixMiddleware: Added text/event-stream to Accept headers")
        
        # Continue with the request
        response = await call_next(request)
        
        # Log the response details for debugging
        logger.debug(f"SSEPathFixMiddleware: Response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
        
        return response"""
    
    # Replace the old middleware with the new one
    updated_content = content.replace(old_middleware, new_middleware)
    return updated_content

def add_middleware(content):
    """Add SSEPathFixMiddleware if it doesn't exist."""
    # Find the right location to add the middleware
    # Look for the line that mounts the MCP SSE app
    mount_pattern = r"# Mount the MCP SSE app at a dedicated path for clarity and to avoid conflicts"
    mount_match = re.search(mount_pattern, content)
    
    if mount_match:
        # Insert the middleware definition before the mount
        insert_pos = mount_match.start()
        
        middleware_def = """
# Create a custom middleware to fix the path routing for SSE endpoints
class SSEPathFixMiddleware(BaseHTTPMiddleware):
    \"\"\"Middleware to fix path routing for SSE endpoints.\"\"\"
    
    async def dispatch(self, request, call_next):
        # Log the request path for debugging
        logger.debug(f"SSEPathFixMiddleware: Request path: {request.url.path}")
        
        # Check if this is a potential SSE endpoint
        is_sse_path = False
        sse_paths = ['/sse', '/messages', '/sse-transport', '/events', '/stream']
        
        # Test if the path contains any of the SSE path patterns
        for sse_path in sse_paths:
            if sse_path in request.url.path:
                is_sse_path = True
                break
        
        if is_sse_path:
            # Create a modified list of headers with text/event-stream
            headers = [(k, v) for k, v in request.headers.items()]
            has_accept = False
            
            # Check if Accept header already exists
            for i, (name, value) in enumerate(headers):
                if name.lower() == b'accept':
                    # Update existing Accept header to include text/event-stream
                    if b'text/event-stream' not in value:
                        headers[i] = (name, value + b', text/event-stream')
                    has_accept = True
                    break
            
            # Add Accept header if it doesn't exist
            if not has_accept:
                headers.append((b'accept', b'text/event-stream'))
            
            # Update the request headers
            request.headers._list = headers
            logger.debug("SSEPathFixMiddleware: Added text/event-stream to Accept headers")
        
        # Continue with the request
        response = await call_next(request)
        
        # Log the response details for debugging
        logger.debug(f"SSEPathFixMiddleware: Response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
        
        return response

"""
        
        updated_content = content[:insert_pos] + middleware_def + content[insert_pos:]
        
        # Now make sure the middleware is applied to the SSE app
        # Look for the line after mcp_sse_app is created
        sse_app_pattern = r"mcp_sse_app = mcp_server\.sse_app\(\)"
        sse_app_match = re.search(sse_app_pattern, updated_content)
        
        if sse_app_match:
            # Add middleware application after the app creation
            add_middleware_code = "\n\n# Add the middleware to the SSE app\nmcp_sse_app.add_middleware(SSEPathFixMiddleware)"
            insert_pos = sse_app_match.end()
            updated_content = updated_content[:insert_pos] + add_middleware_code + updated_content[insert_pos:]
        
        return updated_content
    else:
        print("Warning: Could not find the mount point for the MCP SSE app in app/main.py")
        # Add the middleware at the end of the imports section as a fallback
        import_section_end = content.find("app = FastAPI(")
        if import_section_end == -1:
            import_section_end = 0
        
        middleware_def = """
# Create a custom middleware to fix the path routing for SSE endpoints
class SSEPathFixMiddleware(BaseHTTPMiddleware):
    \"\"\"Middleware to fix path routing for SSE endpoints.\"\"\"
    
    async def dispatch(self, request, call_next):
        # Log the request path for debugging
        logger.debug(f"SSEPathFixMiddleware: Request path: {request.url.path}")
        
        # Check if this is a potential SSE endpoint
        is_sse_path = False
        sse_paths = ['/sse', '/messages', '/sse-transport', '/events', '/stream']
        
        # Test if the path contains any of the SSE path patterns
        for sse_path in sse_paths:
            if sse_path in request.url.path:
                is_sse_path = True
                break
        
        if is_sse_path:
            # Create a modified list of headers with text/event-stream
            headers = [(k, v) for k, v in request.headers.items()]
            has_accept = False
            
            # Check if Accept header already exists
            for i, (name, value) in enumerate(headers):
                if name.lower() == b'accept':
                    # Update existing Accept header to include text/event-stream
                    if b'text/event-stream' not in value:
                        headers[i] = (name, value + b', text/event-stream')
                    has_accept = True
                    break
            
            # Add Accept header if it doesn't exist
            if not has_accept:
                headers.append((b'accept', b'text/event-stream'))
            
            # Update the request headers
            request.headers._list = headers
            logger.debug("SSEPathFixMiddleware: Added text/event-stream to Accept headers")
        
        # Continue with the request
        response = await call_next(request)
        
        # Log the response details for debugging
        logger.debug(f"SSEPathFixMiddleware: Response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
        
        return response

"""
        
        updated_content = content[:import_section_end] + middleware_def + content[import_section_end:]
        
        # Try to find where to add the middleware application
        sse_app_pattern = r"mcp_sse_app = mcp_server\.sse_app\(\)"
        sse_app_match = re.search(sse_app_pattern, updated_content)
        
        if sse_app_match:
            # Add middleware application after the app creation
            add_middleware_code = "\n\n# Add the middleware to the SSE app\nmcp_sse_app.add_middleware(SSEPathFixMiddleware)"
            insert_pos = sse_app_match.end()
            updated_content = updated_content[:insert_pos] + add_middleware_code + updated_content[insert_pos:]
        
        return updated_content

if __name__ == "__main__":
    main() 