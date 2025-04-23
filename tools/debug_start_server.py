#!/usr/bin/env python
"""
tools/debug_start_server.py - Script to start the FastAPI and SSE servers with enhanced debugging.

This script:
1. Sets up detailed logging for SSE and FastMCP related components
2. Modifies settings to improve SSE connectivity
3. Starts both the FastAPI app and SSE server for testing
"""

import importlib
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set up enhanced debugging
os.environ["FASTAPI_DEBUG"] = "1"
os.environ["FASTMCP_DEBUG"] = "1"
os.environ["FASTMCP_LOG_LEVEL"] = "DEBUG"
os.environ["UVICORN_LOG_LEVEL"] = "DEBUG"

# Get ports from environment or use defaults
API_PORT = int(os.getenv("NBA_MCP_PORT", "8000"))
SSE_PORT = int(os.getenv("NBA_MCP_SSE_PORT", "8001"))
HOST = os.getenv("FASTMCP_SSE_HOST", "0.0.0.0")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Store the processes so we can terminate them properly
processes = []

def handle_exit(signum, frame):
    """Handle exit signals by stopping all child processes."""
    print("\nStopping servers...")
    for proc in processes:
        if proc.poll() is None:  # If the process is still running
            proc.terminate()
    sys.exit(0)

def fix_server_settings():
    """Apply fixes to the server settings before starting."""
    print("Applying fixes to server settings...")
    
    try:
        # Import the MCP server
        from app.services.mcp.nba_mcp.nba_server import mcp_server

        # 1. Ensure debug mode is enabled
        mcp_server.settings.debug = True
        print(f"Debug mode: {mcp_server.settings.debug}")
        
        # 2. Ensure paths have leading slashes
        if not mcp_server.settings.message_path.startswith('/'):
            mcp_server.settings.message_path = '/' + mcp_server.settings.message_path
            print(f"Fixed message_path: {mcp_server.settings.message_path}")
        
        if not mcp_server.settings.sse_path.startswith('/'):
            mcp_server.settings.sse_path = '/' + mcp_server.settings.sse_path
            print(f"Fixed sse_path: {mcp_server.settings.sse_path}")
        
        # 3. Force rebuilding the SSE app to apply changes
        sse_app = mcp_server.sse_app()
        print(f"SSE app rebuilt. Routes: {len(sse_app.routes) if hasattr(sse_app, 'routes') else 'unknown'}")
        
        return True
    except Exception as e:
        print(f"Error fixing server settings: {e}")
        return False

def run_servers():
    """Run both the FastAPI and SSE servers with enhanced debugging."""
    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Apply fixes
    fixed = fix_server_settings()
    if not fixed:
        print("Warning: Could not apply all fixes to server settings")
    
    try:
        import uvicorn
        
        print("\nStarting NBA MCP servers with debug settings...")
        print(f"Main API: http://{HOST}:{API_PORT}")
        print(f"SSE Server: http://{HOST}:{SSE_PORT}")
        print("Use Ctrl+C to stop both servers\n")
        
        # Start the main FastAPI server
        api_cmd = [
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", HOST, 
            "--port", str(API_PORT),
            "--reload",
            "--log-level", "debug"
        ]
        api_process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(api_process)
        print(f"Started FastAPI server (PID: {api_process.pid})")
        
        # Give the main server a moment to start
        time.sleep(2)
        
        # Start the SSE server
        sse_cmd = [
            sys.executable, "run_sse.py", 
            "--mode", "local",
            "--debug"
        ]
        sse_process = subprocess.Popen(
            sse_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(sse_process)
        print(f"Started SSE server (PID: {sse_process.pid})")
        
        # Print output in real-time
        try:
            while True:
                # Check if either process has exited
                for proc, name in zip(processes, ["API", "SSE"]):
                    if proc.poll() is not None:
                        print(f"{name} server stopped with exit code {proc.poll()}")
                        handle_exit(None, None)
                
                # Print any available output
                for proc, name in zip(processes, ["API", "SSE"]):
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        print(f"{name}: {line.rstrip()}")
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(0.1)
        except KeyboardInterrupt:
            handle_exit(None, None)
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting servers: {e}")
        sys.exit(1)

def print_import_failure_info():
    """Print detailed information about the Python environment on import failure."""
    print("\n=== Python Environment Information ===")
    print(f"Python executable: {sys.executable}")
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    print("\n=== Installation Hint ===")
    print(f"Try installing dependencies with:")
    print(f"{sys.executable} -m pip install -e \".[examples]\"")
    print("=======================================\n")

if __name__ == "__main__":
    print("=== DEBUG SERVER STARTUP ===")
    print(f"Project root: {project_root}")
    
    # Check if the app module is available
    try:
        importlib.import_module("app.main")
        print("✅ app.main module found")
    except ImportError as e:
        print(f"❌ app.main module not found: {e}")
        print_import_failure_info()
        sys.exit(1)
    
    # Check if the NBA MCP server is available
    try:
        importlib.import_module("app.services.mcp.nba_mcp.nba_server")
        print("✅ NBA MCP server module found")
    except ImportError as e:
        print(f"❌ NBA MCP server module not found: {e}")
        print_import_failure_info()
        sys.exit(1)
    
    # Run the servers
    run_servers() 