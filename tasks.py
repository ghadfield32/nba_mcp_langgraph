# tasks.py

import os

from dotenv import load_dotenv
from invoke import task


# helper to load <env>
def _load_env(env: str):
    path = f".env.{env}"
    load_dotenv(path, override=True)

# ── 2) Figure out whether we can use a PTY (only on Unix) ──
USE_PTY = os.name != "nt"

@task(help={"env": "development|staging|production"})
def dev(c, env="development"):
    """
    Start the API in development mode (uvicorn reload), *embedded* MCP SSE.
    Usage: inv dev --env=development
    """
    _load_env(env)
    
    # On Windows, we need to run these separately
    if os.name == "nt":
        # Install dependencies first
        c.run("uv pip install -e \".[dev,examples]\"", pty=False)
        # Then run the app
        cmd = "uv run uvicorn app.main:app --reload --port 8000"
    else:
        # For Unix systems, we can chain commands
        c.run("uv pip install -e \".[dev,examples]\"", pty=USE_PTY)
        cmd = "uv run uvicorn app.main:app --reload --port 8000"
    
    c.run(cmd, pty=USE_PTY, env={"APP_ENV": env, "MCP_RUN_MODE": "embedded"})

@task(help={"env": "development|staging|production"})
def prod(c, env="production"):
    """
    Start the API in production mode.
    Usage: inv prod --env=production
    """
    _load_env(env)
    cmd = "./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    c.run(cmd, pty=USE_PTY, env={"APP_ENV": env, "MCP_RUN_MODE": "embedded"})

@task(help={"env": "development|staging|production"})
def test(c, env="development"):
    """
    Run tests under given environment.
    Usage: inv test --env=development
    """
    _load_env(env)
    c.run("pytest", pty=USE_PTY, env={"APP_ENV": env})

@task(help={"env": "development|staging|production"})
def up(c, env="development"):
    """
    Bring up full Docker stack: db, app (with MCP), prometheus, grafana.
    Usage: inv up --env=development
    """
    _load_env(env)
    c.run(f"APP_ENV={env} docker compose up -d --build", pty=USE_PTY, env={"APP_ENV": env})

@task(help={"env": "development|staging|production"})
def down(c, env="development"):
    """
    Tear down full Docker stack.
    Usage: inv down --env=development
    """
    _load_env(env)
    c.run(f"APP_ENV={env} docker compose down", pty=USE_PTY, env={"APP_ENV": env})

@task
def lint(c):
    """
    Run code linters (ruff).
    Usage: inv lint
    """
    c.run("ruff check .", pty=USE_PTY)

@task
def fmt(c):
    """
    Auto‑format code (ruff/black).
    Usage: inv fmt
    """
    c.run("ruff format .", pty=USE_PTY)

@task
def clean(c):
    """
    Remove caches and venv.
    Usage: inv clean
    """
    c.run("rm -rf .venv __pycache__ .pytest_cache", warn=True, pty=USE_PTY)
