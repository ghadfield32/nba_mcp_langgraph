# tasks.py

import os

from dotenv import load_dotenv
from invoke import task

# ── 1) Load the right .env file ────────────────────────────
ENV = os.environ.get("APP_ENV", "development")
load_dotenv(f".env.{ENV}")

# ── 2) Figure out whether we can use a PTY (only on Unix) ──
USE_PTY = os.name != "nt"

@task
def dev(c):
    """
    Start the API in development mode (uvicorn reload).
    Usage: inv dev
    """
    # On Windows, 'env=...' will set the variable for the child process.
    cmd = "uv run uvicorn app.main:app --reload --port 8000"
    c.run(cmd, pty=USE_PTY, env={"APP_ENV": ENV})

@task
def prod(c):
    """
    Start the API in production mode.
    Usage: inv prod
    """
    # We still want APP_ENV=production for our code
    cmd = "./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    c.run(cmd, pty=USE_PTY, env={"APP_ENV": "production"})

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
