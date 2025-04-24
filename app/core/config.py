"""Application configuration management.

location: app\core\config.py
This module handles environment-specific configuration loading, parsing, and management
for the application. It includes environment detection, .env file loading, and
configuration value parsing.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List

from dotenv import load_dotenv


# Define environment types
class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"
    TEST        = "test"

# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod": return Environment.PRODUCTION
        case "staging" | "stage":  return Environment.STAGING
        case "test":                return Environment.TEST
        case _:                     return Environment.DEVELOPMENT

# Load appropriate .env file based on environment
def load_env_file():
    """Load environment-specific .env file."""
    env = get_environment().value
    candidates = [f".env.{env}.local", f".env.{env}", ".env.local", ".env"]
    for fn in candidates:
        if Path(fn).is_file():
            load_dotenv(fn)
            print(f"Loaded {fn}")
            return
    print("No .env file loaded")

load_env_file()

def parse_list(value: str, default: List[str]):
    """Parse a list from a string value with robust handling."""
    if not value:
        return default
    # strip quotes
    value = value.strip('"').strip("'")
    # JSON-style?
    if value.startswith("[") and value.endswith("]"):
        return [x.strip().strip('"').strip("'") for x in value[1:-1].split(",") if x.strip()]
    # comma-separated
    return [x.strip() for x in value.split(",") if x.strip()]

class Settings:
    """Application settings without using pydantic."""

    def __init__(self):
        """Initialize application settings from environment variables.

        Loads and sets all configuration values from environment variables,
        with appropriate defaults for each setting. Also applies
        environment-specific overrides based on the current environment.
        """
        # ---- Core ----
        self.ENVIRONMENT = get_environment()
        # for backward-compatibility with any code still using .app_env:
        self.app_env = self.ENVIRONMENT.value
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "FastAPI LangGraph Template")
        self.VERSION      = os.getenv("VERSION", "1.0.0")
        self.DESCRIPTION = os.getenv(
            "DESCRIPTION", "A production-ready FastAPI template with LangGraph and Langfuse integration"
        )
        self.API_V1_STR   = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG        = os.getenv("DEBUG", "false").lower() in ("true","1")
        
        # ---- CORS ----
        raw = os.getenv("ALLOWED_ORIGINS", "")
        self.ALLOWED_ORIGINS = parse_list(raw, ["*"])
        
        # ---- Langfuse ----
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY","")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY","")
        self.LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # ---- LLM Providers ----
        self.llm_provider      = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_api_key    = os.getenv("OPENAI_API_KEY","")
        self.groq_api_key      = os.getenv("GROQ_API_KEY","")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY","")
        self.ollama_host       = os.getenv("OLLAMA_HOST","http://localhost:11434")
        
        # ---- Model settings ----
        self.model_name              = os.getenv("LLM_MODEL","gpt-4o-mini")
        self.default_llm_temperature = float(os.getenv("DEFAULT_LLM_TEMPERATURE","0.2"))
        self.max_tokens              = int(os.getenv("MAX_TOKENS","2000"))
        self.max_llm_call_retries    = int(os.getenv("MAX_LLM_CALL_RETRIES","3"))
        
        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_DAYS", "30"))

        # Logging Configuration
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"

        # Postgres Configuration
        self.POSTGRES_URL = os.getenv("POSTGRES_URL", "")
        self.POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
        self.POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.CHECKPOINT_TABLES = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]

        # Rate Limiting Configuration
        self.RATE_LIMIT_DEFAULT = parse_list(os.getenv("RATE_LIMIT_DEFAULT", ""), ["200 per day", "50 per hour"])

        # Rate limit endpoints defaults
        default_endpoints = {
            "chat": ["30 per minute"],
            "chat_stream": ["20 per minute"],
            "messages": ["50 per minute"],
            "register": ["10 per hour"],
            "login": ["20 per minute"],
            "root": ["10 per minute"],
            "health": ["20 per minute"],
        }

        # Update rate limit endpoints from environment variables
        self.RATE_LIMIT_ENDPOINTS = default_endpoints.copy()
        for endpoint in default_endpoints:
            env_key = f"RATE_LIMIT_{endpoint.upper()}"
            value = parse_list(os.getenv(env_key, ""), [])
            if value:
                self.RATE_LIMIT_ENDPOINTS[endpoint] = value

        # Evaluation Configuration
        self.EVALUATION_LLM = os.getenv("EVALUATION_LLM", "gpt-4o-mini")
        self.EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", "https://api.openai.com/v1")
        self.EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY", self.openai_api_key)
        self.EVALUATION_SLEEP_TIME = int(os.getenv("EVALUATION_SLEEP_TIME", "10"))

        # Finally, adjust per‐environment overrides:
        self.apply_environment_settings()

    def apply_environment_settings(self):
        """Apply environment-specific settings based on the current environment."""
        if self.ENVIRONMENT is Environment.DEVELOPMENT:
            self.DEBUG = True
            self.LOG_LEVEL = "DEBUG"
            self.LOG_FORMAT = "console"
            self.RATE_LIMIT_DEFAULT = ["1000 per day", "200 per hour"]
        elif self.ENVIRONMENT is Environment.STAGING:
            self.DEBUG = False
            self.LOG_LEVEL = "INFO"
            self.RATE_LIMIT_DEFAULT = ["500 per day", "100 per hour"]
        elif self.ENVIRONMENT is Environment.PRODUCTION:
            self.DEBUG = False
            self.LOG_LEVEL = "WARNING"
            self.RATE_LIMIT_DEFAULT = ["200 per day", "50 per hour"]
        elif self.ENVIRONMENT is Environment.TEST:
            self.DEBUG = True
            self.LOG_LEVEL = "DEBUG"
            self.LOG_FORMAT = "console"
            self.RATE_LIMIT_DEFAULT = ["1000 per day", "1000 per hour"]  # Relaxed for testing


# instantiate a global
settings = Settings()
