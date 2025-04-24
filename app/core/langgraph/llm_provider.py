"""
LLM provider module for dynamically loading different LLM backends.

location: app\core\langgraph\llm_provider.py
This module provides a unified interface for loading different LLM providers
based on configuration settings.
"""

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm():
    """
    Get a LangChain LLM instance based on the configured provider.
    
    Returns:
        A LangChain Chat model instance configured according to settings.
        
    Raises:
        ValueError: If the configured provider is not supported.
    """
    p = settings.llm_provider

    if p == "ollama":
        return ChatOllama(
            base_url=settings.ollama_host,
            model=settings.model_name,
            temperature=settings.default_llm_temperature
        )
    
    if p == "openai":
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=settings.default_llm_temperature,
            max_tokens=settings.max_tokens
        )

    if p == "groq":
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.model_name,
            temperature=settings.default_llm_temperature,
            max_tokens=settings.max_tokens
        )

    if p == "anthropic":
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.model_name,
            temperature=settings.default_llm_temperature,
            max_tokens=settings.max_tokens
        )

    raise ValueError(f"Unsupported LLM provider: {p}") 