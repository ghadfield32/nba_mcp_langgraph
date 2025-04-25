"""This file contains the graph utilities for the application."""

import asyncio
import json
import logging
from copy import deepcopy
from functools import wraps
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages import trim_messages as _trim_messages

from app.core.config import settings
from app.core.logging import logger
from app.schemas import Message


def dump_messages(messages: list[Message]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[Message]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
    """
    return [message.model_dump() for message in messages]


def prepare_messages(messages: list[Message], llm: BaseChatModel, system_prompt: str) -> list[Message]:
    """Prepare the messages for the LLM.

    Args:
        messages (list[Message]): The messages to prepare.
        llm (BaseChatModel): The LLM to use.
        system_prompt (str): The system prompt to use.

    Returns:
        list[Message]: The prepared messages.
    """
    # DEBUGGING: Skip trimming to preserve all messages including tool calls
    # trimmed_messages = _trim_messages(
    #     dump_messages(messages),
    #     strategy="last",
    #     token_counter=llm,
    #     max_tokens=settings.max_tokens,
    #     start_on="human",
    #     include_system=False,
    #     allow_partial=False,
    # )
    logger.debug(f"TRIMMING DISABLED for debugging - passing all {len(messages)} messages")
    trimmed_messages = dump_messages(messages)
    return [Message(role="system", content=system_prompt)] + trimmed_messages


def fix_messages_for_ollama(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Fix messages for Ollama consumption.
    
    This handles two specific issues in Ollama:
    1. When content is None but tool_calls exist, it changes content to an empty string
    2. When content is a list of parts, it extracts and joins the text parts
    
    Args:
        messages: The LangChain messages to fix.
        
    Returns:
        The fixed messages.
    """
    logger.debug(f"Fixing {len(messages)} messages for Ollama")
    
    # Check what types we're dealing with
    for i, msg in enumerate(messages):
        logger.debug(f"Message {i}: type={type(msg)}, content type={type(msg.content)}")
        if isinstance(msg.content, list):
            logger.debug(f"  Content is a list with {len(msg.content)} items")
            for j, item in enumerate(msg.content):
                logger.debug(f"    Item {j}: type={type(item)}")
                
    fixed_messages = []
    
    for msg in messages:
        # Create a copy to avoid modifying the original
        fixed_msg = deepcopy(msg)

        # Handle common Ollama issues
        if hasattr(fixed_msg, "content"):
            # Debug the content type
            logger.debug(f"Original content type: {type(fixed_msg.content)}")
            
            # Issue 1: None content with tool_calls
            if fixed_msg.content is None and hasattr(fixed_msg, "additional_kwargs") and fixed_msg.additional_kwargs.get("tool_calls"):
                fixed_msg.content = ""  # Change None to empty string
                logger.debug("Changed None content to empty string")
            
            # Issue 2: Content is a list (structured content)
            elif isinstance(fixed_msg.content, list):
                # Extract text from parts
                text_parts = []
                for part in fixed_msg.content:
                    logger.debug(f"Processing content part: {type(part)}")
                    if isinstance(part, dict) and "type" in part and part["type"] == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                    else:
                        logger.warning(f"Unhandled content part: {part}")
                
                # Join the text parts
                fixed_msg.content = " ".join(text_parts) if text_parts else ""
                logger.debug(f"Converted list content to string: {fixed_msg.content[:50]}...")
                
        fixed_messages.append(fixed_msg)
    
    logger.debug(f"Fixed {len(fixed_messages)} messages")
    return fixed_messages


# DISABLED FOR DEBUGGING
# def retry_on_ratelimit(max_retries=3, initial_delay=1):
#     def decorator(fn):
#         @wraps(fn)
#         async def wrapper(*args, **kwargs):
#             delay = initial_delay
#             for _ in range(max_retries):
#                 result = await fn(*args, **kwargs)
#                 if isinstance(result, str) and "202" in result:
#                     await asyncio.sleep(delay)
#                     delay *= 2
#                     continue
#                 return result
#             return result
#         return wrapper
#     return decorator

# No-op replacement for debugging
def retry_on_ratelimit(max_retries=3, initial_delay=1):
    """Disabled retry decorator for debugging - passes through to original function"""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            logger.debug(f"DISABLED RETRY (graph.py): Direct call to {fn.__name__}")
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
