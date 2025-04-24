"""This file contains message conversion utilities for different LLM providers."""

from typing import (
    Any,
    Dict,
    List,
)


def to_ollama_parts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every message.content is a list of {type,text} parts.

    Ollama's streaming helper expects this even for plain text.
    
    Args:
        messages (List[Dict[str, Any]]): Messages to normalize
        
    Returns:
        List[Dict[str, Any]]: Messages with content normalized for Ollama
        
    Notes:
        The problem occurs in langchain_ollama.chat_models._convert_messages_to_ollama_messages
        where it iterates over message["content"] and calls content_part.get("type").
        If content is a string, it fails with AttributeError: 'str' object has no attribute 'get'.
    """
    normalised = []
    for msg in messages:
        # Clone the message to avoid modifying the original
        new_msg = msg.copy()
        
        # Handle the case of string content by converting to the format Ollama expects
        if isinstance(new_msg.get("content"), str):
            new_msg["content"] = [{"type": "text", "text": new_msg["content"]}]
        # Handle the case of None content for tool calls
        elif new_msg.get("content") is None and "tool_calls" in new_msg:
            # Keep tool_calls but ensure content is an empty list, not None
            new_msg["content"] = []
        
        normalised.append(new_msg)
    return normalised 