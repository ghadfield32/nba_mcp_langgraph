"""Tests for the message utils."""

import pytest

from app.utils.message_utils import to_ollama_parts


def test_to_ollama_parts_converts_string_content():
    """Test that to_ollama_parts converts string content to list of parts."""
    messages = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    
    result = to_ollama_parts(messages)
    
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Hello, world!"
    
    assert result[1]["role"] == "assistant"
    assert isinstance(result[1]["content"], list)
    assert result[1]["content"][0]["type"] == "text"
    assert result[1]["content"][0]["text"] == "Hi there!"


def test_to_ollama_parts_preserves_tool_calls():
    """Test that to_ollama_parts preserves tool calls."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"name": "search", "id": "123", "args": {"query": "weather"}}
            ]
        }
    ]
    
    result = to_ollama_parts(messages)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] is None
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["name"] == "search"
    assert result[0]["tool_calls"][0]["args"]["query"] == "weather"


def test_to_ollama_parts_preserves_existing_parts():
    """Test that to_ollama_parts preserves existing parts structure."""
    messages = [
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": "Here's the information you requested:"},
                {"type": "image", "url": "http://example.com/image.jpg"}
            ]
        }
    ]
    
    result = to_ollama_parts(messages)
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][1]["type"] == "image" 