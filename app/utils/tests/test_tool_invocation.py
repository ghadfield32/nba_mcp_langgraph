"""
Tests for the tool invocation helpers.

This module verifies that the tool invocation helpers correctly handle
different tool signatures.
"""

import inspect
import sys
import unittest
from typing import (
    Any,
    Dict,
    Optional,
)

sys.path.append(".")  # Ensure we can import from the app package

from app.core.langgraph.graph import (
    _invoke_tool_async,
    _invoke_tool_sync,
)


class DummyToolRunArgs:
    """A tool that expects an 'arguments' parameter in run()."""
    def run(self, arguments: Dict[str, Any]):
        return f"Tool with 'arguments' param: {arguments}"


class DummyToolRunInput:
    """A tool that expects an 'input' parameter in run()."""
    def run(self, input: Dict[str, Any]):
        return f"Tool with 'input' param: {input}"


class DummyToolRunKwargs:
    """A tool that expects specific kwargs in run()."""
    def run(self, date: str, team: Optional[str] = None):
        return f"Tool with kwargs: {date}, {team}"
        
        
class DummyToolWithAsyncMethod:
    """A tool that has both run() and ainvoke()."""
    def run(self, arguments: Dict[str, Any]):
        return f"Tool run with args: {arguments}"
        
    async def ainvoke(self, input: Dict[str, Any]):
        return f"Tool ainvoke with input: {input}"


class TestToolInvocation(unittest.TestCase):
    """Test the tool invocation helpers."""
    
    async def test_invoke_tool_sync_with_arguments_param(self):
        """Test _invoke_tool_sync with a tool that expects an 'arguments' parameter."""
        tool = DummyToolRunArgs()
        args = {"date": "2021-01-01", "team": "Lakers"}
        result = await _invoke_tool_sync(tool, args)
        self.assertEqual(result, "Tool with 'arguments' param: {'date': '2021-01-01', 'team': 'Lakers'}")
        
    async def test_invoke_tool_sync_with_input_param(self):
        """Test _invoke_tool_sync with a tool that expects an 'input' parameter."""
        tool = DummyToolRunInput()
        args = {"date": "2021-01-01", "team": "Lakers"}
        result = await _invoke_tool_sync(tool, args)
        self.assertEqual(result, "Tool with 'input' param: {'date': '2021-01-01', 'team': 'Lakers'}")
        
    async def test_invoke_tool_sync_with_kwargs(self):
        """Test _invoke_tool_sync with a tool that expects specific kwargs."""
        tool = DummyToolRunKwargs()
        args = {"date": "2021-01-01", "team": "Lakers"}
        result = await _invoke_tool_sync(tool, args)
        self.assertEqual(result, "Tool with kwargs: 2021-01-01, Lakers")
        
    async def test_invoke_tool_async_prefers_ainvoke(self):
        """Test _invoke_tool_async prefers ainvoke over run."""
        tool = DummyToolWithAsyncMethod()
        args = {"date": "2021-01-01", "team": "Lakers"}
        result = await _invoke_tool_async(tool, args)
        self.assertEqual(result, "Tool ainvoke with input: {'date': '2021-01-01', 'team': 'Lakers'}")


if __name__ == "__main__":
    unittest.main() 