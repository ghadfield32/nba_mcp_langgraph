# NBA MCP Examples

This directory contains examples for using the NBA MCP (Multimodal Contextual Protocol) API in different ways.

## LangGraph Ollama Agent with NBA MCP Tools

The `langgraph_ollama_agent_w_tools.py` example demonstrates how to create a simple LangGraph agent that connects to the NBA MCP server and uses Ollama as the LLM backend.

### Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -r examples/requirements_examples.txt
   ```

2. Make sure you have [Ollama](https://ollama.ai/) installed and running with the `llama3.2:3b` model (or update the script to use a different model).
   ```bash
   ollama pull llama3.2:3b
   ```

### Running the Example

```bash
# Run with default settings (local mode on port 8001)
python examples/langgraph_ollama_agent_w_tools.py

# Run in claude mode (port 8000)
python examples/langgraph_ollama_agent_w_tools.py --mode claude
```

The script will:
1. Start the NBA MCP server in the specified mode
2. Create a LangGraph agent with the NBA MCP tools
3. Connect to Ollama for LLM queries
4. Let you chat with the agent and use the NBA tools

### Available Tools

The agent has access to the following NBA data tools:

- `get_league_leaders_info` - Get top players for a specific stat category
- `get_player_career_information` - Get a player's career statistics
- `get_live_scores` - Get live or historical NBA game scores
- `get_date_range_game_log_or_team_game_log` - Get game logs for a specific time period
- `play_by_play` - Get play-by-play data for games

### Example Queries

Try asking the agent:
- "Who are the top 10 scorers in the NBA this season?"
- "What were LeBron James' stats in the 2023-24 season?"
- "Show me yesterday's NBA scores"
- "Give me the play-by-play for the Lakers' last game" 