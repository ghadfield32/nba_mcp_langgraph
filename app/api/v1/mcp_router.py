from typing import Optional

from fastapi import (
    APIRouter,
    Query,
)

from app.services.mcp.nba_mcp.nba_server import (
    LeagueLeadersParams,
    get_date_range_game_log_or_team_game_log,
    get_league_leaders_info,
    get_live_scores,
    get_player_career_information,
    play_by_play,
)

router = APIRouter()

@router.get("/player-career")
async def player_career_info(player_name: str, season: Optional[str] = None):
    """Fetch player career information via MCP."""
    result = await get_player_career_information(player_name, season)
    return {"result": result}

@router.post("/league-leaders")
async def league_leaders_info(params: LeagueLeadersParams):
    """Get top 10 league leaders based on stats."""
    result = await get_league_leaders_info(params)
    return {"result": result}

@router.get("/live-scores")
async def live_scores(target_date: Optional[str] = Query(None)):
    """Fetch live or historical scores."""
    result = await get_live_scores(target_date)
    return {"result": result}

@router.get("/game-logs")
async def game_logs(season: str, team: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None):
    """Get game logs for date range or team."""
    result = await get_date_range_game_log_or_team_game_log(season, team, date_from, date_to)
    return {"result": result}

@router.get("/play-by-play")
async def play_by_play_endpoint(
    game_date: Optional[str] = None,
    team: Optional[str] = None,
    start_period: int = 1,
    end_period: int = 4,
    start_clock: Optional[str] = None,
    recent_n: int = 5,
    max_lines: int = 200
):
    """Get play-by-play data."""
    result = await play_by_play(
        game_date, team, start_period, end_period,
        start_clock, recent_n, max_lines
    )
    return {"result": result}

@router.get("/diagnostic")
async def mcp_router_diagnostic():
    """Diagnostic endpoint to check MCP tools directly from the MCP router."""
    try:
        tools = await mcp_server.get_tools()
        
        # Handle both string tools and object tools
        tool_names = []
        for t in tools:
            if isinstance(t, str):
                tool_names.append(t)
            else:
                # Use getattr with fallback to handle objects that might not have a name attr
                tool_names.append(getattr(t, "name", str(t)))
                
        return {
            "tools": tool_names,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
