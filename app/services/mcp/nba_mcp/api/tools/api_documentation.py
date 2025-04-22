"""

location: app\services\mcp\nba_mcp\api\tools\api_documentation.py

"""

import sys
from pathlib import Path

print("=== DEBUG ENTRY ===")
print("  __name__   =", __name__)
print("  __package__=", __package__)
print("  sys.path[0] points to:", sys.path[0])
print("  CWD:", Path().resolve())
print("===================")
    
# tools.api_documentation.py
import inspect
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Optional,
)

import pandas as pd

# Import NBA API modules
from nba_api.stats import endpoints
from nba_api.stats.static import (
    players,
    teams,
)

from .nba_api_utils import (
    get_player_id,
    get_team_id,
    normalize_per_mode,
    normalize_season,
    normalize_stat_category,
)

logger = logging.getLogger(__name__)


def get_endpoint_data_structure(endpoint_class):
    """Get the detailed data structure for an endpoint including metrics and column info.
    This is intended to run once so we can cache the result.
    """
    try:
        # Get the required parameters for the endpoint (if any)
        required_params = getattr(endpoint_class, "_required_parameters", [])

        # Initialize parameters dictionary with default sample values
        params = {}
        for param in required_params:
            param_lower = param.lower()
            if "player_id" in param_lower:
                # Use Nikola Jokić as default example
                params[param] = get_player_id("Nikola Jokić")
            elif "team_id" in param_lower:
                # Use Denver Nuggets as default example
                params[param] = get_team_id("Denver Nuggets")
            elif "game_id" in param_lower:
                # Use a sample game_id (for example, from a recent playoff game)
                params[param] = "0042200401"
            elif "league_id" in param_lower:
                params[param] = "00"  # NBA league ID
            elif "season" in param_lower:
                params[param] = "2022-23"  # Use most recent completed season
            else:
                params[param] = "0"  # Use a generic default value

        # Create an instance of the endpoint
        instance = endpoint_class(**params)

        data_sets = {}
        # Get all available data frames from the endpoint
        all_frames = instance.get_data_frames()
        raw_data = instance.get_dict()

        for idx, df in enumerate(all_frames):
            if df is not None and not df.empty:
                result_set = raw_data["resultSets"][idx]
                data_sets[f"dataset_{idx}"] = {
                    "name": result_set["name"],
                    "headers": result_set["headers"],
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
                    "sample_data": df.head(2).to_dict("records"),
                    "row_count": len(df),
                }

        return {"parameters_used": params, "datasets": data_sets}
    except Exception as e:
        return {"error": str(e)}


def analyze_api_structure() -> dict:
    """
    Analyze the NBA API structure and generate a quick guide for each endpoint.
    The quick guide includes the endpoint URL, parameters, and datasets.
    """
    logger.info("Analyzing NBA API structure…")

    # Get all endpoint classes from the endpoints module
    endpoint_classes = inspect.getmembers(endpoints, inspect.isclass)
    logger.info("Found %d potential endpoint classes", len(endpoint_classes))

    api_docs = {"endpoints": {}, "static_data": {"teams": teams.get_teams(), "players": players.get_players()}}

    for endpoint_name, endpoint_class in endpoint_classes:
        try:
            if not hasattr(endpoint_class, "endpoint"):
                continue

            detailed_data = get_endpoint_data_structure(endpoint_class)

            dataset_names = [ds["name"] for ds in detailed_data.get("datasets", {}).values()]
            description = (
                f"This endpoint returns data related to {', '.join(dataset_names)}."
                if dataset_names
                else "No dataset information available."
            )

            api_docs["endpoints"][endpoint_name] = {
                "endpoint_url": endpoint_class.endpoint,
                "parameters": getattr(endpoint_class, "_required_parameters", []),
                "optional_parameters": getattr(endpoint_class, "_optional_parameters", []),
                "default_parameters": getattr(endpoint_class, "_default_parameters", {}),
                "quick_description": description,
                "data_structure": detailed_data,
            }

        except Exception as e:
            logger.error("Error processing endpoint %s: %s", endpoint_name, e, exc_info=True)

    logger.info("Successfully documented %d endpoints", len(api_docs["endpoints"]))
    return api_docs


def save_documentation(api_docs: dict, output_dir: str = None):
    """
    Save NBA API docs as JSON and Markdown in expected locations.
    
    Files will be saved to:
    1. First location: <workspace_root>/app/services/mcp/api_documentation
    2. Second location: <workspace_root>/app/services/mcp/nba_mcp/api_documentation
    
    This ensures files are available in both locations where the system is looking for them.
    """
    # ─── Add debug information ───────────────────────────────────────────────
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {Path(__file__).resolve()}")
    
    # ─── Determine output folders ────────────────────────────────────────────
    if output_dir:
        # Use explicit output directory if provided
        primary_output_path = Path(output_dir)
        logger.info(f"Using explicitly provided output directory: {primary_output_path.resolve()}")
    else:
        # First expected location (relative to workspace)
        workspace_root = Path(__file__).resolve().parents[5]  # Go up to app/services/mcp/nba_mcp/api/tools
        primary_output_path = workspace_root / "app" / "services" / "mcp" / "api_documentation"
        logger.info(f"Primary output path: {primary_output_path.resolve()}")
        
        # Second expected location (derived from script location)
        script_based_path = Path(__file__).resolve().parent.parent.parent / "api_documentation"
        logger.info(f"Secondary output path (script-based): {script_based_path.resolve()}")
        
        # Save to both locations to ensure availability
        secondary_output_path = workspace_root / "app" / "services" / "mcp" / "nba_mcp" / "api_documentation"
        logger.info(f"Secondary output path (workspace-based): {secondary_output_path.resolve()}")

    # ─── Prepare primary directory ────────────────────────────────────────────
    logger.info(f"Creating output directory: {primary_output_path.resolve()}")
    primary_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Prepare secondary directory (if using default paths) ─────────────────
    if not output_dir:
        logger.info(f"Creating secondary output directory: {secondary_output_path.resolve()}")
        secondary_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Write JSON files to primary location ─────────────────────────────────
    endpoints_file = primary_output_path / "endpoints.json"
    endpoints_file.write_text(
        json.dumps(api_docs["endpoints"], indent=2),
        encoding="utf-8",
    )
    logger.info(f"Primary endpoints.json written to: {endpoints_file.resolve()}")
    
    static_data_file = primary_output_path / "static_data.json"
    static_data_file.write_text(
        json.dumps(api_docs["static_data"], indent=2),
        encoding="utf-8",
    )
    logger.info(f"Primary static_data.json written to: {static_data_file.resolve()}")

    # ─── Write JSON files to secondary location (if using default paths) ──────
    if not output_dir:
        endpoints_file_secondary = secondary_output_path / "endpoints.json"
        endpoints_file_secondary.write_text(
            json.dumps(api_docs["endpoints"], indent=2),
            encoding="utf-8",
        )
        logger.info(f"Secondary endpoints.json written to: {endpoints_file_secondary.resolve()}")
        
        static_data_file_secondary = secondary_output_path / "static_data.json"
        static_data_file_secondary.write_text(
            json.dumps(api_docs["static_data"], indent=2),
            encoding="utf-8",
        )
        logger.info(f"Secondary static_data.json written to: {static_data_file_secondary.resolve()}")

    # ─── Build Markdown content ───────────────────────────────────────────────
    markdown_lines = [
        "# NBA API Documentation Quick Guide\n",
        f"**Generated on:** {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "## Available Endpoints\n",
    ]
    for name, info in api_docs["endpoints"].items():
        markdown_lines += [
            f"### {name}\n",
            f"- **Endpoint URL:** `{info['endpoint_url']}`",
            f"- **Quick Description:** {info['quick_description']}\n",
        ]
        # — parameters
        if info["parameters"]:
            markdown_lines += ["#### Required Parameters\n```json",
                json.dumps(info["parameters"], indent=2),
                "```\n"
            ]
        if info["optional_parameters"]:
            markdown_lines += ["#### Optional Parameters\n```json",
                json.dumps(info["optional_parameters"], indent=2),
                "```\n"
            ]
        if info["default_parameters"]:
            markdown_lines += ["#### Default Parameters\n```json",
                json.dumps(info["default_parameters"], indent=2),
                "```\n"
            ]
        # — example params
        params_used = info["data_structure"].get("parameters_used", {})
        if params_used:
            markdown_lines += ["#### Example Parameters Used\n```json",
                json.dumps(params_used, indent=2),
                "```\n"
            ]
        # — datasets
        datasets = info["data_structure"].get("datasets", {})
        if datasets:
            markdown_lines.append("#### Available Datasets\n")
            for ds in datasets.values():
                markdown_lines += [
                    f"- **{ds['name']}** (Rows: {ds['row_count']})",
                    "```json",
                    json.dumps({"headers": ds["headers"], "sample_data": ds["sample_data"]}, indent=2),
                    "```\n"
                ]

    # ─── Write Markdown file to primary location ────────────────────────────
    md_file = primary_output_path / "api_documentation.md"
    md_file.write_text("\n".join(markdown_lines), encoding="utf-8")
    logger.info(f"Primary Markdown doc written to: {md_file.resolve()}")
    
    # ─── Write Markdown file to secondary location (if using default paths) ─
    if not output_dir:
        md_file_secondary = secondary_output_path / "api_documentation.md"
        md_file_secondary.write_text("\n".join(markdown_lines), encoding="utf-8")
        logger.info(f"Secondary Markdown doc written to: {md_file_secondary.resolve()}")
        


import sys
from pathlib import Path

if __name__ == "__main__":
    print("=== DEBUG ENTRY ===")
    print("  __name__   =", __name__)
    print("  __package__=", __package__)
    print("  sys.path[0] points to:", sys.path[0])
    print("  CWD:", Path().resolve())
    print("===================")
    # Generate the documentation quickly
    api_docs = analyze_api_structure()

    # Display a summary for the console
    print("\nDocumentation Summary:")
    print(f"Total endpoints documented: {len(api_docs['endpoints'])}")
    print(f"Total teams in static data: {len(api_docs['static_data']['teams'])}")
    print(f"Total players in static data: {len(api_docs['static_data']['players'])}")
    print(api_docs["endpoints"])

    # Save to files for quick later retrieval
    save_documentation(api_docs)

    # Display first endpoint details
    # if api_docs['endpoints']:
    #     first_endpoint = next(iter(api_docs['endpoints']))
    #     print(f"\nSample endpoint ({first_endpoint}):")
    #     print(json.dumps(api_docs['endpoints'][first_endpoint], indent=2))
