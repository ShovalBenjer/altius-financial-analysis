#!/usr/bin/env python3
"""Run the dashboard API server."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.dashboard_api import app as fastapi_app
import uvicorn

if __name__ == "__main__":
    import json
    config_path = project_root / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        dashboard_config = config.get("dashboard", {})
        host = dashboard_config.get("host", "0.0.0.0")
        port = dashboard_config.get("port", 8000)
        debug = dashboard_config.get("debug", False)
    else:
        host, port, debug = "0.0.0.0", 8000, False

    print(f"Starting Dashboard API on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port, reload=debug)