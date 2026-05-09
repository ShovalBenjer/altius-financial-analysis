"""Dashboard API for auto-forecast agent data."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

import polars as pl
import pandas as pd

app = FastAPI(
    title="Auto-Forecast Agent API",
    description="Dashboard data endpoints for autonomous forecasting",
    version="1.0.0"
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastResponse(BaseModel):
    """Response model for forecast data."""
    deal_name: str
    forecast_date: str
    forecast_value: float
    lower_bound: float
    upper_bound: float
    method: str
    confidence: float
    regime: str


class DashboardSummary(BaseModel):
    """Summary statistics for dashboard."""
    total_forecasts: int
    total_deviations: int
    last_run: str
    regimes: Dict[str, int]


class RunRequest(BaseModel):
    """Request to trigger a forecast run."""
    deals: Optional[List[str]] = None
    send_alerts: bool = True
    horizon_months: Optional[int] = None


def get_config() -> Dict[str, Any]:
    """Load configuration."""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_forecast_output_path() -> Path:
    """Get the forecast output directory."""
    config = get_config()
    return Path(config.get("data", {}).get("forecast_output_path", "results/forecasts"))


def get_history_path() -> Path:
    """Get the history directory."""
    config = get_config()
    return Path(config.get("data", {}).get("history_path", "results/history"))


@app.get("/")
def root() -> Dict[str, Any]:
    """API root with info."""
    return {
        "service": "Auto-Forecast Agent API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "forecasts": "/api/forecasts",
            "dashboard": "/api/dashboard",
            "deviations": "/api/deviations",
            "run": "/api/run (POST)",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/forecasts", response_model=List[ForecastResponse])
def get_forecasts(
    deal: Optional[str] = Query(None, description="Filter by deal name"),
    limit: int = Query(100, description="Maximum number of results"),
    regime: Optional[str] = Query(None, description="Filter by regime")
) -> List[Dict[str, Any]]:
    """Get latest forecasts, optionally filtered."""
    dashboard_path = get_forecast_output_path() / "dashboard.json"

    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="No forecast data available. Run the agent first.")

    with open(dashboard_path, 'r') as f:
        data = json.load(f)

    forecasts = data.get("forecasts", [])

    # Apply filters
    if deal:
        forecasts = [f for f in forecasts if f["deal_name"] == deal]
    if regime:
        forecasts = [f for f in forecasts if f["regime"] == regime]

    # Return latest N
    return forecasts[-limit:]


@app.get("/api/dashboard", response_model=DashboardSummary)
def get_dashboard_summary() -> Dict[str, Any]:
    """Get dashboard summary statistics."""
    dashboard_path = get_forecast_output_path() / "dashboard.json"

    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="No dashboard data available")

    with open(dashboard_path, 'r') as f:
        data = json.load(f)

    summary = data.get("summary", {})
    summary["last_run"] = data.get("last_run", "")

    # Also load deviations count from history if not in dashboard
    deviation_path = get_history_path() / "deviations.csv"
    if deviation_path.exists():
        dev_df = pd.read_csv(deviation_path)
        summary["total_deviations"] = len(dev_df)

    return summary


@app.get("/api/deviations")
def get_deviations(
    limit: int = Query(50, description="Maximum number of results"),
    deal: Optional[str] = Query(None, description="Filter by deal name")
) -> List[Dict[str, Any]]:
    """Get historical forecast deviations."""
    deviation_path = get_history_path() / "deviations.csv"

    if not deviation_path.exists():
        return []

    df = pd.read_csv(deviation_path)
    deviations = df.to_dict('records')

    if deal:
        deviations = [d for d in deviations if d.get("deal_name") == deal]

    return deviations[-limit:]


@app.get("/api/history/forecasts")
def get_forecast_history(
    deal: Optional[str] = Query(None, description="Filter by deal name"),
    days: int = Query(30, description="Number of days to look back")
) -> List[Dict[str, Any]]:
    """Get historical forecast data."""
    history_path = get_history_path() / "forecasts.csv"

    if not history_path.exists():
        return []

    df = pd.read_csv(history_path)
    df["generated_at"] = pd.to_datetime(df["generated_at"])

    cutoff = datetime.utcnow() - pd.Timedelta(days=days)
    df = df[df["generated_at"] > cutoff]

    if deal:
        df = df[df["deal_name"] == deal]

    return df.to_dict('records')


@app.get("/api/regimes")
def get_regime_transitions() -> List[Dict[str, Any]]:
    """Get regime transition history."""
    history_path = get_history_path() / "forecasts.csv"

    if not history_path.exists():
        return []

    df = pd.read_csv(history_path)
    df["generated_at"] = pd.to_datetime(df["generated_at"])

    # Find regime changes per deal
    transitions = []
    for deal in df["deal_name"].unique():
        deal_df = df[df["deal_name"] == deal].sort_values("generated_at")
        if len(deal_df) < 2:
            continue

        prev_regime = None
        for _, row in deal_df.iterrows():
            if prev_regime and row["regime"] != prev_regime:
                transitions.append({
                    "deal_name": deal,
                    "timestamp": row["generated_at"].isoformat() + "Z",
                    "from_regime": prev_regime,
                    "to_regime": row["regime"]
                })
            prev_regime = row["regime"]

    return transitions


@app.post("/api/run")
def trigger_forecast_run(request: RunRequest) -> Dict[str, Any]:
    """Trigger a new forecast run."""
    try:
        from src.agents.auto_forecast import AutoForecastAgent

        agent = AutoForecastAgent()
        result = agent.run(
            run_all=request.deals is None,
            specific_deals=request.deals,
            send_alerts=request.send_alerts
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Manual run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/history")
def get_alert_history(limit: int = Query(50)) -> List[Dict[str, Any]]:
    """Get historical alerts."""
    history_path = get_history_path() / "alerts.csv"

    if not history_path.exists():
        return []

    df = pd.read_csv(history_path)
    return df.tail(limit).to_dict('records')


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    dashboard_config = config.get("dashboard", {})
    host = dashboard_config.get("host", "0.0.0.0")
    port = dashboard_config.get("port", 8000)

    uvicorn.run(app, host=host, port=port)