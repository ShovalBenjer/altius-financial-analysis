import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

import polars as pl
from loguru import logger

from src.agents.auto_forecast import AutoForecastAgent, ScheduleFrequency, RegimeState
from src.agents.alert_manager import AlertManager, AlertSeverity
from src.agents.historical_tracker import HistoricalTracker


class DashboardHandler(BaseHTTPRequestHandler):
    agent: Optional[AutoForecastAgent] = None

    def log_message(self, format, *args):
        logger.debug(f"Dashboard {self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        routes = {
            "/api/status": self._handle_status,
            "/api/forecasts": self._handle_forecasts,
            "/api/forecasts/latest": self._handle_latest_forecasts,
            "/api/deviations": self._handle_deviations,
            "/api/deviations/stats": self._handle_deviation_stats,
            "/api/regime": self._handle_regime,
            "/api/regime/history": self._handle_regime_history,
            "/api/alerts": self._handle_alerts,
            "/api/health": self._handle_health,
        }

        handler = routes.get(path)
        if handler:
            try:
                handler(params)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {
            "/api/forecast/run": self._handle_run_forecast,
            "/api/forecast/compare": self._handle_compare_actuals,
            "/api/alert/test": self._handle_test_alert,
        }

        handler = routes.get(path)
        if handler:
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
                handler(body)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)

    def _handle_status(self, params):
        status = self.agent.get_status()
        status["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._send_json(status)

    def _handle_forecasts(self, params):
        deal_name = params.get("deal_name", [None])[0]
        limit = int(params.get("limit", ["100"])[0])
        tracker = self.agent.historical_tracker
        records = tracker.load_forecasts(deal_name=deal_name, limit=limit)
        self._send_json({"count": len(records), "forecasts": records})

    def _handle_latest_forecasts(self, params):
        n = int(params.get("n", ["5"])[0])
        tracker = self.agent.historical_tracker
        records = tracker.get_latest_forecasts(n=n)
        self._send_json({"count": len(records), "forecasts": records})

    def _handle_deviations(self, params):
        deal_name = params.get("deal_name", [None])[0]
        tracker = self.agent.historical_tracker
        df = tracker.get_deviations_as_df(deal_name)
        if df.is_empty():
            self._send_json({"count": 0, "deviations": []})
        else:
            records = df.to_dicts()
            self._send_json({"count": len(records), "deviations": records})

    def _handle_deviation_stats(self, params):
        deal_name = params.get("deal_name", [None])[0]
        stats = self.agent.historical_tracker.compute_deviation_stats(deal_name)
        self._send_json(stats)

    def _handle_regime(self, params):
        current_regime = self.agent.regime_detector._previous_regime
        self._send_json({
            "current_regime": current_regime.value if current_regime else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_regime_history(self, params):
        limit = int(params.get("limit", ["50"])[0])
        tracker = self.agent.historical_tracker
        records = tracker.load_regime_changes(limit=limit)
        self._send_json({"count": len(records), "regime_changes": records})

    def _handle_alerts(self, params):
        limit = int(params.get("limit", ["50"])[0])
        history = self.agent.alert_manager.get_alert_history(limit=limit)
        self._send_json({"count": len(history), "alerts": history})

    def _handle_health(self, params):
        self._send_json({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_schedule": self.agent.schedule.value,
        })

    def _handle_run_forecast(self, body):
        config = json.loads(body) if body else {}
        measures_path = config.get("measures_path", "measures.csv")
        metadata_path = config.get("metadata_path", "metadata.csv")
        macro_data = config.get("macro_data")
        results = self.agent.run_forecast_cycle(
            measures_path=measures_path,
            metadata_path=metadata_path,
            macro_data=macro_data,
        )
        response = {
            "run_id": HistoricalTracker.generate_run_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "forecast_count": len(results),
            "forecasts": [
                {
                    "deal_name": r.deal_name,
                    "p10": r.p10,
                    "p50": r.p50,
                    "p90": r.p90,
                    "mean": r.mean,
                    "regime": r.regime,
                    "model_name": r.model_name,
                }
                for r in results
            ],
        }
        self._send_json(response, status=201)

    def _handle_compare_actuals(self, body):
        config = json.loads(body) if body else {}
        actuals_path = config.get("actuals_path")
        deviations = self.agent.compare_forecasts_to_actuals(actuals_path=actuals_path)
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deviation_count": len(deviations),
            "deviations": [
                {
                    "deal_name": d.deal_name,
                    "forecast_value": d.forecast_value,
                    "actual_value": d.actual_value,
                    "deviation_pct": d.deviation_pct,
                    "severity": d.severity,
                }
                for d in deviations
            ],
        }
        self._send_json(response)

    def _handle_test_alert(self, body):
        config = json.loads(body) if body else {}
        from src.agents.alert_manager import Alert, AlertChannel
        channel_str = config.get("channel", "slack")
        channel = AlertChannel(channel_str)
        alert = Alert(
            title=config.get("title", "Test Alert"),
            message=config.get("message", "This is a test alert from Chronos AutoForecast"),
            severity=AlertSeverity(config.get("severity", "info")),
            channel=channel,
        )
        success = self.agent.alert_manager.send_alert(alert)
        self._send_json({"sent": success, "alert": alert.to_dict()})

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))


class DashboardServer:
    def __init__(
        self,
        agent: AutoForecastAgent,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.agent = agent
        self.host = host
        self.port = port

    def serve(self):
        DashboardHandler.agent = self.agent
        server = HTTPServer((self.host, self.port), DashboardHandler)
        logger.info(f"DashboardServer starting on {self.host}:{self.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("DashboardServer shutting down")
            server.shutdown()

    def serve_background(self):
        import threading
        DashboardHandler.agent = self.agent
        server = HTTPServer((self.host, self.port), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"DashboardServer running in background on {self.host}:{self.port}")
        return server


def main():
    logger.configure(
        handlers=[{"sink": sys.stderr, "format": "<green>{time}</green> <level>{message}</level>"}]
    )
    logger.info("Initializing AutoForecastAgent for dashboard...")
    agent = AutoForecastAgent(
        schedule=ScheduleFrequency.WEEKLY,
        storage_dir="results",
    )
    server = DashboardServer(agent, host="0.0.0.0", port=8080)
    logger.info("Dashboard endpoints available:")
    for endpoint in [
        "GET  /api/status          - Agent status",
        "GET  /api/forecasts       - Forecast history",
        "GET  /api/forecasts/latest - Latest forecasts",
        "GET  /api/deviations      - Deviation history",
        "GET  /api/deviations/stats - Deviation statistics",
        "GET  /api/regime           - Current regime",
        "GET  /api/regime/history   - Regime change history",
        "GET  /api/alerts           - Alert history",
        "GET  /api/health           - Health check",
        "POST /api/forecast/run     - Trigger forecast cycle",
        "POST /api/forecast/compare - Compare forecasts to actuals",
        "POST /api/alert/test       - Send test alert",
    ]:
        logger.info(f"  {endpoint}")
    server.serve()


if __name__ == "__main__":
    main()