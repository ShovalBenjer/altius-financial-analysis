# Auto-Forecast Agent with Alerting System

Autonomous forecasting with notifications for private equity risk management.

## Overview

The Auto-Forecast Agent runs scheduled forecasts on PE deals, compares predictions against actuals, detects regime changes, and sends alerts on significant deviations. Built to replace manual forecasting runs.

## Architecture

```
src/agents/
├── auto_forecast.py      # Main agent (forecasting, regime detection, orchestration)
├── alert_manager.py      # Multi-channel alerting (Slack, Email, Webhook)
├── dashboard_api.py      # FastAPI endpoints for dashboard integration
└── __init__.py           # Package exports

results/
├── forecasts/            # Generated forecast CSVs
├── history/              # Historical tracking (forecasts.csv, deviations.csv, alerts.csv)
└── auto_forecast.log     # Agent execution logs
```

## Features

### 1. Forecasting Workflow

- **Weekly/Monthly runs**: Execute via cron/systemd scheduler
- **Model selection**: Uses SARIMAX (when data sufficient) with fallback to simple trend
- **Time series features**: Automatic seasonal decomposition, PACF-based lag selection
- **Confidence intervals**: 90% prediction intervals via statsmodels

### 2. Forecast Deviation Detection

- Compares historical forecasts against actualized values
- Flags when deviation exceeds configurable threshold (default: 15%)
- Automatic alert generation on significant deviations

### 3. Regime Change Detection

Real-time regime classification based on:
- Rolling volatility (annualized)
- Trend direction and magnitude
- Adaptive thresholds

Regimes: `growth`, `decline`, `stable`, `elevated_volatility`, `high_volatility`

### 4. AlertManager

Multi-channel notifications:
- **Slack**: Rich attachments with color-coded severity
- **Email**: HTML emails via SMTP (Gmail, SendGrid, etc.)
- **Webhook**: Generic HTTP POST for custom integrations (PagerDuty, Teams, etc.)

Each alert is logged to `results/history/alerts.csv`.

### 5. Dashboard API

FastAPI server exposing JSON endpoints:
- `GET /api/forecasts` — Latest forecast data
- `GET /api/dashboard` — Summary statistics
- `GET /api/deviations` — Historical deviations
- `GET /api/history/forecasts` — All forecast history
- `GET /api/regimes` — Regime transition events
- `POST /api/run` — Manually trigger a forecast run

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "data": {
    "measures_path": "results/measures.csv",
    "metadata_path": "results/metadata.csv"
  },
  "forecasting": {
    "horizon_months": 12,
    "confidence_level": 0.9,
    "min_history_points": 24,
    "actuals_vs_forecast_deviation_threshold_pct": 15.0,
    "regime_change_threshold": 2.0
  },
  "alerts": {
    "email": {"enabled": false, ...},
    "slack": {"enabled": false, ...},
    "webhook": {"enabled": false, ...}
  },
  "dashboard": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

Sensitive credentials (SMTP password, webhook URLs) can be set via environment variables:
- `SMTP_PASSWORD`
- `SLACK_WEBHOOK_URL`
- `WEBHOOK_URL`

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies:
- `polars` — Fast data processing
- `pandas` — Data handling
- `statsmodels` — SARIMAX forecasting
- `fastapi` + `uvicorn` — Dashboard API
- `loguru` — Structured logging

Optional: `transformers` + `torch` for Chronos LLM-based forecasting (future feature).

## Usage

### Run a single forecast (CLI)

```bash
python run_auto_forecast.py
python run_auto_forecast.py --deals "Deal A" "Deal B"  # Specific deals
python run_auto_forecast.py --no-alerts  # Run without sending alerts
```

### Schedule with cron (recommended)

```bash
# Weekly forecast on Monday at 2 AM
0 2 * * 1 cd /path/to/project && python run_auto_forecast.py >> results/cron.log 2>&1

# Monthly forecast on the 1st
0 2 1 * * cd /path/to/project && python run_auto_forecast.py >> results/cron.log 2>&1
```

### Start the dashboard API

```bash
python run_dashboard.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Trigger a run via API

```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"send_alerts": true, "deals": null}'
```

## Outputs

After each run:

- `results/forecasts/forecasts_YYYYMMDD_HHMMSS.csv` — Raw forecast values
- `results/forecasts/dashboard.json` — Latest consolidated data (served by API)
- `results/history/forecasts.csv` — Append-only forecast history
- `results/history/deviations.csv` — Recorded forecast errors
- `results/history/alerts.csv` — Alert delivery log
- `results/auto_forecast.log` — Execution logs

## Alert Types

1. **Forecast Deviation** — Actual value differs from forecast by > threshold
2. **Regime Change** — Time series regime transition detected
3. **Forecast Summary** — Periodic summary report (info level)
4. **Critical** — Agent execution failure

## Integration with Existing Pipeline

The agent consumes:
- `results/measures.csv` — Output from `transform.py`
- `results/metadata.csv` — Deal metadata

It produces forecast data that can be:
- Viewed in Looker Studio via JSON/REST connector
- Consumed by downstream risk models
- Tracked for historical audit

## Troubleshooting

**Insufficient data warning**: Agent needs at least 3 data points; SARIMAX needs 12+ points.

**Import errors**: Install all deps from `requirements.txt`.

**No forecasts generated**: Verify measures CSV exists and contains "Capital Call" or "NAV" measures.

**Alerts not firing**: Check alert channel config in `config.json` and environment variables.

## Extending

Add new alert channels by extending `AlertManager._send_*` methods.

Add new forecasting models by implementing a method on `AutoForecastAgent` and dispatching in `forecast_deal()`.

Create custom regime detectors by modifying `detect_regime()`.