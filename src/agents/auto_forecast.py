"""Auto-Forecast Agent: Autonomous forecasting with alerts."""
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import polars as pl
import pandas as pd
import numpy as np
from loguru import logger

# Conditional imports for advanced models
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    logger.warning("Chronos/transformers not available; using statistical models")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available; install for full forecasting")

from .alert_manager import AlertManager, get_alert_manager

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")


class ForecastResult:
    """Container for a single forecast result."""

    def __init__(
        self,
        deal_name: str,
        forecast_date: date,
        forecast_value: float,
        lower_bound: float,
        upper_bound: float,
        method: str,
        confidence: float,
        regime: str = "normal"
    ):
        self.deal_name = deal_name
        self.forecast_date = forecast_date
        self.forecast_value = forecast_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.method = method
        self.confidence = confidence
        self.regime = regime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deal_name": self.deal_name,
            "forecast_date": self.forecast_date.isoformat(),
            "forecast_value": self.forecast_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "method": self.method,
            "confidence": self.confidence,
            "regime": self.regime,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }


class AutoForecastAgent:
    """Autonomous forecasting agent with alerting capabilities."""

    def __init__(
        self,
        config_path: str = "config.json",
        measures_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """Initialize the auto-forecast agent."""
        self.config = self._load_config(config_path)
        self.forecast_config = self.config.get("forecasting", {})
        self.data_config = self.config.get("data", {})

        # Set paths
        self.measures_path = measures_path or self.data_config.get("measures_path", "results/measures.csv")
        self.metadata_path = metadata_path or self.data_config.get("metadata_path", "results/metadata.csv")
        self.forecast_output_path = Path(self.data_config.get("forecast_output_path", "results/forecasts"))
        self.history_path = Path(self.data_config.get("history_path", "results/history"))
        self.forecast_output_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.alert_manager = get_alert_manager(config_path)
        self.measures: Optional[pl.DataFrame] = None
        self.metadata: Optional[pl.DataFrame] = None
        self.forecast_history: List[ForecastResult] = []
        self.run_timestamp = datetime.utcnow()

        logger.info(f"AutoForecastAgent initialized at {self.run_timestamp.isoformat()}Z")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Config file {config_path} not found; using defaults")
        return {}

    def load_data(self) -> None:
        """Load measures and metadata."""
        logger.info(f"Loading measures from {self.measures_path}")
        if not os.path.exists(self.measures_path):
            raise FileNotFoundError(f"Measures file not found: {self.measures_path}")

        self.measures = pl.read_csv(self.measures_path)
        logger.success(f"Loaded {len(self.measures)} measure records")

        if os.path.exists(self.metadata_path):
            self.metadata = pl.read_csv(self.metadata_path)
            logger.success(f"Loaded {len(self.metadata)} deals metadata")
        else:
            logger.warning(f"Metadata not found at {self.metadata_path}")

    def prepare_timeseries(
        self,
        deal_name: str,
        measure_filter: Optional[str] = None
    ) -> Tuple[np.ndarray, List[date]]:
        """Extract time series for a specific deal."""
        deal_data = self.measures.filter(pl.col("Deal Name") == deal_name)

        if measure_filter:
            deal_data = deal_data.filter(pl.col("Measure") == measure_filter)

        # Sort by date
        deal_data = deal_data.sort("Date")

        # Extract values and dates
        values = deal_data["Amount"].to_numpy()
        dates = [datetime.strptime(d, "%Y-%m-%d").date()
                 for d in deal_data["Date"].to_list()]

        return values, dates

    def detect_regime(
        self,
        values: np.ndarray,
        window: int = 12
    ) -> Tuple[str, Dict[str, float]]:
        """Detect current regime based on volatility and trend.

        Returns:
            Tuple of (regime_name, metrics_dict)
        """
        if len(values) < window * 2:
            return "insufficient_data", {"data_points": len(values)}

        # Calculate rolling statistics
        rolling_vol = np.std(np.diff(values[-window*2:])) / np.abs(np.mean(values[-window:]) + 1e-8)
        rolling_trend = np.polyfit(range(window), values[-window:], 1)[0]

        # Regime classification
        metrics = {
            "volatility": float(rolling_vol),
            "trend": float(rolling_trend),
            "data_points": len(values)
        }

        threshold = self.forecast_config.get("regime_change_threshold", 2.0)

        if rolling_vol > threshold * 2:
            regime = "high_volatility"
        elif rolling_vol > threshold:
            regime = "elevated_volatility"
        elif rolling_trend > 0:
            regime = "growth"
        elif rolling_trend < 0:
            regime = "decline"
        else:
            regime = "stable"

        return regime, metrics

    def forecast_sarimax(
        self,
        values: np.ndarray,
        horizon: int = 12,
        confidence: float = 0.9
    ) -> Tuple[float, float, float]:
        """Forecast using SARIMAX model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for SARIMAX")

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12) if len(values) >= 24 else (0, 0, 0, 0)

        try:
            model = SARIMAX(
                values,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False, maxiter=100)

            forecast = results.get_forecast(steps=horizon)
            mean_forecast = forecast.predicted_mean[-1]
            conf_int = forecast.conf_int(alpha=1 - confidence)
            lower = conf_int.iloc[-1, 0]
            upper = conf_int.iloc[-1, 1]

            return mean_forecast, lower, upper
        except Exception as e:
            logger.error(f"SARIMAX failed: {e}")
            # Fallback to simple exponential smoothing
            return self._forecast_simple(values, horizon, confidence)

    def _forecast_simple(
        self,
        values: np.ndarray,
        horizon: int,
        confidence: float
    ) -> Tuple[float, float, float]:
        """Simple fallback forecast using last value + trend."""
        if len(values) < 2:
            return values[-1] if len(values) > 0 else 0.0, 0.0, 0.0

        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend = coeffs[0]
        last_value = values[-1]

        forecast = last_value + trend * horizon

        # Confidence interval based on recent residuals
        residuals = values - (coeffs[0] * x + coeffs[1])
        std_resid = np.std(residuals)
        z_score = 1.645 if confidence == 0.9 else 1.96  # 90% or 95%
        margin = z_score * std_resid * np.sqrt(horizon)

        return forecast, forecast - margin, forecast + margin

    def forecast_deal(
        self,
        deal_name: str,
        horizon_months: Optional[int] = None
    ) -> Optional[ForecastResult]:
        """Generate forecast for a single deal."""
        if self.measures is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        horizon = horizon_months or self.forecast_config.get("horizon_months", 12)

        # Get capital call/commitment time series (most relevant for forecasting)
        values, dates = self.prepare_timeseries(deal_name, measure_filter="Capital Call")

        if len(values) < 3:
            logger.warning(f"Insufficient data for {deal_name}: {len(values)} points")
            return None

        # Detect current regime
        regime, metrics = self.detect_regime(values)
        logger.info(f"{deal_name}: regime={regime}, metrics={metrics}")

        # Generate forecast
        try:
            if STATSMODELS_AVAILABLE and len(values) >= 12:
                forecast_val, lower, upper = self.forecast_sarimax(values, horizon)
                method = "sarimax"
            else:
                forecast_val, lower, upper = self._forecast_simple(values, horizon)
                method = "simple_trend"
        except Exception as e:
            logger.error(f"Forecast failed for {deal_name}: {e}")
            return None

        # The forecast date is the last date + horizon months
        forecast_date = dates[-1]
        # Simple month addition (approximate)
        month = forecast_date.month + horizon
        year = forecast_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        forecast_date = date(year, month, 1)

        result = ForecastResult(
            deal_name=deal_name,
            forecast_date=forecast_date,
            forecast_value=forecast_val,
            lower_bound=lower,
            upper_bound=upper,
            method=method,
            confidence=self.forecast_config.get("confidence_level", 0.9),
            regime=regime
        )

        logger.success(
            f"Forecast for {deal_name}: ${forecast_val:,.2f} "
            f"(90% CI: ${lower:,.2f} - ${upper:,.2f}) [{method}]"
        )
        return result

    def compare_with_actuals(
        self,
        forecast_results: List[ForecastResult],
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Compare recent forecasts with actuals that have materialized.

        Args:
            forecast_results: List of generated forecasts
            lookback_days: How many days back to check for actuals

        Returns:
            List of deviation alerts that exceeded threshold
        """
        if self.measures is None:
            return []

        threshold_pct = self.forecast_config.get("actuals_vs_forecast_deviation_threshold_pct", 15.0)
        deviations = []

        for fr in forecast_results:
            # Check if actuals exist for forecast_date (if it's in the past)
            today = date.today()
            if fr.forecast_date > today:
                continue  # Forecast is in the future, no actuals yet

            actual_data = self.measures.filter(
                (pl.col("Deal Name") == fr.deal_name) &
                (pl.col("Date") == fr.forecast_date.isoformat())
            )

            if len(actual_data) == 0:
                continue

            actual_value = actual_data["Amount"].sum()

            if fr.forecast_value == 0:
                continue

            deviation_pct = ((actual_value - fr.forecast_value) / abs(fr.forecast_value)) * 100

            if abs(deviation_pct) > threshold_pct:
                alert_data = {
                    "deal_name": fr.deal_name,
                    "forecast_date": fr.forecast_date.isoformat(),
                    "forecast_value": float(fr.forecast_value),
                    "actual_value": float(actual_value),
                    "deviation_pct": float(deviation_pct),
                    "threshold_pct": threshold_pct
                }
                deviations.append(alert_data)
                logger.warning(
                    f"Deviation alert for {fr.deal_name}: {deviation_pct:+.1f}% "
                    f"(threshold {threshold_pct:.1f}%)"
                )

        return deviations

    def save_forecasts(
        self,
        forecast_results: List[ForecastResult],
        filename: Optional[str] = None
    ) -> str:
        """Save forecasts to CSV."""
        if filename is None:
            timestamp = self.run_timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"forecasts_{timestamp}.csv"

        output_path = self.forecast_output_path / filename

        records = [fr.to_dict() for fr in forecast_results]
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        logger.success(f"Saved {len(forecast_results)} forecasts to {output_path}")
        return str(output_path)

    def update_history(
        self,
        forecast_results: List[ForecastResult],
        deviations: List[Dict[str, Any]]
    ) -> None:
        """Append forecast results to historical tracking."""
        # Forecast history
        forecast_history_path = self.history_path / "forecasts.csv"
        records = [fr.to_dict() for fr in forecast_results]

        if records:
            df = pd.DataFrame(records)
            header = not forecast_history_path.exists()
            df.to_csv(forecast_history_path, mode='a', header=header, index=False)

        # Deviation history
        if deviations:
            deviation_path = self.history_path / "deviations.csv"
            dev_df = pd.DataFrame(deviations)
            dev_df["run_timestamp"] = self.run_timestamp.isoformat() + "Z"
            header = not deviation_path.exists()
            dev_df.to_csv(deviation_path, mode='a', header=header, index=False)

        logger.success("History updated")

    def run(
        self,
        run_all: bool = True,
        specific_deals: Optional[List[str]] = None,
        send_alerts: bool = True
    ) -> Dict[str, Any]:
        """Main entry point - run the forecasting workflow."""
        logger.info("=" * 60)
        logger.info("AUTO-FORECAST AGENT: Starting run")
        logger.info("=" * 60)

        try:
            # 1. Load data
            self.load_data()

            # 2. Determine which deals to forecast
            if specific_deals:
                deals_to_process = specific_deals
            elif run_all and self.metadata is not None:
                deals_to_process = self.metadata["Deal Name"].unique().to_list()
            else:
                logger.error("No deals specified for forecasting")
                return {"error": "no_deals_specified", "forecasts": []}

            logger.info(f"Processing {len(deals_to_process)} deals")

            # 3. Generate forecasts
            forecast_results = []
            for deal in deals_to_process:
                try:
                    fr = self.forecast_deal(deal)
                    if fr:
                        forecast_results.append(fr)
                except Exception as e:
                    logger.error(f"Failed to forecast {deal}: {e}")

            logger.success(f"Generated {len(forecast_results)} forecasts")

            # 4. Compare with actuals (for past forecasts)
            deviations = self.compare_with_actuals(forecast_results)
            logger.info(f"Found {len(deviations)} significant deviations")

            # 5. Send alerts if enabled
            alerts_triggered = 0
            if send_alerts:
                # Alert for deviations
                for dev in deviations:
                    result = self.alert_manager.send_forecast_deviation_alert(
                        deal_name=dev["deal_name"],
                        forecast_value=dev["forecast_value"],
                        actual_value=dev["actual_value"],
                        deviation_pct=dev["deviation_pct"],
                        threshold_pct=dev["threshold_pct"]
                    )
                    if any(result.values()):
                        alerts_triggered += 1

                # Send summary report
                summary = {
                    "deals_count": len(forecast_results),
                    "forecasts_generated": len(forecast_results),
                    "deviations_found": len(deviations),
                    "run_timestamp": self.run_timestamp.isoformat()
                }
                self.alert_manager.send_forecast_summary_report(
                    summary_stats=summary,
                    alerts_triggered=alerts_triggered
                )

            # 6. Save forecasts
            forecast_file = self.save_forecasts(forecast_results)

            # 7. Update history
            self.update_history(forecast_results, deviations)

            # 8. Generate dashboard data
            self._generate_dashboard_data(forecast_results, deviations)

            logger.success("Auto-forecast run completed successfully")
            return {
                "run_timestamp": self.run_timestamp.isoformat() + "Z",
                "forecasts_generated": len(forecast_results),
                "deviations_detected": len(deviations),
                "alerts_sent": alerts_triggered,
                "forecast_file": forecast_file
            }

        except Exception as e:
            logger.error(f"Auto-forecast run failed: {e}")
            self.alert_manager.send_alert(
                title="Auto-Forecast Agent: RUN FAILED",
                message=f"The autonomous forecasting agent encountered an error:\n\n{e}",
                severity="critical",
                metadata={"error": str(e), "timestamp": self.run_timestamp.isoformat() + "Z"}
            )
            raise

    def _generate_dashboard_data(
        self,
        forecast_results: List[ForecastResult],
        deviations: List[Dict[str, Any]]
    ) -> None:
        """Generate JSON data for dashboard consumption."""
        dashboard_data = {
            "last_run": self.run_timestamp.isoformat() + "Z",
            "summary": {
                "total_forecasts": len(forecast_results),
                "deviations": len(deviations),
                "regimes": {}
            },
            "forecasts": [fr.to_dict() for fr in forecast_results],
            "deviations": deviations
        }

        # Count regimes
        for fr in forecast_results:
            reg = fr.regime
            dashboard_data["summary"]["regimes"][reg] = dashboard_data["summary"]["regimes"].get(reg, 0) + 1

        # Save dashboard data
        dashboard_path = self.forecast_output_path / "dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        logger.success(f"Dashboard data saved to {dashboard_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Forecast Agent")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--measures", help="Path to measures CSV")
    parser.add_argument("--deals", nargs="+", help="Specific deals to forecast")
    parser.add_argument("--no-alerts", action="store_true", help="Disable alerts for this run")
    args = parser.parse_args()

    agent = AutoForecastAgent(
        config_path=args.config,
        measures_path=args.measures
    )

    result = agent.run(
        run_all=args.deals is None,
        specific_deals=args.deals,
        send_alerts=not args.no_alerts
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()