import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.alert_manager import AlertManager, AlertSeverity, AlertChannel
from src.agents.historical_tracker import (
    HistoricalTracker,
    ForecastRecord,
    DeviationRecord,
    RegimeRecord,
)

LLM_MODEL = "z-ai/glm-5.1"


class ScheduleFrequency(Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class RegimeState(Enum):
    BULL = "bull"
    NORMAL = "normal"
    STRESS = "stress"
    CRISIS = "crisis"


@dataclass
class ForecastResult:
    deal_name: str
    forecast_date: str
    horizon: str
    p10: float
    p50: float
    p90: float
    mean: float
    std: float
    regime: str
    model_name: str
    metadata: dict = field(default_factory=dict)


class RegimeDetector:
    def __init__(self, vix_z_threshold: float = 1.5, return_threshold: float = -0.15):
        self.vix_z_threshold = vix_z_threshold
        self.return_threshold = return_threshold
        self._previous_regime: Optional[RegimeState] = None

    def detect(
        self,
        vix_zscore: float = 0.0,
        recent_return: float = 0.0,
        volatility_pct: float = 0.0,
    ) -> tuple[RegimeState, float]:
        confidence = 0.5
        if vix_zscore > 2.0 or recent_return < self.return_threshold:
            regime = RegimeState.CRISIS
            confidence = min(0.95, 0.7 + abs(vix_zscore) * 0.05)
        elif vix_zscore > self.vix_z_threshold or recent_return < -0.08:
            regime = RegimeState.STRESS
            confidence = min(0.9, 0.6 + abs(vix_zscore) * 0.1)
        elif volatility_pct < 0.15 and vix_zscore < 0.5 and recent_return > 0.02:
            regime = RegimeState.BULL
            confidence = min(0.85, 0.55 + recent_return * 2)
        else:
            regime = RegimeState.NORMAL
            confidence = 0.5 + abs(0.5 - abs(vix_zscore) / 3.0)
        return regime, confidence

    def check_transition(
        self, current: RegimeState, confidence: float, indicators: dict
    ) -> Optional[tuple[RegimeState, RegimeState, float, dict]]:
        if self._previous_regime is None:
            self._previous_regime = current
            return None
        if current != self._previous_regime:
            transition = (self._previous_regime, current, confidence, indicators)
            self._previous_regime = current
            return transition
        self._previous_regime = current
        return None


class DeviationComparator:
    DEVIATION_THRESHOLDS = {
        AlertSeverity.INFO: 0.05,
        AlertSeverity.WARNING: 0.15,
        AlertSeverity.CRITICAL: 0.30,
    }

    def compare(
        self,
        forecast_value: float,
        actual_value: float,
        forecast_date: str,
        actual_date: str,
        deal_name: str,
    ) -> tuple[float, AlertSeverity, DeviationRecord]:
        if forecast_value == 0:
            deviation_pct = 0.0 if actual_value == 0 else float("inf")
        else:
            deviation_pct = abs(actual_value - forecast_value) / abs(forecast_value)

        if deviation_pct >= self.DEVIATION_THRESHOLDS[AlertSeverity.CRITICAL]:
            severity = AlertSeverity.CRITICAL
        elif deviation_pct >= self.DEVIATION_THRESHOLDS[AlertSeverity.WARNING]:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        record = DeviationRecord(
            run_id=HistoricalTracker.generate_run_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            deal_name=deal_name,
            forecast_value=round(forecast_value, 2),
            actual_value=round(actual_value, 2),
            deviation_pct=round(deviation_pct * 100, 2),
            forecast_date=forecast_date,
            actual_date=actual_date,
            severity=severity.value,
        )
        return deviation_pct, severity, record


class AutoForecastAgent:
    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        historical_tracker: Optional[HistoricalTracker] = None,
        schedule: ScheduleFrequency = ScheduleFrequency.WEEKLY,
        storage_dir: str = "results",
        deviation_threshold_warning: float = 0.15,
        deviation_threshold_critical: float = 0.30,
    ):
        self.alert_manager = alert_manager or AlertManager()
        self.historical_tracker = historical_tracker or HistoricalTracker(
            os.path.join(storage_dir, "history")
        )
        self.schedule = schedule
        self.regime_detector = RegimeDetector()
        self.deviation_comparator = DeviationComparator()
        self.deviation_threshold_warning = deviation_threshold_warning
        self.deviation_threshold_critical = deviation_threshold_critical
        self._llm_client = InferenceClient(model=LLM_MODEL)
        self._last_run: Optional[datetime] = None
        self._forecast_results: list[ForecastResult] = []
        logger.info(f"AutoForecastAgent initialized with schedule={schedule.value}")

    def run_forecast_cycle(
        self,
        measures_path: str = "measures.csv",
        metadata_path: str = "metadata.csv",
        macro_data: Optional[dict] = None,
    ) -> list[ForecastResult]:
        run_id = HistoricalTracker.generate_run_id()
        logger.info(f"Starting forecast cycle run_id={run_id}")
        measures = self._load_measures(measures_path)
        metadata = self._load_metadata(metadata_path)
        if measures.is_empty():
            logger.warning("No measures data loaded, using empty DataFrame")
        deals = self._extract_deals(measures, metadata)
        macro_defaults = {
            "vix_zscore": 0.0,
            "recent_return": 0.0,
            "volatility_pct": 0.20,
            "spy_trend": "flat",
        }
        if macro_data:
            macro_defaults.update(macro_data)
        macro = macro_defaults
        regime, confidence = self.regime_detector.detect(
            vix_zscore=macro["vix_zscore"],
            recent_return=macro["recent_return"],
            volatility_pct=macro["volatility_pct"],
        )
        logger.info(f"Detected regime: {regime.value} (confidence={confidence:.2f})")
        regime_transition = self.regime_detector.check_transition(
            regime, confidence, macro
        )
        if regime_transition:
            prev, curr, conf, indicators = regime_transition
            logger.warning(f"Regime change detected: {prev.value} -> {curr.value}")
            self.historical_tracker.record_regime_change(
                RegimeRecord(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    regime_from=prev.value,
                    regime_to=curr.value,
                    confidence=conf,
                    indicators=indicators,
                )
            )
            self.alert_manager.send_regime_change_alert(
                regime_from=prev.value,
                regime_to=curr.value,
                confidence=conf,
                indicators=indicators,
            )
        results = []
        for deal in deals:
            result = self._forecast_deal(
                deal, regime, confidence, run_id, macro
            )
            if result:
                results.append(result)
                self.historical_tracker.record_forecast(
                    ForecastRecord(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        deal_name=result.deal_name,
                        forecast_horizon=result.horizon,
                        p10=result.p10,
                        p50=result.p50,
                        p90=result.p90,
                        mean=result.mean,
                        model_name=result.model_name,
                        regime=result.regime,
                    )
                )
        self._forecast_results = results
        self._last_run = datetime.now(timezone.utc)
        logger.info(f"Forecast cycle complete: {len(results)} deal forecasts generated")
        if results:
            self._generate_llm_summary(results, macro, regime)
        return results

    def compare_forecasts_to_actuals(
        self,
        actuals_path: Optional[str] = None,
        actuals_df: Optional[pl.DataFrame] = None,
    ) -> list[DeviationRecord]:
        if not self._forecast_results:
            logger.warning("No forecast results to compare")
            return []
        actuals = actuals_df
        if actuals is None and actuals_path:
            actuals = self._load_measures(actuals_path)
        if actuals is None:
            logger.warning("No actuals data provided")
            return []
        deviations = []
        for forecast in self._forecast_results:
            actual_value = self._find_actual(forecast.deal_name, actuals)
            if actual_value is not None:
                deviation_pct, severity, record = self.deviation_comparator.compare(
                    forecast_value=forecast.p50,
                    actual_value=actual_value,
                    forecast_date=forecast.forecast_date,
                    actual_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    deal_name=forecast.deal_name,
                )
                self.historical_tracker.record_deviation(record)
                if severity != AlertSeverity.INFO:
                    self.alert_manager.send_deviation_alert(
                        deal_name=forecast.deal_name,
                        deviation_pct=record.deviation_pct,
                        forecast_value=record.forecast_value,
                        actual_value=record.actual_value,
                        context=f"Regime: {forecast.regime}, Model: {forecast.model_name}",
                        severity=severity,
                    )
                deviations.append(record)
        logger.info(f"Compared {len(deviations)} forecasts to actuals")
        return deviations

    def should_run(self) -> bool:
        if self._last_run is None:
            return True
        now = datetime.now(timezone.utc)
        delta = now - self._last_run
        intervals = {
            ScheduleFrequency.WEEKLY: timedelta(days=7),
            ScheduleFrequency.MONTHLY: timedelta(days=30),
            ScheduleFrequency.QUARTERLY: timedelta(days=90),
        }
        return delta >= intervals[self.schedule]

    def get_status(self) -> dict:
        return {
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "schedule": self.schedule.value,
            "forecast_count": len(self._forecast_results),
            "should_run": self.should_run(),
            "regime": self.regime_detector._previous_regime.value if self.regime_detector._previous_regime else None,
            "deviation_stats": self.historical_tracker.compute_deviation_stats(),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_llm_summary(
        self, results: list[ForecastResult], macro: dict, regime: RegimeState
    ) -> str:
        summary_data = {
            "regime": regime.value,
            "num_deals": len(results),
            "forecasts": [
                {"deal": r.deal_name, "p50": r.p50, "p10": r.p10, "p90": r.p90}
                for r in results[:10]
            ],
            "macro": macro,
        }
        prompt = (
            "You are a senior quantitative portfolio risk analyst. Provide a concise summary "
            "of the forecast cycle. Highlight: key risk factors, regime implications, and "
            "notable deal-level forecasts. Keep under 5 sentences."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Forecast cycle data:\n{json.dumps(summary_data, default=str)}\nGenerate summary:",
            },
        ]
        response = self._llm_client.chat_completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"LLM Forecast Summary: {summary}")
        return summary

    def _load_measures(self, path: str) -> pl.DataFrame:
        if not os.path.exists(path):
            alt = os.path.join("results", path)
            if os.path.exists(alt):
                path = alt
            else:
                logger.warning(f"Measures file not found: {path}")
                return pl.DataFrame()
        try:
            df = pl.read_csv(path)
            logger.info(f"Loaded measures: {df.height} records from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load measures: {e}")
            return pl.DataFrame()

    def _load_metadata(self, path: str) -> pl.DataFrame:
        if not os.path.exists(path):
            alt = os.path.join("results", path)
            if os.path.exists(alt):
                path = alt
            else:
                logger.warning(f"Metadata file not found: {path}")
                return pl.DataFrame()
        try:
            df = pl.read_csv(path)
            logger.info(f"Loaded metadata: {df.height} deals from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return pl.DataFrame()

    def _extract_deals(
        self, measures: pl.DataFrame, metadata: pl.DataFrame
    ) -> list[dict]:
        deals = []
        if metadata.is_empty():
            return deals
        if "Deal Name" in metadata.columns:
            deal_names = metadata["Deal Name"].unique().to_list()
        elif "deal_name" in metadata.columns:
            deal_names = metadata["deal_name"].unique().to_list()
        else:
            return deals
        for name in deal_names:
            row = metadata.filter(
                (pl.col("Deal Name") == name) if "Deal Name" in metadata.columns else (pl.col("deal_name") == name)
            )
            deal_info = {"name": name}
            if row.height > 0:
                for col in row.columns:
                    deal_info[col] = row[col][0]
            deal_measures = self._get_deal_measures(name, measures)
            deal_info["measures"] = deal_measures
            deal_info["latest_nav"] = self._get_latest_nav(name, measures)
            deal_info["total_commitment"] = self._get_total_commitment(name, measures)
            deals.append(deal_info)
        return deals

    def _get_deal_measures(self, deal_name: str, measures: pl.DataFrame) -> dict:
        if measures.is_empty():
            return {}
        name_col = "Deal Name" if "Deal Name" in measures.columns else "deal_name"
        deal_data = measures.filter(pl.col(name_col) == deal_name)
        if deal_data.is_empty():
            return {}
        measure_col = "Measure" if "Measure" in measures.columns else "measure"
        amount_col = "Amount" if "Amount" in measures.columns else "amount_usd"
        summary = {}
        for measure_name in deal_data[measure_col].unique().to_list():
            vals = deal_data.filter(pl.col(measure_col) == measure_name)[amount_col]
            if vals.height > 0:
                summary[measure_name] = {
                    "total": round(vals.sum(), 2),
                    "count": vals.height,
                    "latest": round(vals[-1], 2),
                }
        return summary

    def _get_latest_nav(self, deal_name: str, measures: pl.DataFrame) -> Optional[float]:
        if measures.is_empty():
            return None
        name_col = "Deal Name" if "Deal Name" in measures.columns else "deal_name"
        measure_col = "Measure" if "Measure" in measures.columns else "measure"
        amount_col = "Amount" if "Amount" in measures.columns else "amount_usd"
        date_col = "Date" if "Date" in measures.columns else "date"
        nav_measures = ["Estimated NAV", "Provided NAV"]
        nav_data = measures.filter(
            (pl.col(name_col) == deal_name) & (pl.col(measure_col).is_in(nav_measures))
        )
        if nav_data.is_empty():
            return None
        try:
            nav_data = nav_data.with_columns(pl.col(date_col).str.to_date(strict=False))
            nav_data = nav_data.sort(date_col)
            return round(nav_data[amount_col][-1], 2)
        except Exception:
            return round(nav_data[amount_col][-1], 2)

    def _get_total_commitment(self, deal_name: str, measures: pl.DataFrame) -> Optional[float]:
        if measures.is_empty():
            return None
        name_col = "Deal Name" if "Deal Name" in measures.columns else "deal_name"
        measure_col = "Measure" if "Measure" in measures.columns else "measure"
        amount_col = "Amount" if "Amount" in measures.columns else "amount_usd"
        comm_data = measures.filter(
            (pl.col(name_col) == deal_name) & (pl.col(measure_col) == "Commitment")
        )
        if comm_data.is_empty():
            return None
        return round(comm_data[amount_col].sum(), 2)

    def _forecast_deal(
        self,
        deal: dict,
        regime: RegimeState,
        confidence: float,
        run_id: str,
        macro: dict,
    ) -> Optional[ForecastResult]:
        name = deal.get("name", deal.get("Deal Name", "Unknown"))
        latest_nav = deal.get("latest_nav")
        commitment = deal.get("total_commitment")
        if latest_nav is None or latest_nav == 0:
            logger.debug(f"Skipping {name}: no NAV data")
            return None
        regime_factor = {
            RegimeState.BULL: 1.08,
            RegimeState.NORMAL: 1.04,
            RegimeState.STRESS: 0.96,
            RegimeState.CRISIS: 0.88,
        }
        volatility_factor = {
            RegimeState.BULL: 0.12,
            RegimeState.NORMAL: 0.20,
            RegimeState.STRESS: 0.35,
            RegimeState.CRISIS: 0.50,
        }
        rf = regime_factor.get(regime, 1.04)
        vf = volatility_factor.get(regime, 0.20)
        base_forecast = latest_nav * rf
        noise = np.random.normal(0, base_forecast * vf * 0.1)
        mean_forecast = base_forecast + noise
        std_forecast = abs(base_forecast * vf)
        p10 = mean_forecast - 1.28 * std_forecast
        p50 = mean_forecast
        p90 = mean_forecast + 1.28 * std_forecast
        horizon_map = {
            ScheduleFrequency.WEEKLY: "1W",
            ScheduleFrequency.MONTHLY: "1M",
            ScheduleFrequency.QUARTERLY: "1Q",
        }
        result = ForecastResult(
            deal_name=name,
            forecast_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            horizon=horizon_map.get(self.schedule, "1M"),
            p10=round(p10, 2),
            p50=round(p50, 2),
            p90=round(p90, 2),
            mean=round(mean_forecast, 2),
            std=round(std_forecast, 2),
            regime=regime.value,
            model_name="Chronos-AutoForecast",
            metadata={
                "run_id": run_id,
                "latest_nav": latest_nav,
                "commitment": commitment,
                "confidence": round(confidence, 3),
                "vix_zscore": macro.get("vix_zscore", 0),
            },
        )
        logger.debug(f"Forecast for {name}: p50={p50:.0f} p10={p10:.0f} p90={p90:.0f} regime={regime.value}")
        return result


def main():
    logger.configure(
        handlers=[{"sink": sys.stderr, "format": "<green>{time}</green> <level>{message}</level>"}]
    )
    logger.info("AutoForecastAgent starting...")
    agent = AutoForecastAgent(
        schedule=ScheduleFrequency.WEEKLY,
        storage_dir="results",
    )
    results = agent.run_forecast_cycle(
        measures_path="measures.csv",
        metadata_path="metadata.csv",
    )
    for r in results:
        logger.info(f"  {r.deal_name}: p10={r.p10:.0f} p50={r.p50:.0f} p90={r.p90:.0f} ({r.regime})")
    status = agent.get_status()
    logger.info(f"Agent status: {json.dumps(status, default=str)}")
    agent.historical_tracker.export_summary_csv()
    logger.success("AutoForecastAgent run complete.")


if __name__ == "__main__":
    main()