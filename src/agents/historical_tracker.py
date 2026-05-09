import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import polars as pl
from loguru import logger


@dataclass
class ForecastRecord:
    run_id: str
    timestamp: str
    deal_name: str
    forecast_horizon: str
    p10: float
    p50: float
    p90: float
    mean: float
    model_name: str
    regime: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeviationRecord:
    run_id: str
    timestamp: str
    deal_name: str
    forecast_value: float
    actual_value: float
    deviation_pct: float
    forecast_date: str
    actual_date: str
    severity: str = "info"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegimeRecord:
    run_id: str
    timestamp: str
    regime_from: str
    regime_to: str
    confidence: float
    indicators: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HistoricalTracker:
    def __init__(self, storage_dir: str = "results/history"):
        self.storage_dir = storage_dir
        self._forecasts_path = os.path.join(storage_dir, "forecasts.jsonl")
        self._deviations_path = os.path.join(storage_dir, "deviations.jsonl")
        self._regimes_path = os.path.join(storage_dir, "regimes.jsonl")
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"HistoricalTracker initialized with storage_dir={storage_dir}")

    def record_forecast(self, record: ForecastRecord) -> None:
        with open(self._forecasts_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.debug(f"Recorded forecast: {record.run_id} / {record.deal_name}")

    def record_deviation(self, record: DeviationRecord) -> None:
        with open(self._deviations_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.debug(f"Recorded deviation: {record.deal_name} deviation={record.deviation_pct:.1f}%")

    def record_regime_change(self, record: RegimeRecord) -> None:
        with open(self._regimes_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(f"Recorded regime change: {record.regime_from} -> {record.regime_to}")

    def load_forecasts(self, deal_name: Optional[str] = None, limit: int = 1000) -> list[dict]:
        records = self._load_jsonl(self._forecasts_path, limit)
        if deal_name:
            records = [r for r in records if r.get("deal_name") == deal_name]
        return records

    def load_deviations(self, deal_name: Optional[str] = None, limit: int = 1000) -> list[dict]:
        records = self._load_jsonl(self._deviations_path, limit)
        if deal_name:
            records = [r for r in records if r.get("deal_name") == deal_name]
        return records

    def load_regime_changes(self, limit: int = 100) -> list[dict]:
        return self._load_jsonl(self._regimes_path, limit)

    def get_deviations_as_df(self, deal_name: Optional[str] = None) -> pl.DataFrame:
        records = self.load_deviations(deal_name)
        if not records:
            return pl.DataFrame(schema={
                "run_id": pl.Utf8, "timestamp": pl.Utf8, "deal_name": pl.Utf8,
                "forecast_value": pl.Float64, "actual_value": pl.Float64,
                "deviation_pct": pl.Float64, "forecast_date": pl.Utf8,
                "actual_date": pl.Utf8, "severity": pl.Utf8,
            })
        return pl.from_dicts(records)

    def get_forecasts_as_df(self, deal_name: Optional[str] = None) -> pl.DataFrame:
        records = self.load_forecasts(deal_name)
        if not records:
            return pl.DataFrame(schema={
                "run_id": pl.Utf8, "timestamp": pl.Utf8, "deal_name": pl.Utf8,
                "forecast_horizon": pl.Utf8, "p10": pl.Float64, "p50": pl.Float64,
                "p90": pl.Float64, "mean": pl.Float64, "model_name": pl.Utf8,
                "regime": pl.Utf8,
            })
        df = pl.from_dicts(records)
        if "metadata" in df.columns:
            df = df.drop("metadata")
        return df

    def get_regime_changes_as_df(self) -> pl.DataFrame:
        records = self.load_regime_changes()
        if not records:
            return pl.DataFrame(schema={
                "run_id": pl.Utf8, "timestamp": pl.Utf8, "regime_from": pl.Utf8,
                "regime_to": pl.Utf8, "confidence": pl.Float64,
            })
        return pl.from_dicts(records)

    def get_latest_forecasts(self, n: int = 5) -> list[dict]:
        all_records = self._load_jsonl(self._forecasts_path, limit=n * 10)
        by_deal = {}
        for r in all_records:
            deal = r.get("deal_name", "unknown")
            if deal not in by_deal:
                by_deal[deal] = []
            by_deal[deal].append(r)
        latest = []
        for deal, records in by_deal.items():
            records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            latest.extend(records[:n])
        return latest

    def compute_deviation_stats(self, deal_name: Optional[str] = None) -> dict:
        df = self.get_deviations_as_df(deal_name)
        if df.is_empty():
            return {"count": 0, "mean_deviation_pct": 0.0, "max_deviation_pct": 0.0, "critical_count": 0}
        stats = {
            "count": df.height,
            "mean_deviation_pct": round(df["deviation_pct"].mean(), 2),
            "max_deviation_pct": round(df["deviation_pct"].max(), 2),
            "critical_count": df.filter(pl.col("severity") == "critical").height,
        }
        return stats

    def export_summary_csv(self, output_dir: Optional[str] = None) -> dict:
        output_dir = output_dir or self.storage_dir
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for name, loader in [
            ("forecasts", self.get_forecasts_as_df),
            ("deviations", self.get_deviations_as_df),
            ("regime_changes", self.get_regime_changes_as_df),
        ]:
            df = loader()
            path = os.path.join(output_dir, f"{name}_history.csv")
            df.write_csv(path)
            paths[name] = path
            logger.info(f"Exported {name} history to {path}")
        return paths

    @staticmethod
    def _load_jsonl(path: str, limit: int = 1000) -> list[dict]:
        if not os.path.exists(path):
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-limit:]

    @staticmethod
    def generate_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")