# Private Equity Risk & Forecasting Engine

<p align="center">
  <img src="https://github.com/user-attachments/assets/bcedddfb-d07e-4f42-883a-926ea3564446" width="800" alt="Risk Command Center Dashboard">
</p>

<p align="center">
  <a href="https://lookerstudio.google.com/reporting/7b2515d4-9975-484f-a77f-1402f9e6d9b4" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%93%8A%20Live%20Dashboard-Looker%20Studio-blue?style=for-the-badge"/>
  </a>
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%92%BB%20Code-Python%20%7C%20Polars%20%7C%20AutoGluon-green?style=for-the-badge"/>
  </a>
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Model-Chronos%20Tiny-orange?style=for-the-badge"/>
  </a>
</p>

OLS missed the PE valuation lag (R²=0.049). Chronos didn't. This pipeline catches what linear models can't by de-smoothing stale private equity valuations and running probabilistic forecasts with a pre-trained time series transformer.

---

## What It Does

Private equity valuations are smoothed and lagged. That masks real volatility. This pipeline audits data integrity, strips accounting artifacts, and models tail risk using AutoGluon's ensemble, catching a $34M valuation discrepancy in the source data along the way.

**Key numbers:**
- OLS R² = 0.049 (proved linear models can't explain PE returns)
- AutoGluon winner: Chronos[tiny], validation WQL = -1.48
- Dynamic Rho = 0.056 (Geltner de-smoothing via AR(1))
- Lag 4 confirmed as primary driver via PACF (1-year valuation lag)
- Volatility scalar = 0.40 (private fund = 40% of public proxy volatility)
- Q1 2026 forecast: defensive posture, significant P10 downside

---

## Architecture

<p align="center">
  <img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6ec32191-13a0-4c5d-b77f-8a3b68f2d72a" />
</p>

**Data Engineering (Polars + DuckDB):** Fast ETL with SQL-based auditing. Flagged commitment discrepancies in LOTUS II and ALPHA funds. Normalized cross-currency cash flows to USD.

**Proxy Strategy:** The fund's history predates modern rate regimes (ZIRP, COVID, inflation). Engineered a Modern Proxy using PSP (Global Listed PE ETF) + Russell 2000 + HYG alongside S&P 500 and 10Y Treasuries.

**Feature Engineering:** Geltner de-smoothing, PACF lag selection, VIX Z-score regime detection.

**Modeling:** Two tracks in parallel -- OLS for inference (confirmed non-linearity), AutoGluon for probabilistic forecasting (optimized for WQL to weight tail risk). Chronos beat DeepAR and PatchTST.

---

## Repository Structure

```
.
├── transform.py          # ETL: ingestion, cleaning, auditing
├── model_risk.py         # Risk engine: feature eng, OLS, AutoGluon, forecasting
├── requirements.txt      # Core deps (Torch, AutoGluon, Polars)
├── metadata.csv          # Cleaned deal-level static data
├── measures.csv          # Normalized transaction log
├── macro_regimes.csv     # Historical regime indicators (VIX Z-scores)
└── scenario_stress.csv   # Probabilistic forecast outputs
```

---

## Setup

Python 3.10+. Use a virtual environment -- Torch version conflicts are real.

```bash
pip install -r requirements.txt
python transform.py    # outputs metadata.csv, measures.csv, portfolio_sunburst.html
python model_risk.py   # fetches live market data, trains ensemble, outputs forecast CSVs
```

---

## Dashboard

The Looker Studio dashboard visualizes the P10/P50/P90 fan chart, a 3D risk cube clustering crisis regimes, and capital concentration by vintage and geography.

<p align="center">
  <a href="https://lookerstudio.google.com/reporting/7b2515d4-9975-484f-a77f-1402f9e6d9b4">
    <img src="https://img.shields.io/badge/VIEW%20DASHBOARD-CLICK%20HERE-red?style=for-the-badge&logo=google-looker"/>
  </a>
</p>
