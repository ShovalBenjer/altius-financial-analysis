# **Private Equity Risk & Forecasting Engine**


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

A state-of-the-art quantitative risk analysis pipeline designed to uncover hidden downside risks in illiquid Private Equity portfolios. This project moves beyond traditional linear regression, utilizing a hybrid approach of **Foundation Models (Chronos)** and **Econometric De-smoothing** to forecast Q1 2026 performance under stress scenarios.

---

# **1. Executive Summary**

### **The Challenge: "The Stale Price Problem"**
Private Equity (PE) valuations are smoothed and lagged, masking true economic volatility. Traditional linear models (OLS) fail to detect correlations with public markets, creating a false sense of security.

### **The Solution: "The Big Short" Architecture**
We implemented a rigorous pipeline to audit data integrity, strip away accounting artifacts, and model non-linear risk using Deep Learning.

*   **Linear Failure:** OLS regression yielded a negligible **$R^2$ of 0.049**, proving macro factors alone cannot explain PE returns via simple linear relationships.
*   **Deep Learning Success:** The **AutoGluon** ensemble identified **Chronos[tiny]** (a Pre-trained Time Series Transformer) as the superior predictor, achieving a validation score of **-1.48 WQL**.
*   **Outcome:** The model predicts a **defensive outlook** for Q1 2026, identifying significant downside risk in stress scenarios that standard models miss.

---

# **2. System Architecture**
<p align="center">
  <img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6ec32191-13a0-4c5d-b77f-8a3b68f2d72a" />
</p>

The pipeline is built on a modern, high-performance stack designed for financial data integrity.

### **A. Data Engineering (Polars + DuckDB)**
*   **Tech Stack:** `Polars` for lightning-fast ETL; `DuckDB` for SQL-based auditing.
*   **Integrity Check:** Automated audit logic flagged discrepancies in `LOTUS II` and `ALPHA` (Reported vs. Calculated Commitment), preventing a $34M valuation error.
*   **Normalization:** Standardized cross-currency cash flows to USD base.

### **B. The Proxy Strategy (Data Enrichment)**
To model modern market regimes (ZIRP, COVID, Inflation) absent from the legacy fund data, we engineered a "Modern Proxy" dataset:
*   **Proxy Asset:** **PSP** (Global Listed Private Equity ETF).
*   **Market Ensemble:** Added **Russell 2000 (`^RUT`)** and **High Yield Spreads (`HYG`)** alongside S&P 500 and 10Y Treasuries.

### **C. Quantitative Feature Engineering**
*   **Geltner De-smoothing:** Calculated a **Dynamic Rho (0.056)** using AR(1) regression to mathematically "unsmooth" returns.
*   **PACF Lag Selection:** Statistically identified **Lag 4** as the primary driver, confirming a 1-year valuation lag.
*   **Regime Detection:** Calculated **VIX Z-Scores** to mathematically segregate "Crisis" vs. "Normal" market regimes.

---

# **3. Modeling Strategy**

We deployed two parallel modeling tracks to answer "Why?" and "What Next?"

### **Track 1: Statistical Inference (OLS)**
*   **Goal:** Explainability.
*   **Result:** **$R^2$ of 0.029**.
*   **Insight:** Validated that PE returns are idiosyncratic and non-linear. This failure of linear models necessitated the move to AI.

### **Track 2: SOTA Deep Learning (AutoGluon)**
*   **Goal:** Probabilistic Forecasting (Tail Risk).
*   **Metric:** Optimized for **Weighted Quantile Loss (WQL)** to prioritize accurate tail-risk estimation over median error.
*   **Model Zoo:**
    *   **Chronos:** Pre-trained Transformer (Winner).
    *   **DeepAR:** Probabilistic RNN.
    *   **PatchTST:** Transformer-based segmentation.

---

# **4. Key Findings**

### **1. The "Volatility Scalar" (0.40)**
The private fund exhibits only **40% of the volatility** of the public market proxy. We applied this scalar to our forecasts to translate public market volatility into realistic private fund marks.

### **2. Q1 2026 Forecast (The Fan Chart)**
The probabilistic forecast indicates a **defensive posture**:
*   **Bull Case:** Returns remain muted due to the lag effect of previous rate hikes.
*   **Stress Case:** Significant downside risk identified in the P10 quantile.

---

# **5. Repository Structure**

```
.
├── transform.py          # Core ETL: Ingestion, Cleaning, Auditing
├── model_risk.py         # Risk Engine: Feature Eng, OLS, AutoGluon, Forecasting
├── requirements.txt      # Dependencies (Torch, AutoGluon, Polars)
├── metadata.csv          # Cleaned deal-level static data
├── measures.csv          # Normalized transaction log
├── macro_regimes.csv     # Historical regime indicators (VIX Z-scores)
└── scenario_stress.csv   # Probabilistic forecast outputs
```

---

# **6. Installation & Usage**

### **Prerequisites**
*   Python 3.10+
*   pip package manager

### **Setup**
1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Use a virtual environment to avoid Torch/Torchvision version conflicts.*

### **Execution**
1.  **Run ETL Pipeline:**
    ```bash
    python transform.py
    ```
    *Outputs `metadata.csv`, `measures.csv`, and `portfolio_sunburst.html`.*

2.  **Run Risk Engine:**
    ```bash
    python model_risk.py
    ```
    *Fetches live market data, trains the ensemble, and outputs forecast CSVs.*

---

# **7. The Risk Command Center**

The final output is a comprehensive **Looker Studio Dashboard** visualizing the "Unseen Risk."

*   **Risk Cube Proxy:** A 3D scatter plot visually clustering "Crisis" regimes (High Fear + Credit Stress).
*   **Capital Concentration:** Interactive Treemaps highlighting exposure by Vintage and Geography.
*   **Probabilistic Fan Chart:** Visualizes the P10/P50/P90 return scenarios.

<p align="center">
  <a href="https://lookerstudio.google.com/reporting/7b2515d4-9975-484f-a77f-1402f9e6d9b4">
    <img src="https://img.shields.io/badge/VIEW%20DASHBOARD-CLICK%20HERE-red?style=for-the-badge&logo=google-looker"/>
  </a>
</p>

---

# **Contact**

For questions regarding the methodology or codebase, please contact the repository maintainer.
