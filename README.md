# Systematic Alternative Data Trading Pipeline

A quantitative trading pipeline that ingests high-frequency options data, engineers path-dependent targets, trains machine learning models without look-ahead bias, and executes automated orders via Interactive Brokers.

## Quantitative Thesis
Sudden shifts in options order flow—such as anomalous volume spikes or rapid changes in the Put/Call ratio—often reflect informed trading activity that precedes short-term price movements in the underlying equity. By tracking these high-frequency signals and applying a strict stop-loss framework to our historical data, we can train a model to identify momentum setups that are realistic to execute and survive market volatility.

---

## System Architecture

```text
Pipeline/
│
├── .env.example          # Template for required environment variables
├── .gitignore            # Excludes sensitive data, parquets, and local checkpoints
├── requirements.txt      # Core project dependencies
├── README.md             # Project documentation and thesis
│
├── data/                 # Local directory for cached binary/parquet tapes
├── models/               # Saved production model checkpoints (.json)
│
└── src/                  # Core modules
    ├── options_pipeline.py   # Databento OPRA ingestion & Polars feature engineering
    ├── model_training.py     # TimeSeriesSplit cross-validation & XGBoost optimization
    └── live_execution.py     # IBKR TWS live socket integration & execution loop

```

---

## Core Components & Engineering Rationale

### 1. Data Ingestion & Caching (`options_pipeline.py`)

* **Databento (OPRA.PILLAR):** We use Databento to pull raw binary tick data. This provides exact nanosecond-precision timestamps for options trades, which is necessary for aligning high-frequency options flow with standard 1-minute equity bars.
* **Polars for Aggregation:** Because tick data is massive, we use Polars instead of Pandas. Polars is built on Rust and uses multi-threading to process millions of rows significantly faster, preventing memory bottlenecks during feature engineering.
* **Local Parquet Caching:** The script saves the initial raw data pull as a compressed `.parquet` file. This prevents the system from having to re-download gigabytes of data every time the script is run, saving time and API costs.

### 2. Path-Dependent Target Engineering

* **Filtering "Ghost MFE":** In traditional backtesting, models often calculate the Maximum Favorable Excursion (MFE)—the absolute peak price an asset reached after a signal. However, if the asset dropped 5% before reaching that peak, a real trader would have been stopped out. This is "Ghost MFE."
* **Trailing Stop-Loss:** To fix this, our target variable is path-dependent. We simulate a strict 0.5% trailing stop-loss on the historical data. If the asset hits the stop-loss before generating a return, the target is capped at a loss. This forces the model to learn which setups survive the execution path.

### 3. Walk-Forward Cross-Validation (`model_training.py`)

* **TimeSeriesSplit:** Standard machine learning validation scrambles data randomly, which allows the model to use "future" data to predict the past. We use Scikit-learn's `TimeSeriesSplit` to enforce a strict chronological expanding window. The model only ever trains on past data to predict unseen future data.
* **Algorithm Selection:** We use an XGBoost Regressor to predict trade expectancy. XGBoost is highly efficient for tabular numerical data. We intentionally limit the tree depth (`max_depth=3`) to prevent the model from over-fitting to the inevitable noise of high-frequency market data.

### 4. Live Execution Bridge (`live_execution.py`)

* **Socket Integration:** The execution script interfaces with Interactive Brokers Trader Workstation (TWS) via local sockets. We use the `ib_async` library because it natively handles the complex asynchronous event loops required to maintain a stable connection to the broker.
* **Dynamic Evaluation:** The script polls the latest options metrics and feeds them into the saved XGBoost model (`options_flow_model_v1.json`). If the predicted Expected Value (EV) clears our confidence threshold, the script automatically sends a bracket order to the market.

---

## Model Validation & Risk Summary

### 1. Initial Thesis & Conceptual Soundness

* **Economic Rationale:** High-frequency options market microstructure acts as a leading indicator. Sudden imbalances in put/call ratios and anomalous call volume shocks reflect aggressive directional positioning prior to equity price discovery.
* **Risk & Limitations:** The strategy is vulnerable to liquidity gaps and spread widening during major macroeconomic news releases. Furthermore, high-frequency signals can decay rapidly, meaning execution latency is a primary risk factor.

### 2. General Model Description

* **Target Engineering:** The target variable is the realized percentage return, strictly bounded by a 0.5% trailing stop-loss constraint to accurately reflect live execution.
* **Feature Space:**
* `opt_put_call_ratio`: The rolling 1-minute volume ratio of put versus call options contracts.
* `opt_call_shock`: The normalized deviation of call volume relative to baseline historical averages.


* **Algorithm:** XGBoost Regressor optimized for Expected Value (EV) estimation.

### 3. Code Implementation & Architecture

* **`src/options_pipeline.py`:** Handles the API requests, manages local data caching, and performs the mathematical aggregations required to map options trades to equity timestamps.
* **`src/model_training.py`:** Conducts the expanding-window cross-validation, generates the error metrics (MAE), and exports the final fitted model for production use.
* **`src/live_execution.py`:** Manages the real-time continuous loop, evaluating incoming data against the model and pushing execution instructions to the broker API.

```

```
