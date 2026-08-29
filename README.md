# Systematic Alternative Data Trading Pipeline

This project is an exploration into building a complete quantitative trading pipeline from scratch. It walks through ingesting high-frequency options data, engineering targets that survive real-world execution, training a machine learning model without look-ahead bias, and finally deploying it to a live brokerage environment.

## The Core Concept
The underlying hypothesis of this pipeline is that options market microstructure acts as a leading indicator. When unusual volume spikes or put/call ratios shift rapidly, it often signals aggressive, informed positioning just before the underlying equity moves. This project tests whether tracking these high-frequency shifts—paired with strict risk management—can reliably identify short-term momentum setups.

---

## System Architecture

```text
Pipeline/
│
├── .env.example          # Template for required environment variables (Alpaca/IBKR)
├── .gitignore            # Excludes sensitive data, parquets, and local checkpoints
├── requirements.txt      # Core project dependencies
├── README.md             # Project documentation and thesis
│
├── data/                 # Local directory for cached binary/parquet tapes
├── models/               # Saved production model checkpoints (.json)
├── execution_log.csv     # Automated out-of-sample forward-testing ledger
│
└── src/                  # Core modules
    ├── options_pipeline.py   # Databento OPRA ingestion & Polars feature engineering
    ├── model_training.py     # TimeSeriesSplit cross-validation & XGBoost optimization
    └── live_execution.py     # Dual-broker async execution loop (Alpaca Data + IBKR Routing)

```

---

## How It Works (And Why It Was Built This Way)

### 1. Data Ingestion & Caching (`options_pipeline.py`)

* **Nanosecond Precision:** To accurately align options flow with 1-minute equity bars, you need exact timestamps. The pipeline uses Databento to pull OPRA binary tick data, providing the nanosecond resolution required for high-frequency analysis.
* **Beating Memory Bottlenecks:** Options tick data is notoriously massive, and standard Pandas DataFrames often choke on it. The script leverages Polars—a DataFrame library built on Rust—to utilize multi-threading and process millions of rows without crashing system memory.
* **Parquet Caching:** Downloading gigabytes of raw data on every test run is slow and expensive. The script automatically caches initial pulls locally as compressed `.parquet` files to drastically speed up feature engineering iteration.

### 2. Path-Dependent Target Engineering

* **The "Ghost MFE" Trap:** A common flaw in backtesting is calculating Maximum Favorable Excursion (MFE)—the absolute peak price an asset reached after a buy signal. But if the asset dropped 5% *before* rocketing up 10%, a real trader would have stopped out. Giving a model credit for that unreachable peak creates "Ghost MFE."
* **Enforcing Reality:** To fix this, the target variable in this pipeline is path-dependent. It simulates a strict 0.5% trailing stop-loss directly on the historical tape. If the price path hits the stop-loss before generating a return, the target records a loss. This forces the model to learn which setups actually survive the execution path.

### 3. Walk-Forward Cross-Validation (`model_training.py`)

* **Preventing Look-Ahead Bias:** Randomly shuffling time-series data for cross-validation ruins the timeline, allowing the model to accidentally learn from "future" data. The pipeline uses Scikit-learn's `TimeSeriesSplit` to enforce a chronological expanding window, ensuring the model is only ever trained on past events to predict unseen future events.
* **Managing Noise:** Financial data is incredibly noisy. We use an XGBoost Regressor for its efficiency with tabular data, but intentionally restrict the tree depth (`max_depth=3`). This acts as a regularizer, preventing the algorithm from memorizing the noise and overfitting the training set.

### 4. Live Execution Bridge (`live_execution.py`)

* **Separation of Concerns:** Moving from a Jupyter notebook to live execution requires managing API limits and latency. This script splits the workload: Alpaca WebSockets stream the high-throughput equity pricing, while the Interactive Brokers Gateway (IBKR) handles the actual order routing and portfolio logic.
* **Thread Safety:** Blocking WebSockets can easily crash Python's `asyncio` event loop. By wrapping the Alpaca stream in a background daemon thread, it isolates the connection. This prevents loop collisions and automatically kills "zombie" processes during a graceful `Ctrl+C` shutdown.
* **Bypassing Rate Limits:** IBKR heavily restricts how often you can request account snapshots. Instead of polling, the script uses a low-level socket request (`ib.client.reqAccountUpdates`) to quietly cache liquidity updates in the background. This gives the execution logic instant access to account balances without halting the event loop.
* **Dynamic Sizing:** Position sizes scale linearly based on the expected value (EV) predicted by the model. When a signal clears the confidence threshold, the script fires a bracket order (Market Entry + GTC Trailing Stop) and logs the trade to a local CSV to track out-of-sample forward testing.

---

## Strategy Limitations & Risk Profile

* **Market Mechanics:** No model is immune to physical market realities. This specific strategy relies on high-frequency signals that decay rapidly, making it highly sensitive to execution latency, liquidity gaps, and spread widening during macroeconomic news events.
* **Absolute Returns vs. Risk Parity:** Because the pipeline relies on a tight 0.5% trailing stop to protect capital, it is designed to get stopped out frequently. Consequently, the absolute returns will likely underperform a simple passive "buy-and-hold" strategy during massive, volatile bull runs. The trade-off is significant downside protection and capital preservation during choppy or bearish regimes.

```

```
