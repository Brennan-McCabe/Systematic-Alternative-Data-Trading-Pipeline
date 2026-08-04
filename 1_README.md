# Systematic Alternative Data Trading Pipeline

**Disclaimer:** This is an educational project I built to learn machine learning, data engineering, and quantitative finance. It is currently running in a paper-trading sandbox. Please do not use this code to trade real money.

### Overview

The current iteration of this project is a high-frequency statistical arbitrage pipeline. The XGBoost model monitors a basket of 50 volatile small-cap stocks, feeding on live institutional options flow (Call/Put volume and implied volatility shocks) via Polygon.io.

If the model detects an Expected Value (EV) edge above a strict 2% threshold, it automatically executes a dynamically sized position. Risk is managed natively on the broker side using tight, dynamic trailing stops to lock in profits, while a secondary exit engine continuously prunes stagnant, sideways-trading positions to free up capital. All actions are performed fully autonomously in an Alpaca paper account using scheduled cron tasks.

---

### The Evolution: Hard Lessons in Algorithmic Trading

This project was originally supposed to be a simple stock prediction script. I wanted to feed 5 years of stock price data into an XGBoost model and see if it could predict a moving average crossover.

It worked on paper, but I quickly learned a series of hard lessons about the reality of algorithmic trading. Every single complex feature in this repository was built out of necessity to fix a fatal flaw I discovered in earlier models. Here is how a simple script turned into a fully automated, alternative-data quantitative pipeline.

#### Issue 1: The "Priced In" Reality (Asset Class Selection)

**The Problem:** I originally built this model to trade large-cap stocks (like Apple) and ETFs (like SPY). The backtests were terrible. I realized that the large-cap market is hyper-efficient; massive firms have already arbitraged away any edge you can find in basic price data.
**The Solution:** I pivoted to the "Wild West" of the market: a basket of 50 Small-Cap stocks. Small-caps are illiquid, highly volatile, and heavily driven by retail sentiment and options flow, leaving actual inefficiencies that a retail algorithmic model can exploit.

#### Issue 2: The 79% Mirage (Data Leakage)

**The Problem:** After pivoting to small-caps, my XGBoost model suddenly showed a ridiculous 79% annual return in backtesting. It meant I had either cracked the market or something was broken. In reality, the model was cheating. By using a standard ML `train_test_split`, the model was training on Wednesday's data, testing on Monday's data, and using the "future" to predict the past.
**The Solution:** I threw out standard ML validation and implemented Scikit-learn's `TimeSeriesSplit`. This forces the model to strictly "walk forward" in time, never seeing future data. My returns crashed back down to reality, but it forced me to build a system that actually worked instead of a time machine.

#### Issue 3: Price Data is "Too Slow"

**The Problem:** The AI kept betting on losing outcomes. I realized that by the time a chart pattern forms, the institutions are already taking profit. Predicting based on chart data was like driving by looking in the rearview mirror.
**The Solution:** I threw away the technical indicators and integrated the Polygon.io API to pull real-time Alternative Data. Now, the model looks at Institutional Options Flow (Put/Call ratios and extreme Call Volume shocks). Instead of trying to predict the chart, the AI tracks "smart money" and follows before the shift is completely priced in.

#### Issue 4: The Train-Serve Skew (The Proxy Problem)

**The Problem:** Because historical options data is incredibly expensive, I tried to backtest my model using synthetic equity volume proxies to simulate history. I realized this created a massive statistical disconnect—my model was training on fake proxies but trading on real options data in production.
**The Solution:** I threw out the proxies and built a forward-logger (`options_ingestion.py`). Now, the pipeline runs daily to collect and log pristine, live-sampled Polygon data. The model strictly trains on the exact same data distributions it sees in the live market.

#### Issue 5: Hunting Black Swans is a Trap (The Strategy Pivot)

**The Problem:** For a long time, the pipeline used a custom asymmetric loss function to hunt for massive 20% "fat tail" breakouts. But relying on rare black swans meant suffering through a terrible win rate and heavy drawdowns.
**The Solution:** I pivoted the architecture to a higher-frequency statistical edge model. I lowered the execution threshold to 2% and replaced static hard stops with dynamic trailing stops via Alpaca. Instead of hunting for massive home runs, the bot now aggressively locks in smaller, frequent wins as the options flow pushes the stock up.

#### Issue 6: Exit Engine Bloat (State Management)

**The Problem:** My original exit engine aggressively queried Alpaca's API history to manage complex take-profit brackets and hard stops, which was brittle and prone to failure.
**The Solution:** I handed risk management entirely over to Alpaca's native servers using `TrailingStopLossRequest`. Now, the `exit_engine.py` is just a lightweight stagnation pruner that automatically frees up capital if a stock trades sideways for two days.

---

### Tech Stack

* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-learn (`TimeSeriesSplit`)
* **Data Ingestion & Engineering:** Polars, Pandas, Numpy, Requests
* **APIs:** Polygon.io (Options Flow), Alpaca Trade API (Execution), Yahoo Finance (Macro/VIX)
* **Automation:** PythonAnywhere scheduled tasks

---

### How to Run the Pipeline

**1. Clone the repository**

```bash
git clone https://github.com/Brennan-McCabe/Systematic-Alternative-Data-Trading-Pipeline.git
cd Systematic-Alternative-Data-Trading-Pipeline

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Set up your API Keys**
Create a `.env` file in the root directory. You will need free paper-trading keys from Alpaca and an Options API key from Polygon ($29/mo).

```env
ALPACA_API_KEY="your_paper_key"
ALPACA_SECRET_KEY="your_secret_key"
OPTIONS_API_KEY="your_polygon_key"

```

**4. Build the Dataset (Daily)**
Schedule `options_ingestion.py` to run daily after market close. This builds the historical `.csv` of pristine options flow required for training.

**5. Train the Model**
Once you have enough logged data, run `research_pipeline.py`. This utilizes strict walk-forward cross-validation to evaluate the strategy and exports the production `options_flow_model_v1.json` file.

**6. Live Execution**
Schedule `Main.py` to run autonomously (e.g., via PythonAnywhere).

**7. Stagnation Management**
Run `exit_engine.py` as a continuous or highly frequent scheduled task to prune trades that have stalled out.

---

### Next Goals

* **Slippage Management:** Small-caps are illiquid and bid-ask spreads aren't factored into the current market order execution. The next major update will replace `MarketOrderRequest` with `LimitOrderRequest` dynamically pegged to the NBBO midpoint.
* **Options Illiquidity:** Small actions may be amplified by using the put/call ratio on highly illiquid small-cap options chains, which requires a new smoothing function in the ingestion script.

*This was really just a project with the goal of learning and I've certainly accomplished that, but there's still a lot to learn.*
