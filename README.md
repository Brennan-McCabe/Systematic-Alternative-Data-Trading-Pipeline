The Fat-Tail Hunter: My Journey Building a Quant Trading Bot
Disclaimer: This is an educational project I built to learn machine learning and data engineering. It is currently running in a paper-trading sandbox. Please do not use this code to trade real money.

This project was originally supposed to be a simple stock prediction script. I wanted to feed 5 years of stock price data (Open, High, Low, Close, Volume) into an XGBoost model and see if it could predict a moving average crossover.

It worked on paper, but I quickly learned a series of hard lessons about the reality of algorithmic trading. Every single complex feature in this repository was built out of necessity to fix a fatal flaw I discovered in my earlier models.

Here is how a simple script turned into a fully automated, alternative-data quantitative pipeline.

Issue 1: The "Priced In" Reality (Asset Class Selection)
The Problem: I originally built this model to trade large-cap stocks (like Apple) and ETFs (like SPY). The backtests were terrible. I realized that the large-cap market is hyper-efficient; massive firms like Citadel and Jane Street have already arbitraged away any edge you can find in basic price data.The Solution: I pivoted to the "Wild West" of the market: a basket of 25 Small-Cap stocks. Small-caps are illiquid, highly volatile, and heavily driven by retail sentiment and options flow, leaving actual inefficiencies that a retail algorithmic model can exploit.

Issue 2: The 79% Mirage (Data Leakage)
The Problem: After pivoting to small-caps, my XGBoost model suddenly showed a ridiculous 79% annual return in backtesting. I thought I had cracked the market. In reality, the model was cheating. Because I was using a standard ML train_test_split (random K-Fold cross-validation), the model was training on Wednesday's data, testing on Monday's data, and using the "future" to predict the past.The Solution: I threw out standard ML validation and implemented Scikit-learn's TimeSeriesSplit. This forces the model to strictly "walk forward" in time, never seeing future data. My returns crashed back down to reality, but it forced me to build a system that actually worked instead of a time machine.

Issue 3: The ML Model was a "Coward" (The MSE Problem)
The Problem: Once the time-travel bug was fixed, the model used Mean Squared Error (MSE) to evaluate its predictions. MSE treats missing a massive 30% breakout exactly the same as missing a 2% normal daily fluctuation. Because massive breakouts are rare (fat tails), the AI learned the safest mathematical route was to just predict a stock would do nothing. It vetoed every single trade.The Solution: I dove into the math of XGBoost and wrote a Custom Asymmetric Loss Function. I engineered the gradient and hessian calculations to apply a 50x mathematical penalty if the AI missed a massive runner. This cured the AI's "cowardice" and forced it to optimize for Expected Value (EV) rather than just "safe" accuracy.

Issue 4: Price Data is "Too Slow"
The Problem: Even with the new loss function, the AI kept getting trapped in false breakouts. I realized that by the time a Moving Average crosses or a chart pattern forms, the institutions are already taking profit. Predicting based on chart data was like driving by looking in the rearview mirror.The Solution: I threw away the technical indicators and integrated the Polygon.io API to pull real-time Alternative Data. Now, the model looks at Institutional Options Flow (Put/Call ratios and extreme Call Volume shocks). Instead of trying to predict the chart, the AI tracks "smart money" urgency before the stock actually moves.

Issue 5: Manual Execution and Blown Accounts
The Problem: Even when the AI found a good setup, I couldn't sit at my computer all day to execute it. Furthermore, small-cap stocks are incredibly volatile. A stock could gap down 15% in ten minutes, blowing up my paper account.The Solution: I automated the execution and built a mathematical risk manager. I connected the Alpaca Trading API and scheduled the Python script to run automatically via a daily Cron Job. If the AI approves a trade, it dynamically scales the bet size based on Expected Value, calculates the 14-day Average True Range (ATR), and pegs a Good 'Til Canceled (GTC) 1.5x ATR hard stop-loss directly to the broker to protect capital.

Tech Stack
Language: Python (I used VS)

Machine Learning: XGBoost, Scikit-learn (TimeSeriesSplit)

Data Ingestion: Pandas, Numpy, Requests, Polygon.io Options API

Live Execution: Alpaca Trade API

To run it:

1. Clone the repository

git clone https://github.com/Brennan-McCabe/Systematic-Alternative-Data-Trading-Pipeline.git
cd Systematic-Alternative-Data-Trading-Pipeline/AltData-Quant-Pipeline

2. Install dependencies

pip install -r requirements.txt

3. Set up your API Keys
Create a .env file in the root directory. You will need free paper-trading keys from Alpaca and an Options API key from Polygon.

ALPACA_API_KEY="your_paper_key"
ALPACA_SECRET_KEY="your_secret_key"
OPTIONS_API_KEY="your_polygon_key"

4. Run the Pipeline

python quant_bot_live.py

Next Goals:

I'm working on the exit script and there's a few "realism" issues that I've yet to tackle:
Small-caps are illiquid and bid-ask spreads aren't considered at all yet
Options have a similar issue with illiquidity, small actions may be amplified by using the put/call ratio on small-caps

This was really just a project with the goal of learning and I've certainly accomplished that, there's still a lot to learn, though.
