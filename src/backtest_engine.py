import os
import polars as pl
import xgboost as xgb
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Dynamically locate the data folder relative to this exact script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "xgboost_training_data.parquet")

CONFIDENCE_THRESHOLD = 0.001  # 0.1% EV prediction required to trade
TRAIN_SPLIT = 0.80            
FRICTION_BPS = 0.0005         # 5 basis points round-trip penalty

# --- DYNAMIC CAPITAL ALLOCATION CONFIG ---
BASE_ALLOCATION = 0.30        # 30% of portfolio capital for baseline conviction
MAX_ALLOCATION = 0.95         # 95% hard cap for maximum conviction

def run_out_of_sample_backtest():
    print("Loading historical data for out-of-sample backtest...")
    
    # 1. Load Data
    df = pl.read_parquet(DATASET_PATH)
    df = df.sort("timestamp_1m")
    
    # 2. Chronological Train/Test Split
    split_index = int(len(df) * TRAIN_SPLIT)
    train_df = df.head(split_index)
    test_df = df.tail(len(df) - split_index)
    
    features = ["opt_put_call_ratio", "opt_call_shock"]
    target = "Realized_Target_EV"
    
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select(target).to_numpy().ravel()
    X_test = test_df.select(features).to_numpy()
    
    # 3. Train the Simulation Model
    print("Training simulation model on the past 80% of data...")
    sim_model = xgb.XGBRegressor(max_depth=3)
    sim_model.fit(X_train, y_train)
    
    # 4. Predict the Future (Unseen 20%)
    print("Predicting unseen holdout set...")
    predictions = sim_model.predict(X_test)
    test_df = test_df.with_columns(pl.Series("Predicted_EV", predictions))
    
    # 5. Simulate Trading Logic with Slippage/Friction
    test_df = test_df.with_columns(
        pl.when(pl.col("Predicted_EV") >= CONFIDENCE_THRESHOLD)
        .then(pl.col("Realized_Target_EV") - FRICTION_BPS)
        .otherwise(0.0)
        .alias("Net_Return")
    )
    
    # 6. Calculate Dynamic Position Sizing
    # Scales linearly from BASE_ALLOCATION up to MAX_ALLOCATION based on prediction strength
    test_df = test_df.with_columns(
        pl.when(pl.col("Predicted_EV") >= CONFIDENCE_THRESHOLD)
        .then(
            pl.min_horizontal(
                pl.lit(MAX_ALLOCATION),
                pl.lit(BASE_ALLOCATION) * (pl.col("Predicted_EV") / CONFIDENCE_THRESHOLD)
            )
        )
        .otherwise(0.0)
        .alias("Position_Size")
    )
    
    # 7. Calculate Portfolio Equity Curve 
    # Actual impact on the portfolio = the asset's net return * the % of capital allocated
    test_df = test_df.with_columns(
        (pl.col("Net_Return") * pl.col("Position_Size")).alias("Portfolio_Contribution")
    ).with_columns(
        (1 + pl.col("Portfolio_Contribution")).cum_prod().alias("Equity_Curve")
    )
    
    # 8. Extract Quantifiable Metrics
    # We analyze the portfolio contributions rather than the raw asset returns
    portfolio_returns = test_df.filter(pl.col("Position_Size") > 0.0)["Portfolio_Contribution"].to_numpy()
    position_sizes = test_df.filter(pl.col("Position_Size") > 0.0)["Position_Size"].to_numpy()
    
    if len(portfolio_returns) == 0:
        print("\nNo trades triggered in the holdout set.")
        return
        
    total_trades = len(portfolio_returns)
    winning_trades = len(portfolio_returns[portfolio_returns > 0])
    win_rate = winning_trades / total_trades
    
    avg_allocation = np.mean(position_sizes)
    net_cumulative = test_df["Equity_Curve"][-1] - 1.0
    
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 390) if std_return > 0 else 0
    
    equity_curve = test_df["Equity_Curve"].to_numpy()
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)
    
    # ==========================================
    # ALLOCATED OOS TEAR SHEET OUTPUT
    # ==========================================
    print("\n" + "="*40)
    print("DYNAMIC ALLOCATION TEAR SHEET".center(40))
    print("="*40)
    print(f"Total Trades Taken:   {total_trades}")
    print(f"Win Rate (Net):       {win_rate * 100:.2f}%")
    print(f"Avg Capital Risked:   {avg_allocation * 100:.2f}% per trade")
    print("-" * 40)
    print(f"Portfolio Net Return: {net_cumulative * 100:.2f}%")
    print(f"Portfolio Max Drawdown: {max_drawdown * 100:.3f}%")
    print(f"Est. Sharpe Ratio:    {sharpe_ratio:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_out_of_sample_backtest()
