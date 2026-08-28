import polars as pl
import xgboost as xgb
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "data/xgboost_training_data.parquet"
CONFIDENCE_THRESHOLD = 0.001  # 0.1% EV prediction required to trade
TRAIN_SPLIT = 0.80            # Train on first 80%, test on last 20%
FRICTION_BPS = 0.0005         # 5 basis points (0.05%) round-trip penalty

def run_out_of_sample_backtest():
    print("Loading historical data for out-of-sample backtest...")
    
    # 1. Load Data
    df = pl.read_parquet(DATASET_PATH)
    
    # Ensure data is sorted chronologically
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
    # Calculate Theoretical (Gross) Return
    test_df = test_df.with_columns(
        pl.when(pl.col("Predicted_EV") >= CONFIDENCE_THRESHOLD)
        .then(pl.col("Realized_Target_EV"))
        .otherwise(0.0)
        .alias("Gross_Return")
    )
    
    # Calculate Realistic (Net) Return after the 5 bps tax
    test_df = test_df.with_columns(
        pl.when(pl.col("Predicted_EV") >= CONFIDENCE_THRESHOLD)
        .then(pl.col("Realized_Target_EV") - FRICTION_BPS)
        .otherwise(0.0)
        .alias("Net_Return")
    )
    
    # 6. Calculate Portfolio Equity Curve (Based on Net Returns)
    test_df = test_df.with_columns(
        (1 + pl.col("Net_Return")).cum_prod().alias("Equity_Curve")
    )
    
    # 7. Extract Quantifiable Metrics
    net_returns = test_df.filter(pl.col("Net_Return") != 0.0)["Net_Return"].to_numpy()
    
    if len(net_returns) == 0:
        print("\nNo trades triggered in the holdout set.")
        return
        
    total_trades = len(net_returns)
    winning_trades = len(net_returns[net_returns > 0])
    win_rate = winning_trades / total_trades
    
    gross_cumulative = (1 + test_df["Gross_Return"]).cum_prod()[-1] - 1.0
    net_cumulative = test_df["Equity_Curve"][-1] - 1.0
    
    # Annualized Sharpe Ratio (Net)
    mean_return = np.mean(net_returns)
    std_return = np.std(net_returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 390) if std_return > 0 else 0
    
    # Maximum Drawdown (Net)
    equity_curve = test_df["Equity_Curve"].to_numpy()
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)
    
    # ==========================================
    # OOS TEAR SHEET OUTPUT
    # ==========================================
    print("\n" + "="*40)
    print("OUT-OF-SAMPLE TEAR SHEET (FRICTION APPLIED)".center(40))
    print("="*40)
    print(f"Total Trades Taken:   {total_trades}")
    print(f"Win Rate (Net):       {win_rate * 100:.2f}%")
    print("-" * 40)
    print(f"Gross Return:         {gross_cumulative * 100:.2f}%")
    print(f"Net Return:           {net_cumulative * 100:.2f}%")
    print(f"Lost to Friction:     {(gross_cumulative - net_cumulative) * 100:.2f}%")
    print("-" * 40)
    print(f"Max Drawdown (Net):   {max_drawdown * 100:.2f}%")
    print(f"Est. Sharpe Ratio:    {sharpe_ratio:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_out_of_sample_backtest()
