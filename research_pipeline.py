import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

DATASET_FILE = "options_master_dataset.csv"
HOLD_DAYS = 2  # Reduced hold time for tighter trades
MIN_EV_THRESHOLD = 0.02

def run_research():
    print("--- Loading Options Dataset ---")
    try:
        df = pd.read_csv(DATASET_FILE, parse_dates=["Date"])
    except FileNotFoundError:
        print("Dataset not found. Run options_ingestion.py to build history first.")
        return

    df = df.sort_values(by=["Ticker", "Date"])
    
    # Calculate Target: Forward 2-day return
    df['Target_EV'] = df.groupby('Ticker')['Close'].transform(lambda x: (x.shift(-HOLD_DAYS) - x) / x)
    df = df.dropna(subset=['Target_EV'])
    
    # Sort chronologically for strict walk-forward validation
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    features = ['Opt_Put_Call_Ratio', 'Opt_Call_Shock', 'Opt_IV_Rank', 'VIX_Fear_Index']
    X = df[features]
    y = df['Target_EV']
    
    print("\n--- Starting Walk-Forward Backtest ---")
    tscv = TimeSeriesSplit(n_splits=5)
    
    total_trades, winning_trades, total_pnl = 0, 0, 0.0
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        for i in range(len(preds)):
            if preds[i] >= MIN_EV_THRESHOLD:
                total_trades += 1
                actual_ret = y_test.iloc[i]
                
                # Simulate Trailing Stop approximation (cap max loss at -2%)
                realized_ret = max(actual_ret, -0.02)
                
                total_pnl += realized_ret
                if realized_ret > 0:
                    winning_trades += 1
                    
        print(f"Fold {fold} complete.")
        
    if total_trades > 0:
        print("\n==========================================")
        print("OUT-OF-SAMPLE BACKTEST RESULTS")
        print("==========================================")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {(winning_trades/total_trades)*100:.2f}%")
        print(f"Avg PnL per Trade: {(total_pnl/total_trades)*100:.2f}%")
    else:
        print("No trades met the execution threshold.")

    # Train final model on ALL data for deployment
    final_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3)
    final_model.fit(X, y)
    final_model.save_model("options_flow_model_v1.json")
    print("\nProduction model saved as 'options_flow_model_v1.json'.")

if __name__ == "__main__":
    run_research()
