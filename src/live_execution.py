import time
import polars as pl
import xgboost as xgb
from ib_async import IB, Stock, MarketOrder

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "MSFT"
MODEL_PATH = "options_flow_model_v1.json"
DATASET_PATH = "xgboost_training_data.parquet"
STOP_LOSS_PCT = 0.005        # 0.5% trailing stop
CONFIDENCE_THRESHOLD = 0.004 # Minimum predicted EV (0.4%) to trigger entry

def get_latest_market_metrics() -> tuple[float, float]:
    """
    Pulls the most recent high-frequency options metrics from the local parquet cache.
    (In a full live deployment, replace this function with your real-time WebSocket/API listener).
    """
    df = pl.read_parquet(DATASET_PATH)
    # Sort by timestamp and grab the absolute latest row
    latest_row = df.sort("timestamp_1m").tail(1)
    
    ratio = float(latest_row["opt_put_call_ratio"][0])
    shock = float(latest_row["opt_call_shock"][0])
    return ratio, shock

def run_execution_loop():
    print("Initializing Live Execution Engine...")
    
    # 1. Load the XGBoost Model Brain
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"Loaded model successfully from {MODEL_PATH}")
    
    # 2. Connect to IBKR TWS Gateway
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4002, clientId=1)
        print("Connected to TWS Gateway for Live Execution.")
    except Exception as e:
        print(f"Failed to connect to IBKR: {e}")
        return

    # Define the underlying asset contract
    contract = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    print(f"\nMonitoring institutional flow for {SYMBOL}...")
    print("Press Ctrl+C to halt execution.\n")

    try:
        while True:
            # 3. Pull real features dynamically instead of using hardcoded values
            try:
                current_put_call_ratio, current_call_shock = get_latest_market_metrics()
            except Exception as read_err:
                print(f"[{time.strftime('%H:%M:%S সিস্ট')}] Error reading local feature feed: {read_err}")
                time.sleep(10)
                continue
            
            # Format features into the shape expected by XGBoost
            features = [[current_put_call_ratio, current_call_shock]]
            
            # 4. Predict Expected Value (EV) via Model
            predicted_ev = model.predict(features)[0]
            
            print(f"[{time.strftime('%H:%M:%S')}] Signal Checked | Ratio: {current_put_call_ratio:.4f} | Shock: {current_call_shock:.4f} | Pred EV: {predicted_ev*100:.3f}%")
            
            # 5. Execution Logic Trigger
            if predicted_ev >= CONFIDENCE_THRESHOLD:
                print(f"*** HIGH-CONVICTION SIGNAL DETECTED (EV: {predicted_ev*100:.3f}%) ***")
                print(f"Executing market order for {SYMBOL} with {STOP_LOSS_PCT*100}% trailing stop...")
                
                # Place a standard market entry order via IBKR
                order = MarketOrder('BUY', 10) # 10 shares test size
                trade = ib.placOrder(contract, order)
                
                ib.sleep(1)
                print(f"Order status: {trade.orderStatus.status}")
                
                # Cooldown period after firing a trade
                time.sleep(60)
            
            # Poll every 60 seconds to match the 1-minute interval structure
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nExecution halted by user. Disconnecting...")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected cleanly from IBKR.")

if __name__ == "__main__":
    run_execution_loop()