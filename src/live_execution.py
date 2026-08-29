import os
import csv
import math
import asyncio
import threading
import polars as pl
import xgboost as xgb
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ib_async import IB, Stock, MarketOrder, Order
from alpaca.data.live import StockDataStream

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "xgboost_training_data.parquet")
LOG_FILE = os.path.join(BASE_DIR, "execution_log.csv")

# Strategy Parameters
SYMBOL = "MSFT"
CONFIDENCE_THRESHOLD = 0.001  # 0.1% expected value required for order execution
BASE_ALLOCATION = 0.30        # Baseline capital deployment
MAX_ALLOCATION = 0.95         # Maximum portfolio exposure limit
STOP_LOSS_PCT = 0.5           # Trailing stop-loss percentage

# Global State Variables
ib = None  
alpaca_stream = None  
model = xgb.XGBRegressor()
is_in_cooldown = False

# ==========================================
# 0. EXECUTION LOGGING
# ==========================================
def initialize_logger():
    """Initializes the CSV execution log and writes headers if the file does not exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Symbol", "Predicted_EV", "Allocation_Pct", "Shares", "Entry_Price"])
        print(f"[SYSTEM] Initialized execution log at: {LOG_FILE}")

# ==========================================
# 1. MARKET DATA STREAM (Daemonized)
# ==========================================
async def alpaca_handler(bar):
    print(f"[MARKET DATA] {SYMBOL} 1m Bar | Close: ${bar.close:.2f} | Vol: {bar.volume}")

def run_alpaca_silently():
    """Executes the Alpaca WebSocket stream within a background daemon thread."""
    try:
        alpaca_stream.run()
    except Exception:
        pass  # Handle thread termination exceptions gracefully during shutdown

async def start_alpaca_stream():
    global alpaca_stream
    try:
        alpaca_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        alpaca_stream.subscribe_bars(alpaca_handler, SYMBOL)
        
        # Initialize the stream as a daemon thread to ensure immediate termination upon script exit
        stream_thread = threading.Thread(target=run_alpaca_silently, daemon=True)
        stream_thread.start()
        
        # Maintain asynchronous task execution state
        while stream_thread.is_alive():
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"[MARKET DATA ERROR] Stream initialization failed: {e}")

# ==========================================
# 2. EXECUTION LOGIC (Sizing & Routing)
# ==========================================
async def cooldown_timer():
    """Prevents duplicate order execution by enforcing a minimum time delay between trades."""
    global is_in_cooldown
    await asyncio.sleep(5)
    is_in_cooldown = False

async def execute_trade(predicted_ev):
    global is_in_cooldown
    
    if is_in_cooldown:
        return
        
    print(f"\n[INFERENCE] Target threshold exceeded. Predicted EV: {predicted_ev*100:.3f}%")
    
    # Retrieve synchronized account state
    available_funds = 0.0
    for item in ib.accountValues():
        if item.tag in ['AvailableFunds', 'BuyingPower', 'NetLiquidation']:
            try:
                val = float(item.value)
                if val > 0:
                    available_funds = val
                    break
            except ValueError:
                continue
                
    if available_funds <= 0:
        print("[EXECUTION ERROR] Failed to retrieve synchronized account state. Order bypassed.")
        return

    # Position Sizing Calculation (Bounded Linear Scaling)
    allocation_pct = min(MAX_ALLOCATION, BASE_ALLOCATION * (predicted_ev / CONFIDENCE_THRESHOLD))
    capital_to_deploy = available_funds * allocation_pct
    
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    tickers = await ib.reqTickersAsync(contract)
    
    # Market price retrieval with offline/NaN contingency handling
    current_price = tickers[0].marketPrice()
    if math.isnan(current_price) or current_price <= 0:
        current_price = tickers[0].close
    if math.isnan(current_price) or current_price <= 0:
        current_price = 415.0  
    
    shares_to_buy = int(capital_to_deploy // current_price)
    
    if shares_to_buy <= 0:
        print("[EXECUTION WARNING] Insufficient allocated capital for minimum position size. Order bypassed.")
        return

    print(f"[EXECUTION] Routing {shares_to_buy} shares @ ~${current_price:.2f} | Allocation: {allocation_pct*100:.1f}%")
    
    # Construct Bracket Order (Includes Extended Hours Flags)
    parent = MarketOrder('BUY', shares_to_buy, transmit=False, outsideRth=True)
    stop = Order(
        action='SELL',
        orderType='TRAIL',
        totalQuantity=shares_to_buy,
        trailingPercent=STOP_LOSS_PCT,
        parentId=parent.orderId,
        tif='GTC',
        outsideRth=True,
        transmit=True
    )
    
    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, stop)
    
    # Append execution data to local CSV log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, 
            SYMBOL, 
            f"{predicted_ev*100:.3f}%", 
            f"{allocation_pct*100:.1f}%", 
            shares_to_buy, 
            f"${current_price:.2f}"
        ])
        
    print(f"[EXECUTION] Order transmitted. Trailing stop configured at {STOP_LOSS_PCT}%. Log appended.")
    
    is_in_cooldown = True
    asyncio.create_task(cooldown_timer())

# ==========================================
# 3. DATA REPLAY ENGINE 
# ==========================================
async def start_parquet_replay_stream():
    print("[SYSTEM] Initializing historical data replay engine...")
    
    df = pl.read_parquet(DATASET_PATH).sort("timestamp_1m")
    X_train = df.select(["opt_put_call_ratio", "opt_call_shock"]).to_numpy()
    y_train = df.select("Realized_Target_EV").to_numpy().ravel()
    model.fit(X_train, y_train)
    
    print("[INFERENCE] XGBoost model fitted and initialized for evaluation.")
    
    for row in df.iter_rows(named=True):
        ratio = row["opt_put_call_ratio"]
        shock = row["opt_call_shock"]
        
        features = np.array([[ratio, shock]])
        predicted_ev = model.predict(features)[0]
        
        print(f"[MARKET TICK] P/C Ratio: {ratio:.3f} | Call Shock: {shock:.3f} | Predicted EV: {predicted_ev*100:.3f}%")
        
        if predicted_ev >= CONFIDENCE_THRESHOLD:
            await execute_trade(predicted_ev)
            
        await asyncio.sleep(0.5) 

# ==========================================
# MAIN EVENT LOOP
# ==========================================
async def main():
    print("========================================")
    print(" INITIALIZING ALGORITHMIC EXECUTION PIPELINE")
    print("========================================")
    
    initialize_logger()
    
    global ib
    ib = IB()
    
    try:
        await ib.connectAsync('127.0.0.1', 4002, clientId=1)
        print("[BROKER] Connection to Interactive Brokers established.")
        
        await asyncio.sleep(1)
        accounts = ib.managedAccounts()
        account_id = accounts[0] if accounts else ''
        
        ib.client.reqAccountUpdates(True, account_id)
        print(f"[BROKER] Background account state synchronization activated (Account: {account_id})")
        await asyncio.sleep(2)  
            
    except Exception as e:
        print(f"[BROKER ERROR] Connection failed. Verify TWS/Gateway status and API configuration: {e}")
        return

    await asyncio.gather(
        start_alpaca_stream(),
        start_parquet_replay_stream()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM] KeyboardInterrupt received. Initiating graceful shutdown sequence...")
        
        if ib is not None and ib.isConnected():
            ib.disconnect()
            
        print("[SYSTEM] Shutdown complete. Resources released.")
