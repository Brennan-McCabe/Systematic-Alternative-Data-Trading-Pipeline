import os
import csv
import math
import logging
import asyncio
import threading
import numpy as np
import polars as pl
import xgboost as xgb
from datetime import datetime
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from ib_async import IB, Stock, MarketOrder, Order

# ==========================================
# 0. MUTE VERBOSE LIBRARY LOGGING
# ==========================================
logging.getLogger('ib_async').setLevel(logging.CRITICAL)

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
CONFIDENCE_THRESHOLD = 0.001  
BASE_ALLOCATION = 0.30        
MAX_ALLOCATION = 0.95         
STOP_LOSS_PCT = 0.5           

# Global State Variables
ib = None  
alpaca_stream = None  
model = xgb.XGBRegressor()
is_in_cooldown = False

# ==========================================
# 1. CUSTOM ERROR HANDLERS (Clean Output)
# ==========================================
def on_broker_error(reqId, errorCode, errorString, contract):
    """Filters out expected noise and cleanly formats real broker errors."""
    # 10089: Delayed market data (expected for paper accounts)
    # 2109: Outside RTH ignored (expected for Market orders)
    # 10349: TIF DAY canceled (expected on weekends)
    ignored_codes = [10089, 2109, 10349]
    
    if errorCode in ignored_codes:
        return
        
    print(f"[BROKER MESSAGE] Code {errorCode}: {errorString}")

def on_order_cancel(trade):
    """Replaces massive Trade object dumps with a clean cancellation notice."""
    print(f"[EXECUTION CANCELLED] Exchange rejected order {trade.order.orderId} (Market Closed).")

def initialize_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Symbol", "Predicted_EV", "Allocation_Pct", "Shares", "Entry_Price"])
        print(f"[SYSTEM] Initialized execution log at: {LOG_FILE}")

# ==========================================
# 2. MARKET DATA STREAM (Daemonized)
# ==========================================
async def alpaca_handler(bar):
    print(f"[MARKET DATA] {SYMBOL} 1m Bar | Close: ${bar.close:.2f} | Vol: {bar.volume}")

def run_alpaca_silently():
    try:
        alpaca_stream.run()
    except Exception:
        pass  

async def start_alpaca_stream():
    global alpaca_stream
    try:
        alpaca_stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        alpaca_stream.subscribe_bars(alpaca_handler, SYMBOL)
        
        stream_thread = threading.Thread(target=run_alpaca_silently, daemon=True)
        stream_thread.start()
        
        while stream_thread.is_alive():
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"[MARKET DATA ERROR] Stream initialization failed: {e}")

# ==========================================
# 3. EXECUTION LOGIC (Sizing & Routing)
# ==========================================
async def cooldown_timer():
    global is_in_cooldown
    await asyncio.sleep(5)
    is_in_cooldown = False

async def execute_trade(predicted_ev):
    global is_in_cooldown
    
    if is_in_cooldown:
        return
        
    print(f"\n[INFERENCE] Target threshold exceeded. Predicted EV: {predicted_ev*100:.3f}%")
    
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

    allocation_pct = min(MAX_ALLOCATION, BASE_ALLOCATION * (predicted_ev / CONFIDENCE_THRESHOLD))
    capital_to_deploy = available_funds * allocation_pct
    
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    tickers = await ib.reqTickersAsync(contract)
    
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
# 4. DATA REPLAY ENGINE 
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
    
    # Wire the custom event handlers
    ib.errorEvent += on_broker_error
    ib.cancelOrderEvent += on_order_cancel
    
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
