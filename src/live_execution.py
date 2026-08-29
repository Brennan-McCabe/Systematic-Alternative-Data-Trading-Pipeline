import os
import math
import asyncio
import polars as pl
import xgboost as xgb
import numpy as np
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

# Trading Parameters
SYMBOL = "MSFT"
CONFIDENCE_THRESHOLD = 0.001  # 0.1% EV prediction required
BASE_ALLOCATION = 0.30        # 30% baseline capital
MAX_ALLOCATION = 0.95         # 95% max capital cap
STOP_LOSS_PCT = 0.5           # 0.5% trailing stop

# Initialize Global Variables safely
ib = None  # Initialized dynamically inside the event loop
model = xgb.XGBRegressor()

# State Tracking
is_in_cooldown = False

# ==========================================
# 1. ALPACA EQUITY STREAM (Silent on Weekends)
# ==========================================
async def alpaca_handler(bar):
    print(f"[ALPACA] {SYMBOL} 1m Bar | Close: ${bar.close:.2f} | Vol: {bar.volume}")

async def start_alpaca_stream():
    try:
        stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        stream.subscribe_bars(alpaca_handler, SYMBOL)
        await asyncio.to_thread(stream.run)
    except Exception as e:
        print(f" -> [ALPACA] Stream Error: {e}")

# ==========================================
# 2. EXECUTION LOGIC (Dynamic Sizing & Routing)
# ==========================================
async def cooldown_timer():
    global is_in_cooldown
    # 5-second cooldown for the replay test (900s in live production)
    await asyncio.sleep(5)
    is_in_cooldown = False

async def execute_trade(predicted_ev):
    global is_in_cooldown
    
    if is_in_cooldown:
        return
        
    print(f"\n*** HIGH-CONVICTION SIGNAL: EV {predicted_ev*100:.3f}% ***")
    
    # Pull live account balance safely via async pull
    try:
        account_summary = await ib.reqAccountSummaryAsync()
    except Exception as e:
        print(f"[ERROR] Account summary request failed: {e}")
        return

    available_funds = 0.0
    for item in account_summary:
        if item.tag in ['AvailableFunds', 'BuyingPower', 'NetLiquidation']:
            try:
                available_funds = float(item.value)
                if available_funds > 0:
                    break
            except ValueError:
                continue
                
    if available_funds <= 0:
        raw_tags = list(set([item.tag for item in account_summary]))
        print(f"[ERROR] Could not locate capital. Available IBKR tags: {raw_tags[:15]}...")
        return

    # Bounded Linear Scaling Math
    allocation_pct = min(MAX_ALLOCATION, BASE_ALLOCATION * (predicted_ev / CONFIDENCE_THRESHOLD))
    capital_to_deploy = available_funds * allocation_pct
    
    # Get latest price to calculate share quantity
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    tickers = await ib.reqTickersAsync(contract)
    
    # Safe price extraction with weekend NaN handling
    current_price = tickers[0].marketPrice()
    if math.isnan(current_price) or current_price <= 0:
        current_price = tickers[0].close
    if math.isnan(current_price) or current_price <= 0:
        current_price = 415.0  # Hard fallback for offline weekend execution
    
    shares_to_buy = int(capital_to_deploy // current_price)
    
    if shares_to_buy <= 0:
        print("[WARNING] Insufficient capital to execute trade.")
        return

    print(f" -> Routing order: {shares_to_buy} shares @ ~${current_price:.2f} (Allocated {allocation_pct*100:.1f}%)")
    
    # Bracket Order: Market Entry + Trailing Stop
    parent = MarketOrder('BUY', shares_to_buy, transmit=False)
    stop = Order(
        action='SELL',
        orderType='TRAIL',
        totalQuantity=shares_to_buy,
        trailingPercent=STOP_LOSS_PCT,
        parentId=parent.orderId,
        transmit=True
    )
    
    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, stop)
    print(f" -> Execution complete. Trailing stop armed at {STOP_LOSS_PCT}%.")
    
    # Trigger non-blocking cooldown
    is_in_cooldown = True
    asyncio.create_task(cooldown_timer())

# ==========================================
# 3. MARKET REPLAY ENGINE (The AI Brain)
# ==========================================
async def start_parquet_replay_stream():
    print(" -> [REPLAY] Initializing Local Data Replay Engine...")
    
    # Train/load the model temporarily
    df = pl.read_parquet(DATASET_PATH).sort("timestamp_1m")
    X_train = df.select(["opt_put_call_ratio", "opt_call_shock"]).to_numpy()
    y_train = df.select("Realized_Target_EV").to_numpy().ravel()
    model.fit(X_train, y_train)
    
    print(" -> [BRAIN] XGBoost Model Armed and Ready.")
    
    for row in df.iter_rows(named=True):
        ratio = row["opt_put_call_ratio"]
        shock = row["opt_call_shock"]
        
        # Predict EV
        features = np.array([[ratio, shock]])
        predicted_ev = model.predict(features)[0]
        
        print(f"[TICK] Ratio: {ratio:.3f} | Shock: {shock:.3f} | Pred EV: {predicted_ev*100:.3f}%")
        
        if predicted_ev >= CONFIDENCE_THRESHOLD:
            await execute_trade(predicted_ev)
            
        await asyncio.sleep(0.5)  # Stream at 2 ticks per second

# ==========================================
# MAIN EVENT LOOP
# ==========================================
async def main():
    print("========================================")
    print(" STARTING LIVE EXECUTION PIPELINE")
    print("========================================")
    
    # Initialize IB strictly inside the active event loop
    global ib
    ib = IB()
    
    try:
        # Connect to IBKR Gateway Paper Trading
        await ib.connectAsync('127.0.0.1', 4002, clientId=1)
        print(" -> [BROKER] Connected to Interactive Brokers.")
        
    except Exception as e:
        print(f" -> [BROKER ERROR] Ensure TWS/Gateway is open and API is enabled: {e}")
        return

    await asyncio.gather(
        start_alpaca_stream(),
        start_parquet_replay_stream()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
        if ib is not None and ib.isConnected():
            ib.disconnect()
