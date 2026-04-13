import os
import datetime
import time
import numpy as np
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide
import warnings

warnings.filterwarnings('ignore')

# Load the hidden .env file (Same one your Entry Script uses)
load_dotenv()

# ==========================================
# 1. CONFIGURATION & API KEYS
# ==========================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Risk & Exit Parameters
HOLD_DAYS_TRADING = 5      # Replaced calendar days with actual market days
TAKE_PROFIT_PCT = 20.0     # Liquidate immediately if profit hits 20%
CHECK_INTERVAL_SEC = 300   # How often the bot checks the portfolio (300s = 5 mins)

# Initialize Alpaca Client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# ==========================================
# 2. THE EXIT ORCHESTRATOR
# ==========================================
def scan_and_liquidate():
    """Single pass of the portfolio to check for exits."""
    try:
        open_positions = trading_client.get_all_positions()
        
        if not open_positions:
            print("No open positions found. Portfolio is clean.")
            return
            
        print(f"Found {len(open_positions)} open positions. Evaluating...")

        for position in open_positions:
            ticker = position.symbol
            current_profit = float(position.unrealized_plpc) * 100 
            
            print(f"\nAnalyzing [{ticker}] | Current PnL: {current_profit:.2f}%")
            
            # --- EXIT TRIGGER 1: THE PARABOLIC TAKE-PROFIT ---
            if current_profit >= TAKE_PROFIT_PCT:
                print(f"[{ticker}] -> TAKE PROFIT TRIGGERED! Up {current_profit:.2f}%. Liquidating early.")
                try:
                    closed_order = trading_client.close_position(symbol_or_asset_id=ticker)
                    print(f"[{ticker}] -> SUCCESS. Secured the bag. Order ID: {closed_order.id}")
                except Exception as e:
                    print(f"[{ticker}] -> FAILED TO LIQUIDATE: {e}")
                continue # Skip the time-check and move to the next stock
            
            # --- EXIT TRIGGER 2: THE TIME LIMIT ---
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[ticker],
                side=OrderSide.BUY,
                limit=1 
            )
            
            order_history = trading_client.get_orders(req)
            
            if not order_history:
                print(f"[{ticker}] Warning: Could not locate original buy order. Skipping.")
                continue
                
            entry_date = order_history[0].filled_at.date()
            now_date = datetime.datetime.now(datetime.timezone.utc).date()
            
            # Calculate actual business/trading days held (ignores weekends)
            days_held = np.busday_count(entry_date, now_date)
            
            print(f"[{ticker}] Held for {days_held} trading days (Target: {HOLD_DAYS_TRADING} days).")
            
            if days_held >= HOLD_DAYS_TRADING:
                print(f"[{ticker}] -> TIME LIMIT REACHED. Initiating Liquidation.")
                try:
                    closed_order = trading_client.close_position(symbol_or_asset_id=ticker)
                    print(f"[{ticker}] -> SUCCESS! Position liquidated. Order ID: {closed_order.id}")
                except Exception as e:
                    print(f"[{ticker}] -> FAILED TO LIQUIDATE: {e}")
            else:
                print(f"[{ticker}] Holding position. Needs {HOLD_DAYS_TRADING - days_held} more trading days.")
                
            time.sleep(0.5) # API Rate Limit Buffer

    except Exception as e:
        print(f"Fatal Error in Exit Scan: {e}")

def main():
    print(f"--- Starting Autonomous Exit Engine (Continuous Mode) ---")
    while True:
        print(f"\n--- New Scan Cycle: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        scan_and_liquidate()
        print(f"--- Cycle Complete. Sleeping for {CHECK_INTERVAL_SEC / 60} minutes ---")
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
