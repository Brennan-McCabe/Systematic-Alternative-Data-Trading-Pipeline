import os
import datetime
import numpy as np
import time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

HOLD_DAYS_TRADING = 2  # Shorter leash for high-frequency trades
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def scan_and_liquidate():
    open_positions = trading_client.get_all_positions()
    
    if not open_positions:
        print(f"[{datetime.datetime.now()}] Portfolio clean. No open positions.")
        return
        
    print(f"[{datetime.datetime.now()}] Scanning {len(open_positions)} positions for stagnation...")

    for position in open_positions:
        ticker = position.symbol
        
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[ticker],
            side=OrderSide.BUY,
            limit=1 
        )
        order_history = trading_client.get_orders(req)
        
        if not order_history:
            continue
            
        entry_date = order_history[0].filled_at.date()
        now_date = datetime.datetime.now(datetime.timezone.utc).date()
        days_held = np.busday_count(entry_date, now_date)
        
        if days_held >= HOLD_DAYS_TRADING:
            print(f"[{ticker}] -> Stagnation Limit Reached ({days_held} days). Liquidating.")
            try:
                trading_client.close_position(symbol_or_asset_id=ticker)
            except Exception as e:
                print(f"[{ticker}] -> Failed to close: {e}")
        
        time.sleep(0.2) # API Rate Limit Buffer

if __name__ == "__main__":
    scan_and_liquidate()
