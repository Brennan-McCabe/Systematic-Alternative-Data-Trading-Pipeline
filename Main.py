import yfinance as yf
import pandas as pd
import polars as pl
import os
import datetime
import requests
from xgboost import XGBRegressor
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
OPTIONS_API_KEY = os.getenv("OPTIONS_API_KEY")

MIN_EV_THRESHOLD = 0.02  # Lower threshold for high-frequency setups
BET_SIZE = 1000          # Flat allocation per trade
TRAIL_PERCENT = 1.5      # 1.5% trailing stop

TICKERS = [
    "RIOT", "MARA", "PLUG", "FCEL", "RUN", "AMSC", "PACB", "EDIT", "CRSP", "GPRO", 
    "GOLF", "YELP", "EXTR", "AGYS", "PRTS", "BKE", "SAH", "RICK", "WNC", "GATX", 
    "AAON", "MTH", "LGIH", "WASH", "BANF", "UPST", "SOFI", "PLTR", "AFRM", "HOOD",  
    "AMC", "GME", "BYND", "CVNA", "SPCE", "QS", "LCID", "CHPT", "BLNK", "MVIS",  
    "DKNG", "PENN", "FUBO", "OPEN", "SOUN", "RKLB", "IONQ", "ROOT", "LMND", "AI"
]

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def fetch_options_features(ticker: str) -> dict:
    """Pulls live Options Flow data from Polygon.io."""
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={OPTIONS_API_KEY}"
        response = requests.get(url).json()
        results = response.get("results", [])
        
        if not results:
            return {"Opt_Put_Call_Ratio": 1.0, "Opt_Call_Shock": 1.0, "Opt_IV_Rank": 50.0}

        call_vol, put_vol = 0, 0
        iv_list = []
        
        for contract in results:
            details = contract.get("details", {})
            day_data = contract.get("day", {})
            iv = contract.get("implied_volatility")
            
            if details.get("contract_type") == "call":
                call_vol += day_data.get("volume", 0)
            elif details.get("contract_type") == "put":
                put_vol += day_data.get("volume", 0)
                
            if iv:
                iv_list.append(iv)
                
        p_c_ratio = min((put_vol / call_vol) if call_vol > 0 else 5.0, 5.0)
        call_shock = call_vol / 2000
        avg_iv = sum(iv_list) / len(iv_list) if iv_list else 0.50
        iv_rank = min(max(((avg_iv - 0.20) / (1.20 - 0.20)) * 100, 0.0), 100.0)
        
        return {"Opt_Put_Call_Ratio": p_c_ratio, "Opt_Call_Shock": call_shock, "Opt_IV_Rank": iv_rank}
    except Exception:
        return {"Opt_Put_Call_Ratio": 1.0, "Opt_Call_Shock": 1.0, "Opt_IV_Rank": 50.0}

def execute_trade(ticker: str, current_price: float):
    qty = int(BET_SIZE // current_price)
    if qty <= 0: return

    try:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            trailing_stop_loss=TrailingStopLossRequest(trail_percent=TRAIL_PERCENT)
        )
        submitted = trading_client.submit_order(order_data=order)
        print(f"[{ticker}] -> EXECUTED: {qty} shares @ ~${current_price:.2f} | Trail: {TRAIL_PERCENT}% | ID: {submitted.id}")
    except Exception as e:
        print(f"[{ticker}] -> EXECUTION FAILED: {e}")

def main():
    print(f"--- Quant Engine Started: {datetime.datetime.now()} ---")
    
    try:
        model = XGBRegressor()
        model.load_model("options_flow_model_v1.json")
    except Exception:
        print("Model file not found. Ensure research_pipeline.py has been run.")
        return

    # I/O Optimization: Batch Download
    print("Fetching batch market data...")
    batch_data = yf.download(TICKERS, period="1mo", progress=False)
    vix_data = yf.download("^VIX", period="5d", progress=False)
    current_vix = float(vix_data['Close'].iloc[-1].item())
    
    for ticker in TICKERS:
        try:
            # Extract single ticker series from batch
            close_series = batch_data['Close'][ticker].dropna()
            if len(close_series) < 30:
                continue
                
            current_price = float(close_series.iloc[-1])
            ma_30 = close_series.rolling(30).mean().iloc[-1]
            
            # Regime Filter Shield
            if current_price < ma_30:
                continue
                
            options_data = fetch_options_features(ticker)
            
            live_features = pl.DataFrame([{
                "Opt_Put_Call_Ratio": options_data["Opt_Put_Call_Ratio"],
                "Opt_Call_Shock": options_data["Opt_Call_Shock"],
                "Opt_IV_Rank": options_data["Opt_IV_Rank"],
                "VIX_Fear_Index": current_vix
            }])
            
            predicted_ev = float(model.predict(live_features)[0])
            
            if predicted_ev >= MIN_EV_THRESHOLD:
                print(f"[{ticker}] EV: {predicted_ev*100:.2f}% (Threshold Met)")
                execute_trade(ticker, current_price)
                
        except Exception as e:
            print(f"[{ticker}] Error processing pipeline: {e}")

if __name__ == "__main__":
    main()
