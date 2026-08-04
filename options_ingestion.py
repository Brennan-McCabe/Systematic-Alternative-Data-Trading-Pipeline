import yfinance as yf
import pandas as pd
import requests
import os
import datetime
from dotenv import load_dotenv

load_dotenv()
OPTIONS_API_KEY = os.getenv("OPTIONS_API_KEY")

TICKERS = [
    "RIOT", "MARA", "PLUG", "FCEL", "RUN", "AMSC", "PACB", "EDIT", "CRSP", "GPRO", 
    "GOLF", "YELP", "EXTR", "AGYS", "PRTS", "BKE", "SAH", "RICK", "WNC", "GATX", 
    "AAON", "MTH", "LGIH", "WASH", "BANF", "UPST", "SOFI", "PLTR", "AFRM", "HOOD",  
    "AMC", "GME", "BYND", "CVNA", "SPCE", "QS", "LCID", "CHPT", "BLNK", "MVIS",  
    "DKNG", "PENN", "FUBO", "OPEN", "SOUN", "RKLB", "IONQ", "ROOT", "LMND", "AI"     
]

DATASET_FILE = "options_master_dataset.csv"

def fetch_options_features(ticker: str) -> dict:
    """Pulls end-of-day options flow from Polygon.io."""
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={OPTIONS_API_KEY}"
        response = requests.get(url).json()
        results = response.get("results", [])
        
        if not results:
            return None

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
        call_shock = call_vol / 2000  # Baseline expected volume scaling
        avg_iv = sum(iv_list) / len(iv_list) if iv_list else 0.50
        iv_rank = min(max(((avg_iv - 0.20) / (1.20 - 0.20)) * 100, 0.0), 100.0)
        
        return {"Opt_Put_Call_Ratio": p_c_ratio, "Opt_Call_Shock": call_shock, "Opt_IV_Rank": iv_rank}
    except Exception as e:
        print(f"[{ticker}] Options API Error: {e}")
        return None

def main():
    print(f"--- Running Daily Data Ingestion: {datetime.date.today()} ---")
    
    # Batch download daily prices and VIX
    prices = yf.download(TICKERS, period="5d", progress=False)['Close']
    vix = yf.download("^VIX", period="5d", progress=False)['Close'].iloc[-1].item()
    
    records = []
    for ticker in TICKERS:
        try:
            current_price = float(prices[ticker].iloc[-1])
            opt_data = fetch_options_features(ticker)
            
            if opt_data and current_price > 0:
                record = {
                    "Date": datetime.date.today(),
                    "Ticker": ticker,
                    "Close": current_price,
                    "VIX_Fear_Index": vix,
                    **opt_data
                }
                records.append(record)
                print(f"Logged [{ticker}]")
        except Exception as e:
            print(f"Skipped [{ticker}]: {e}")
            
    # Append to CSV
    df = pd.DataFrame(records)
    if os.path.exists(DATASET_FILE):
        df.to_csv(DATASET_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(DATASET_FILE, index=False)
        
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
