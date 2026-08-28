import os
from datetime import datetime
import databento as db
import polars as pl
import yfinance as yf
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================
load_dotenv()
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Ensure local data directory exists to prevent path errors
os.makedirs("data", exist_ok=True)

# Define the targeted market regimes for stratified sampling
REGIMES = [
    # 1. Baseline Flat Period (Low Volatility / Sideways Chop)
    {"start": "2023-08-14", "end": "2023-08-19", "label": "flat_baseline"},
    
    # 2. Grind-Down Bear (Persistent distribution)
    {"start": "2022-10-10", "end": "2022-10-15", "label": "bear_grind"},
    
    # 3. Euphoric Bull Run-Up (Preceding late Jan 2024 tech melt-up)
    {"start": "2024-01-22", "end": "2024-01-27", "label": "bull_meltup"},
    
    # 4. Macro Shock Run-Up (Preceding the Aug 5, 2026 VIX spike)
    {"start": "2026-07-31", "end": "2026-08-06", "label": "macro_shock"}
]

def build_regime_dataset():
    db_client = db.Historical(DATABENTO_API_KEY)
    alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    
    all_regime_frames = []
    
    for regime in REGIMES:
        print(f"\n[{regime['label'].upper()}] Fetching data from {regime['start']} to {regime['end']}...")
        
        # ==========================================
        # 1. FETCH UNDERLYING EQUITY (ALPACA)
        # ==========================================
        print(" -> Downloading MSFT equity prices (1m) via Alpaca...")
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols="MSFT",
                timeframe=TimeFrame.Minute,
                start=datetime.strptime(regime["start"], "%Y-%m-%d"),
                end=datetime.strptime(regime["end"], "%Y-%m-%d")
            )
            bars = alpaca_client.get_stock_bars(request_params)
            
            if bars.df.empty:
                print(" -> WARNING: No equity data found for this range. Skipping.")
                continue
                
            alpaca_df = bars.df.reset_index()
            alpaca_df.rename(columns={"timestamp": "timestamp_1m"}, inplace=True)
            equity_df = pl.from_pandas(alpaca_df)
            
            # Strip timezone metadata to enforce safe, naive UTC joins
            equity_df = equity_df.with_columns(
                pl.col("timestamp_1m").dt.replace_time_zone(None)
            )
            
        except Exception as e:
            print(f" -> ERROR fetching Alpaca data: {e}")
            continue

        # ==========================================
        # 2. FETCH OPRA TAPE (DATABENTO) & CACHE
        # ==========================================
        opra_cache_path = f"data/MSFT_{regime['label']}_opra.parquet"
        
        if os.path.exists(opra_cache_path):
            print(" -> Loading OPRA tape from FREE local cache...")
            options_df = pl.read_parquet(opra_cache_path)
        else:
            print(" -> Downloading MSFT options tape via Databento...")
            try:
                db_data = db_client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    symbols="MSFT.OPT",
                    stype_in="parent",
                    schema="trades",
                    start=regime["start"],
                    end=regime["end"],
                    limit=None        
                )
                options_df = pl.from_pandas(db_data.to_df())
                
                # Instantly cache to hard drive to prevent re-billing!
                options_df.write_parquet(opra_cache_path) 
                
            except Exception as e:
                print(f" -> ERROR fetching Databento data: {e}")
                continue
                
        # ==========================================
        # 3. FEATURE ENGINEERING & TARGET LABELING
        # ==========================================
        print(" -> Applying feature engineering and stop-loss logic...")
        
        try:
            # Format Options Timestamps and Identify Put vs Call
            options_df = options_df.with_columns(
                pl.col("ts_event").dt.replace_time_zone(None).dt.truncate("1m").alias("timestamp_1m"),
                pl.when(pl.col("symbol").str.contains(r"C[0-9]{8}$")).then(pl.lit("C"))
                  .when(pl.col("symbol").str.contains(r"P[0-9]{8}$")).then(pl.lit("P"))
                  .otherwise(pl.lit("OTHER")).alias("option_type")
            )
            
            # Group OPRA tick data into 1-minute volume bins
            opt_grouped = options_df.group_by("timestamp_1m").agg([
                pl.col("size").filter(pl.col("option_type") == "C").sum().alias("call_volume"),
                pl.col("size").filter(pl.col("option_type") == "P").sum().alias("put_volume")
            ])
            
            # --- THE FIX: Align Time Resolutions ---
            equity_df = equity_df.with_columns(pl.col("timestamp_1m").cast(pl.Datetime("ns")))
            opt_grouped = opt_grouped.with_columns(pl.col("timestamp_1m").cast(pl.Datetime("ns")))
            
            # Join Equity and Options data
            engineered_df = equity_df.join(opt_grouped, on="timestamp_1m", how="inner")
            
            # Calculate Market Microstructure Features
            engineered_df = engineered_df.sort("timestamp_1m")
            engineered_df = engineered_df.with_columns(
                (pl.col("put_volume") / (pl.col("call_volume") + 1.0)).alias("opt_put_call_ratio"),
                pl.col("call_volume").rolling_mean(30).alias("call_mean"),
                pl.col("call_volume").rolling_std(30).alias("call_std")
            ).with_columns(
                ((pl.col("call_volume") - pl.col("call_mean")) / (pl.col("call_std") + 1.0)).alias("opt_call_shock")
            )
            
            # Path-Dependent Target Engineering (15-Minute Forward Horizon)
            # Create a list of the next 15 minute closing prices
            engineered_df = engineered_df.with_columns(
                pl.concat_list([pl.col("close").shift(-i) for i in range(1, 16)]).alias("future_prices")
            )
            
            # Define trailing stop-loss execution
            def calc_ev(row):
                entry = row["close"]
                futures = row["future_prices"]
                if entry is None or futures is None or None in futures:
                    return None
                
                running_max = entry
                for price in futures:
                    if price > running_max:
                        running_max = price
                    if price <= running_max * 0.995:  # 0.5% Trailing Stop Trigger
                        return -0.005
                
                return (futures[-1] - entry) / entry
                
            engineered_df = engineered_df.with_columns(
                pl.struct(["close", "future_prices"])
                .map_elements(calc_ev, return_dtype=pl.Float64)
                .alias("Realized_Target_EV")
            )
            
            # Cleanup burn-in and forward-horizon nulls
            engineered_df = engineered_df.select([
                "timestamp_1m", "close", "opt_put_call_ratio", "opt_call_shock", "Realized_Target_EV"
            ]).drop_nulls()
            
            all_regime_frames.append(engineered_df)
            print(f" -> {regime['label']} successfully processed!")
            
        except Exception as e:
            print(f" -> ERROR in feature engineering for {regime['label']}: {e}")
            continue
        
    # ==========================================
    # 4. MASTER DATASET CONCATENATION
    # ==========================================
    if all_regime_frames:
        print("\nConcatenating all market regimes into master dataset...")
        master_df = pl.concat(all_regime_frames)
        
        # Enforce strict chronological sorting to prevent cross-regime look-ahead bias
        master_df = master_df.sort("timestamp_1m")
        
        # Save to the pipeline data folder
        master_df.write_parquet("data/xgboost_training_data.parquet")
        print("SUCCESS: Master dataset saved as 'data/xgboost_training_data.parquet'.")
    else:
        print("\nFAILED: No regime data was successfully processed.")

if __name__ == "__main__":
    build_regime_dataset()
