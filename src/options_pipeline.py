import databento as db
import polars as pl
import pandas as pd
import yfinance as yf
import os
import datetime
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

# ==========================================
# 1. CONFIGURATION 
# ==========================================
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")

EQUITY_TICKER = "MSFT"
DATABENTO_TICKER = f"{EQUITY_TICKER}.OPT"

# yfinance limits 1m data to the last 7 days. Use recent trading days.
START_DATE = "2026-08-24" 
END_DATE = "2026-08-25"

# The UTC timestamps for the Databento query (e.g., 9:30 AM to 4:00 PM EST)
START_TIME_UTC = f"{START_DATE}T13:30:00"
END_TIME_UTC = f"{START_DATE}T20:00:00"

LOCAL_TAPE_FILE = f"{DATABENTO_TICKER}_opra_tape.parquet"
STOP_LOSS_PCT = -0.005  # 0.5% tight trailing stop

# ==========================================
# 2. ZERO-COPY INGESTION (Credit Protected)
# ==========================================
def fetch_institutional_tape() -> pl.DataFrame:
    """Fetches OPRA options tape or loads it from local disk to save API credits."""
    if os.path.exists(LOCAL_TAPE_FILE):
        print(f"Loading tick tape from local cache: {LOCAL_TAPE_FILE}")
        return pl.read_parquet(LOCAL_TAPE_FILE)
        
    print(f"Querying Databento OPRA API for {DATABENTO_TICKER}...")
    client = db.Historical(DATABENTO_API_KEY)
    
    data = client.timeseries.get_range(
        dataset='OPRA.PILLAR', 
        symbols=DATABENTO_TICKER, 
        stype_in='parent',
        schema='trades',
        start=START_TIME_UTC,
        end=END_TIME_UTC,
    )
    
    data.to_parquet(LOCAL_TAPE_FILE)
    print(f"Saved binary stream to {LOCAL_TAPE_FILE}")
    
    return pl.read_parquet(LOCAL_TAPE_FILE)

def fetch_equity_prices() -> pl.DataFrame:
    """Pulls 1-minute underlying stock prices to calculate target MFE."""
    print("Fetching 1-minute underlying equity data...")
    df_pd = yf.download(EQUITY_TICKER, start=START_DATE, end=END_DATE, interval="1m", progress=False)
    
    if isinstance(df_pd.columns, pd.MultiIndex):
        df_pd.columns = df_pd.columns.get_level_values(0)
        
    df_pd = df_pd.reset_index()
    df = pl.from_pandas(df_pd)
    
    # FIX: Cast the yfinance timestamp to nanoseconds (ns) to match Databento's precision
    df = df.with_columns(
        pl.col("Datetime")
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("ns", "UTC"))
        .alias("timestamp_1m")
    )
    
    df = df.rename({
        "High": "stock_high", 
        "Low": "stock_low", 
        "Close": "stock_close"
    })
    
    return df.select(["timestamp_1m", "stock_high", "stock_low", "stock_close"])
# ==========================================
# 3. FEATURE & TARGET ENGINEERING
# ==========================================
def engineer_pipeline(options_df: pl.DataFrame, equity_df: pl.DataFrame) -> pl.DataFrame:
    print("Engineering high-frequency features and path-dependent targets...")
    
    # A. Parse OCC Symbology & Floor timestamps to 1-minute bins
    options_df = options_df.with_columns([
        pl.col("symbol").str.slice(12, 1).alias("option_type"),
        pl.col("ts_event").dt.truncate("1m").alias("timestamp_1m")
    ])
    
    # B. Group tick data into 1-minute signals
    flow_df = options_df.group_by("timestamp_1m").agg([
        pl.when(pl.col("option_type") == "C").then(pl.col("size")).otherwise(0).sum().alias("call_vol"),
        pl.when(pl.col("option_type") == "P").then(pl.col("size")).otherwise(0).sum().alias("put_vol"),
        pl.len().alias("trade_count")
    ]).sort("timestamp_1m")
    
    # C. Calculate Put/Call Ratio and Volume Shocks
    safe_call = pl.when(pl.col("call_vol") == 0).then(1).otherwise(pl.col("call_vol"))
    
    flow_df = flow_df.with_columns([
        (pl.col("put_vol") / safe_call).clip(0.0, 5.0).alias("opt_put_call_ratio"),
        pl.col("call_vol").rolling_mean(window_size=15).alias("call_vol_15m_avg")
    ])
    
    safe_avg = pl.when(pl.col("call_vol_15m_avg") == 0).then(1).otherwise(pl.col("call_vol_15m_avg"))
    flow_df = flow_df.with_columns(
        (pl.col("call_vol") / safe_avg).alias("opt_call_shock")
    )
    
    # D. Merge Options Flow with Underlying Equity Prices
    master_df = flow_df.join(equity_df, on="timestamp_1m", how="inner")
    
    # E. Target Engineering: The 60-Minute MFE / MAE Trailing Stop Logic
    master_df = master_df.reverse().with_columns([
        pl.col("stock_high").rolling_max(window_size=60, min_periods=1).alias("future_60m_high"),
        pl.col("stock_low").rolling_min(window_size=60, min_periods=1).alias("future_60m_low")
    ]).reverse()
    
    master_df = master_df.with_columns([
        ((pl.col("future_60m_high") - pl.col("stock_close")) / pl.col("stock_close")).alias("MFE_pct"),
        ((pl.col("future_60m_low") - pl.col("stock_close")) / pl.col("stock_close")).alias("MAE_pct")
    ])
    
    # The Path-Dependent Target: If the stop loss is hit, cap the target return at the loss
    master_df = master_df.with_columns(
        pl.when(pl.col("MAE_pct") <= STOP_LOSS_PCT)
        .then(STOP_LOSS_PCT)
        .otherwise(pl.col("MFE_pct"))
        .alias("Realized_Target_EV")
    )
    
    # Drop rows without a full 15-minute history (for the rolling average) 
    # and drop the final 60 rows which don't have a full future target window
    master_df = master_df.drop_nulls().slice(15, -60)
    
    return master_df

# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    try:
        # 1. Ingest Data
        options_tape = fetch_institutional_tape()
        equity_data = fetch_equity_prices()
        
        # 2. Engineer Pipeline
        final_dataset = engineer_pipeline(options_tape, equity_data)
        
        # 3. Output
        print("\n==========================================")
        print("PIPELINE GENERATION SUCCESSFUL")
        print("==========================================")
        print(final_dataset.select([
            "timestamp_1m", 
            "opt_put_call_ratio", 
            "opt_call_shock", 
            "Realized_Target_EV"
        ]).head(10))
        
        # Ready for XGBoost Model Training
        final_dataset.write_parquet("xgboost_training_data.parquet")
        print("\nFinal training dataset saved to 'xgboost_training_data.parquet'")
        
    except Exception as e:
        print(f"Pipeline Error: {e}")

if __name__ == "__main__":
    main()
