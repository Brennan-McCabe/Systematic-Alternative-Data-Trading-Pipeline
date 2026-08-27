import polars as pl

def engineer_institutional_flow(parquet_file: str) -> pl.DataFrame:
    print(f"Loading local tick tape: {parquet_file}")
    df = pl.read_parquet(parquet_file)
    
    # 1. OCC Symbology Parsing (Extracting Put vs Call)
    # The 'P' or 'C' is always at index 12 in a standard OCC string
    df = df.with_columns(
        pl.col("symbol").str.slice(12, 1).alias("option_type")
    )
    
    # 2. High-Frequency Binning (1-Minute Intervals)
    # Grouping the tick-tape to simulate intraday momentum
    df = df.sort("ts_event")
    flow_df = df.group_by_dynamic("ts_event", every="1m").agg([
        pl.when(pl.col("option_type") == "C").then(pl.col("size")).otherwise(0).sum().alias("call_vol"),
        pl.when(pl.col("option_type") == "P").then(pl.col("size")).otherwise(0).sum().alias("put_vol"),
        pl.len().alias("trade_count") # Proxy for institutional activity speed
    ])
    
    # 3. Feature Engineering (The Signals)
    # Prevent division-by-zero errors in the math
    safe_call = pl.when(pl.col("call_vol") == 0).then(1).otherwise(pl.col("call_vol"))
    
    flow_df = flow_df.with_columns([
        # Put/Call Ratio (Capped at 5.0 to prevent outlier explosions)
        (pl.col("put_vol") / safe_call).clip(0.0, 5.0).alias("opt_put_call_ratio"),
        
        # 15-Minute Rolling Average for Baseline Volume
        pl.col("call_vol").rolling_mean(window_size=15).alias("call_vol_15m_avg")
    ])
    
    safe_avg = pl.when(pl.col("call_vol_15m_avg") == 0).then(1).otherwise(pl.col("call_vol_15m_avg"))
    
    flow_df = flow_df.with_columns(
        # Call Shock: Current Minute Volume vs 15-Minute Moving Average
        (pl.col("call_vol") / safe_avg).alias("opt_call_shock")
    )
    
    # Drop the first 15 minutes to clear out the rolling-average nulls
    return flow_df.drop_nulls()

if __name__ == "__main__":
    # Runs entirely locally, costing $0
    signals = engineer_institutional_flow("MARA_opra_tape.parquet")
    print(signals.head())
