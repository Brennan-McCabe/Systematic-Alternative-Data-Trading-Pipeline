import polars as pl
import numpy as np

def engineer_mfe_targets(df: pl.DataFrame, stop_loss_pct: float = -0.005) -> pl.DataFrame:
    """
    Calculates the 60-minute forward MFE and MAE, then applies a strict 
    trailing stop logic to create the final XGBoost target variable.
    """
    # 1. The Forward-Looking Window Trick
    # To look exactly 60 rows into the future efficiently, we reverse the DataFrame, 
    # apply a standard backward rolling max/min, and then reverse it back.
    df = df.reverse().with_columns([
        pl.col("stock_high").rolling_max(window_size=60, min_periods=1).alias("future_60m_high"),
        pl.col("stock_low").rolling_min(window_size=60, min_periods=1).alias("future_60m_low")
    ]).reverse()
    
    # 2. Calculate the Excursions (Returns)
    df = df.with_columns([
        ((pl.col("future_60m_high") - pl.col("stock_close")) / pl.col("stock_close")).alias("MFE_pct"),
        ((pl.col("future_60m_low") - pl.col("stock_close")) / pl.col("stock_close")).alias("MAE_pct")
    ])
    
    # 3. The Path-Dependent Trailing Stop Filter
    # If the MAE breaches our tight stop loss, we assume the trade was killed.
    # Therefore, the realized target for XGBoost becomes the stop loss, NOT the MFE.
    df = df.with_columns(
        pl.when(pl.col("MAE_pct") <= stop_loss_pct)
        .then(stop_loss_pct)
        .otherwise(pl.col("MFE_pct"))
        .alias("Realized_Target_EV")
    )
    
    # Drop rows at the very end of the dataset that don't have a full 60 minutes of future data
    return df.drop_nulls()
