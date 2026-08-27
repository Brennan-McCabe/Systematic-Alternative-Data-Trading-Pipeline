import databento as db
import polars as pl
import os
from dotenv import load_dotenv

load_dotenv()
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")

def fetch_institutional_options_tape(parent_ticker: str, start_time: str, end_time: str) -> pl.DataFrame:
    """
    Queries Databento for tick-level options trades (OPRA).
    Converts the zero-copy DBN format directly into a Polars DataFrame.
    """
    # Initialize the Databento Historical Client
    client = db.Historical(DATABENTO_API_KEY)
    
    print(f"Requesting tick-level OPRA flow for {parent_ticker}...")
    
    # The get_range method handles the DBN stream
    # stype_in='parent' tells DB to fetch all options contracts for the underlying stock
    data = client.timeseries.get_range(
        dataset='OPRA.PILLAR', 
        symbols=parent_ticker, 
        stype_in='parent',
        schema='trades', # Pulling the execution tape. Can swap to 'bbo' for bid/ask data
        start=start_time,
        end=end_time,
    )
    
    # Directly convert the binary stream to Polars (Zero-Copy)
    # This avoids standard Pandas iteration bottlenecks
    df = data.to_polars()
    
    return df

def main():
    # SURGICAL QUERY: 1 single hour for 1 single ticker to protect free credits
    ticker = "MARA"
    start = "2023-10-04T13:30:00" # 9:30 AM EST (Market Open) in UTC
    end = "2023-10-04T14:30:00"   # 10:30 AM EST in UTC
    
    try:
        options_df = fetch_institutional_options_tape(ticker, start, end)
        
        print("\n--- OPRA Ingestion Complete ---")
        print(f"Total options trades captured in 1 hour: {options_df.height}")
        
        # Display the first few nanosecond-stamped trades
        print(options_df.head())
        
        # Save to a local Parquet file for model training (faster than CSV)
        options_df.write_parquet(f"{ticker}_opra_tape.parquet")
        print(f"\nSaved to {ticker}_opra_tape.parquet for XGBoost training.")
        
    except Exception as e:
        print(f"Databento Query Failed: {e}")

if __name__ == "__main__":
    main()
