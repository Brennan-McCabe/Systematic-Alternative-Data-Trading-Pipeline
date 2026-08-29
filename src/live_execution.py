import os
import asyncio
import databento as db
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================
load_dotenv()
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Global counter to prevent terminal freeze from OPRA firehose
opra_tick_count = 0

# ==========================================
# 1. ALPACA EQUITY STREAM (1-Minute Bars)
# ==========================================
async def alpaca_handler(bar):
    # This fires exactly once per minute when Alpaca closes a candle
    print(f"[ALPACA] MSFT 1m Bar | Close: ${bar.close:.2f} | Vol: {bar.volume}")

async def start_alpaca_stream():
    print(" -> [ALPACA] Initializing Equity WebSocket...")
    try:
        stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        stream.subscribe_bars(alpaca_handler, "MSFT")
        
        # Run stream in a background thread to prevent blocking the async loop
        await asyncio.to_thread(stream.run)
    except Exception as e:
        print(f" -> [ALPACA] Stream Error: {e}")

# ==========================================
# 2. DATABENTO OPRA STREAM (Tick Data)
# ==========================================
def databento_worker():
    global opra_tick_count
    print(" -> [DATABENTO] Initializing Live OPRA Tape...")
    try:
        # NOTE: This requires the $199/mo standard live plan to authenticate properly
        client = db.Live(DATABENTO_API_KEY)
        client.subscribe(
            dataset="OPRA.PILLAR",
            symbols="MSFT.OPT",
            stype_in="parent",
            schema="trades"
        )
        client.start()
        
        for record in client:
            if isinstance(record, db.TradeMsg):
                opra_tick_count += 1
                
                # Console Throttle: Print a heartbeat every 5,000 trades
                if opra_tick_count % 5000 == 0:
                    print(f"[DATABENTO] OPRA Heartbeat | Processed {opra_tick_count} options trades...")
                    
    except Exception as e:
        print(f" -> [DATABENTO] Stream Error: {e}")

async def start_databento_stream():
    # Databento's standard client is blocking, so we isolate it in a thread
    await asyncio.to_thread(databento_worker)

# ==========================================
# MAIN EVENT LOOP
# ==========================================
async def main():
    print("========================================")
    print(" STARTING LIVE INGESTION ENGINE")
    print("========================================")
    print("Listening for real-time market data... (Awaiting market open)")
    
    # Launch both WebSocket listeners concurrently
    await asyncio.gather(
        start_alpaca_stream(),
        start_databento_stream()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down live ingestion engine...")
