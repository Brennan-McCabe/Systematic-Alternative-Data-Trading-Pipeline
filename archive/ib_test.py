from ib_async import IB

def connect_to_gateway():
    ib = IB()
    
    try:
        # Connect to TWS Paper Trading on port 4002
        ib.connect('127.0.0.1', 4002, clientId=1)
        print("Successfully connected to TWS Gateway!")
        
        summary = ib.accountSummary()
        if summary:
            print(f"Connected Account ID: {summary[0].account}")
            
    except Exception as e:
        print(f"Connection failed: {e}")
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected cleanly.")

if __name__ == "__main__":
    connect_to_gateway()
