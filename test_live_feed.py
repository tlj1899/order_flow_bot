"""
Test script to see what's actually coming from Databento
"""
import databento as db
import asyncio

async def test_feed():
    try:
        import config
        api_key = config.DATABENTO_API_KEY
    except:
        print("Error: Could not load config.py")
        return
    
    print("Connecting to Databento live feed...")
    
    client = db.Live(key=api_key)
    
    # Subscribe to one symbol for testing
    client.subscribe(
        dataset='GLBX.MDP3',
        schema='mbp-1',
        symbols=['GCZ5']
    )
    
    client.subscribe(
        dataset='GLBX.MDP3',
        schema='trades',
        symbols=['GCZ5']
    )
    
    print("✓ Subscribed to GCZ5")
    print("Waiting for data (will show first 10 records)...\n")
    
    count = 0
    async for record in client:
        count += 1
        
        print(f"━━━ Record #{count} ━━━")
        print(f"  Type: {type(record).__name__}")
        
        # Check for symbol fields
        if hasattr(record, 'raw_symbol'):
            print(f"  ✓ raw_symbol: '{record.raw_symbol}'")
        else:
            print(f"  ✗ No raw_symbol attribute")
            
        if hasattr(record, 'symbol'):
            print(f"  ✓ symbol: '{record.symbol}'")
        else:
            print(f"  ✗ No symbol attribute")
        
        if hasattr(record, 'instrument_id'):
            print(f"  ✓ instrument_id: {record.instrument_id}")
        
        # Check if it's MBP or Trade
        if hasattr(record, 'bid_px') and hasattr(record, 'ask_px'):
            print(f"  → MBP Record: Bid={record.bid_px}, Ask={record.ask_px}")
        elif hasattr(record, 'price'):
            print(f"  → Trade Record: Price={record.price}, Size={record.size}")
        
        print()
        
        if count >= 10:
            print("Stopping after 10 records...")
            break
    
    await client.close()
    print("\n✓ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_feed())
