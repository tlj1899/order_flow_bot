"""
Inspect MBP1Msg records to see what attributes are available
"""
import databento as db
import asyncio

async def inspect_mbp():
    try:
        import config
        api_key = config.DATABENTO_API_KEY
    except:
        print("Error: Could not load config.py")
        return
    
    print("Connecting to Databento...")
    
    client = db.Live(key=api_key)
    
    client.subscribe(
        dataset='GLBX.MDP3',
        schema='mbp-1',
        symbols=['GCZ5']
    )
    
    print("✓ Subscribed, waiting for MBP1Msg records...\n")
    
    count = 0
    async for record in client:
        record_type = type(record).__name__
        
        if record_type == 'MBP1Msg':
            count += 1
            print(f"━━━ MBP1Msg #{count} ━━━")
            
            # Show all non-private attributes
            attrs = [a for a in dir(record) if not a.startswith('_')]
            print(f"All attributes: {attrs}\n")
            
            # Show key values
            for attr in ['instrument_id', 'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 
                        'bid_ct', 'ask_ct', 'levels']:
                if hasattr(record, attr):
                    val = getattr(record, attr)
                    print(f"  {attr}: {val}")
            
            print()
            
            if count >= 3:
                print("Got 3 MBP records, stopping...")
                break
    
    print("\n✓ Done!")

if __name__ == "__main__":
    asyncio.run(inspect_mbp())
