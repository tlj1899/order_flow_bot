"""
Check the actual timezone of trades
"""
import json
from datetime import datetime

# Load trades
with open('closed_trades.json', 'r') as f:
    trades = json.load(f)

print("="*70)
print("TRADE TIMESTAMP ANALYSIS")
print("="*70)
print()

for i, trade in enumerate(trades, 1):
    ts_str = trade['timestamp']
    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    
    print(f"Trade {i}:")
    print(f"  Symbol: {trade['symbol']}")
    print(f"  Direction: {trade['direction']}")
    print(f"  PnL: ${trade['pnl']:.2f}")
    print(f"  Timestamp (from file): {ts_str}")
    print(f"  Hour (from timestamp): {ts.hour}")
    
    # If timezone aware
    if ts.tzinfo:
        print(f"  Timezone: {ts.tzinfo}")
    else:
        print(f"  Timezone: NAIVE (no timezone info)")
    
    print()

print("="*70)
print("EXPLANATION:")
print("="*70)
print()
print("You said trades were at 18:00 EST (your local time)")
print("18:00 EST = 23:00 UTC (EST is UTC-5)")
print()
print("If hour shows 18, timestamps were stored as LOCAL time")
print("If hour shows 23, timestamps were stored as UTC (correct)")
