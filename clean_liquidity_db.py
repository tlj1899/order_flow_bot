#!/usr/bin/env python3
"""
Clean the liquidity zones database of incorrectly formatted prices
"""

import sqlite3

db_path = 'liquidity_zones.db'

print("Cleaning liquidity zones database...")

with sqlite3.connect(db_path) as conn:
    # Check current zones
    cursor = conn.execute('SELECT COUNT(*) FROM liquidity_zones')
    count_before = cursor.fetchone()[0]
    print(f"Zones before cleanup: {count_before}")
    
    # Delete zones with crazy prices (anything > 1 million is definitely wrong)
    conn.execute('DELETE FROM liquidity_zones WHERE price > 1000000')
    conn.commit()
    
    cursor = conn.execute('SELECT COUNT(*) FROM liquidity_zones')
    count_after = cursor.fetchone()[0]
    print(f"Zones after cleanup: {count_after}")
    print(f"Removed: {count_before - count_after} bad zones")
    
    # Show remaining zones
    cursor = conn.execute('SELECT symbol, COUNT(*) FROM liquidity_zones GROUP BY symbol')
    print("\nRemaining zones by symbol:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]} zones")
    
    # Clean interactions table too
    cursor = conn.execute('SELECT COUNT(*) FROM zone_interactions')
    count_before = cursor.fetchone()[0]
    print(f"\nInteractions before cleanup: {count_before}")
    
    conn.execute('DELETE FROM zone_interactions WHERE price > 1000000')
    conn.commit()
    
    cursor = conn.execute('SELECT COUNT(*) FROM zone_interactions')
    count_after = cursor.fetchone()[0]
    print(f"Interactions after cleanup: {count_after}")
    print(f"Removed: {count_before - count_after} bad interactions")

print("\n✓ Database cleaned!")
print("You can now restart the trading system with correct price formatting.")
