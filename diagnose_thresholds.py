"""
Diagnostic Script - Analyze Actual Price Movements

This script examines your zone_interactions to understand what the
actual price movements look like, which helps determine if thresholds
need adjustment.
"""

import sqlite3
from collections import defaultdict
import sys


def analyze_database(db_path):
    """Analyze actual price movements in the database"""
    
    conn = sqlite3.connect(db_path)
    
    print("="*80)
    print("DATABASE ANALYSIS - Understanding Your Price Movements")
    print("="*80)
    
    # Get all interactions with price data
    cursor = conn.execute('''
        SELECT symbol, price, price_before, price_after, interaction_type
        FROM zone_interactions
        WHERE price_before IS NOT NULL AND price_after IS NOT NULL
        ORDER BY symbol, price
    ''')
    
    data_by_symbol = defaultdict(list)
    
    for row in cursor:
        symbol = row[0]
        zone_price = row[1]
        price_before = row[2]
        price_after = row[3]
        old_type = row[4]
        
        # Calculate penetrations
        pen_before = abs(price_before - zone_price)
        pen_after = abs(price_after - zone_price)
        max_pen = max(pen_before, pen_after)
        
        # Calculate movement
        movement = abs(price_after - price_before)
        
        data_by_symbol[symbol].append({
            'zone_price': zone_price,
            'price_before': price_before,
            'price_after': price_after,
            'max_penetration': max_pen,
            'movement': movement,
            'old_type': old_type
        })
    
    # Current thresholds
    THRESHOLDS = {
        'NQ': 20.0,
        'GC': 4.0,
        'SI': 0.20,
        'PL': 10.0,
        'HG': 0.04,
    }
    
    print("\n📊 ANALYSIS BY INSTRUMENT")
    print("="*80)
    
    for symbol in sorted(data_by_symbol.keys()):
        interactions = data_by_symbol[symbol]
        threshold = THRESHOLDS.get(symbol, 10.0)
        
        print(f"\n{symbol} (Breakthrough Threshold: ${threshold})")
        print("-" * 80)
        
        # Calculate penetration statistics
        penetrations = [i['max_penetration'] for i in interactions]
        movements = [i['movement'] for i in interactions]
        
        # Categorize based on current threshold
        would_be_breakthroughs = sum(1 for p in penetrations if p >= threshold)
        would_be_rejections = len(penetrations) - would_be_breakthroughs
        
        print(f"  Total Interactions: {len(interactions)}")
        print(f"\n  Penetration Stats:")
        print(f"    Min:     ${min(penetrations):.4f}")
        print(f"    Max:     ${max(penetrations):.4f}")
        print(f"    Average: ${sum(penetrations)/len(penetrations):.4f}")
        print(f"    Median:  ${sorted(penetrations)[len(penetrations)//2]:.4f}")
        
        print(f"\n  Movement Stats (price_before → price_after):")
        print(f"    Min:     ${min(movements):.4f}")
        print(f"    Max:     ${max(movements):.4f}")
        print(f"    Average: ${sum(movements)/len(movements):.4f}")
        
        print(f"\n  With Current Threshold (${threshold}):")
        print(f"    Would be Breakthroughs: {would_be_breakthroughs} "
              f"({would_be_breakthroughs/len(interactions)*100:.1f}%)")
        print(f"    Would be Rejections:    {would_be_rejections} "
              f"({would_be_rejections/len(interactions)*100:.1f}%)")
        
        # Show distribution of penetrations
        print(f"\n  Penetration Distribution:")
        
        # Create bins
        if threshold > 1:
            bins = [0, threshold*0.25, threshold*0.5, threshold*0.75, threshold, threshold*1.5, threshold*2, float('inf')]
            bin_labels = [
                f'0 - ${threshold*0.25:.2f}',
                f'${threshold*0.25:.2f} - ${threshold*0.5:.2f}',
                f'${threshold*0.5:.2f} - ${threshold*0.75:.2f}',
                f'${threshold*0.75:.2f} - ${threshold:.2f}',
                f'${threshold:.2f} - ${threshold*1.5:.2f}',
                f'${threshold*1.5:.2f} - ${threshold*2:.2f}',
                f'>${threshold*2:.2f}'
            ]
        else:
            bins = [0, threshold*0.25, threshold*0.5, threshold*0.75, threshold, threshold*1.5, float('inf')]
            bin_labels = [
                f'0 - ${threshold*0.25:.4f}',
                f'${threshold*0.25:.4f} - ${threshold*0.5:.4f}',
                f'${threshold*0.5:.4f} - ${threshold*0.75:.4f}',
                f'${threshold*0.75:.4f} - ${threshold:.4f}',
                f'${threshold:.4f} - ${threshold*1.5:.4f}',
                f'>${threshold*1.5:.4f}'
            ]
        
        for i in range(len(bins)-1):
            count = sum(1 for p in penetrations if bins[i] <= p < bins[i+1])
            if count > 0:
                pct = count / len(penetrations) * 100
                bar = '█' * int(pct / 2)
                print(f"    {bin_labels[i]:25s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Recommendation
        print(f"\n  💡 RECOMMENDATION:")
        if would_be_breakthroughs == 0:
            # Find what threshold would give ~30% breakthroughs
            sorted_pens = sorted(penetrations, reverse=True)
            target_idx = int(len(sorted_pens) * 0.3)
            suggested = sorted_pens[target_idx] if target_idx < len(sorted_pens) else threshold / 2
            
            print(f"    ⚠️  ZERO breakthroughs with current threshold (${threshold})")
            print(f"    ⚠️  Maximum penetration seen: ${max(penetrations):.4f}")
            print(f"    ⚠️  This threshold is TOO HIGH for your data")
            print(f"    ✅ Suggested threshold: ${suggested:.4f} (would give ~30% breakthroughs)")
        elif would_be_breakthroughs / len(interactions) < 0.1:
            print(f"    ⚠️  Very few breakthroughs ({would_be_breakthroughs/len(interactions)*100:.1f}%)")
            print(f"    Consider lowering threshold to ${threshold * 0.75:.4f}")
        elif would_be_breakthroughs / len(interactions) > 0.7:
            print(f"    ⚠️  Too many breakthroughs ({would_be_breakthroughs/len(interactions)*100:.1f}%)")
            print(f"    Consider raising threshold to ${threshold * 1.5:.4f}")
        else:
            print(f"    ✅ Threshold looks reasonable")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_interactions = sum(len(v) for v in data_by_symbol.values())
    total_would_breakthrough = 0
    
    for symbol in data_by_symbol:
        threshold = THRESHOLDS.get(symbol, 10.0)
        penetrations = [i['max_penetration'] for i in data_by_symbol[symbol]]
        total_would_breakthrough += sum(1 for p in penetrations if p >= threshold)
    
    print(f"\nTotal Interactions: {total_interactions}")
    print(f"Total Breakthroughs (with current thresholds): {total_would_breakthrough} "
          f"({total_would_breakthrough/total_interactions*100:.1f}%)")
    
    if total_would_breakthrough == 0:
        print("\n⚠️  CRITICAL: Your thresholds are too high!")
        print("   None of your 1924 interactions would qualify as breakthroughs.")
        print("   This explains why you got 0% breakthroughs.")
        print("\n   ACTION REQUIRED:")
        print("   1. Look at the 'RECOMMENDATION' for each instrument above")
        print("   2. Lower the BREAKTHROUGH_THRESHOLDS in liquidity_zones_WORKING.py")
        print("   3. Re-run the recategorization with adjusted thresholds")
    elif total_would_breakthrough / total_interactions < 0.2:
        print("\n⚠️  Your thresholds may be too high")
        print(f"   Only {total_would_breakthrough/total_interactions*100:.1f}% would be breakthroughs")
        print("   Consider lowering thresholds based on recommendations above")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = input("Enter path to liquidity_zones.db (or press Enter for default): ").strip()
        if not db_path:
            db_path = 'liquidity_zones.db'
    
    analyze_database(db_path)
