"""
Retroactive Zone Interaction Recategorization - FIXED VERSION

This version properly distinguishes between:
- BREAKTHROUGH: Price moved significantly through zone (>= threshold)
- REJECTION: Price touched zone but didn't penetrate far, then reversed
- CONSOLIDATION: Multiple interactions with tight price range (rare)

Key fix: Rejection is DEFAULT behavior, consolidation requires many prices in tight range
"""

import sqlite3
import shutil
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import os


class RetroactiveCategorizer:
    """
    Recategorizes historical zone interactions using corrected logic
    """
    
    # CALIBRATED FOR YOUR ACTUAL DATA (based on diagnostic analysis)
    # Your price movements are much smaller than default thresholds assumed
    ZONE_BUFFERS = {
        'NQ': 3.0,     # Reduced from 12.0
        'GC': 1.0,     # Reduced from 3.0
        'SI': 0.05,    # Reduced from 0.15
        'PL': 2.0,     # Reduced from 8.0
        'HG': 0.01,    # Reduced from 0.03
    }
    
    BREAKTHROUGH_THRESHOLDS = {
        'NQ': 0.25,    # Reduced from 20.0 (your max was $6.13)
        'GC': 0.15,    # Reduced from 4.0 (your max was $1.90)
        'SI': 0.005,   # Reduced from 0.20 (your max was $0.0225)
        'PL': 0.15,    # Reduced from 10.0 (your max was $4.10)
        'HG': 0.0003,  # Reduced from 0.04 (your max was $0.0017)
    }
    
    # Need MANY interactions to be consolidation (not just 4 prices)
    MIN_CONSOLIDATION_INTERACTIONS = 8  # At least 8 separate interactions
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
        # Create backup
        backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        
        self.conn = sqlite3.connect(db_path)
        
    def load_all_interactions(self) -> Dict[Tuple[str, float], List[dict]]:
        """Load all zone interactions from database"""
        cursor = self.conn.execute('''
            SELECT symbol, price, interaction_type, timestamp, 
                   price_before, price_after, consolidation_time
            FROM zone_interactions
            ORDER BY symbol, price, timestamp
        ''')
        
        interactions_by_zone = defaultdict(list)
        
        for row in cursor:
            interaction = {
                'symbol': row[0],
                'price': row[1],
                'old_type': row[2],
                'timestamp': row[3],
                'price_before': row[4],
                'price_after': row[5],
                'consolidation_time': row[6]
            }
            
            zone_key = (row[0], row[1])
            interactions_by_zone[zone_key].append(interaction)
        
        total = sum(len(v) for v in interactions_by_zone.values())
        print(f"\n📊 Loaded {total} interactions across {len(interactions_by_zone)} zones")
        
        return interactions_by_zone
    
    def recategorize_zone_interactions(self, interactions: List[dict]) -> List[str]:
        """
        CORRECTED LOGIC for recategorization
        
        For each interaction:
        1. Calculate max_penetration from price_before and price_after
        2. If >= breakthrough_threshold → BREAKTHROUGH
        3. Otherwise → REJECTION (default)
        
        Consolidation only detected across entire zone history if:
        - Many interactions (8+) 
        - AND all have small penetrations
        - AND tight overall price range
        """
        if not interactions:
            return []
        
        symbol = interactions[0]['symbol']
        zone_price = interactions[0]['price']
        
        breakthrough_threshold = self.BREAKTHROUGH_THRESHOLDS.get(symbol, 10.0)
        buffer = self.ZONE_BUFFERS.get(symbol, 5.0)
        
        new_categories = []
        all_prices = []  # Track all prices for consolidation check
        
        # First pass: categorize each interaction individually
        for inter in interactions:
            price_before = inter['price_before']
            price_after = inter['price_after']
            
            if price_before is None or price_after is None:
                # No price data, keep old category
                new_categories.append(inter['old_type'])
                continue
            
            # Calculate maximum penetration
            penetration_before = abs(price_before - zone_price)
            penetration_after = abs(price_after - zone_price)
            max_penetration = max(penetration_before, penetration_after)
            
            # Track prices for consolidation detection
            all_prices.extend([price_before, price_after])
            
            # CORRECTED CATEGORIZATION LOGIC
            if max_penetration >= breakthrough_threshold:
                # Price moved significantly through zone
                category = 'breakthrough'
            else:
                # Default to rejection (touched zone, didn't break through)
                category = 'rejection'
            
            new_categories.append(category)
        
        # Second pass: Check if ENTIRE zone history looks like consolidation
        # This is RARE - only happens when many interactions all have tiny movements
        if len(interactions) >= self.MIN_CONSOLIDATION_INTERACTIONS and all_prices:
            price_range = max(all_prices) - min(all_prices)
            avg_penetration = sum(abs(p - zone_price) for p in all_prices) / len(all_prices)
            
            # Very tight range AND small average penetration = consolidation zone
            consolidation_threshold = buffer * 0.3  # Even tighter than before
            
            if (price_range <= consolidation_threshold and 
                avg_penetration <= consolidation_threshold and
                all(cat == 'rejection' for cat in new_categories)):
                # All interactions were rejections with tiny movements
                # This is actually a consolidation zone
                print(f"  💫 Detected consolidation zone: {symbol} ${zone_price} "
                      f"({len(interactions)} interactions, range ${price_range:.2f})")
                new_categories = ['consolidation'] * len(interactions)
        
        return new_categories
    
    def update_database(self):
        """
        Main function: Recategorize everything and update database
        """
        print("\n🔄 Starting recategorization process...")
        
        # Load all interactions
        interactions_by_zone = self.load_all_interactions()
        
        # Track changes
        changes = {
            'rejection': 0,
            'breakthrough': 0,
            'consolidation': 0,
            'unchanged': 0
        }
        
        old_distribution = defaultdict(int)
        new_distribution = defaultdict(int)
        
        # Process each zone
        for zone_key, interactions in interactions_by_zone.items():
            symbol, zone_price = zone_key
            
            # Get new categories
            new_categories = self.recategorize_zone_interactions(interactions)
            
            # Update each interaction
            for interaction, new_category in zip(interactions, new_categories):
                old_category = interaction['old_type']
                
                old_distribution[old_category] += 1
                new_distribution[new_category] += 1
                
                if old_category != new_category:
                    changes[new_category] += 1
                    
                    # Update in database
                    self.conn.execute('''
                        UPDATE zone_interactions
                        SET interaction_type = ?
                        WHERE symbol = ? AND price = ? AND timestamp = ?
                    ''', (new_category, symbol, zone_price, interaction['timestamp']))
                else:
                    changes['unchanged'] += 1
        
        # Recalculate zone statistics
        print("\n📊 Recalculating zone statistics...")
        self._recalculate_zone_stats()
        
        # Commit changes
        self.conn.commit()
        
        # Print summary
        print("\n" + "="*80)
        print("RECATEGORIZATION COMPLETE")
        print("="*80)
        
        print("\n📈 Old Distribution:")
        total_old = sum(old_distribution.values())
        for cat in ['rejection', 'breakthrough', 'consolidation']:
            count = old_distribution.get(cat, 0)
            pct = count / total_old * 100 if total_old > 0 else 0
            print(f"  {cat.capitalize():15s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n📈 New Distribution:")
        total_new = sum(new_distribution.values())
        for cat in ['rejection', 'breakthrough', 'consolidation']:
            count = new_distribution.get(cat, 0)
            pct = count / total_new * 100 if total_new > 0 else 0
            print(f"  {cat.capitalize():15s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n🔄 Changes Made:")
        print(f"  Changed to rejection:     {changes['rejection']:4d}")
        print(f"  Changed to breakthrough:  {changes['breakthrough']:4d}")
        print(f"  Changed to consolidation: {changes['consolidation']:4d}")
        print(f"  Unchanged:                {changes['unchanged']:4d}")
        
        total_changed = sum(v for k, v in changes.items() if k != 'unchanged')
        total = total_changed + changes['unchanged']
        pct_changed = total_changed / total * 100 if total > 0 else 0
        print(f"\n  Total changed: {total_changed}/{total} ({pct_changed:.1f}%)")
        
        # Health check
        rej_pct = new_distribution['rejection'] / total_new * 100
        bt_pct = new_distribution['breakthrough'] / total_new * 100
        cons_pct = new_distribution['consolidation'] / total_new * 100
        
        print("\n" + "="*80)
        print("HEALTH CHECK")
        print("="*80)
        
        if bt_pct == 0:
            print("⚠️  WARNING: 0% breakthroughs detected!")
            print("   This suggests thresholds may be too high for your data.")
            print("   Try lowering BREAKTHROUGH_THRESHOLDS in the working tracker.")
        elif bt_pct < 10:
            print(f"⚠️  Low breakthrough rate ({bt_pct:.1f}%)")
            print("   Most zones are holding. This might be correct for ranging markets.")
        elif bt_pct > 70:
            print(f"⚠️  High breakthrough rate ({bt_pct:.1f}%)")
            print("   Most zones are breaking. This might be correct for trending markets.")
        else:
            print(f"✅ Healthy breakthrough rate ({bt_pct:.1f}%)")
        
        if cons_pct > 20:
            print(f"⚠️  High consolidation rate ({cons_pct:.1f}%)")
            print("   This is unusual. Check MIN_CONSOLIDATION_INTERACTIONS setting.")
        elif cons_pct > 0:
            print(f"✅ Normal consolidation rate ({cons_pct:.1f}%)")
        
        print("\n✅ Database updated successfully!")
        
    def _recalculate_zone_stats(self):
        """Recalculate zone statistics"""
        cursor = self.conn.execute('SELECT DISTINCT symbol, price FROM zone_interactions')
        zones = cursor.fetchall()
        
        for symbol, price in zones:
            cursor = self.conn.execute('''
                SELECT interaction_type, COUNT(*) 
                FROM zone_interactions 
                WHERE symbol = ? AND price = ?
                GROUP BY interaction_type
            ''', (symbol, price))
            
            counts = {'rejection': 0, 'breakthrough': 0, 'consolidation': 0}
            for interaction_type, count in cursor:
                counts[interaction_type] = count
            
            total_hits = sum(counts.values())
            
            tends_to_reject = counts['rejection'] > counts['breakthrough']
            tends_to_melt_through = counts['breakthrough'] > counts['rejection']
            tends_to_consolidate = counts['consolidation'] > max(counts['rejection'], counts['breakthrough'])
            
            if total_hits > 0:
                max_count = max(counts.values())
                consistency = max_count / total_hits
                hit_factor = min(1.0, total_hits / 10.0)
                confidence = 0.5 + (consistency * hit_factor * 0.5)
            else:
                confidence = 0.5
            
            self.conn.execute('''
                UPDATE liquidity_zones
                SET rejection_count = ?,
                    breakthrough_count = ?,
                    consolidation_count = ?,
                    hit_count = ?,
                    tends_to_reject_first = ?,
                    tends_to_consolidate = ?,
                    tends_to_melt_through = ?,
                    confidence = ?
                WHERE symbol = ? AND price = ?
            ''', (
                counts['rejection'],
                counts['breakthrough'],
                counts['consolidation'],
                total_hits,
                int(tends_to_reject),
                int(tends_to_consolidate),
                int(tends_to_melt_through),
                confidence,
                symbol,
                price
            ))
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Main entry point"""
    import sys
    
    print("="*80)
    print("RETROACTIVE ZONE INTERACTION RECATEGORIZATION - FIXED")
    print("="*80)
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = input("\nEnter path to liquidity_zones.db (or press Enter for default): ").strip()
        if not db_path:
            db_path = 'liquidity_zones.db'
    
    if not os.path.exists(db_path):
        print(f"\n❌ Error: Database file not found: {db_path}")
        return
    
    print(f"\n📁 Using database: {db_path}")
    
    response = input("\n⚠️  This will modify your database (backup will be created). Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    categorizer = RetroactiveCategorizer(db_path)
    categorizer.update_database()
    categorizer.close()
    
    print("\n" + "="*80)
    print("All done! Your historical interactions have been recategorized.")
    print("="*80)


if __name__ == "__main__":
    main()
