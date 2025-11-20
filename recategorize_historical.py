"""
Retroactive Zone Interaction Recategorization

This script reads your existing zone_interactions table, applies the new
categorization logic, and updates the liquidity_zones table with corrected
rejection/breakthrough/consolidation counts.

IMPORTANT: This creates a backup of your database before making changes.
"""

import sqlite3
import shutil
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import os


class RetroactiveCategorizer:
    """
    Recategorizes historical zone interactions using the new logic
    """
    
    # Same thresholds as the fixed version
    ZONE_BUFFERS = {
        'NQ': 12.0,
        'GC': 3.0,
        'SI': 0.15,
        'PL': 8.0,
        'HG': 0.03,
    }
    
    BREAKTHROUGH_THRESHOLDS = {
        'NQ': 20.0,
        'GC': 4.0,
        'SI': 0.20,
        'PL': 10.0,
        'HG': 0.04,
    }
    
    MIN_CONSOLIDATION_PRICES = 4
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
        # Create backup
        backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        
        self.conn = sqlite3.connect(db_path)
        
    def load_all_interactions(self) -> Dict[Tuple[str, float], List[dict]]:
        """
        Load all zone interactions from database, grouped by (symbol, zone_price)
        """
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
        
        print(f"\n📊 Loaded {sum(len(v) for v in interactions_by_zone.values())} " 
              f"interactions across {len(interactions_by_zone)} zones")
        
        return interactions_by_zone
    
    def recategorize_interaction(self, interaction: dict) -> str:
        """
        Apply new categorization logic to a single interaction
        
        Uses price_before and price_after to determine the movement through the zone
        """
        symbol = interaction['symbol']
        zone_price = interaction['price']
        price_before = interaction['price_before']
        price_after = interaction['price_after']
        
        if price_before is None or price_after is None:
            # Can't recategorize without price data
            return interaction['old_type']
        
        # Calculate maximum penetration through zone
        # Check both before and after to see which was further from zone
        penetration_before = abs(price_before - zone_price)
        penetration_after = abs(price_after - zone_price)
        max_penetration = max(penetration_before, penetration_after)
        
        # Get thresholds for this symbol
        breakthrough_threshold = self.BREAKTHROUGH_THRESHOLDS.get(symbol, 10.0)
        buffer = self.ZONE_BUFFERS.get(symbol, 5.0)
        consolidation_range = buffer * 0.5
        
        # Apply new categorization logic
        
        # BREAKTHROUGH: Significant movement through zone
        if max_penetration >= breakthrough_threshold:
            return 'breakthrough'
        
        # CONSOLIDATION: Small movement, stayed near zone
        # Note: Hard to detect from single interaction, use price range
        price_range = abs(price_after - price_before)
        if price_range <= consolidation_range:
            # This might be consolidation, but we can't be sure from one interaction
            # Default to old type if it was consolidation, otherwise rejection
            if interaction['old_type'] == 'consolidation':
                return 'consolidation'
            return 'rejection'
        
        # REJECTION: Everything else (touched zone, didn't break through)
        return 'rejection'
    
    def recategorize_zone_visit(self, interactions: List[dict]) -> List[str]:
        """
        BETTER: Look at sequences of interactions to detect patterns
        
        This analyzes multiple interactions at the same zone to better
        identify consolidations
        """
        if not interactions:
            return []
        
        symbol = interactions[0]['symbol']
        zone_price = interactions[0]['price']
        
        breakthrough_threshold = self.BREAKTHROUGH_THRESHOLDS.get(symbol, 10.0)
        buffer = self.ZONE_BUFFERS.get(symbol, 5.0)
        consolidation_range = buffer * 0.5
        
        new_categories = []
        
        # Group interactions into "visits" (sequences close in time)
        visits = []
        current_visit = [interactions[0]]
        
        for i in range(1, len(interactions)):
            prev_time = datetime.fromisoformat(interactions[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(interactions[i]['timestamp'])
            
            # If less than 1 hour apart, consider it the same visit
            if (curr_time - prev_time).total_seconds() < 3600:
                current_visit.append(interactions[i])
            else:
                visits.append(current_visit)
                current_visit = [interactions[i]]
        
        visits.append(current_visit)
        
        # Categorize each visit
        for visit in visits:
            # Collect all prices in this visit
            prices = []
            for inter in visit:
                if inter['price_before']:
                    prices.append(inter['price_before'])
                if inter['price_after']:
                    prices.append(inter['price_after'])
            
            if not prices:
                # No price data, keep old categories
                new_categories.extend([inter['old_type'] for inter in visit])
                continue
            
            # Calculate stats for this visit
            max_price = max(prices)
            min_price = min(prices)
            price_range = max_price - min_price
            max_penetration = max(abs(max_price - zone_price), abs(min_price - zone_price))
            
            # Determine category for this visit
            if max_penetration >= breakthrough_threshold:
                category = 'breakthrough'
            elif len(prices) >= self.MIN_CONSOLIDATION_PRICES and price_range <= consolidation_range:
                category = 'consolidation'
            else:
                category = 'rejection'
            
            # Apply same category to all interactions in this visit
            new_categories.extend([category] * len(visit))
        
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
            
            # Get new categories for all interactions at this zone
            new_categories = self.recategorize_zone_visit(interactions)
            
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
        for cat, count in sorted(old_distribution.items()):
            pct = count / total_old * 100 if total_old > 0 else 0
            print(f"  {cat.capitalize():15s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n📈 New Distribution:")
        total_new = sum(new_distribution.values())
        for cat, count in sorted(new_distribution.items()):
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
        
        print("\n✅ Database updated successfully!")
        
    def _recalculate_zone_stats(self):
        """
        Recalculate rejection_count, breakthrough_count, consolidation_count
        for each zone based on updated interactions
        """
        # Get all zones
        cursor = self.conn.execute('SELECT DISTINCT symbol, price FROM zone_interactions')
        zones = cursor.fetchall()
        
        for symbol, price in zones:
            # Count each type
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
            
            # Determine behavioral tendencies
            tends_to_reject = counts['rejection'] > counts['breakthrough']
            tends_to_melt_through = counts['breakthrough'] > counts['rejection']
            tends_to_consolidate = counts['consolidation'] > max(counts['rejection'], counts['breakthrough'])
            
            # Calculate confidence (consistency score)
            if total_hits > 0:
                max_count = max(counts.values())
                consistency = max_count / total_hits
                hit_factor = min(1.0, total_hits / 10.0)
                confidence = 0.5 + (consistency * hit_factor * 0.5)
            else:
                confidence = 0.5
            
            # Update zone
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
    """
    Main entry point
    """
    import sys
    
    print("="*80)
    print("RETROACTIVE ZONE INTERACTION RECATEGORIZATION")
    print("="*80)
    
    # Get database path
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = input("\nEnter path to liquidity_zones.db (or press Enter for default): ").strip()
        if not db_path:
            db_path = 'liquidity_zones.db'
    
    # Check if file exists
    if not os.path.exists(db_path):
        print(f"\n❌ Error: Database file not found: {db_path}")
        print("\nPlease provide the correct path to your liquidity_zones.db file")
        return
    
    print(f"\n📁 Using database: {db_path}")
    
    # Confirm
    response = input("\n⚠️  This will modify your database (backup will be created). Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Process
    categorizer = RetroactiveCategorizer(db_path)
    categorizer.update_database()
    categorizer.close()
    
    print("\n" + "="*80)
    print("All done! Your historical interactions have been recategorized.")
    print("="*80)


if __name__ == "__main__":
    main()
