"""
Liquidity Zone Tracker - SIMPLIFIED AND FIXED

The problem with previous versions: trying to detect and categorize in one pass.

NEW APPROACH:
1. Track when price ENTERS a zone proximity
2. Wait for price to either:
   a) Cross through zone and move significantly = BREAKTHROUGH
   b) Cross through zone but reverse = REJECTION  
   c) Stay near zone without decisive move = CONSOLIDATION
3. Only record ONCE per zone visit
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidityZone:
    """Represents a tracked liquidity zone"""
    symbol: str
    price: float
    zone_type: str
    confidence: float
    hit_count: int
    rejection_count: int
    breakthrough_count: int
    consolidation_count: int
    last_interaction: datetime
    created_at: datetime
    tends_to_reject_first: bool = False
    tends_to_consolidate: bool = False
    tends_to_melt_through: bool = False


@dataclass
class ZoneVisit:
    """Tracks an ongoing visit to a zone"""
    zone_price: float
    entry_price: float
    entry_time: datetime
    entry_side: str  # 'from_below' or 'from_above'
    max_penetration: float  # Maximum distance moved through zone
    prices_seen: List[float]  # All prices during this visit
    
    
class LiquidityZoneTracker:
    """
    Tracks liquidity zones with proper state-based categorization
    """
    
    ROUND_NUMBER_LEVELS = {
        'NQ': [25, 50, 100],
        'GC': [5, 10],
        'SI': [0.25, 0.50],
        'PL': [10, 25],
        'HG': [0.05, 0.10],
    }
    
    ZONE_BUFFERS = {
        'NQ': 3.0,    # Calibrated based on your data (was 12.0)
        'GC': 1.0,    # Calibrated (was 3.0)
        'SI': 0.05,   # Calibrated (was 0.15)
        'PL': 2.0,    # Calibrated (was 8.0)
        'HG': 0.01,   # Calibrated (was 0.03)
    }
    
    BREAKTHROUGH_THRESHOLDS = {
        'NQ': 0.25,   # Calibrated - your max was $6.13 (was 20.0)
        'GC': 0.15,   # Calibrated - your max was $1.90 (was 4.0)
        'SI': 0.005,  # Calibrated - your max was $0.0225 (was 0.20)
        'PL': 0.15,   # Calibrated - your max was $4.10 (was 10.0)
        'HG': 0.0003, # Calibrated - your max was $0.0017 (was 0.04)
    }
    
    MIN_CONSOLIDATION_PRICES = 4  # Need to see 4+ prices near zone to call it consolidation
    
    def __init__(self, db_path: str = 'liquidity_zones.db'):
        self.db_path = db_path
        self.zones: Dict[str, List[LiquidityZone]] = defaultdict(list)
        self._init_database()
        self._load_zones()
        
        # Track ongoing zone visits
        self.active_visits: Dict[Tuple[str, float], ZoneVisit] = {}
        
        # Price history for context
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
    def _init_database(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS liquidity_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    zone_type TEXT,
                    confidence REAL,
                    hit_count INTEGER,
                    rejection_count INTEGER,
                    breakthrough_count INTEGER,
                    consolidation_count INTEGER,
                    last_interaction TEXT,
                    created_at TEXT,
                    tends_to_reject_first INTEGER,
                    tends_to_consolidate INTEGER,
                    tends_to_melt_through INTEGER,
                    UNIQUE(symbol, price)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS zone_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    interaction_type TEXT,
                    timestamp TEXT,
                    price_before REAL,
                    price_after REAL,
                    consolidation_time INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
    
    def _load_zones(self):
        """Load zones from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM liquidity_zones')
            
            for row in cursor:
                zone = LiquidityZone(
                    symbol=row[1],
                    price=row[2],
                    zone_type=row[3],
                    confidence=row[4],
                    hit_count=row[5],
                    rejection_count=row[6],
                    breakthrough_count=row[7],
                    consolidation_count=row[8],
                    last_interaction=datetime.fromisoformat(row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    tends_to_reject_first=bool(row[11]),
                    tends_to_consolidate=bool(row[12]),
                    tends_to_melt_through=bool(row[13])
                )
                self.zones[zone.symbol].append(zone)
        
        logger.info(f"Loaded {sum(len(z) for z in self.zones.values())} liquidity zones")
    
    def _save_zone(self, zone: LiquidityZone):
        """Save zone to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO liquidity_zones 
                (symbol, price, zone_type, confidence, hit_count, rejection_count, 
                 breakthrough_count, consolidation_count, last_interaction, created_at,
                 tends_to_reject_first, tends_to_consolidate, tends_to_melt_through)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                zone.symbol, zone.price, zone.zone_type, zone.confidence,
                zone.hit_count, zone.rejection_count, zone.breakthrough_count,
                zone.consolidation_count,
                zone.last_interaction.isoformat(), zone.created_at.isoformat(),
                int(zone.tends_to_reject_first), int(zone.tends_to_consolidate),
                int(zone.tends_to_melt_through)
            ))
            conn.commit()
    
    def _log_interaction(self, symbol: str, price: float, interaction_type: str,
                        price_before: Optional[float] = None, price_after: Optional[float] = None):
        """Log interaction to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO zone_interactions
                (symbol, price, interaction_type, timestamp, price_before, price_after)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, price, interaction_type, datetime.now(timezone.utc).isoformat(),
                  price_before, price_after))
            conn.commit()
    
    def detect_zones(self, symbol: str, current_price: float) -> List[LiquidityZone]:
        """Detect and create zones if they don't exist"""
        if symbol not in self.ROUND_NUMBER_LEVELS:
            return []
        
        zones_found = []
        for level in self.ROUND_NUMBER_LEVELS[symbol]:
            zone_price = round(current_price / level) * level
            
            existing_zone = next(
                (z for z in self.zones[symbol] if abs(z.price - zone_price) < 0.01),
                None
            )
            
            if existing_zone:
                zones_found.append(existing_zone)
            else:
                new_zone = LiquidityZone(
                    symbol=symbol,
                    price=zone_price,
                    zone_type='round_number',
                    confidence=0.7,
                    hit_count=0,
                    rejection_count=0,
                    breakthrough_count=0,
                    consolidation_count=0,
                    last_interaction=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc)
                )
                self.zones[symbol].append(new_zone)
                self._save_zone(new_zone)
                zones_found.append(new_zone)
                logger.info(f"Created zone: {symbol} @ {zone_price}")
        
        return zones_found
    
    def record_price_interaction(self, symbol: str, current_price: float) -> Optional[str]:
        """
        COMPLETELY REDESIGNED INTERACTION DETECTION
        
        State machine approach:
        1. Detect when price enters zone proximity
        2. Track the visit until price leaves
        3. Categorize based on complete visit behavior
        """
        # Add to history
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) < 3:
            return None
        
        buffer = self.ZONE_BUFFERS.get(symbol, 5.0)
        breakthrough_threshold = self.BREAKTHROUGH_THRESHOLDS.get(symbol, 10.0)
        
        # Get previous price
        prev_price = list(self.price_history[symbol])[-2]
        
        # Check each zone
        for zone in self.zones[symbol]:
            zone_key = (symbol, zone.price)
            
            # Is price currently near this zone?
            near_zone_now = abs(current_price - zone.price) <= buffer
            was_near_zone = abs(prev_price - zone.price) <= buffer
            
            # STATE 1: Entering zone proximity
            if near_zone_now and not was_near_zone:
                # Starting a new visit
                entry_side = 'from_below' if prev_price < zone.price else 'from_above'
                
                self.active_visits[zone_key] = ZoneVisit(
                    zone_price=zone.price,
                    entry_price=current_price,
                    entry_time=datetime.now(timezone.utc),
                    entry_side=entry_side,
                    max_penetration=abs(current_price - zone.price),
                    prices_seen=[current_price]
                )
                logger.debug(f"Started visit to {symbol} {zone.price} from {entry_side}")
            
            # STATE 2: Continuing visit
            elif near_zone_now and zone_key in self.active_visits:
                visit = self.active_visits[zone_key]
                visit.prices_seen.append(current_price)
                
                # Track maximum penetration through zone
                penetration = abs(current_price - zone.price)
                if penetration > visit.max_penetration:
                    visit.max_penetration = penetration
                
                # Check if we've seen enough prices to detect consolidation WHILE STILL IN ZONE
                # This catches when price consolidates without leaving
                if len(visit.prices_seen) >= self.MIN_CONSOLIDATION_PRICES * 2:  # 2x minimum
                    price_range = max(visit.prices_seen) - min(visit.prices_seen)
                    consolidation_range = buffer * 0.5
                    
                    if price_range <= consolidation_range and visit.max_penetration < breakthrough_threshold:
                        # This is a consolidation!
                        interaction_type = 'consolidation'
                        zone.consolidation_count += 1
                        zone.hit_count += 1
                        zone.tends_to_consolidate = zone.consolidation_count > max(zone.rejection_count, zone.breakthrough_count)
                        zone.last_interaction = datetime.now(timezone.utc)
                        self._update_confidence(zone)
                        self._save_zone(zone)
                        self._log_interaction(symbol, zone.price, interaction_type, visit.entry_price, current_price)
                        
                        # Clean up visit
                        del self.active_visits[zone_key]
                        
                        logger.info(f"💫 CONSOLIDATION at {symbol} {zone.price}: "
                                   f"{len(visit.prices_seen)} prices, range ${price_range:.2f} (<= ${consolidation_range:.2f})")
                        return interaction_type
            
            # STATE 3: Exiting zone - but only if we've truly left (beyond 2x buffer)
            # This prevents premature categorization
            elif not near_zone_now and zone_key in self.active_visits:
                visit = self.active_visits[zone_key]
                
                # Check if we've REALLY left (moved beyond extended zone)
                # Use 2x buffer to ensure we capture full breakthrough moves
                extended_buffer = buffer * 2
                truly_left = abs(current_price - zone.price) > extended_buffer
                
                if truly_left:
                    # Add final price
                    visit.prices_seen.append(current_price)
                    
                    # Now categorize based on complete visit
                    interaction_type = self._categorize_visit(visit, zone, symbol, breakthrough_threshold)
                    
                    # Update zone
                    if interaction_type == 'rejection':
                        zone.rejection_count += 1
                        zone.tends_to_reject_first = zone.rejection_count > zone.breakthrough_count
                    elif interaction_type == 'breakthrough':
                        zone.breakthrough_count += 1
                        zone.tends_to_melt_through = zone.breakthrough_count > zone.rejection_count
                    elif interaction_type == 'consolidation':
                        zone.consolidation_count += 1
                        zone.tends_to_consolidate = zone.consolidation_count > max(zone.rejection_count, zone.breakthrough_count)
                    
                    zone.hit_count += 1
                    zone.last_interaction = datetime.now(timezone.utc)
                    self._update_confidence(zone)
                    self._save_zone(zone)
                    self._log_interaction(symbol, zone.price, interaction_type, visit.entry_price, current_price)
                    
                    # Clean up visit
                    del self.active_visits[zone_key]
                    
                    return interaction_type
                else:
                    # Not far enough away yet, keep tracking
                    visit.prices_seen.append(current_price)
                    penetration = abs(current_price - zone.price)
                    if penetration > visit.max_penetration:
                        visit.max_penetration = penetration
        
        return None
    
    def _categorize_visit(self, visit: ZoneVisit, zone: LiquidityZone, 
                         symbol: str, breakthrough_threshold: float) -> str:
        """
        Categorize a completed zone visit
        
        BREAKTHROUGH: Max penetration >= threshold
        CONSOLIDATION: Many prices seen (4+) with small range
        REJECTION: Touched zone, reversed back
        """
        # Check if price broke through significantly
        if visit.max_penetration >= breakthrough_threshold:
            logger.info(f"🚀 BREAKTHROUGH at {symbol} {zone.price}: "
                       f"Penetrated ${visit.max_penetration:.2f} ({visit.entry_side})")
            return 'breakthrough'
        
        # Check if consolidated (stayed near zone for multiple prices)
        if len(visit.prices_seen) >= self.MIN_CONSOLIDATION_PRICES:
            price_range = max(visit.prices_seen) - min(visit.prices_seen)
            # Use tighter range for consolidation (half the buffer)
            consolidation_range = self.ZONE_BUFFERS.get(symbol, 5.0) * 0.5
            if price_range <= consolidation_range:
                logger.info(f"💫 CONSOLIDATION at {symbol} {zone.price}: "
                           f"{len(visit.prices_seen)} prices, range ${price_range:.2f} (<= ${consolidation_range:.2f})")
                return 'consolidation'
        
        # Otherwise it's a rejection
        logger.info(f"🛑 REJECTION at {symbol} {zone.price}: "
                   f"Max penetration ${visit.max_penetration:.2f}, then reversed ({visit.entry_side})")
        return 'rejection'
    
    def _update_confidence(self, zone: LiquidityZone):
        """Update zone confidence based on consistency"""
        if zone.hit_count < 2:
            return
        
        total = zone.rejection_count + zone.breakthrough_count + zone.consolidation_count
        if total == 0:
            return
        
        max_count = max(zone.rejection_count, zone.breakthrough_count, zone.consolidation_count)
        consistency = max_count / total
        hit_factor = min(1.0, zone.hit_count / 10.0)
        zone.confidence = 0.5 + (consistency * hit_factor * 0.5)
    
    def get_zone_summary(self, symbol: str) -> str:
        """Get formatted summary"""
        zones = sorted([z for z in self.zones[symbol] if z.hit_count > 0], key=lambda z: z.price)
        
        if not zones:
            return f"No active zones for {symbol}"
        
        summary = f"\n{'='*80}\nLiquidity Zones for {symbol}:\n{'='*80}\n"
        
        for zone in zones:
            total = zone.rejection_count + zone.breakthrough_count + zone.consolidation_count
            if total == 0:
                continue
                
            r_pct = zone.rejection_count / total * 100
            b_pct = zone.breakthrough_count / total * 100
            c_pct = zone.consolidation_count / total * 100
            
            behavior = "🛑 Rejects" if zone.tends_to_reject_first else \
                      "🚀 Breaks" if zone.tends_to_melt_through else "💫 Consolidates"
            
            summary += f"\n${zone.price:.2f} | Conf: {zone.confidence:.2f} | Hits: {zone.hit_count} | {behavior}\n"
            summary += f"  R: {zone.rejection_count} ({r_pct:.0f}%) | "
            summary += f"B: {zone.breakthrough_count} ({b_pct:.0f}%) | "
            summary += f"C: {zone.consolidation_count} ({c_pct:.0f}%)\n"
        
        return summary + f"{'='*80}\n"

    def get_optimal_stop_loss(self, symbol: str, entry_price: float, 
                              direction: str, default_distance: float) -> float:
        """
        Get optimal stop loss placement beyond strong liquidity zones
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of the trade
            direction: 'LONG' or 'SHORT'
            default_distance: Default stop distance if no zones found
            
        Returns:
            Optimal stop loss price
        """
        zones = self.zones.get(symbol, [])
        
        if not zones:
            # No zones detected yet, use default
            if direction == 'LONG':
                return entry_price - default_distance
            else:
                return entry_price + default_distance
        
        # Find strong zones that could act as support/resistance
        if direction == 'LONG':
            # For LONG, look for strong support zones below entry
            support_zones = [
                z for z in zones 
                if z.price < entry_price 
                and z.tends_to_reject_first  # Strong zones that reject price
                and z.confidence > 0.6  # High confidence
                and abs(z.price - entry_price) < entry_price * 0.05  # Within 5%
            ]
            
            if support_zones:
                # Sort by distance from entry (closest first)
                support_zones.sort(key=lambda z: abs(z.price - entry_price))
                
                # Place stop just below the nearest strong support
                nearest_support = support_zones[0]
                
                # Use buffer based on symbol
                buffer = self.ZONE_BUFFERS.get(symbol, 1.0) * 0.5
                optimal_stop = nearest_support.price - buffer
                
                logger.info(f"🛡️  {symbol} LONG: Stop at ${optimal_stop:.2f} "
                          f"(below support ${nearest_support.price:.2f}, conf={nearest_support.confidence:.2f})")
                
                return optimal_stop
            else:
                # No strong support found, use default
                return entry_price - default_distance
                
        else:  # SHORT
            # For SHORT, look for strong resistance zones above entry
            resistance_zones = [
                z for z in zones 
                if z.price > entry_price 
                and z.tends_to_reject_first  # Strong zones that reject price
                and z.confidence > 0.6  # High confidence
                and abs(z.price - entry_price) < entry_price * 0.05  # Within 5%
            ]
            
            if resistance_zones:
                # Sort by distance from entry (closest first)
                resistance_zones.sort(key=lambda z: abs(z.price - entry_price))
                
                # Place stop just above the nearest strong resistance
                nearest_resistance = resistance_zones[0]
                
                # Use buffer based on symbol
                buffer = self.ZONE_BUFFERS.get(symbol, 1.0) * 0.5
                optimal_stop = nearest_resistance.price + buffer
                
                logger.info(f"🛡️  {symbol} SHORT: Stop at ${optimal_stop:.2f} "
                          f"(above resistance ${nearest_resistance.price:.2f}, conf={nearest_resistance.confidence:.2f})")
                
                return optimal_stop
            else:
                # No strong resistance found, use default
                return entry_price + default_distance
    
    def get_optimal_take_profit(self, symbol: str, entry_price: float, 
                               direction: str, risk_reward: float = 2.0,
                               stop_loss: float = None) -> float:
        """
        Get optimal take profit placement EXCLUSIVELY based on liquidity zones
        NO fallback to R:R ratios - always uses zone-based placement
        
        Strategy:
        1. Find strong zones that could act as resistance/support (where price reverses)
        2. Place take profit BEFORE reaching those zones
        3. If no strong zones found, progressively relax criteria
        4. If still no zones, create synthetic zone based on round numbers
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of the trade
            direction: 'LONG' or 'SHORT'
            risk_reward: Risk/reward ratio (only used for minimum profit distance)
            stop_loss: Stop loss price (for calculating minimum profit)
            
        Returns:
            Optimal take profit price based on liquidity zones
        """
        zones = self.zones.get(symbol, [])
        buffer = self.ZONE_BUFFERS.get(symbol, 1.0)
        
        # Calculate minimum profit distance (50% of R:R, just as a floor)
        if stop_loss:
            risk = abs(entry_price - stop_loss)
            min_profit_distance = risk * 0.5  # At least 1:2 R:R minimum
        else:
            min_profit_distance = buffer * 1.0  # Default minimum
        
        if direction == 'LONG':
            # For LONG: Look for resistance zones ABOVE entry where price might reverse
            # We'll place take profit BELOW those zones
            
            # Phase 1: Try strong resistance zones (confidence > 0.7, within 20%)
            resistance_zones = [
                z for z in zones 
                if z.price > entry_price  # Above entry
                and z.tends_to_reject_first  # Strong resistance
                and z.confidence > 0.7
                and abs(z.price - entry_price) < entry_price * 0.20  # Within 20%
            ]
            
            # Phase 2: If none found, relax confidence requirement
            if not resistance_zones:
                resistance_zones = [
                    z for z in zones 
                    if z.price > entry_price
                    and z.tends_to_reject_first
                    and z.confidence > 0.5  # Lower confidence
                    and abs(z.price - entry_price) < entry_price * 0.30  # Within 30%
                ]
            
            # Phase 3: If still none, use ANY zone that rejects (any confidence)
            if not resistance_zones:
                resistance_zones = [
                    z for z in zones 
                    if z.price > entry_price
                    and z.rejection_count > 0  # Has rejected at least once
                    and abs(z.price - entry_price) < entry_price * 0.50  # Within 50%
                ]
            
            # Phase 4: If STILL none, create synthetic zone at next round number
            if not resistance_zones:
                round_levels = self.ROUND_NUMBER_LEVELS.get(symbol, [5])
                for level in sorted(round_levels, reverse=True):  # Try largest levels first
                    synthetic_zone_price = round((entry_price + min_profit_distance * 2) / level) * level
                    if synthetic_zone_price > entry_price + min_profit_distance:
                        logger.info(f"🎯 {symbol} LONG: Using synthetic zone at ${synthetic_zone_price:.2f} (next ${level} level)")
                        return synthetic_zone_price - (buffer * 0.3)
                
                # Absolute fallback: entry + (min_profit * 2)
                synthetic_target = entry_price + (min_profit_distance * 2)
                logger.info(f"🎯 {symbol} LONG: Using calculated target at ${synthetic_target:.2f} (no zones found)")
                return synthetic_target
            
            # We have resistance zones - find the nearest one
            resistance_zones.sort(key=lambda z: abs(z.price - entry_price))
            nearest_resistance = resistance_zones[0]
            
            # Place take profit BELOW the resistance zone
            optimal_target = nearest_resistance.price - (buffer * 0.3)
            
            # Ensure minimum profit distance
            if optimal_target < entry_price + min_profit_distance:
                # Zone is too close, move to minimum profit distance
                optimal_target = entry_price + min_profit_distance
                logger.info(f"🎯 {symbol} LONG: Target at ${optimal_target:.2f} "
                          f"(adjusted for min profit, resistance ${nearest_resistance.price:.2f} too close)")
            else:
                logger.info(f"🎯 {symbol} LONG: Target at ${optimal_target:.2f} "
                          f"(below resistance ${nearest_resistance.price:.2f}, conf={nearest_resistance.confidence:.2f})")
            
            return optimal_target
                
        else:  # SHORT
            # For SHORT: Look for support zones BELOW entry where price might reverse
            # We'll place take profit ABOVE those zones
            
            # Phase 1: Try strong support zones (confidence > 0.7, within 20%)
            support_zones = [
                z for z in zones 
                if z.price < entry_price  # Below entry
                and z.tends_to_reject_first  # Strong support
                and z.confidence > 0.7
                and abs(z.price - entry_price) < entry_price * 0.20  # Within 20%
            ]
            
            # Phase 2: If none found, relax confidence requirement
            if not support_zones:
                support_zones = [
                    z for z in zones 
                    if z.price < entry_price
                    and z.tends_to_reject_first
                    and z.confidence > 0.5  # Lower confidence
                    and abs(z.price - entry_price) < entry_price * 0.30  # Within 30%
                ]
            
            # Phase 3: If still none, use ANY zone that rejects (any confidence)
            if not support_zones:
                support_zones = [
                    z for z in zones 
                    if z.price < entry_price
                    and z.rejection_count > 0  # Has rejected at least once
                    and abs(z.price - entry_price) < entry_price * 0.50  # Within 50%
                ]
            
            # Phase 4: If STILL none, create synthetic zone at next round number
            if not support_zones:
                round_levels = self.ROUND_NUMBER_LEVELS.get(symbol, [5])
                for level in sorted(round_levels, reverse=True):  # Try largest levels first
                    synthetic_zone_price = round((entry_price - min_profit_distance * 2) / level) * level
                    if synthetic_zone_price < entry_price - min_profit_distance:
                        logger.info(f"🎯 {symbol} SHORT: Using synthetic zone at ${synthetic_zone_price:.2f} (next ${level} level)")
                        return synthetic_zone_price + (buffer * 0.3)
                
                # Absolute fallback: entry - (min_profit * 2)
                synthetic_target = entry_price - (min_profit_distance * 2)
                logger.info(f"🎯 {symbol} SHORT: Using calculated target at ${synthetic_target:.2f} (no zones found)")
                return synthetic_target
            
            # We have support zones - find the nearest one
            support_zones.sort(key=lambda z: abs(z.price - entry_price))
            nearest_support = support_zones[0]
            
            # Place take profit ABOVE the support zone
            optimal_target = nearest_support.price + (buffer * 0.3)
            
            # Ensure minimum profit distance
            if optimal_target > entry_price - min_profit_distance:
                # Zone is too close, move to minimum profit distance
                optimal_target = entry_price - min_profit_distance
                logger.info(f"🎯 {symbol} SHORT: Target at ${optimal_target:.2f} "
                          f"(adjusted for min profit, support ${nearest_support.price:.2f} too close)")
            else:
                logger.info(f"🎯 {symbol} SHORT: Target at ${optimal_target:.2f} "
                          f"(above support ${nearest_support.price:.2f}, conf={nearest_support.confidence:.2f})")
            
            return optimal_target

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    tracker = LiquidityZoneTracker(db_path='test.db')
    
    # Simulate price movements
    print("\nTest: Price rejecting at $4090")
    for p in [4085, 4087, 4089, 4090, 4089, 4087, 4085]:
        tracker.detect_zones('GC', p)
        result = tracker.record_price_interaction('GC', p)
        if result:
            print(f"  → {result}")
    
    print(tracker.get_zone_summary('GC'))
