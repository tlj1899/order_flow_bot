"""
Order Flow Trading System - LIVE DATA ONLY
Streams real-time order flow data from Databento API for futures contracts (GC, SI, PL, HG, NQ)
Detects order flow imbalances and places trades via Topstep API
NO HISTORICAL DATA - Uses only live streaming market data
"""

import databento as db
import asyncio
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, timezone
from collections import deque, defaultdict
import logging
import requests
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json, csv
from io import StringIO
from topstepapi import TopstepClient
import paho.mqtt.client as mqtt

# Import custom modules
try:
    from liquidity_zones import LiquidityZoneTracker
    LIQUIDITY_ZONES_AVAILABLE = True
except ImportError:
    LIQUIDITY_ZONES_AVAILABLE = False
    logging.warning("Liquidity zone module not available - using default stops/targets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OrderFlowMetrics:
    """Store order flow metrics for analysis"""
    symbol: str
    timestamp: datetime
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float
    cumulative_delta: float
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    trade_intensity: float
    level_2_imbalance: float


@dataclass
class TradeSignal:
    """Trade signal generated from order flow analysis"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    reason: str


class OrderFlowAnalyzer:
    """Analyze order flow data to detect imbalances"""
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.volume_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.delta_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.cumulative_delta = defaultdict(float)
        
    def calculate_imbalance(self, bid_volume: float, ask_volume: float) -> float:
        """Calculate order flow imbalance ratio"""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_volume
    
    def update_cumulative_delta(self, symbol: str, delta: float):
        """Update cumulative delta for a symbol"""
        self.cumulative_delta[symbol] += delta
        self.delta_history[symbol].append(delta)
    
    def detect_aggressive_trades(self, trades: List[Dict]) -> Tuple[float, float]:
        """Detect aggressive buying and selling from trade data"""
        aggressive_buy = 0.0
        aggressive_sell = 0.0
        
        for trade in trades:
            # Trades at ask are aggressive buys, trades at bid are aggressive sells
            if trade.get('side') == 'buy' or trade.get('aggressor_side') == 'buy':
                aggressive_buy += trade.get('size', 0)
            elif trade.get('side') == 'sell' or trade.get('aggressor_side') == 'sell':
                aggressive_sell += trade.get('size', 0)
        
        return aggressive_buy, aggressive_sell
    
    def calculate_level_2_imbalance(self, orderbook: Dict) -> float:
        """Calculate imbalance from level 2 order book data"""
        bid_depth = sum([level['size'] for level in orderbook.get('bids', [])[:5]])
        ask_depth = sum([level['size'] for level in orderbook.get('asks', [])[:5]])
        
        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return 0.0
        
        return (bid_depth - ask_depth) / total_depth
    
    def analyze_order_flow(self, symbol: str, market_data: Dict) -> OrderFlowMetrics:
        """Comprehensive order flow analysis"""
        bid_volume = market_data.get('bid_volume', 0)
        ask_volume = market_data.get('ask_volume', 0)
        
        # Calculate metrics
        imbalance = self.calculate_imbalance(bid_volume, ask_volume)
        delta = bid_volume - ask_volume
        self.update_cumulative_delta(symbol, delta)
        
        aggressive_buy, aggressive_sell = self.detect_aggressive_trades(
            market_data.get('trades', [])
        )
        
        level_2_imbalance = self.calculate_level_2_imbalance(
            market_data.get('orderbook', {})
        )
        
        # Calculate trade intensity (trades per minute)
        trade_intensity = len(market_data.get('trades', [])) / 1.0  # 1 minute window
        
        return OrderFlowMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance,
            cumulative_delta=self.cumulative_delta[symbol],
            aggressive_buy_volume=aggressive_buy,
            aggressive_sell_volume=aggressive_sell,
            trade_intensity=trade_intensity,
            level_2_imbalance=level_2_imbalance
        )


class SignalGenerator:
    """Generate trading signals based on order flow analysis"""
    
    def __init__(self, imbalance_threshold: float = 0.6, 
                 cumulative_delta_threshold: float = 1000):
        self.imbalance_threshold = imbalance_threshold
        self.cumulative_delta_threshold = cumulative_delta_threshold
        
    def generate_signal(self, metrics: OrderFlowMetrics, 
                       current_price: float) -> Optional[TradeSignal]:
        """Generate trading signal from order flow metrics"""
        confidence = 0.0
        direction = None
        reasons = []
        
        # Check imbalance ratio
        if metrics.imbalance_ratio > self.imbalance_threshold:
            confidence += 0.3
            direction = 'LONG'
            reasons.append(f'Strong buy imbalance: {metrics.imbalance_ratio:.2f}')
        elif metrics.imbalance_ratio < -self.imbalance_threshold:
            confidence += 0.3
            direction = 'SHORT'
            reasons.append(f'Strong sell imbalance: {metrics.imbalance_ratio:.2f}')
        
        # Check cumulative delta
        if metrics.cumulative_delta > self.cumulative_delta_threshold:
            if direction == 'LONG' or direction is None:
                confidence += 0.25
                direction = 'LONG'
                reasons.append(f'Positive cumulative delta: {metrics.cumulative_delta:.0f}')
        elif metrics.cumulative_delta < -self.cumulative_delta_threshold:
            if direction == 'SHORT' or direction is None:
                confidence += 0.25
                direction = 'SHORT'
                reasons.append(f'Negative cumulative delta: {metrics.cumulative_delta:.0f}')
        
        # Check aggressive volume
        aggr_ratio = (metrics.aggressive_buy_volume - metrics.aggressive_sell_volume) / \
                     max(metrics.aggressive_buy_volume + metrics.aggressive_sell_volume, 1)
        
        if abs(aggr_ratio) > 0.5:
            confidence += 0.2
            if aggr_ratio > 0 and direction == 'LONG':
                reasons.append(f'Aggressive buying: {aggr_ratio:.2f}')
            elif aggr_ratio < 0 and direction == 'SHORT':
                reasons.append(f'Aggressive selling: {aggr_ratio:.2f}')
        
        # Check level 2 imbalance
        if abs(metrics.level_2_imbalance) > 0.4:
            confidence += 0.15
            if metrics.level_2_imbalance > 0 and direction == 'LONG':
                reasons.append(f'Order book bid depth: {metrics.level_2_imbalance:.2f}')
            elif metrics.level_2_imbalance < 0 and direction == 'SHORT':
                reasons.append(f'Order book ask depth: {metrics.level_2_imbalance:.2f}')
        
        # Trade intensity bonus
        if metrics.trade_intensity > 50:  # High activity
            confidence += 0.1
            reasons.append(f'High trade intensity: {metrics.trade_intensity:.0f}')
        
        # Only generate signal if confidence threshold met
        if confidence >= 0.6 and direction:
            # Calculate stop loss and take profit based on contract specifications
            risk_reward_ratio = 2.0
            
            # Contract-specific stop distances (in price units)
            # These are reasonable distances based on typical volatility
            if metrics.symbol == 'GC':  # Gold (trades around $2000-4000)
                stop_distance = 5.0  # $5.00 per troy ounce
            elif metrics.symbol == 'SI':  # Silver (trades around $20-50)
                stop_distance = 0.10  # $0.10 per troy ounce (10 cents)
            elif metrics.symbol == 'PL':  # Platinum (trades around $900-1600)
                stop_distance = 10.0  # $10.00 per troy ounce
            elif metrics.symbol == 'HG':  # Copper (trades around $3-5)
                stop_distance = 0.02  # $0.02 per pound (2 cents)
            elif metrics.symbol == 'NQ':  # Nasdaq (trades around 15000-25000)
                stop_distance = 20.0  # 20 index points
            else:
                stop_distance = 10.0  # Default
            
            if direction == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * risk_reward_ratio)
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * risk_reward_ratio)
            
            return TradeSignal(
                symbol=metrics.symbol,
                direction=direction,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=metrics.timestamp,
                reason='; '.join(reasons)
            )
        
        return None


class DatabentoClient:
    """Client for Databento API to fetch live order flow data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.live_client = None  # For live streaming only

        # Databento symbols for futures
        self.symbols = {
            'GC': 'GC.v.0',  # Gold
            'SI': 'SI.v.0',  # Silver
            'PL': 'PL.v.0',  # Platinum
            'HG': 'HG.v.0',  # Copper
            #'NQ': 'NQZ5'   # Nasdaq-100 E-mini
        }
    
    async def stream_market_data(self, symbols: List[str], callback):
        """Stream real-time market data - LIVE ONLY"""
        try:
            # Initialize live client
            self.live_client = db.Live(key=self.api_key)
            
            # Subscribe to MBP-1 (best bid/offer) and trades
            databento_symbols = [self.symbols[sym] for sym in symbols]
            
            logger.info("Subscribing to LIVE market data streams...")
            
            self.live_client.subscribe(
                dataset='GLBX.MDP3',  # CME Globex
                schema='mbp-1',
                stype_in='continuous',
                symbols=databento_symbols
            )
            
            self.live_client.subscribe(
                dataset='GLBX.MDP3',
                schema='trades',
                stype_in='continuous',
                symbols=databento_symbols
            )
            
            logger.info("✓ Subscribed to live MBP-1 and Trades data")
            logger.info("✓ Streaming live market data only (no historical data)")
            
            # Start streaming
            async for record in self.live_client:
                await callback(record)
                
        except Exception as e:
            logger.error(f"Error streaming market data: {e}")
            raise


class TopstepClientClass:
    """Client for Topstep API to execute trades"""
    
    def __init__(self, api_key: str, username: str, account: str, mll: int, position_size: int, paper_mode: bool = False):
        self.api_key = api_key
        self.username = username
        self.account = account
        self.account_name = None
        self.account_balance = 150000
        self.can_trade = True
        self.mll = mll
        self.dmll = 0
        self.position_size = position_size
        self.account_type = "Practice"
        self.paper_mode = paper_mode
        
        # Position tracking
        self.positions = {}
        
        if self.paper_mode:
            logger.warning("Running in PAPER TRADING mode - no real orders will be placed")

    def authenticate(self):
        try:
            client = TopstepClient(self.username, self.api_key)
            client.account.search_accounts()
            return client
        except Exception as e:
            print(f"Topstep authentication failed: {e}")

    def get_account_info(self) -> Dict:
        """Get account information"""
        if self.paper_mode:
            return {
                'account_id': self.account,
                'balance': 150000,
                'mode': 'PAPER'
            }
        
        try:
            accounts = self.authenticate().account.search_accounts()
            df_accounts = pd.DataFrame(accounts)
            for i in range(len(df_accounts)):
                if str(self.account) == str(df_accounts['id'].iloc[i]):
                    self.account_name = df_accounts['name'].iloc[i]
                    if "KTC" in df_accounts['name'].iloc[i]:
                        self.account_type = "Combine"
                    elif "XFA" in df_accounts['name'].iloc[i]:
                        self.account_type = "XFA"
                    elif "LIVE" in df_accounts['name'].iloc[i]:
                        self.account_type = "Live"
                    else:
                        self.account_type = "CAUTION: UNABLE TO DETERMINE ACCOUNT TYPE"
                    self.account_balance = df_accounts['balance'].iloc[i]
                    self.can_trade = df_accounts['canTrade'].iloc[i]
                    self.dmll = self.get_dmll()
                    print("=" * 70)
                    print("TOPSTEP ACCOUNT INFORMATION:")
                    print("=" * 70)
                    print(f"ACCOUNT ID: {self.account}")
                    print(f"ACCOUNT NAME: {self.account_name}")
                    print(f"ACCOUNT TYPE: {self.account_type}")
                    print(f"ACCOUNT BALANCE: {self.account_balance}")
                    print(f"ACCOUNT DMLL: {self.dmll}")
                    print(f"TRADING ALLOWED: {self.can_trade}") 
                    print("=" * 70)
                    return
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account info: {e}")
            logger.warning("Topstep API may not be configured correctly. Check your API endpoint and credentials.")
            logger.info("Tip: Use paper_mode=True to test without connecting to Topstep API")
            return {}

    def get_dmll(self):
        return self.get_account_balance() - self.mll

    def get_account_balance(self):
        try:
            accounts = self.authenticate().account.search_accounts()
            df_accounts = pd.DataFrame(accounts)
            for i in range(len(df_accounts)):
                if str(self.account) == str(df_accounts['id'].iloc[i]):
                    return df_accounts['balance'].iloc[i]
        except Exception as e:
            print(f"Error retrieving Topstep account balance: {e}")

    def get_contract_id(self, instrument):
        try:
            if instrument == "GC":
                return self.authenticate().contract.search_contracts('gce')[0]['id']
            if instrument == "SI":
                return self.authenticate().contract.search_contracts('sie')[0]['id']
            if instrument == "PL":
                return self.authenticate().contract.search_contracts('pl')[0]['id']
            if instrument == "HG":
                return self.authenticate().contract.search_contracts('cpe')[0]['id']
            print("ERROR! Invalid instrument.")
        except Exception as e:
            print(f"Error retrieving contract id: {e}")

    def get_contract_symbol(self, instrument):
        try:
            if instrument == "GC":
                return self.authenticate().contract.search_contracts('gce')[0]['name']
            if instrument == "SI":
                return self.authenticate().contract.search_contracts('sie')[0]['name']
            if instrument == "PL":
                return self.authenticate().contract.search_contracts('pl')[0]['name']
            if instrument == "HG":
                return self.authenticate().contract.search_contracts('cpe')[0]['name']
            print("ERROR! Invalid instrument.")
        except Exception as e:
            print(f"Error retrieving contract name: {e}")
    
    def place_order(self, signal: TradeSignal) -> Optional[str]:
        """Place order based on trade signal"""
        
        order_direction = 1
        close_order_direction = 0

        order_data = {
            'account_id': self.account,
            'symbol': signal.symbol,
            'side': 'BUY' if signal.direction == 'LONG' else 'SELL',
            'order_type': 'LIMIT',
            'quantity': self.position_size,
            'price': signal.entry_price,
            'time_in_force': 'DAY',
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }

        contract = self.get_contract_id(signal.symbol)
        if signal.direction == 'LONG':
            order_direction = 0
            close_order_direction = 1 
        
        logger.info(f"{'[PAPER] ' if self.paper_mode else ''}Placing order: {order_data}")
        
        if self.paper_mode:
            # Simulate order placement
            import uuid
            order_id = f"PAPER-{uuid.uuid4().hex[:8]}"
            
            self.positions[signal.symbol] = {
                'order_id': order_id,
                'direction': signal.direction,
                'quantity': self.position_size,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            logger.info(f"[PAPER] Order simulated successfully: {order_id}")
            return order_id
        
        try:
            order_id = self.authenticate().order.place_order(
                account_id=self.account,
                contract_id=contract,
                type=2,
                side=order_direction,
                size=self.position_size)

            logger.info(f"Order placed successfully: {order_id}")
            
            # Track position
            self.positions[signal.symbol] = {
                'order_id': order_id,
                'direction': signal.direction,
                'quantity': self.position_size,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def publish_mqtt(self, topic, message):

        broker_address = "homeassistant.local"
        broker_port = 1883
        username = "tapette"
        password = "N1pples!"
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(username, password)
        client.connect(broker_address, broker_port, 60)
        client.loop_start()
        client.publish(topic, message)
        time.sleep(2)
        client.loop_stop()
        client.disconnect()
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str = 'MANUAL') -> Optional[Dict]:
        """Close existing position and return PnL info"""
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return None
            
            position = self.positions[symbol]

            trade_info = {
                'symbol': symbol,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'quantity': position['quantity']
            }
            
            if self.paper_mode:
                # Simulate position close
                logger.info(f"[PAPER] Closing {position['direction']} position on {symbol}")
                
                del self.positions[symbol]
                return trade_info
            
            contract=self.get_contract_id(symbol)
            positions_df = pd.DataFrame(self.authenticate().position.search_open_positions(account_id=self.account))
            order_id = None
            for i in range(len(positions_df)):
                if positions_df['contractId'].iloc[i] == contract:
                    order_id = positions_df['id'].iloc[i]
                    break
            self.authenticate().position.close_position(account_id=self.account, contract_id=contract) 
            time.sleep(5)
            trades_df = pd.DataFrame(self.authenticate().trade.search_trades(account_id=self.account, start_timestamp="2025-01-01T00:00:00Z"))
            for i in range(len(trades_df)):
                if trades_df['id'].iloc[i] == order_id:
                    profit = trades_df['profitAndLoss'].iloc[i] - trades_df['fees'].iloc[i]
                    if profit >= 0:
                        self.publish_mqtt("trading", "profit")
                    else:
                        self.publish_mqtt("trading", "loss")
                break
            
            logger.info(f"Position closed for {symbol}")
            
            trade_info = {
                'symbol': symbol,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'quantity': position['quantity']
            }
            
            del self.positions[symbol]
            return trade_info
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            return self.authenticate().position.search_open_positions(account_id=self.account)
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []


class OrderFlowTradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, databento_api_key: str, topstep_api_key: str, topstep_username: str, 
                 topstep_account_id: str, topstep_mll: int, topstep_position_size: int, paper_mode: bool = False,
                 enable_order_flow_exits: bool = True, signal_cooldown_minutes: int = 5,
                 imbalance_threshold: float = 0.8, min_confidence: float = 0.8, orderflow_exit_threshold: float = 0.5,
                 enable_session_filter: bool = False):
        self.databento = DatabentoClient(databento_api_key)
        self.topstep = TopstepClientClass(topstep_api_key, topstep_username, topstep_account_id, topstep_mll, topstep_position_size, paper_mode=paper_mode)
        self.analyzer = OrderFlowAnalyzer(lookback_periods=20)
        self.signal_generator = SignalGenerator(
            imbalance_threshold=imbalance_threshold,
            cumulative_delta_threshold=1000
        )

        # Store thresholds for use in exit logic
        self.orderflow_exit_threshold = orderflow_exit_threshold
        self.min_confidence = min_confidence
        
        self.symbols = ['GC', 'SI', 'PL', 'HG']
        self.market_data_cache = defaultdict(lambda: {
            'trades': deque(maxlen=100),
            'orderbook': {},
            'last_price': 0.0,
            'bid_volume': 0.0,
            'ask_volume': 0.0,
            'price_history': deque(maxlen=20)  # Track price history for momentum
        })
        
        self.last_position_close_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))  # ✅ Track close time
        self.signal_cooldown = timedelta(minutes=signal_cooldown_minutes)  # Cooldown between signals
        logger.info(f"⏱️  Signal cooldown: {signal_cooldown_minutes} minutes after position close")
        
        # Exit strategy configuration
        self.enable_order_flow_exits = enable_order_flow_exits
        
        # Session filtering - Trade only profitable hours
        self.enable_session_filter = enable_session_filter  # Set to False to disable
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.closed_trades = []
        
        # Activity tracking
        self.total_records_received = 0
        self.record_count = defaultdict(int)
        self.last_status_update = datetime.now()
        self.status_update_interval = timedelta(seconds=30)
        
        # Symbology mapping: instrument_id -> symbol
        self.instrument_to_symbol = {}
        
        # Contract multipliers for PnL calculation
        self.contract_multipliers = {
            'GC': 100,   # Gold: 100 troy oz
            'SI': 5000,  # Silver: 5000 troy oz
            'PL': 50,    # Platinum: 50 troy oz
            'HG': 25000, # Copper: 25000 lbs
            'NQ': 20     # E-mini Nasdaq: $20 per point
        }
        
        # Liquidity zone tracking
        if LIQUIDITY_ZONES_AVAILABLE:
            self.liquidity_tracker = LiquidityZoneTracker()
            self.use_liquidity_zones = True
            logger.info("✅ Liquidity zone tracking ENABLED")
        else:
            self.liquidity_tracker = None
            self.use_liquidity_zones = False
            logger.warning("⚠️  Liquidity zone tracking DISABLED")
        
    def should_trade_now(self) -> bool:
        """Check if current time is within profitable trading hours"""
        if not self.enable_session_filter:
            return True
        
        # Get current time in both UTC and EST for clarity
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        
        # Trading window: 18:00 - 08:00 EST
        # Converting to UTC (EST = UTC-5):
        # 18:00 EST = 23:00 UTC (start)
        # 08:00 EST = 13:00 UTC (end)
        
        # Trade from 23:00 UTC to 13:00 UTC (next day)
        # This is: 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (14 hours)
        
        if hour_utc >= 23 or hour_utc < 13:
            return True
        
        # Outside trading window (13:00-23:00 UTC = 08:00-18:00 EST)
        logger.debug(f"⏸️  Outside trading window (hour {hour_utc:02d}:00 UTC)")
        return False
    
    async def process_market_data(self, record):
        """Process incoming market data record"""
        try:
            # Count every single record received from Databento
            self.total_records_received += 1
            
            # Every 100 records, show we're receiving data
            #if self.total_records_received % 100 == 0:
                #logger.info(f"📡 Received {self.total_records_received} total records from Databento")
            
            record_type = type(record).__name__
            
            # Handle SymbolMappingMsg to build our mapping
            if record_type == 'SymbolMappingMsg':
                # These records tell us instrument_id -> raw_symbol mapping
                if hasattr(record, 'stype_in_symbol') and hasattr(record, 'instrument_id'):
                    raw_symbol = record.stype_in_symbol
                    instrument_id = record.instrument_id
                    
                    # Parse raw_symbol (like 'GCZ5') to our format ('GC')
                    symbol = self._parse_raw_symbol(raw_symbol)
                    
                    if symbol:
                        self.instrument_to_symbol[instrument_id] = symbol
                        logger.info(f"📋 Mapped instrument {instrument_id} -> {raw_symbol} -> {symbol}")
                return
            
            # Skip system messages
            if record_type in ['SystemMsg', 'ErrorMsg']:
                return
            
            # Get instrument_id from record
            if not hasattr(record, 'instrument_id'):
                return
            
            instrument_id = record.instrument_id
            
            # Look up symbol from our mapping
            symbol = self.instrument_to_symbol.get(instrument_id)
            
            if not symbol:
                # We haven't received the symbol mapping yet for this instrument
                logger.debug(f"No symbol mapping for instrument_id {instrument_id} yet")
                return
            
            # SUCCESS - we have a mapped symbol
            logger.debug(f"✓ Processing record for {symbol} (instrument {instrument_id})")
            
            # Track activity
            self.record_count[symbol] += 1
            
            # Update market data cache based on record type
            if record_type == 'MBP1Msg':
                # MBP (Market by Price) record
                # Data is in the 'levels' array as BidAskPair objects
                if hasattr(record, 'levels') and record.levels and len(record.levels) > 0:
                    # Get first level (best bid/ask)
                    level = record.levels[0]
                    
                    # Prices are in fixed-point format, need to normalize
                    bid_px = level.bid_px / 1e9 if level.bid_px > 1e6 else level.bid_px
                    ask_px = level.ask_px / 1e9 if level.ask_px > 1e6 else level.ask_px
                    bid_sz = level.bid_sz
                    ask_sz = level.ask_sz
                    
                    mid_price = (bid_px + ask_px) / 2
                    self.market_data_cache[symbol]['last_price'] = mid_price
                    self.market_data_cache[symbol]['bid_volume'] = bid_sz
                    self.market_data_cache[symbol]['ask_volume'] = ask_sz

                    # Track price history for momentum detection
                    self.market_data_cache[symbol]['price_history'].append(mid_price)
                    
                    # Update liquidity zone tracker with NORMALIZED prices
                    if self.use_liquidity_zones:
                        total_volume = bid_sz + ask_sz
                        self.liquidity_tracker.detect_zones(symbol, mid_price)
                        self.liquidity_tracker.record_price_interaction(symbol, mid_price)
                        interaction = self.liquidity_tracker.record_price_interaction(symbol, mid_price)
                    
                    # Update orderbook
                    self.market_data_cache[symbol]['orderbook'] = {
                        'bids': [{'price': bid_px, 'size': bid_sz}],
                        'asks': [{'price': ask_px, 'size': ask_sz}]
                    }
                    
                    logger.debug(f"{symbol}: Bid={bid_px:.2f}({bid_sz}), Ask={ask_px:.2f}({ask_sz})")
                
            elif record_type == 'TradeMsg':
                # Trade record - most reliable for order flow analysis
                if hasattr(record, 'price') and hasattr(record, 'size'):
                    # Trade prices are in fixed-point format, need to convert
                    price = record.price / 1e9 if record.price > 1e6 else record.price
                    size = record.size
                    
                    # Determine if this is aggressive buy or sell
                    # Side: 'A'=Ask (buyer aggressor), 'B'=Bid (seller aggressor)
                    side = getattr(record, 'side', None)
                    action = getattr(record, 'action', None)
                    
                    # Infer aggressor side
                    aggressor_side = None
                    if side == 'A' or side == b'A':  # Trade at ask = buyer aggressive
                        aggressor_side = 'buy'
                    elif side == 'B' or side == b'B':  # Trade at bid = seller aggressive
                        aggressor_side = 'sell'
                    
                    trade_data = {
                        'price': price,
                        'size': size,
                        'side': side,
                        'aggressor_side': aggressor_side,
                        'timestamp': record.ts_event
                    }
                    self.market_data_cache[symbol]['trades'].append(trade_data)
                    
                    # Update last price (prefer MBP price if available, otherwise use trade price)
                    if self.market_data_cache[symbol]['last_price'] == 0:
                        self.market_data_cache[symbol]['last_price'] = price
                    
                    # If we don't have bid/ask volume from MBP, estimate from recent trades
                    if self.market_data_cache[symbol]['bid_volume'] == 0:
                        # Calculate volume from recent trades
                        recent_trades = list(self.market_data_cache[symbol]['trades'])
                        buy_vol = sum(t['size'] for t in recent_trades if t.get('aggressor_side') == 'buy')
                        sell_vol = sum(t['size'] for t in recent_trades if t.get('aggressor_side') == 'sell')
                        
                        self.market_data_cache[symbol]['bid_volume'] = buy_vol
                        self.market_data_cache[symbol]['ask_volume'] = sell_vol
                    
                    logger.debug(f"{symbol}: Trade at {price:.2f}, size={size}, side={side}, aggressor={aggressor_side}")
            
            # Periodic status update
            self._print_status_update()
            
            # Analyze order flow every few records
            await self.analyze_and_trade(symbol)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _parse_raw_symbol(self, raw_symbol: str) -> Optional[str]:
        """Parse Databento raw symbol (like 'GCZ5') to our symbol format (like 'GC')"""
        if not raw_symbol:
            return None
            
        # Extract the root symbol (first 2-3 characters before numbers)
        # GCZ5 -> GC, SIZ5 -> SI, NQZ5 -> NQ, etc.
        for symbol in self.symbols:
            if raw_symbol.startswith(symbol):
                return symbol
        
        return None
    
    def _calculate_unrealized_pnl(self) -> Dict[str, float]:
        """Calculate unrealized PnL for all open positions"""
        unrealized_pnls = {}
        
        for symbol, position in self.topstep.positions.items():
            current_price = self.market_data_cache[symbol]['last_price']
            
            # Normalize if needed
            if current_price > 1e6:
                current_price = current_price / 1e9
            
            entry_price = position['entry_price']
            if entry_price > 1e6:
                entry_price = entry_price / 1e9
            
            direction = position['direction']
            
            # Calculate price difference
            if direction == 'LONG':
                price_diff = current_price - entry_price
            else:  # SHORT
                price_diff = entry_price - current_price
            
            # Convert to dollar PnL
            multiplier = self.contract_multipliers.get(symbol, 1)
            pnl = price_diff * multiplier
            
            unrealized_pnls[symbol] = pnl
        
        return unrealized_pnls

def _check_price_momentum(self, symbol: str, direction: str, 
                          confirmation_periods: int = 8,
                          method: str = 'net_change') -> Tuple[bool, str]:
    """
    Check if recent price action confirms the signal direction using multiple detection methods
    
    This prevents "catching falling knives" and false entries.
    
    Methods:
    - 'tick_count': Original method - counts up vs down ticks (can miss trends with noise)
    - 'net_change': Net price change from N periods ago (best for clear trends)
    - 'ema': Compare current price to exponential moving average (smooth trend detection)
    - 'linear_regression': Calculate trend slope (most sophisticated)
    - 'combined': Require multiple methods to agree (most conservative)
    
    Args:
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT' 
        confirmation_periods: Number of recent prices to check
        method: Detection method to use
        
    Returns:
        (confirmed: bool, reason: str)
    """
    price_history = self.market_data_cache[symbol]['price_history']
    
    if len(price_history) < confirmation_periods:
        return False, f"Insufficient price history ({len(price_history)}/{confirmation_periods})"
    
    # Get last N prices
    recent_prices = list(price_history)[-confirmation_periods:]
    
    # Import config for thresholds
    try:
        import config
        tick_threshold = config.PRICE_CONFIRMATION_THRESHOLD
        net_change_threshold = config.NET_CHANGE_THRESHOLD
        ema_periods = config.EMA_PERIODS
        lr_threshold = config.LINEAR_REGRESSION_THRESHOLD
        combined_min = config.COMBINED_MIN_CONFIRMATIONS
    except (ImportError, AttributeError):
        tick_threshold = 0.6
        net_change_threshold = 0.0003
        ema_periods = 5
        lr_threshold = 0.0001
        combined_min = 2
    
    # ============================================
    # METHOD 1: TICK COUNT (Original)
    # ============================================
    def check_tick_count():
        bullish_moves = 0
        bearish_moves = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                bullish_moves += 1
            elif recent_prices[i] < recent_prices[i-1]:
                bearish_moves += 1
        
        total_moves = bullish_moves + bearish_moves
        if total_moves == 0:
            return False, "No price movement detected"
        
        bullish_pct = bullish_moves / total_moves
        bearish_pct = bearish_moves / total_moves
        
        if direction == 'LONG':
            if bullish_pct >= tick_threshold:
                return True, f"Tick count: {bullish_pct:.0%} bullish ticks"
            return False, f"Tick count: Only {bullish_pct:.0%} bullish (need {tick_threshold:.0%})"
        else:
            if bearish_pct >= tick_threshold:
                return True, f"Tick count: {bearish_pct:.0%} bearish ticks"
            return False, f"Tick count: Only {bearish_pct:.0%} bearish (need {tick_threshold:.0%})"
    
    # ============================================
    # METHOD 2: NET CHANGE (Simple & Effective)
    # ============================================
    def check_net_change():
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        net_change = (end_price - start_price) / start_price
        
        if direction == 'LONG':
            if net_change >= net_change_threshold:
                return True, f"Net change: +{net_change:.2%} (need +{net_change_threshold:.2%})"
            return False, f"Net change: {net_change:+.2%} insufficient for LONG (need +{net_change_threshold:.2%})"
        else:  # SHORT
            if net_change <= -net_change_threshold:
                return True, f"Net change: {net_change:.2%} (need -{net_change_threshold:.2%})"
            return False, f"Net change: {net_change:+.2%} insufficient for SHORT (need -{net_change_threshold:.2%})"
    
    # ============================================
    # METHOD 3: EMA COMPARISON
    # ============================================
    def check_ema():
        # Calculate EMA
        if len(recent_prices) < ema_periods:
            return False, f"Insufficient data for EMA-{ema_periods}"
        
        # Simple EMA calculation
        ema = recent_prices[0]
        multiplier = 2 / (ema_periods + 1)
        
        for price in recent_prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        current_price = recent_prices[-1]
        ema_diff = (current_price - ema) / ema
        
        if direction == 'LONG':
            if current_price > ema and ema_diff > 0.0001:  # Price above EMA
                return True, f"Price ${current_price:.2f} above EMA-{ema_periods} ${ema:.2f} ({ema_diff:+.2%})"
            return False, f"Price ${current_price:.2f} not above EMA-{ema_periods} ${ema:.2f}"
        else:  # SHORT
            if current_price < ema and ema_diff < -0.0001:  # Price below EMA
                return True, f"Price ${current_price:.2f} below EMA-{ema_periods} ${ema:.2f} ({ema_diff:+.2%})"
            return False, f"Price ${current_price:.2f} not below EMA-{ema_periods} ${ema:.2f}"
    
    # ============================================
    # METHOD 4: LINEAR REGRESSION SLOPE
    # ============================================
    def check_linear_regression():
        import numpy as np
        
        # Create x-axis (time indices)
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        # Calculate linear regression slope
        # slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        # Normalize slope by average price to get percentage
        avg_price = np.mean(y)
        normalized_slope = slope / avg_price
        
        if direction == 'LONG':
            if normalized_slope >= lr_threshold:
                return True, f"Linear regression: +{normalized_slope:.4%} upward slope"
            return False, f"Linear regression: {normalized_slope:+.4%} slope insufficient for LONG"
        else:  # SHORT
            if normalized_slope <= -lr_threshold:
                return True, f"Linear regression: {normalized_slope:.4%} downward slope"
            return False, f"Linear regression: {normalized_slope:+.4%} slope insufficient for SHORT"
    
    # ============================================
    # METHOD 5: COMBINED (Multiple Confirmations)
    # ============================================
    def check_combined():
        confirmations = []
        reasons = []
        
        # Test each method
        methods = [
            ('Net Change', check_net_change),
            ('EMA', check_ema),
            ('Linear Regression', check_linear_regression)
        ]
        
        for name, method_func in methods:
            try:
                confirmed, reason = method_func()
                if confirmed:
                    confirmations.append(name)
                reasons.append(f"{name}: {reason}")
            except Exception as e:
                reasons.append(f"{name}: Error ({str(e)})")
        
        num_confirmations = len(confirmations)
        
        if num_confirmations >= combined_min:
            return True, f"{num_confirmations}/{len(methods)} methods confirm ({', '.join(confirmations)})"
        else:
            return False, f"Only {num_confirmations}/{len(methods)} methods confirm (need {combined_min}). " + "; ".join(reasons)
    
    # ============================================
    # EXECUTE SELECTED METHOD
    # ============================================
    try:
        if method == 'tick_count':
            return check_tick_count()
        elif method == 'net_change':
            return check_net_change()
        elif method == 'ema':
            return check_ema()
        elif method == 'linear_regression':
            return check_linear_regression()
        elif method == 'combined':
            return check_combined()
        else:
            logger.warning(f"Unknown momentum method '{method}', falling back to 'net_change'")
            return check_net_change()
    except Exception as e:
        logger.error(f"Error in momentum detection: {e}")
        return False, f"Error in {method} detection: {str(e)}"

    def _calculate_tight_stop(self, symbol: str, entry_price: float, 
                              direction: str, percentage: float = 0.003) -> float:
        """
        Calculate tight static stop loss for anticipatory entries
    
        Used when entering against momentum (not waiting for price confirmation).
        Caps downside risk on counter-trend entries.
    
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            percentage: Stop distance as percentage of entry (default 0.3%)
        
        Returns:
            Tight stop loss price
        """
        stop_distance = entry_price * percentage
    
        if direction == 'LONG':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def _apply_max_risk_cap(self, symbol: str, entry_price: float, stop_loss: float, 
                        direction: str) -> float:
    """
    Apply maximum risk cap to prevent one big loser from wiping out winners
    
    CRITICAL: Limits risk per trade to MAX_RISK_TICKS configuration.
    This prevents scenarios like: 5 winners × $200 = $1000, then 1 loser × $1500 = -$500 net
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        stop_loss: Original stop loss from liquidity zones
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Adjusted stop loss (closer to entry if needed)
    """
    try:
        import config
        if not config.ENABLE_MAX_RISK_CAP:
            return stop_loss
        
        max_risk_ticks = config.MAX_RISK_TICKS.get(symbol)
        tick_size = config.TICK_SIZES.get(symbol)
        tick_value = config.TICK_VALUES.get(symbol)
        
        if not all([max_risk_ticks, tick_size, tick_value]):
            logger.warning(f"{symbol}: Missing max risk configuration, using original stop")
            return stop_loss
        
    except (ImportError, AttributeError):
        return stop_loss
    
    # Calculate current risk in ticks
    risk_distance = abs(entry_price - stop_loss)
    current_risk_ticks = risk_distance / tick_size
    current_risk_dollars = current_risk_ticks * tick_value
    
    # If current risk exceeds max, tighten stop
    if current_risk_ticks > max_risk_ticks:
        max_risk_distance = max_risk_ticks * tick_size
        max_risk_dollars = max_risk_ticks * tick_value
        
        if direction == 'LONG':
            adjusted_stop = entry_price - max_risk_distance
        else:
            adjusted_stop = entry_price + max_risk_distance
        
        logger.warning(f"⚠️  {symbol} {direction}: Risk cap applied!")
        logger.warning(f"   Original stop: ${stop_loss:.2f} ({current_risk_ticks:.0f} ticks = ${current_risk_dollars:.2f} risk)")
        logger.warning(f"   Adjusted stop: ${adjusted_stop:.2f} ({max_risk_ticks} ticks = ${max_risk_dollars:.2f} risk MAX)")
        logger.warning(f"   This prevents excessive losses that wipe out winning streaks")
        
        return adjusted_stop
    else:
        logger.info(f"✅ {symbol}: Risk within limit ({current_risk_ticks:.0f}/{max_risk_ticks} ticks = ${current_risk_dollars:.2f})")
        return stop_loss
    
    def _check_position_exits(self):
        """Check if any positions should be closed based on stop loss, take profit, or order flow reversal"""
        now = datetime.now(timezone.utc)
        hour_utc = now.hour
        
        # CRITICAL: Force close ALL positions at session end (08:00 EST = 13:00 UTC)
        # This prevents positions from running into volatile NY session
        if hour_utc == 13 and self.topstep.positions and self.enable_session_filter:
            logger.warning("⏰ SESSION END at 08:00 EST - Force closing ALL open positions")
            positions_to_force_close = list(self.topstep.positions.keys())
            
            for symbol in positions_to_force_close:
                current_price = self.market_data_cache[symbol]['last_price']
                if current_price > 1e6:
                    current_price = current_price / 1e9
                
                trade_info = self.topstep.close_position(symbol, current_price, 'SESSION_END')
                if trade_info:
                    entry = trade_info['entry_price']
                    direction = trade_info['direction']
                    price_diff = (current_price - entry) if direction == 'LONG' else (entry - current_price)
                    multiplier = self.contract_multipliers.get(symbol, 1)
                    pnl = price_diff * multiplier
                    
                    self.realized_pnl += pnl
                    self.closed_trades.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry': entry,
                        'exit': current_price,
                        'pnl': pnl,
                        'reason': 'SESSION_END',
                        'timestamp': now
                    })
                    self._save_closed_trades()
                    
                    logger.warning(f"⏰ FORCED: {symbol} {direction} @ {current_price:.2f} | PnL: ${pnl:+,.2f}")
            
            logger.warning(f"✅ All positions closed at session end | Total PnL: ${self.realized_pnl:,.2f}")
            return  # Exit early, no need to check normal exits
        
        # Normal position exit checking
        positions_to_close = []
        
        for symbol, position in self.topstep.positions.items():
            current_price = self.market_data_cache[symbol]['last_price']
            
            # Normalize if needed
            if current_price > 1e6:
                current_price = current_price / 1e9
            
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if entry_price > 1e6:
                entry_price = entry_price / 1e9
                stop_loss = stop_loss / 1e9
                take_profit = take_profit / 1e9
            
            direction = position['direction']
            exit_reason = None
            
            # Check for order flow reversal first
            if not exit_reason and self.enable_order_flow_exits:
                market_data = self.market_data_cache[symbol]
                
                # Analyze current order flow
                metrics = self.analyzer.analyze_order_flow(symbol, market_data)
                
                # Check for strong opposing order flow
                opposing_signal = False
                reversal_reasons = []
                
                # For LONG positions, look for strong selling pressure
                if direction == 'LONG':
                    # Strong sell imbalance (opposite of entry)
                    if metrics.imbalance_ratio < -self.orderflow_exit_threshold:
                        opposing_signal = True
                        reversal_reasons.append(f'Strong sell imbalance: {metrics.imbalance_ratio:.2f}')
                    
                    # Negative cumulative delta (persistent selling)
                    if metrics.cumulative_delta < -1000:
                        opposing_signal = True
                        reversal_reasons.append(f'Negative cumulative delta: {metrics.cumulative_delta:.0f}')
                    
                    # Aggressive selling detected
                    if metrics.aggressive_sell_volume > metrics.aggressive_buy_volume * 1.5:
                        opposing_signal = True
                        reversal_reasons.append('Aggressive selling detected')
                
                # For SHORT positions, look for strong buying pressure
                else:  # SHORT
                    # Strong buy imbalance (opposite of entry)
                    if metrics.imbalance_ratio > self.orderflow_exit_threshold:
                        opposing_signal = True
                        reversal_reasons.append(f'Strong buy imbalance: {metrics.imbalance_ratio:.2f}')
                    
                    # Positive cumulative delta (persistent buying)
                    if metrics.cumulative_delta > 1000:
                        opposing_signal = True
                        reversal_reasons.append(f'Positive cumulative delta: {metrics.cumulative_delta:.0f}')
                    
                    # Aggressive buying detected
                    if metrics.aggressive_buy_volume > metrics.aggressive_sell_volume * 1.5:
                        opposing_signal = True
                        reversal_reasons.append('Aggressive buying detected')
                
                # Exit if we detect strong opposing flow
                if opposing_signal and len(reversal_reasons) >= 2:  # Need at least 2 confirming signals
                    exit_reason = f'ORDER_FLOW_REVERSAL: {"; ".join(reversal_reasons)}'
                    logger.info(f"🔄 Order flow reversal detected for {symbol} {direction} position")
            
            # Check traditional exit conditions last
            if direction == 'LONG':
                if current_price <= stop_loss:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= take_profit:
                    exit_reason = 'TAKE_PROFIT'
            else:  # SHORT
                if current_price >= stop_loss:
                    exit_reason = 'STOP_LOSS'
                elif current_price <= take_profit:
                    exit_reason = 'TAKE_PROFIT'

            if exit_reason:
                positions_to_close.append((symbol, current_price, exit_reason))
        
        # Close positions and track PnL
        for symbol, exit_price, exit_reason in positions_to_close:
            trade_info = self.topstep.close_position(symbol, exit_price, exit_reason)
            
            if trade_info:
                # Calculate PnL
                entry = trade_info['entry_price']
                exit = trade_info['exit_price']
                
                if entry > 1e6:
                    entry = entry / 1e9
                if exit > 1e6:
                    exit = exit / 1e9
                
                if trade_info['direction'] == 'LONG':
                    price_diff = exit - entry
                else:
                    price_diff = entry - exit
                
                multiplier = self.contract_multipliers.get(symbol, 1)
                pnl = price_diff * multiplier
                
                # Track realized PnL
                self.realized_pnl += pnl
                self.topstep.publish_mqtt("trading/pnl", self.realized_pnl)
                self.closed_trades.append({
                    'symbol': symbol,
                    'direction': trade_info['direction'],
                    'entry': entry,
                    'exit': exit,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'timestamp': datetime.now(timezone.utc)  # Use UTC time
                })
                
                # Auto-save trades to file
                self._save_closed_trades()
                
                logger.info(f"✅ Position closed: {symbol} {trade_info['direction']} @ {exit:.2f} "
                          f"(Entry: {entry:.2f}) | PnL: ${pnl:+,.2f} | Reason: {exit_reason}")
                logger.info(f"💰 Total Realized PnL: ${self.realized_pnl:,.2f}")

                # ✅ ADD THIS: Start cooldown timer
                self.last_position_close_time[symbol] = datetime.now(timezone.utc)
                cooldown_mins = self.signal_cooldown.total_seconds() / 60
                logger.info(f"⏱️  {symbol}: Cooldown started - no new trades for {cooldown_mins:.0f} minutes")
    
    def _save_closed_trades(self):
        """Save closed trades to JSON file for analysis"""
        try:
            import json
            
            # Convert trades to serializable format
            trades_data = []
            for trade in self.closed_trades:
                trade_copy = trade.copy()
                if isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                trades_data.append(trade_copy)
            
            with open('closed_trades.json', 'w') as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _print_status_update(self):
        """Print periodic status updates showing system is active"""
        now = datetime.now()
        if now - self.last_status_update >= self.status_update_interval:
            # Check for position exits first
            self._check_position_exits()
            
            # Build status message
            status_lines = ["\n" + "="*70]
            status_lines.append(f"📊 Order Flow Analysis Status - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Show trading status
            trading_active = self.should_trade_now()
            status = "🟢 ACTIVE" if trading_active else "🔴 PAUSED (Outside Hours)"
            status_lines.append(f"Trading Status: {status}")
            status_lines.append("="*70)
            
            for symbol in self.symbols:
                data = self.market_data_cache[symbol]
                records = self.record_count[symbol]
                price = data['last_price']
                
                # Normalize price if it's still in raw format
                if price > 1e6:
                    price = price / 1e9
                
                bid_vol = data['bid_volume']
                ask_vol = data['ask_volume']
                trades = len(data['trades'])
                
                if price > 0:
                    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                    cum_delta = self.analyzer.cumulative_delta.get(symbol, 0)
                    
                    status_lines.append(
                        f"{symbol:3s} | Price: {price:8.2f} | "
                        f"Bid: {bid_vol:6.0f} Ask: {ask_vol:6.0f} | "
                        f"Imb: {imbalance:+.2f} | "
                        f"ΔCum: {cum_delta:+8.0f} | "
                        f"Trades: {trades:3d} | "
                        f"Records: {records:5d}"
                    )
                else:
                    status_lines.append(f"{symbol:3s} | Waiting for data...")
            
            status_lines.append("="*70)
            
            # Calculate PnLs
            unrealized_pnls = self._calculate_unrealized_pnl()
            total_unrealized = sum(unrealized_pnls.values())
            total_pnl = self.realized_pnl + total_unrealized
            
            # Show PnL Summary
            status_lines.append("💰 PnL Summary:")
            status_lines.append(f"   Realized PnL:    ${self.realized_pnl:>10,.2f}")
            status_lines.append(f"   Unrealized PnL:  ${total_unrealized:>10,.2f}")
            status_lines.append(f"   Total PnL:       ${total_pnl:>10,.2f}")
            status_lines.append(f"   Open Positions:  {len(self.topstep.positions)}/{len(self.symbols)}")
            
            if len(self.closed_trades) > 0:
                wins = sum(1 for t in self.closed_trades if t['pnl'] > 0)
                losses = len(self.closed_trades) - wins
                win_rate = wins / len(self.closed_trades) * 100 if len(self.closed_trades) > 0 else 0
                status_lines.append(f"   Closed Trades:   {len(self.closed_trades)} (W:{wins} L:{losses} WR:{win_rate:.0f}%)")
            
            status_lines.append("="*70)
            
            # Show open positions with individual PnL
            if self.topstep.positions:
                status_lines.append("📈 Open Positions:")
                for sym, pos in self.topstep.positions.items():
                    entry = pos['entry_price']
                    sl = pos['stop_loss']
                    tp = pos['take_profit']
                    
                    # Normalize if needed
                    if entry > 1e6:
                        entry = entry / 1e9
                        sl = sl / 1e9
                        tp = tp / 1e9
                    
                    # Get current price and PnL
                    current = self.market_data_cache[sym]['last_price']
                    if current > 1e6:
                        current = current / 1e9
                    
                    pnl = unrealized_pnls.get(sym, 0)
                    pnl_str = f"${pnl:+,.2f}" if pnl != 0 else "$0.00"
                    
                    status_lines.append(
                        f"  {sym}: {pos['direction']} @ {entry:.2f} (Now: {current:.2f}) | "
                        f"PnL: {pnl_str} | SL: {sl:.2f} TP: {tp:.2f}"
                    )
            else:
                status_lines.append("📈 Open Positions: None")
            
            status_lines.append("="*70 + "\n")
            
            logger.info("\n".join(status_lines))
            
            self.last_status_update = now
    
    async def analyze_and_trade(self, symbol: str):
        """Analyze order flow and execute trades"""
        try:
            # Check if we're in trading hours
            if not self.should_trade_now():
                return
        
            # Check if cooldown period has passed since last position close
            time_since_close = datetime.now(timezone.utc) - self.last_position_close_time[symbol]
            if time_since_close < self.signal_cooldown:
                remaining = self.signal_cooldown - time_since_close
                logger.debug(f"⏸️  {symbol}: Cooldown active ({remaining.total_seconds()/60:.1f} min remaining)")
                return

            # Get market data
            market_data = self.market_data_cache[symbol]
        
            if market_data['last_price'] == 0:
                return
        
            # Normalize price if needed
            current_price = market_data['last_price']
            if current_price > 1e6:
                current_price = current_price / 1e9
        
            # Analyze order flow
            metrics = self.analyzer.analyze_order_flow(symbol, market_data)
        
            # Generate signal
            signal = self.signal_generator.generate_signal(
                metrics,
                current_price
            )
        
            # If we have a signal, check price confirmation BEFORE modifying stops/targets
            if signal:
                # Import config dynamically to get latest settings
                try:
                    import config
                    require_confirmation = config.REQUIRE_PRICE_CONFIRMATION
                    confirmation_periods = config.PRICE_CONFIRMATION_PERIODS
                    confirmation_threshold = config.PRICE_CONFIRMATION_THRESHOLD
                    use_tight_stop = config.USE_TIGHT_STOP_ON_ANTICIPATORY
                    tight_stop_pct = config.TIGHT_STOP_PERCENTAGE
                except (ImportError, AttributeError):
                    # Defaults if config not available
                    require_confirmation = True
                    confirmation_periods = 5
                    confirmation_threshold = 0.6
                    use_tight_stop = False
                    tight_stop_pct = 0.003
            
                # Check price momentum confirmation
                if require_confirmation:
                    # Get momentum detection method from config
                    try:
                        confirmation_method = config.PRICE_CONFIRMATION_METHOD
                    except AttributeError:
                        confirmation_method = 'net_change'
                    
                    confirmed, reason = self._check_price_momentum(
                        symbol, signal.direction, 
                        confirmation_periods, confirmation_method  # ADD THIS PARAMETER
                    )
                
                    if not confirmed:
                        logger.info(f"⚠️  {symbol} {signal.direction} signal REJECTED: {reason}")
                        logger.info(f"   Orderflow confidence was {signal.confidence:.2f} but price not confirming")
                        return  # Don't trade - price action doesn't confirm signal
                    else:
                        logger.info(f"✅ {symbol} {signal.direction} signal CONFIRMED: {reason}")
            
                # Use tight stop if configured and not using price confirmation
                elif use_tight_stop:
                    tight_stop = self._calculate_tight_stop(
                        symbol, signal.entry_price, signal.direction, tight_stop_pct
                    )
                    logger.info(f"🎯 {symbol}: Using tight anticipatory stop at ${tight_stop:.2f} "
                              f"({tight_stop_pct*100:.1f}% from entry)")
                    signal.stop_loss = tight_stop
                    # Recalculate take profit based on new stop
                    if self.use_liquidity_zones:
                        optimal_target = self.liquidity_tracker.get_optimal_take_profit(
                            symbol, signal.entry_price, signal.direction,
                            risk_reward=2.0, stop_loss=tight_stop
                        )
                        signal.take_profit = optimal_target
        
            # Override stops/targets with liquidity zone-based placement if available
            if signal and self.use_liquidity_zones:
                default_stop_distances = {
                    'GC': 5.0, 'SI': 0.10, 'PL': 10.0, 'HG': 0.02, 'NQ': 20.0
                }
                default_distance = default_stop_distances.get(symbol, 10.0)
            
                # Get optimal stop loss beyond liquidity zone (only if not using tight stop)
                if not (use_tight_stop and not require_confirmation):
                    optimal_stop = self.liquidity_tracker.get_optimal_stop_loss(
                        symbol, signal.entry_price, signal.direction, default_distance
                    )
                    signal.stop_loss = optimal_stop

                # 🆕 ADD THIS: Apply maximum risk cap to prevent excessive losses
                signal.stop_loss = self._apply_max_risk_cap(
                    symbol, signal.entry_price, signal.stop_loss, signal.direction
                )
            
                # Get optimal take profit before liquidity zone
                optimal_target = self.liquidity_tracker.get_optimal_take_profit(
                    symbol, signal.entry_price, signal.direction, 
                    risk_reward=2.0, stop_loss=signal.stop_loss
                )
            
                signal.take_profit = optimal_target
            
                logger.info(f"📍 Using liquidity-zone-based stops/targets for {symbol}")
        
            if signal:
                logger.info(f"Signal generated for {symbol}: {signal.direction} "
                          f"(Confidence: {signal.confidence:.2f})")
                logger.info(f"Reason: {signal.reason}")
            
                # Check if we already have a position for this symbol
                # Use the local positions dictionary, not API call
                has_position = symbol in self.topstep.positions
            
                if has_position:
                    logger.info(f"⚠️  Position already exists for {symbol}, skipping new signal")
                    logger.info(f"   Current position: {self.topstep.positions[symbol]['direction']} "
                              f"@ {self.topstep.positions[symbol]['entry_price']}")
                    return
            
                if not has_position:
                    # Place trade
                    order_id = self.topstep.place_order(signal)
                
                    if order_id:
                        logger.info(f"✅ Trade executed for {symbol}: Order ID {order_id}")
        
        except Exception as e: 
            logger.error(f"Error in analyze_and_trade for {symbol}: {e}")
    
    async def run(self):
        """Main run loop"""
        logger.info("="*70)
        logger.info("🔴 LIVE DATA ONLY MODE - No historical data will be used")
        logger.info("="*70)
        logger.info("Starting Order Flow Trading System")
        logger.info(f"Monitoring symbols: {', '.join(self.symbols)}")
        logger.info("System will only operate during CME Globex market hours")
        logger.info("Sunday 5:00 PM - Friday 4:00 PM Central Time")
        logger.info("="*70)
        
        # Get account info
        account_info = self.topstep.get_account_info()
        logger.info(f"Account info: {account_info}")
        
        try:
            # Start streaming market data
            await self.databento.stream_market_data(
                self.symbols,
                self.process_market_data
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
            raise


async def main():
    """Main entry point"""
    # Import configuration
    try:
        import config
        DATABENTO_API_KEY = config.DATABENTO_API_KEY
        TOPSTEP_API_KEY = config.TOPSTEP_API_KEY
        TOPSTEP_USERNAME = config.TOPSTEP_USERNAME
        TOPSTEP_ACCOUNT_ID = config.TOPSTEP_ACCOUNT_ID
        TOPSTEP_MLL = config.TOPSTEP_MLL
        TOPSTEP_POSITION_SIZE = config.TOPSTEP_POSITION_SIZE
        SIGNAL_COOLDOWN_MINUTES = config.SIGNAL_COOLDOWN_MINUTES
        # Check if paper mode is configured (default to True for safety)
        PAPER_MODE = config.PAPER_MODE
        # Check if order flow exits are enabled (default to True)
        ENABLE_ORDER_FLOW_EXITS = config.ENABLE_ORDER_FLOW_EXITS
        # Check if session filter is enabled (default to True based on analysis)
        ENABLE_SESSION_FILTER = config.ENABLE_SESSION_FILTER
        # New parameters for reducing losing trades
        IMBALANCE_THRESHOLD = config.IMBALANCE_THRESHOLD
        MIN_CONFIDENCE = config.MIN_CONFIDENCE 
        ORDER_FLOW_EXIT_THRESHOLD = config.ORDER_FLOW_EXIT_THRESHOLD
    except ImportError:
        logger.error("config.py not found. Please copy config_template.py to config.py and add your API keys")
        return
    except AttributeError as e:
        logger.error(f"Missing configuration in config.py: {e}")
        return
    
    # Validate configuration
    if "your_" in DATABENTO_API_KEY or "your_" in TOPSTEP_API_KEY:
        logger.error("Please update your API keys in config.py (remove placeholder text)")
        return
    
    if not DATABENTO_API_KEY or not TOPSTEP_API_KEY or not TOPSTEP_ACCOUNT_ID:
        logger.error("API keys cannot be empty. Please configure config.py")
        return
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Trading mode: {'PAPER TRADING' if PAPER_MODE else 'LIVE TRADING'}")
    logger.info(f"Order flow exits: {'ENABLED' if ENABLE_ORDER_FLOW_EXITS else 'DISABLED'}")
    
    # Show trading hours
    logger.info("="*70)
    logger.info("📅 TRADING SCHEDULE")
    logger.info("="*70)
    logger.info("Trading Hours: 18:00 - 08:00 EST (6 PM - 8 AM)")
    logger.info("              23:00 - 13:00 UTC")
    logger.info("Sessions:     Asian + London (No NY/After Hours)")
    logger.info("Blocked:      08:00 - 18:00 EST (NY Session)")
    logger.info("="*70)
    
    if not PAPER_MODE:
        logger.warning("⚠️  LIVE TRADING MODE ENABLED - Real money will be traded!")
        logger.warning("⚠️  Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)
    
    # Initialize and run trading system
    system = OrderFlowTradingSystem(
        databento_api_key=DATABENTO_API_KEY,
        topstep_api_key=TOPSTEP_API_KEY,
        topstep_username=TOPSTEP_USERNAME,
        topstep_account_id=TOPSTEP_ACCOUNT_ID,
        topstep_mll=TOPSTEP_MLL,
        topstep_position_size=TOPSTEP_POSITION_SIZE,
        paper_mode=PAPER_MODE,
        enable_order_flow_exits=ENABLE_ORDER_FLOW_EXITS,
        signal_cooldown_minutes=SIGNAL_COOLDOWN_MINUTES,
        imbalance_threshold=IMBALANCE_THRESHOLD,
        min_confidence=MIN_CONFIDENCE,
        orderflow_exit_threshold=ORDER_FLOW_EXIT_THRESHOLD,
        enable_session_filter=ENABLE_SESSION_FILTER
    )    
    # Set session filter
    system.enable_session_filter = ENABLE_SESSION_FILTER
    
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
