# Configuration file for Order Flow Trading System
# Copy this to config.py and fill in your credentials

# API Credentials
DATABENTO_API_KEY = "db-GN9KE3vmDYRM3TvvjaSuEJH5px99Y"
TOPSTEP_API_KEY = "4SVowkutjEhlr1TIrjm5w3p4SbdrhPEsWna4UVeKt3Q="
TOPSTEP_USERNAME = "tapey"
TOPSTEP_ACCOUNT_ID = "14544972"
TOPSTEP_MLL = 0

# Trading Mode
PAPER_MODE = False  # Set to False for live trading (use with extreme caution!)

# Trading Parameters
SYMBOLS = ['GC', 'SI', 'PL', 'HG']

# Order Flow Analysis Parameters
LOOKBACK_PERIODS = 30  # Number of periods to look back for analysis DEFAULT: 20
IMBALANCE_THRESHOLD = 0.8  # Threshold for order flow imbalance (0-1)
CUMULATIVE_DELTA_THRESHOLD = 1000  # Cumulative delta threshold

# Risk Management
RISK_REWARD_RATIO = 2.0  # Take profit / Stop loss ratio
TOPSTEP_POSITION_SIZE = 1  # Number of contracts per trade

# Price Action Confirmation

REQUIRE_PRICE_CONFIRMATION = True   # Wait for price to move in signal direction
PRICE_CONFIRMATION_PERIODS = 5      # Check last 5 prices
PRICE_CONFIRMATION_THRESHOLD = 0.8  # Need 60% moves confirming direction

# Anticipatory Entries

USE_TIGHT_STOP_ON_ANTICIPATORY = False
TIGHT_STOP_PERCENTAGE = 0.003  # 0.3% stop

# Stop Loss Distances (in points/ticks)
STOP_LOSS_DISTANCES = {
    'GC': 5.0,   # Gold - $5
    'SI': 0.10,  # Silver - $0.10
    'PL': 5.0,  # Platinum - $5
    'HG': 0.02  # Copper - $0.02
    #'ES': 5.0   # S&P 500 - 5 points
}

# Signal Parameters
SIGNAL_COOLDOWN_MINUTES = 2  # Minutes between signals for same symbol DEFAULT: 5
MIN_CONFIDENCE = 0.8  # Minimum confidence level for signal generation
TRADE_INTENSITY_THRESHOLD = 50  # Trades per minute threshold

# Exit Strategy
ENABLE_ORDER_FLOW_EXITS = True  # Exit positions on order flow reversals
ORDER_FLOW_EXIT_THRESHOLD = 0.4  # Imbalance threshold for reversal exits
ORDER_FLOW_EXIT_MIN_SIGNALS = 2  # Minimum confirming signals for reversal exit

# Session Filtering (Based on 150-trade analysis)
ENABLE_SESSION_FILTER = False  # Only trade during profitable hours
# Trading Hours: 18:00 - 08:00 EST (23:00 - 13:00 UTC)
# Covers: Asian Session (full) + London Session (pre-NY open)
# Blocks: NY Session (08:00-18:00 EST) which showed -$2,135 on 66 trades

# Aggressive Trade Detection
AGGRESSIVE_RATIO_THRESHOLD = 0.5  # Threshold for aggressive buy/sell ratio
LEVEL_2_IMBALANCE_THRESHOLD = 0.4  # Order book depth imbalance threshold

# Databento Configuration
DATABENTO_DATASET = 'GLBX.MDP3'  # CME Globex Market Data

# Symbol Mappings (Databento format)
DATABENTO_SYMBOLS = {
    'GC': 'GC.v.0',  # Gold Futures
    'SI': 'SI.v.0',  # Silver Futures
    'PL': 'PL.v.0',  # Platinum Futures
    'HG': 'HG.v.0'  # Copper Futures
    #'ES': 'ES.v.0'   # Nasdaq-100 E-mini Futures
}

# Topstep Configuration
TOPSTEP_BASE_URL = "https://api.topstepx.com/v1"

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "orderflow_trading.log"
