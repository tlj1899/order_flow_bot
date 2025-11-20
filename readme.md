# Order Flow Trading System

An automated trading system that analyzes order flow data from Databento API and executes trades via Topstep API based on detected order flow imbalances.

## Features

- **Real-time Order Flow Analysis**: Monitors GC (Gold), SI (Silver), PL (Platinum), HG (Copper), and NQ (Nasdaq-100) futures
- **Multiple Imbalance Detection Methods**:
  - Bid/Ask volume imbalance ratio
  - Cumulative delta analysis
  - Aggressive buy/sell detection
  - Level 2 order book depth analysis
  - Trade intensity monitoring
- **Automated Trade Execution**: Places trades with stop loss and take profit orders
- **Risk Management**: Built-in position sizing and cooldown periods
- **Confidence-Based Signals**: Only trades when multiple indicators align

## Order Flow Metrics

The system analyzes several key order flow metrics:

1. **Imbalance Ratio**: (Bid Volume - Ask Volume) / Total Volume
2. **Cumulative Delta**: Running sum of volume deltas (buy volume - sell volume)
3. **Aggressive Trades**: Distinguishes between passive and aggressive orders
4. **Level 2 Imbalance**: Order book depth on bid vs ask side
5. **Trade Intensity**: Number of trades per minute

## Signal Generation Logic

Signals are generated when multiple conditions align:

- **Primary**: Order flow imbalance > 60% (configurable)
- **Confirming**: Cumulative delta threshold exceeded
- **Confirming**: Aggressive buying/selling detected
- **Confirming**: Order book depth imbalance
- **Bonus**: High trade intensity

Minimum confidence threshold: 0.6 (signals require 60%+ confidence)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys:
```bash
cp config_template.py config.py
# Edit config.py and add your API keys
```

## Configuration

Edit `config.py` to set:

- **API Credentials**: Databento and Topstep API keys
- **Trading Parameters**: Symbols, position sizes, risk management
- **Signal Thresholds**: Imbalance ratios, delta thresholds, confidence levels
- **Stop Loss Distances**: Per-symbol stop loss distances

### Key Parameters

```python
# Order Flow Thresholds
IMBALANCE_THRESHOLD = 0.6  # 60% imbalance triggers signal
CUMULATIVE_DELTA_THRESHOLD = 1000  # Cumulative delta threshold
MIN_CONFIDENCE = 0.6  # Minimum 60% confidence for trade

# Risk Management
RISK_REWARD_RATIO = 2.0  # 2:1 reward to risk
SIGNAL_COOLDOWN_MINUTES = 5  # Wait 5 minutes between signals
```

## Usage

### Basic Usage

```bash
python orderflow_trading_system.py
```

### Running with Custom Config

```python
from orderflow_trading_system import OrderFlowTradingSystem
import config

system = OrderFlowTradingSystem(
    databento_api_key=config.DATABENTO_API_KEY,
    topstep_api_key=config.TOPSTEP_API_KEY,
    topstep_account_id=config.TOPSTEP_ACCOUNT_ID
)

await system.run()
```

## API Requirements

### Databento API

- Account with historical and live data access
- Access to CME Globex (GLBX.MDP3) dataset
- MBP-1 (Market by Price) and Trades schemas

Website: https://databento.com

### Topstep API

- Active trading account
- API access enabled
- Valid API key and account ID

Website: https://www.topstep.com

**Note**: The Topstep API integration in this script is based on typical REST API patterns. You'll need to verify the actual Topstep API endpoints and authentication methods as their API documentation may differ.

## How It Works

### 1. Data Collection
```
Databento → Stream Market Data → Process Records
           ↓
      MBP-1 Data (Best Bid/Offer)
      Trade Data (Execution prices)
```

### 2. Order Flow Analysis
```
Market Data → Calculate Metrics → Detect Imbalances
            ↓
    - Bid/Ask Imbalance
    - Cumulative Delta
    - Aggressive Trades
    - Level 2 Depth
    - Trade Intensity
```

### 3. Signal Generation
```
Order Flow Metrics → Score Confidence → Generate Signal
                   ↓
    - Direction (LONG/SHORT)
    - Entry Price
    - Stop Loss
    - Take Profit
    - Confidence Score
```

### 4. Trade Execution
```
Signal → Check Position → Place Order → Monitor
       ↓                ↓
   Cooldown Timer   Risk Management
```

## Example Signals

### Long Signal Example
```
Signal: LONG NQ
Confidence: 0.85
Entry: 15,250
Stop Loss: 15,230
Take Profit: 15,290
Reason: Strong buy imbalance: 0.72; 
        Positive cumulative delta: 1,500; 
        Aggressive buying: 0.65
```

### Short Signal Example
```
Signal: SHORT GC
Confidence: 0.75
Entry: 2,050.50
Stop Loss: 2,055.50
Take Profit: 2,040.50
Reason: Strong sell imbalance: -0.68; 
        Negative cumulative delta: -1,200; 
        Order book ask depth: -0.55
```

## Risk Management

The system includes several risk management features:

1. **Position Limits**: Only one position per symbol at a time
2. **Signal Cooldown**: Prevents over-trading (default 5 minutes)
3. **Confidence Threshold**: Only trades high-confidence signals
4. **Stop Loss Orders**: Automatic stop loss on every trade
5. **Take Profit Orders**: Defined profit targets

### Per-Symbol Stop Losses

```python
GC (Gold):     5.0 points  = $5.00
SI (Silver):   0.10 points = $0.10
PL (Platinum): 10.0 points = $10.00
HG (Copper):   0.02 points = $0.02
NQ (Nasdaq):   20.0 points = 20 index points
```

## Monitoring and Logging

The system logs all activity:

- Market data processing
- Order flow analysis
- Signal generation with reasons
- Trade execution
- Errors and warnings

Log file: `orderflow_trading.log`

## Customization

### Adjusting Sensitivity

To make the system more/less aggressive:

```python
# More aggressive (more trades)
IMBALANCE_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.5
SIGNAL_COOLDOWN_MINUTES = 3

# More conservative (fewer trades)
IMBALANCE_THRESHOLD = 0.7
MIN_CONFIDENCE = 0.7
SIGNAL_COOLDOWN_MINUTES = 10
```

### Adding New Symbols

```python
# In config.py
SYMBOLS = ['GC', 'SI', 'PL', 'HG', 'NQ', 'ES']  # Add ES (S&P 500)

DATABENTO_SYMBOLS = {
    # ... existing symbols ...
    'ES': 'ES.FUT'
}

STOP_LOSS_DISTANCES = {
    # ... existing symbols ...
    'ES': 10.0  # 10 points
}
```

## Safety Notes

⚠️ **IMPORTANT WARNINGS**:

1. **Paper Trading First**: Test thoroughly with paper trading before using real money
2. **API Key Security**: Never commit API keys to version control
3. **Position Monitoring**: Always monitor positions manually as a backup
4. **Market Hours**: Ensure you're trading during appropriate market hours
5. **Risk Management**: Never risk more than you can afford to lose
6. **Slippage**: Actual fills may differ from limit prices in fast markets
7. **API Limits**: Be aware of API rate limits and data costs

## Troubleshooting

### Common Issues

**"No market data received"**
- Check Databento API key and permissions
- Verify market is open
- Check symbol format

**"Order placement failed"**
- Verify Topstep API credentials
- Check account balance and margin
- Ensure account is in good standing

**"Too many signals"**
- Increase `MIN_CONFIDENCE` threshold
- Increase `SIGNAL_COOLDOWN_MINUTES`
- Adjust `IMBALANCE_THRESHOLD`

## Performance Optimization

For better performance:

1. **Reduce Lookback Periods**: Lower `LOOKBACK_PERIODS` for faster analysis
2. **Filter Symbols**: Trade only 1-2 symbols instead of all 5
3. **Increase Cooldown**: Reduce trade frequency
4. **Cache Market Data**: Optimize data structure sizes

## License

This code is provided as-is for educational purposes. Use at your own risk.

## Disclaimer

This trading system is for educational and informational purposes only. Trading futures involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always consult with a qualified financial advisor before trading.

The author and contributors are not responsible for any financial losses incurred through the use of this software.
