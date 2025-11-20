# Order Flow Trading System (order_flow_bot)

This module implements a live-only order-flow trading system that:

- Streams MBP-1 (best bid/offer) and trade messages from Databento (`GLBX.MDP3`).
- Computes order-flow metrics (imbalance ratio, cumulative delta, aggressive trades, level-2 depth, trade intensity).
- Generates confidence-scored trade signals and executes trades via the included `topstepapi` client.
- Supports `paper_mode` for simulation and optional liquidity-zone stop/target placement when `liquidity_zones.py` is available.

This README focuses on how to run, configure, and safely operate the code contained in this folder.

## Architecture & data flow

- Databento (live) → `DatabentoClient.stream_market_data()` → `OrderFlowTradingSystem.process_market_data()`
- Market data is cached per symbol in `market_data_cache` (`last_price`, `bid_volume`, `ask_volume`, `trades`, `price_history`).
- `OrderFlowAnalyzer` computes `OrderFlowMetrics` from the cached data.
- `SignalGenerator` scores metrics and returns `TradeSignal` (direction, confidence, entry, SL, TP).
- `TopstepClientClass` places orders via the `topstepapi` package or simulates them when `paper_mode=True`.

Key files

- `orderflow_trading_system.py` — main orchestrator (streaming, analysis, signal generation, execution).
- `config.py` — live configuration; imports credentials from `config_secret.py` in this setup.
- `config_secret.py` — local credentials (present in the folder during development). Do NOT commit this file.
- `topstepapi/` — bundled Topstep client and unit tests.
- `requirements.txt` — Python dependencies.
- `orderflow/` — an embedded virtualenv (do not edit; create your own venv instead).

## Quick start

1) Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure credentials and parameters

- By default this repository expects `config.py` which imports `config_secret.py`. Edit `config_secret.py` (or replace it with environment-backed secrets) to provide:
	- `DATABENTO_API_KEY`
	- `TOPSTEP_API_KEY`
	- `TOPSTEP_USERNAME`
	- `TOPSTEP_ACCOUNT_ID`

- Important safety flag: set `PAPER_MODE = True` (in `config.py`) while testing.

3) Run (paper mode recommended first):

```bash
cd order_flow_bot
python orderflow_trading_system.py
```

## Tests

- `topstepapi/` contains unit tests. Run them from that directory:

```bash
cd order_flow_bot/topstepapi
pytest
```

There are currently no dedicated unit tests for `OrderFlowAnalyzer` or `SignalGenerator` — adding them is recommended.

## Important configuration highlights

- Tunables in `config.py` include: `IMBALANCE_THRESHOLD`, `MIN_CONFIDENCE`, `SIGNAL_COOLDOWN_MINUTES`, `ENABLE_ORDER_FLOW_EXITS`, `ENABLE_SESSION_FILTER`, and price-confirmation settings (`PRICE_CONFIRMATION_METHOD`, `PRICE_CONFIRMATION_PERIODS`).
- The system dynamically imports `config` during signal evaluation so small runtime edits can be picked up without restarting, but a restart is recommended after major changes.

## Security & safety notes (must-read)

- This code interacts with real capital when `PAPER_MODE=False`. Always start with `PAPER_MODE=True`.
- Do not commit `config_secret.py` or actual API keys. Move secrets to environment variables or a secrets manager for production.
- The `TopstepClientClass.publish_mqtt()` method currently contains hard-coded MQTT credentials and a broker address. Treat these as secrets and move them into `config.py` or environment variables before using in a networked environment.
- There are several places where API results are indexed (e.g., `search_contracts(...)[0]`) without full checks — expect to harden these calls for production to avoid IndexErrors.

## Recent fixes & notable implementation details

- Fixed a bug in `TopstepClientClass.close_position()` where the loop searching open positions had an unconditional `break`. The loop now breaks only after a matching contract is found.
- Momentum detection has multiple methods implemented (`net_change`, `ema`, `linear_regression`, `combined`) selectable via `PRICE_CONFIRMATION_METHOD`.
- A max-risk cap feature (`ENABLE_MAX_RISK_CAP`, `MAX_RISK_TICKS`) can adjust stop losses to limit dollar risk per trade.

## Operational behaviour

- Session filtering: when `ENABLE_SESSION_FILTER=True`, the system forces a session-end close at 13:00 UTC (08:00 EST).
- Cooldown: after a position close the symbol is unavailable for new signals for `SIGNAL_COOLDOWN_MINUTES`.
- Only one position per symbol is tracked locally; the system relies on `topstepapi` to align with the broker/account state.

## Troubleshooting

- No market data: check `DATABENTO_API_KEY`, dataset permissions, internet connectivity, and that the instruments are supported by your Databento plan.
- Order placement failed: check `TOPSTEP_API_KEY`, `TOPSTEP_ACCOUNT_ID`, account balance, and API limit/errors returned by `topstepapi`.
- Missing liquidity zones: if `liquidity_zones.py` is absent, the system falls back to default stop distances.

## Recommended next steps (for maintainers)

- Replace `config_secret.py` with environment-variable-based secrets and add a small loader (e.g., `python-dotenv` or os.environ).
- Parameterize MQTT broker and credentials in `config.py` rather than hard-coding.
- Add unit tests for `OrderFlowAnalyzer` and `SignalGenerator` to protect core logic.
- Consider splitting `orderflow_trading_system.py` into smaller modules: `client_databento.py`, `analyzer.py`, `executor.py`, `utils.py` for easier testing.

## Contributing

Open issues or PRs on the parent repository. For changes that affect on-device runtime or install flows, include reproduction steps and indicate whether you tested on a Raspberry Pi environment.

# order_flow_bot
Algorithmic futures trading bot utilizing order flow data, liquidity zone tracking and dynamic targets and stops
