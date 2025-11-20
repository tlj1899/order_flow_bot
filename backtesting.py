"""
Backtesting module for Order Flow Trading System
Tests the strategy on historical Databento data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import json

from orderflow_trading_system import (
    OrderFlowAnalyzer,
    SignalGenerator,
    OrderFlowMetrics,
    TradeSignal
)


@dataclass
class BacktestTrade:
    """Record of a backtested trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_points: float
    exit_reason: str
    confidence: float


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period: timedelta


class OrderFlowBacktester:
    """Backtest order flow trading strategy"""
    
    def __init__(self, analyzer: OrderFlowAnalyzer, 
                 signal_generator: SignalGenerator,
                 initial_capital: float = 100000):
        self.analyzer = analyzer
        self.signal_generator = signal_generator
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.trades: List[BacktestTrade] = []
        self.open_positions: Dict[str, Dict] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Contract specifications (profit per point)
        self.contract_specs = {
            'GC': 100,   # Gold: $100 per point
            'SI': 5000,  # Silver: $5,000 per point
            'PL': 50,    # Platinum: $50 per point
            'HG': 25000, # Copper: $25,000 per point
            'NQ': 20     # Nasdaq: $20 per point
        }
        
    def process_bar(self, bar: pd.Series, symbol: str) -> None:
        """Process a single bar of data"""
        current_time = bar.name if isinstance(bar.name, datetime) else bar['timestamp']
        current_price = bar['close'] if 'close' in bar else bar['price']
        
        # Check for exit conditions on open positions
        self._check_exits(symbol, current_time, current_price)
        
        # Create mock market data for analysis
        market_data = {
            'bid_volume': bar.get('bid_volume', bar.get('bid_sz', 0)),
            'ask_volume': bar.get('ask_volume', bar.get('ask_sz', 0)),
            'trades': [],  # Would need to populate from actual trade data
            'orderbook': {
                'bids': [{'price': bar.get('bid_px', current_price), 
                         'size': bar.get('bid_sz', 0)}],
                'asks': [{'price': bar.get('ask_px', current_price), 
                         'size': bar.get('ask_sz', 0)}]
            }
        }
        
        # Analyze order flow
        metrics = self.analyzer.analyze_order_flow(symbol, market_data)
        
        # Generate signal
        signal = self.signal_generator.generate_signal(metrics, current_price)
        
        # Enter position if signal and no existing position
        if signal and symbol not in self.open_positions:
            self._enter_position(signal, current_time)
        
        # Update equity curve
        self._update_equity(current_time)
    
    def _enter_position(self, signal: TradeSignal, entry_time: datetime) -> None:
        """Enter a new position"""
        self.open_positions[signal.symbol] = {
            'signal': signal,
            'entry_time': entry_time,
            'entry_price': signal.entry_price
        }
        
    def _check_exits(self, symbol: str, current_time: datetime, 
                     current_price: float) -> None:
        """Check if open positions should be exited"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        signal = position['signal']
        entry_price = position['entry_price']
        
        exit_reason = None
        exit_price = current_price
        
        # Check stop loss
        if signal.direction == 'LONG':
            if current_price <= signal.stop_loss:
                exit_reason = 'STOP_LOSS'
                exit_price = signal.stop_loss
            elif current_price >= signal.take_profit:
                exit_reason = 'TAKE_PROFIT'
                exit_price = signal.take_profit
        else:  # SHORT
            if current_price >= signal.stop_loss:
                exit_reason = 'STOP_LOSS'
                exit_price = signal.stop_loss
            elif current_price <= signal.take_profit:
                exit_reason = 'TAKE_PROFIT'
                exit_price = signal.take_profit
        
        # Exit if condition met
        if exit_reason:
            self._exit_position(symbol, current_time, exit_price, exit_reason)
    
    def _exit_position(self, symbol: str, exit_time: datetime,
                       exit_price: float, exit_reason: str) -> None:
        """Exit a position and record the trade"""
        position = self.open_positions[symbol]
        signal = position['signal']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate P&L
        if signal.direction == 'LONG':
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price
        
        # Convert to dollar P&L
        contract_multiplier = self.contract_specs.get(symbol, 1)
        pnl = pnl_points * contract_multiplier
        
        # Update capital
        self.current_capital += pnl
        
        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            pnl=pnl,
            pnl_points=pnl_points,
            exit_reason=exit_reason,
            confidence=signal.confidence
        )
        self.trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[symbol]
    
    def _update_equity(self, current_time: datetime) -> None:
        """Update equity curve"""
        self.equity_curve.append((current_time, self.current_capital))
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> BacktestMetrics:
        """Run backtest on historical data"""
        print("Starting backtest...")
        
        # Combine all data and sort by timestamp
        all_bars = []
        for symbol, df in data.items():
            for idx, bar in df.iterrows():
                all_bars.append((idx if isinstance(idx, datetime) else 
                               bar.get('timestamp'), symbol, bar))
        
        all_bars.sort(key=lambda x: x[0])
        
        # Process each bar
        for timestamp, symbol, bar in all_bars:
            self.process_bar(bar, symbol)
        
        # Close any remaining open positions at end
        for symbol in list(self.open_positions.keys()):
            last_price = data[symbol].iloc[-1]['close']
            last_time = data[symbol].index[-1]
            self._exit_position(symbol, last_time, last_price, 'END_OF_DATA')
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        print(f"\nBacktest complete!")
        print(f"Total trades: {metrics.total_trades}")
        print(f"Win rate: {metrics.win_rate:.2%}")
        print(f"Total P&L: ${metrics.total_pnl:,.2f}")
        
        return metrics
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics"""
        if not self.trades:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_win=0, avg_loss=0,
                largest_win=0, largest_loss=0, profit_factor=0,
                sharpe_ratio=0, max_drawdown=0, 
                avg_holding_period=timedelta(0)
            )
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        # Calculate metrics
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl for t in self.trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252) 
                       if len(returns) > 1 and np.std(returns) > 0 else 0)
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Average holding period
        holding_periods = [t.exit_time - t.entry_time for t in self.trades]
        avg_holding_period = sum(holding_periods, timedelta(0)) / len(holding_periods)
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=self.current_capital - self.initial_capital,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_holding_period=avg_holding_period
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [eq[1] for eq in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def plot_results(self, save_path: str = None) -> None:
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Equity Curve
        if self.equity_curve:
            times, equity = zip(*self.equity_curve)
            axes[0].plot(times, equity, label='Equity', linewidth=2)
            axes[0].axhline(y=self.initial_capital, color='r', 
                          linestyle='--', label='Initial Capital')
            axes[0].set_title('Equity Curve')
            axes[0].set_ylabel('Capital ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Trade P&L Distribution
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            axes[1].bar(range(len(pnls)), pnls, 
                       color=['g' if p > 0 else 'r' for p in pnls])
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_title('Trade P&L Distribution')
            axes[1].set_xlabel('Trade Number')
            axes[1].set_ylabel('P&L ($)')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Win Rate by Symbol
        if self.trades:
            symbol_stats = {}
            for trade in self.trades:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {'wins': 0, 'total': 0}
                symbol_stats[trade.symbol]['total'] += 1
                if trade.pnl > 0:
                    symbol_stats[trade.symbol]['wins'] += 1
            
            symbols = list(symbol_stats.keys())
            win_rates = [symbol_stats[s]['wins'] / symbol_stats[s]['total'] * 100 
                        for s in symbols]
            
            axes[2].bar(symbols, win_rates, color='skyblue')
            axes[2].set_title('Win Rate by Symbol')
            axes[2].set_xlabel('Symbol')
            axes[2].set_ylabel('Win Rate (%)')
            axes[2].axhline(y=50, color='r', linestyle='--', label='50%')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        else:
            plt.show()
    
    def export_trades(self, filepath: str) -> None:
        """Export trades to CSV"""
        if not self.trades:
            print("No trades to export")
            return
        
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        trades_df.to_csv(filepath, index=False)
        print(f"Trades exported to {filepath}")
    
    def generate_report(self, filepath: str = None) -> str:
        """Generate detailed backtest report"""
        metrics = self._calculate_metrics()
        
        report = f"""
{'='*60}
ORDER FLOW TRADING SYSTEM - BACKTEST REPORT
{'='*60}

OVERALL PERFORMANCE
-------------------
Initial Capital:        ${self.initial_capital:,.2f}
Final Capital:          ${self.current_capital:,.2f}
Total P&L:              ${metrics.total_pnl:,.2f}
Return:                 {(metrics.total_pnl/self.initial_capital)*100:.2f}%

TRADE STATISTICS
----------------
Total Trades:           {metrics.total_trades}
Winning Trades:         {metrics.winning_trades}
Losing Trades:          {metrics.losing_trades}
Win Rate:               {metrics.win_rate:.2%}

PROFIT/LOSS ANALYSIS
--------------------
Average Win:            ${metrics.avg_win:,.2f}
Average Loss:           ${metrics.avg_loss:,.2f}
Largest Win:            ${metrics.largest_win:,.2f}
Largest Loss:           ${metrics.largest_loss:,.2f}
Profit Factor:          {metrics.profit_factor:.2f}

RISK METRICS
------------
Sharpe Ratio:           {metrics.sharpe_ratio:.2f}
Max Drawdown:           {metrics.max_drawdown:.2%}
Avg Holding Period:     {metrics.avg_holding_period}

PER-SYMBOL BREAKDOWN
--------------------
"""
        # Add per-symbol statistics
        symbol_stats = {}
        for trade in self.trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {
                    'trades': 0, 'wins': 0, 'pnl': 0
                }
            symbol_stats[trade.symbol]['trades'] += 1
            if trade.pnl > 0:
                symbol_stats[trade.symbol]['wins'] += 1
            symbol_stats[trade.symbol]['pnl'] += trade.pnl
        
        for symbol, stats in symbol_stats.items():
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            report += f"{symbol:5s} | Trades: {stats['trades']:3d} | "
            report += f"Win Rate: {wr:5.1f}% | P&L: ${stats['pnl']:10,.2f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"Report saved to {filepath}")
        
        return report


def main():
    """Example usage of backtesting module"""
    import databento as db
    
    # Initialize components
    analyzer = OrderFlowAnalyzer(lookback_periods=20)
    signal_generator = SignalGenerator(
        imbalance_threshold=0.6,
        cumulative_delta_threshold=1000
    )
    
    backtester = OrderFlowBacktester(
        analyzer=analyzer,
        signal_generator=signal_generator,
        initial_capital=100000
    )
    
    # Example: Load historical data (you need to implement data loading)
    # For demonstration, we'll create dummy data structure
    print("Loading historical data...")
    
    # In practice, you'd load from Databento:
    # client = db.Historical("YOUR_API_KEY")
    # data = client.timeseries.get_range(...)
    
    # Dummy data structure for example
    symbols = ['GC', 'SI', 'PL', 'HG', 'NQ']
    historical_data = {}
    
    # You would populate this with actual Databento data
    # historical_data[symbol] = dataframe with columns:
    # timestamp, bid_px, ask_px, bid_sz, ask_sz, close, etc.
    
    # Run backtest
    # metrics = backtester.run_backtest(historical_data)
    
    # Generate visualizations and reports
    # backtester.plot_results('backtest_results.png')
    # backtester.export_trades('trades.csv')
    # report = backtester.generate_report('backtest_report.txt')
    # print(report)
    
    print("\nBacktesting module loaded successfully!")
    print("To run a backtest, provide historical data and call run_backtest()")


if __name__ == "__main__":
    main()
