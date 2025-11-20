"""
Trade Analysis Module
Analyzes trading performance to identify patterns in wins/losses and per-symbol performance
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns

class TradeAnalyzer:
    """Comprehensive trade analysis and diagnostics"""
    
    def __init__(self):
        self.trades = []
        
    def load_trades(self, trades_list: List[Dict]) -> None:
        """Load trades from the trading system"""
        self.trades = trades_list
        
    def load_from_file(self, filepath: str) -> None:
        """Load trades from JSON file"""
        with open(filepath, 'r') as f:
            self.trades = json.load(f)
    
    def save_to_file(self, filepath: str) -> None:
        """Save trades to JSON file"""
        # Convert datetime objects to strings
        trades_serializable = []
        for trade in self.trades:
            trade_copy = trade.copy()
            if isinstance(trade_copy.get('timestamp'), datetime):
                trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
            trades_serializable.append(trade_copy)
        
        with open(filepath, 'w') as f:
            json.dump(trades_serializable, f, indent=2)
    
    def analyze_by_symbol(self) -> pd.DataFrame:
        """Analyze performance by symbol"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        analysis = []
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol]
            
            wins = symbol_trades[symbol_trades['pnl'] > 0]
            losses = symbol_trades[symbol_trades['pnl'] <= 0]
            
            analysis.append({
                'Symbol': symbol,
                'Total Trades': len(symbol_trades),
                'Wins': len(wins),
                'Losses': len(losses),
                'Win Rate': len(wins) / len(symbol_trades) * 100 if len(symbol_trades) > 0 else 0,
                'Total PnL': symbol_trades['pnl'].sum(),
                'Avg Win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'Avg Loss': losses['pnl'].mean() if len(losses) > 0 else 0,
                'Largest Win': wins['pnl'].max() if len(wins) > 0 else 0,
                'Largest Loss': losses['pnl'].min() if len(losses) > 0 else 0,
                'Profit Factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
            })
        
        return pd.DataFrame(analysis).sort_values('Total PnL', ascending=False)
    
    def analyze_by_direction(self) -> pd.DataFrame:
        """Analyze performance by trade direction (LONG vs SHORT)"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        analysis = []
        for direction in df['direction'].unique():
            dir_trades = df[df['direction'] == direction]
            
            wins = dir_trades[dir_trades['pnl'] > 0]
            losses = dir_trades[dir_trades['pnl'] <= 0]
            
            analysis.append({
                'Direction': direction,
                'Total Trades': len(dir_trades),
                'Wins': len(wins),
                'Losses': len(losses),
                'Win Rate': len(wins) / len(dir_trades) * 100 if len(dir_trades) > 0 else 0,
                'Total PnL': dir_trades['pnl'].sum(),
                'Avg PnL': dir_trades['pnl'].mean(),
                'Profit Factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
            })
        
        return pd.DataFrame(analysis)
    
    def analyze_by_exit_reason(self) -> pd.DataFrame:
        """Analyze performance by exit reason"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        analysis = []
        for reason in df['reason'].unique():
            reason_trades = df[df['reason'] == reason]
            
            wins = reason_trades[reason_trades['pnl'] > 0]
            
            analysis.append({
                'Exit Reason': reason,
                'Count': len(reason_trades),
                'Wins': len(wins),
                'Win Rate': len(wins) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0,
                'Total PnL': reason_trades['pnl'].sum(),
                'Avg PnL': reason_trades['pnl'].mean()
            })
        
        return pd.DataFrame(analysis).sort_values('Total PnL', ascending=False)
    
    def analyze_by_time_of_day(self) -> pd.DataFrame:
        """Analyze performance by hour of day"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamp strings to datetime if needed
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        
        analysis = []
        for hour in sorted(df['hour'].unique()):
            hour_trades = df[df['hour'] == hour]
            
            wins = hour_trades[hour_trades['pnl'] > 0]
            
            analysis.append({
                'Hour (UTC)': hour,
                'Trades': len(hour_trades),
                'Wins': len(wins),
                'Win Rate': len(wins) / len(hour_trades) * 100 if len(hour_trades) > 0 else 0,
                'Total PnL': hour_trades['pnl'].sum(),
                'Avg PnL': hour_trades['pnl'].mean()
            })
        
        return pd.DataFrame(analysis)
    
    def identify_losing_patterns(self) -> Dict:
        """Identify common patterns in losing trades"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        losses = df[df['pnl'] < 0]
        
        if len(losses) == 0:
            return {"message": "No losing trades found"}
        
        patterns = {
            'total_losses': len(losses),
            'total_loss_amount': losses['pnl'].sum(),
            'avg_loss': losses['pnl'].mean(),
            'worst_symbol': losses.groupby('symbol')['pnl'].sum().idxmin(),
            'worst_symbol_loss': losses.groupby('symbol')['pnl'].sum().min(),
            'worst_direction': losses.groupby('direction')['pnl'].sum().idxmin() if 'direction' in losses.columns else None,
            'most_common_exit': losses['reason'].mode()[0] if 'reason' in losses.columns and len(losses) > 0 else None,
            'losses_by_symbol': losses.groupby('symbol').size().to_dict(),
            'avg_loss_by_symbol': losses.groupby('symbol')['pnl'].mean().to_dict()
        }
        
        # Time-based analysis
        if isinstance(losses['timestamp'].iloc[0], str):
            losses['timestamp'] = pd.to_datetime(losses['timestamp'])
        
        losses['hour'] = losses['timestamp'].dt.hour
        patterns['worst_hour'] = losses.groupby('hour')['pnl'].sum().idxmin()
        patterns['worst_hour_loss'] = losses.groupby('hour')['pnl'].sum().min()
        
        return patterns
    
    def calculate_session_performance(self) -> pd.DataFrame:
        """Analyze performance by trading session"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamp strings to datetime if needed
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        
        # Define sessions (UTC times)
        # Asian: 23:00-08:00 UTC
        # London: 07:00-16:00 UTC  
        # New York: 13:00-22:00 UTC
        
        def get_session(hour):
            if 23 <= hour or hour < 7:
                return 'Asian'
            elif 7 <= hour < 13:
                return 'London'
            elif 13 <= hour < 20:
                return 'New York'
            else:
                return 'After Hours'
        
        df['session'] = df['hour'].apply(get_session)
        
        analysis = []
        for session in ['Asian', 'London', 'New York', 'After Hours']:
            session_trades = df[df['session'] == session]
            
            if len(session_trades) == 0:
                continue
            
            wins = session_trades[session_trades['pnl'] > 0]
            
            analysis.append({
                'Session': session,
                'Trades': len(session_trades),
                'Wins': len(wins),
                'Win Rate': len(wins) / len(session_trades) * 100,
                'Total PnL': session_trades['pnl'].sum(),
                'Avg PnL': session_trades['pnl'].mean()
            })
        
        return pd.DataFrame(analysis)
    
    def generate_report(self, output_file: str = 'trade_analysis_report.txt') -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE TRADE ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")
        
        # Overall statistics
        df = pd.DataFrame(self.trades)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        report.append("OVERALL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Total Trades:        {len(df)}")
        report.append(f"Winning Trades:      {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
        report.append(f"Losing Trades:       {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
        report.append(f"Total PnL:           ${df['pnl'].sum():,.2f}")
        report.append(f"Average Win:         ${wins['pnl'].mean():,.2f}" if len(wins) > 0 else "Average Win:         N/A")
        report.append(f"Average Loss:        ${losses['pnl'].mean():,.2f}" if len(losses) > 0 else "Average Loss:        N/A")
        report.append(f"Largest Win:         ${wins['pnl'].max():,.2f}" if len(wins) > 0 else "Largest Win:         N/A")
        report.append(f"Largest Loss:        ${losses['pnl'].min():,.2f}" if len(losses) > 0 else "Largest Loss:        N/A")
        
        if len(wins) > 0 and len(losses) > 0 and losses['pnl'].sum() != 0:
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
            report.append(f"Profit Factor:       {profit_factor:.2f}")
        
        report.append("")
        
        # By Symbol
        report.append("PERFORMANCE BY SYMBOL")
        report.append("-"*80)
        symbol_analysis = self.analyze_by_symbol()
        for _, row in symbol_analysis.iterrows():
            report.append(f"\n{row['Symbol']}:")
            report.append(f"  Trades: {row['Total Trades']} | Win Rate: {row['Win Rate']:.1f}% | PnL: ${row['Total PnL']:,.2f}")
            report.append(f"  Avg Win: ${row['Avg Win']:,.2f} | Avg Loss: ${row['Avg Loss']:,.2f}")
            report.append(f"  Profit Factor: {row['Profit Factor']:.2f}")
        
        report.append("")
        
        # By Direction
        report.append("PERFORMANCE BY DIRECTION")
        report.append("-"*80)
        direction_analysis = self.analyze_by_direction()
        report.append(direction_analysis.to_string(index=False))
        report.append("")
        
        # By Exit Reason
        report.append("PERFORMANCE BY EXIT REASON")
        report.append("-"*80)
        exit_analysis = self.analyze_by_exit_reason()
        report.append(exit_analysis.to_string(index=False))
        report.append("")
        
        # By Session
        report.append("PERFORMANCE BY TRADING SESSION")
        report.append("-"*80)
        session_analysis = self.calculate_session_performance()
        report.append(session_analysis.to_string(index=False))
        report.append("")
        
        # Losing Patterns
        report.append("LOSING TRADE PATTERNS")
        report.append("-"*80)
        patterns = self.identify_losing_patterns()
        for key, value in patterns.items():
            if key == 'losses_by_symbol':
                report.append(f"\nLosses by Symbol:")
                for sym, count in value.items():
                    report.append(f"  {sym}: {count} losses")
            elif key == 'avg_loss_by_symbol':
                report.append(f"\nAvg Loss by Symbol:")
                for sym, avg in value.items():
                    report.append(f"  {sym}: ${avg:,.2f}")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value}")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def plot_performance(self, output_file: str = 'trade_analysis.png'):
        """Generate performance visualization charts"""
        if not self.trades:
            print("No trades to plot")
            return
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamp strings to datetime if needed
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trade Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative PnL
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['pnl'].cumsum()
        axes[0, 0].plot(df.index, df['cumulative_pnl'], linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Cumulative PnL Over Time')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Cumulative PnL ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PnL by Symbol
        symbol_pnl = df.groupby('symbol')['pnl'].sum().sort_values()
        colors = ['red' if x < 0 else 'green' for x in symbol_pnl.values]
        axes[0, 1].barh(symbol_pnl.index, symbol_pnl.values, color=colors)
        axes[0, 1].set_title('Total PnL by Symbol')
        axes[0, 1].set_xlabel('PnL ($)')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Win Rate by Symbol
        symbol_stats = df.groupby('symbol').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).sort_values(ascending=False)
        axes[0, 2].bar(symbol_stats.index, symbol_stats.values, color='skyblue')
        axes[0, 2].axhline(y=50, color='r', linestyle='--', label='50%')
        axes[0, 2].set_title('Win Rate by Symbol')
        axes[0, 2].set_ylabel('Win Rate (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. PnL Distribution
        axes[1, 0].hist(df['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('PnL Distribution')
        axes[1, 0].set_xlabel('PnL ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Performance by Hour
        df['hour'] = df['timestamp'].dt.hour
        hourly_pnl = df.groupby('hour')['pnl'].sum()
        axes[1, 1].bar(hourly_pnl.index, hourly_pnl.values, 
                       color=['red' if x < 0 else 'green' for x in hourly_pnl.values])
        axes[1, 1].set_title('PnL by Hour of Day (UTC)')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Total PnL ($)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Win/Loss Streak Analysis
        df['win'] = df['pnl'] > 0
        axes[1, 2].scatter(df.index, df['pnl'], 
                          c=df['win'].map({True: 'green', False: 'red'}), 
                          alpha=0.6)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 2].set_title('Individual Trade PnL')
        axes[1, 2].set_xlabel('Trade Number')
        axes[1, 2].set_ylabel('PnL ($)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Performance charts saved to {output_file}")
        
        return fig


def main():
    """Example usage"""
    analyzer = TradeAnalyzer()
    
    # Example: Load from file
    # analyzer.load_from_file('closed_trades.json')
    
    # Generate report
    # report = analyzer.generate_report('trade_analysis_report.txt')
    # print(report)
    
    # Generate charts
    # analyzer.plot_performance('trade_analysis.png')
    
    print("Trade analyzer module loaded successfully!")
    print("Usage:")
    print("  analyzer = TradeAnalyzer()")
    print("  analyzer.load_from_file('closed_trades.json')")
    print("  analyzer.generate_report()")
    print("  analyzer.plot_performance()")


if __name__ == "__main__":
    main()
