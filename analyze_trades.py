#!/usr/bin/env python3
"""
Quick Trade Analysis Script
Run this after your trading session to diagnose performance
"""

from trade_analyzer import TradeAnalyzer
import sys

def main():
    print("="*80)
    print("TRADE ANALYSIS TOOL")
    print("="*80)
    print()
    
    # Load trades
    analyzer = TradeAnalyzer()
    
    try:
        analyzer.load_from_file('closed_trades.json')
        print(f"✓ Loaded {len(analyzer.trades)} trades from closed_trades.json")
        print()
    except FileNotFoundError:
        print("✗ No closed_trades.json file found!")
        print("  Make sure you've run the trading system first.")
        sys.exit(1)
    
    # Generate full report
    print("Generating comprehensive analysis report...")
    report = analyzer.generate_report('trade_analysis_report.txt')
    print()
    print(report)
    
    # Generate charts
    print("\nGenerating performance charts...")
    try:
        analyzer.plot_performance('trade_analysis.png')
        print("✓ Charts saved to trade_analysis.png")
    except Exception as e:
        print(f"✗ Error generating charts: {e}")
    
    print()
    print("="*80)
    print("KEY RECOMMENDATIONS:")
    print("="*80)
    
    # Analyze and give recommendations
    df = analyzer.analyze_by_symbol()
    
    # Find worst performer
    worst_symbol = df.iloc[-1]
    if worst_symbol['Total PnL'] < -500:
        print(f"\n🚨 DISABLE {worst_symbol['Symbol']}!")
        print(f"   Lost ${abs(worst_symbol['Total PnL']):.0f} with {worst_symbol['Win Rate']:.0f}% win rate")
        print(f"   Add to config.py: '{worst_symbol['Symbol']}': {{'enabled': False}}")
    
    # Check time analysis
    time_df = analyzer.analyze_by_time_of_day()
    worst_hours = time_df[time_df['Total PnL'] < -500].sort_values('Total PnL')
    
    if len(worst_hours) > 0:
        print(f"\n🚨 AVOID THESE HOURS (UTC):")
        for _, row in worst_hours.iterrows():
            hour = int(row['Hour (UTC)'])  # Convert to int for formatting
            print(f"   Hour {hour:02d}:00 - Lost ${abs(row['Total PnL']):.0f}")
        print("\n   ✅ SESSION FILTER NOW IMPLEMENTED!")
        print("   Trading restricted to 18:00-08:00 EST (23:00-13:00 UTC)")
        print("   This should eliminate most of these losses.")
    
    # Check direction bias
    dir_df = analyzer.analyze_by_direction()
    if len(dir_df) == 2:
        long_pnl = dir_df[dir_df['Direction'] == 'LONG']['Total PnL'].values[0]
        short_pnl = dir_df[dir_df['Direction'] == 'SHORT']['Total PnL'].values[0]
        
        if long_pnl < -500 and short_pnl > 0:
            print(f"\n🚨 LONG TRADES LOSING!")
            print(f"   LONG: ${long_pnl:.0f} | SHORT: ${short_pnl:.0f}")
            print("   Consider trading SHORT only")
        elif short_pnl < -500 and long_pnl > 0:
            print(f"\n🚨 SHORT TRADES LOSING!")
            print(f"   SHORT: ${short_pnl:.0f} | LONG: ${long_pnl:.0f}")
            print("   Consider trading LONG only")
    
    print()
    print("="*80)
    print("Full report saved to: trade_analysis_report.txt")
    print("Charts saved to: trade_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()
