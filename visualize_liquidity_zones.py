"""
Liquidity Zone Visualization System

Creates comprehensive graphical analysis of detected liquidity zones
and observed price behavior around each zone across all historical data.
"""

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f0f0f0'


class LiquidityZoneVisualizer:
    """Visualize liquidity zones and price behavior"""
    
    def __init__(self, db_path: str = 'liquidity_zones.db'):
        self.db_path = db_path
        self.zones_by_symbol = defaultdict(list)
        self.interactions_by_symbol = defaultdict(list)
        
    def load_data(self):
        """Load all zones and interactions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load zones
                cursor = conn.execute('''
                    SELECT symbol, price, zone_type, confidence, hit_count, 
                           rejection_count, breakthrough_count, consolidation_count,
                           tends_to_reject_first, tends_to_consolidate, tends_to_melt_through
                    FROM liquidity_zones
                    ORDER BY symbol, price
                ''')
                
                for row in cursor:
                    self.zones_by_symbol[row[0]].append({
                        'price': row[1],
                        'zone_type': row[2],
                        'confidence': row[3],
                        'hit_count': row[4],
                        'rejection_count': row[5],
                        'breakthrough_count': row[6],
                        'consolidation_count': row[7],
                        'tends_to_reject_first': bool(row[8]),
                        'tends_to_consolidate': bool(row[9]),
                        'tends_to_melt_through': bool(row[10])
                    })
                
                # Load interactions
                cursor = conn.execute('''
                    SELECT symbol, price, interaction_type, timestamp,
                           price_before, price_after
                    FROM zone_interactions
                    ORDER BY timestamp
                ''')
                
                for row in cursor:
                    self.interactions_by_symbol[row[0]].append({
                        'price': row[1],
                        'interaction_type': row[2],
                        'timestamp': datetime.fromisoformat(row[3]),
                        'price_before': row[4],
                        'price_after': row[5]
                    })
                
                print(f"Loaded {sum(len(z) for z in self.zones_by_symbol.values())} zones")
                print(f"Loaded {sum(len(i) for i in self.interactions_by_symbol.values())} interactions")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def plot_symbol_zones(self, symbol: str, ax: plt.Axes):
        """Plot liquidity zones for a single symbol"""
        zones = self.zones_by_symbol.get(symbol, [])
        
        if not zones:
            ax.text(0.5, 0.5, f'No zones detected for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        # Sort zones by price
        zones = sorted(zones, key=lambda z: z['price'])
        
        prices = [z['price'] for z in zones]
        confidences = [z['confidence'] for z in zones]
        hit_counts = [z['hit_count'] for z in zones]
        
        # Color by zone type
        colors = []
        for z in zones:
            if z['zone_type'] == 'round_number':
                colors.append('#3498db')  # Blue
            elif z['zone_type'] == 'session_level':
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#95a5a6')  # Gray
        
        # Create scatter plot with size based on hit count
        sizes = [max(50, min(500, h * 10)) for h in hit_counts]
        scatter = ax.scatter(confidences, prices, s=sizes, c=colors, 
                           alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add labels for high-confidence zones
        for zone in zones:
            if zone['confidence'] > 0.8 or zone['hit_count'] > 5:
                ax.annotate(f"${zone['price']:.2f}", 
                          xy=(zone['confidence'], zone['price']),
                          xytext=(5, 0), textcoords='offset points',
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Price Level', fontsize=10)
        ax.set_title(f'{symbol} Liquidity Zones\n({len(zones)} zones detected)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        round_patch = mpatches.Patch(color='#3498db', label='Round Number')
        session_patch = mpatches.Patch(color='#e74c3c', label='Session Level')
        ax.legend(handles=[round_patch, session_patch], loc='upper left', fontsize=8)
    
    def plot_zone_behavior(self, symbol: str, ax: plt.Axes):
        """Plot zone behavior statistics"""
        zones = self.zones_by_symbol.get(symbol, [])
        
        if not zones:
            ax.text(0.5, 0.5, f'No behavior data for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        # Calculate behavior statistics
        behavior_data = {
            'Rejections': sum(z['rejection_count'] for z in zones),
            'Breakthroughs': sum(z['breakthrough_count'] for z in zones),
            'Consolidations': sum(z['consolidation_count'] for z in zones)
        }
        
        # Create bar chart
        behaviors = list(behavior_data.keys())
        counts = list(behavior_data.values())
        colors_bar = ['#2ecc71', '#e74c3c', '#f39c12']
        
        bars = ax.bar(behaviors, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{symbol} Zone Behavior\n(All Historical Data)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = sum(counts)
        if total > 0:
            for i, (behavior, count) in enumerate(zip(behaviors, counts)):
                pct = (count / total) * 100
                ax.text(i, -max(counts) * 0.1, f'{pct:.1f}%', 
                       ha='center', fontsize=9, style='italic')
    
    def plot_zone_heatmap(self, symbol: str, ax: plt.Axes):
        """Plot heatmap of zone interactions over time"""
        interactions = self.interactions_by_symbol.get(symbol, [])
        
        if not interactions or len(interactions) < 5:
            ax.text(0.5, 0.5, f'Insufficient interaction data for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        # Create price bins
        prices = [i['price'] for i in interactions]
        price_min, price_max = min(prices), max(prices)
        price_range = price_max - price_min
        
        if price_range == 0:
            ax.text(0.5, 0.5, 'All interactions at same price', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Create bins
        n_bins = min(20, len(set(prices)))
        bins = np.linspace(price_min, price_max, n_bins + 1)
        
        # Count interactions per bin
        interaction_counts = np.zeros(n_bins)
        for price in prices:
            bin_idx = min(np.searchsorted(bins, price) - 1, n_bins - 1)
            if bin_idx >= 0:
                interaction_counts[bin_idx] += 1
        
        # Create bar chart
        bin_centers = (bins[:-1] + bins[1:]) / 2
        colors_heat = plt.cm.YlOrRd(interaction_counts / max(interaction_counts))
        
        ax.barh(bin_centers, interaction_counts, height=price_range/n_bins * 0.8,
               color=colors_heat, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Interaction Count', fontsize=10)
        ax.set_ylabel('Price Level', fontsize=10)
        ax.set_title(f'{symbol} Price Interaction Density\n({len(interactions)} total interactions)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def plot_confidence_distribution(self, symbol: str, ax: plt.Axes):
        """Plot distribution of zone confidences"""
        zones = self.zones_by_symbol.get(symbol, [])
        
        if not zones:
            ax.text(0.5, 0.5, f'No confidence data for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        confidences = [z['confidence'] for z in zones]
        
        # Create histogram
        ax.hist(confidences, bins=10, range=(0, 1), color='#3498db', 
               alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add mean line
        mean_conf = np.mean(confidences)
        ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_conf:.2f}')
        
        # Add median line
        median_conf = np.median(confidences)
        ax.axvline(median_conf, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_conf:.2f}')
        
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Number of Zones', fontsize=10)
        ax.set_title(f'{symbol} Zone Confidence Distribution\n({len(zones)} zones)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    def plot_zone_timeline(self, symbol: str, ax: plt.Axes):
        """Plot when zones were interacted with over time"""
        interactions = self.interactions_by_symbol.get(symbol, [])
        
        if not interactions:
            ax.text(0.5, 0.5, f'No timeline data for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        # Group interactions by day
        daily_counts = defaultdict(int)
        for interaction in interactions:
            day = interaction['timestamp'].date()
            daily_counts[day] += 1
        
        if not daily_counts:
            ax.text(0.5, 0.5, 'No daily data available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Sort by date
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in dates]
        
        # Plot
        ax.plot(dates, counts, marker='o', linestyle='-', linewidth=2,
               markersize=6, color='#3498db', markerfacecolor='#e74c3c')
        
        # Fill area under curve
        ax.fill_between(dates, counts, alpha=0.3, color='#3498db')
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Interactions', fontsize=10)
        ax.set_title(f'{symbol} Zone Interactions Over Time', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_behavior_patterns(self, symbol: str, ax: plt.Axes):
        """Plot behavioral pattern percentages"""
        zones = self.zones_by_symbol.get(symbol, [])
        
        if not zones:
            ax.text(0.5, 0.5, f'No pattern data for {symbol}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{symbol} - No Data')
            return
        
        # Count behavioral patterns
        patterns = {
            'Rejects First': sum(1 for z in zones if z['tends_to_reject_first']),
            'Consolidates': sum(1 for z in zones if z['tends_to_consolidate']),
            'Melts Through': sum(1 for z in zones if z['tends_to_melt_through']),
            'No Pattern': sum(1 for z in zones if not (z['tends_to_reject_first'] or 
                                                        z['tends_to_consolidate'] or 
                                                        z['tends_to_melt_through']))
        }
        
        # Filter out zero values
        patterns = {k: v for k, v in patterns.items() if v > 0}
        
        if not patterns:
            ax.text(0.5, 0.5, 'No patterns detected yet', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Create pie chart
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
        explode = [0.05 if v == max(patterns.values()) else 0 for v in patterns.values()]
        
        wedges, texts, autotexts = ax.pie(patterns.values(), labels=patterns.keys(),
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_pie, explode=explode,
                                           shadow=True)
        
        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(f'{symbol} Behavioral Patterns\n({len(zones)} zones analyzed)', 
                    fontsize=12, fontweight='bold')
    
    def create_comprehensive_report(self, output_file: str = 'liquidity_zones_analysis.png'):
        """Create comprehensive multi-page visualization"""
        
        if not self.load_data():
            print("Failed to load data from database")
            return
        
        symbols = sorted(self.zones_by_symbol.keys())
        
        if not symbols:
            print("No symbols with zone data found")
            return
        
        # Create multi-page figure (one page per symbol)
        for symbol in symbols:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle(f'{symbol} Liquidity Zone Analysis - Complete Historical Data', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Create grid layout
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                         left=0.05, right=0.95, top=0.93, bottom=0.05)
            
            # Top row - Main visualizations
            ax1 = fig.add_subplot(gs[0, :2])  # Zone scatter plot (2 columns)
            ax2 = fig.add_subplot(gs[0, 2])   # Behavior bars
            
            # Middle row
            ax3 = fig.add_subplot(gs[1, 0])   # Interaction heatmap
            ax4 = fig.add_subplot(gs[1, 1])   # Confidence distribution
            ax5 = fig.add_subplot(gs[1, 2])   # Timeline
            
            # Bottom row
            ax6 = fig.add_subplot(gs[2, 0])   # Behavioral patterns pie
            ax7 = fig.add_subplot(gs[2, 1:])  # Summary statistics
            
            # Generate plots
            self.plot_symbol_zones(symbol, ax1)
            self.plot_zone_behavior(symbol, ax2)
            self.plot_zone_heatmap(symbol, ax3)
            self.plot_confidence_distribution(symbol, ax4)
            self.plot_zone_timeline(symbol, ax5)
            self.plot_behavior_patterns(symbol, ax6)
            self.plot_summary_stats(symbol, ax7)
            
            # Save individual symbol figure
            symbol_file = output_file.replace('.png', f'_{symbol}.png')
            plt.savefig(symbol_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved {symbol_file}")
            
            plt.close()
        
        # Create summary page with all symbols
        self.create_summary_page(symbols, output_file.replace('.png', '_summary.png'))
        
        print(f"\n✓ All visualizations complete!")
    
    def plot_summary_stats(self, symbol: str, ax: plt.Axes):
        """Plot summary statistics table"""
        zones = self.zones_by_symbol.get(symbol, [])
        interactions = self.interactions_by_symbol.get(symbol, [])
        
        # Calculate statistics
        stats = {
            'Total Zones': len(zones),
            'Round Number Zones': sum(1 for z in zones if z['zone_type'] == 'round_number'),
            'Session Level Zones': sum(1 for z in zones if z['zone_type'] == 'session_level'),
            'Average Confidence': f"{np.mean([z['confidence'] for z in zones]):.2f}" if zones else "0.00",
            'Total Interactions': len(interactions),
            'Total Rejections': sum(z['rejection_count'] for z in zones),
            'Total Breakthroughs': sum(z['breakthrough_count'] for z in zones),
            'Most Hit Zone': f"${max((z['price'] for z in zones), key=lambda p: next(z['hit_count'] for z in zones if z['price'] == p)):.2f} ({max(z['hit_count'] for z in zones)} hits)" if zones else "N/A",
            'Highest Confidence Zone': f"${max((z['price'] for z in zones), key=lambda p: next(z['confidence'] for z in zones if z['price'] == p)):.2f} ({max(z['confidence'] for z in zones):.2f})" if zones else "N/A"
        }
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[key, str(value)] for key, value in stats.items()]
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        ax.set_title(f'{symbol} Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    def create_summary_page(self, symbols: List[str], output_file: str):
        """Create summary comparison page across all symbols"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Liquidity Zone Analysis - All Symbols Summary', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Total zones by symbol
        ax1 = axes[0, 0]
        zone_counts = [len(self.zones_by_symbol[s]) for s in symbols]
        ax1.bar(symbols, zone_counts, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Zones')
        ax1.set_title('Total Zones Detected per Symbol')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average confidence by symbol
        ax2 = axes[0, 1]
        avg_confidences = []
        for s in symbols:
            zones = self.zones_by_symbol[s]
            avg_conf = np.mean([z['confidence'] for z in zones]) if zones else 0
            avg_confidences.append(avg_conf)
        ax2.bar(symbols, avg_confidences, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Zone Confidence by Symbol')
        ax2.set_ylim(0, 1)
        ax2.axhline(0.7, color='red', linestyle='--', label='Good Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Total interactions by symbol
        ax3 = axes[1, 0]
        interaction_counts = [len(self.interactions_by_symbol[s]) for s in symbols]
        ax3.bar(symbols, interaction_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Interactions')
        ax3.set_title('Total Zone Interactions per Symbol')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Rejection rate by symbol
        ax4 = axes[1, 1]
        rejection_rates = []
        for s in symbols:
            zones = self.zones_by_symbol[s]
            total_hits = sum(z['hit_count'] for z in zones)
            total_rejections = sum(z['rejection_count'] for z in zones)
            rate = (total_rejections / total_hits * 100) if total_hits > 0 else 0
            rejection_rates.append(rate)
        ax4.bar(symbols, rejection_rates, color='#f39c12', alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Rejection Rate (%)')
        ax4.set_title('Zone Rejection Rate by Symbol')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved summary: {output_file}")
        plt.close()


def main():
    """Main execution"""
    print("="*70)
    print("LIQUIDITY ZONE VISUALIZATION SYSTEM")
    print("="*70)
    print()
    
    visualizer = LiquidityZoneVisualizer()
    
    print("Generating comprehensive liquidity zone analysis...")
    print("This will create detailed charts for each symbol showing:")
    print("  - Zone locations and confidence levels")
    print("  - Price behavior around zones (rejections/breakthroughs)")
    print("  - Interaction density heatmaps")
    print("  - Historical timeline of zone activity")
    print("  - Behavioral pattern analysis")
    print()
    
    visualizer.create_comprehensive_report()
    
    print()
    print("="*70)
    print("Files created:")
    print("  - liquidity_zones_analysis_[SYMBOL].png (one per symbol)")
    print("  - liquidity_zones_analysis_summary.png (cross-symbol comparison)")
    print("="*70)


if __name__ == "__main__":
    main()
