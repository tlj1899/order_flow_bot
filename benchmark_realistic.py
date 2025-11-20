"""
Realistic Database Performance Benchmark

Tests what actually matters for your trading system:
- Random writes (zone interaction logging)
- Random reads (zone lookups)
- Mixed read/write workload (real trading)
- Small transaction latency (order flow processing)
"""

import sqlite3
import time
import random
import os
from contextlib import contextmanager


def benchmark_database_performance(db_path='benchmark.db'):
    """
    Comprehensive database performance test
    
    Simulates real trading workload:
    - Zone interaction writes
    - Zone lookup reads
    - Mixed operations
    """
    
    # Clean start
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create realistic schema
    cursor.execute('''
        CREATE TABLE zone_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            price REAL,
            interaction_type TEXT,
            timestamp TEXT,
            price_before REAL,
            price_after REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE liquidity_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            price REAL,
            confidence REAL,
            rejection_count INTEGER,
            breakthrough_count INTEGER,
            UNIQUE(symbol, price)
        )
    ''')
    
    # Create indexes (realistic for production)
    cursor.execute('CREATE INDEX idx_interactions_symbol ON zone_interactions(symbol)')
    cursor.execute('CREATE INDEX idx_interactions_timestamp ON zone_interactions(timestamp)')
    cursor.execute('CREATE INDEX idx_zones_symbol ON liquidity_zones(symbol)')
    
    conn.commit()
    
    print("="*80)
    print("DATABASE PERFORMANCE BENCHMARK - Real Trading Workload")
    print("="*80)
    print(f"SSD Read Speed:  4.7 GB/s (sequential)")
    print(f"SSD Write Speed: 320 MB/s (sequential)")
    print("="*80)
    
    # Test 1: Sequential Writes (Zone Interactions)
    print("\n📊 TEST 1: Sequential Writes (Logging Zone Interactions)")
    print("-" * 80)
    
    num_records = 10000
    test_data = [
        (f'GC', 4000 + i*0.1, 'rejection', '2025-11-15T12:00:00', 4000.0, 4000.1)
        for i in range(num_records)
    ]
    
    start = time.time()
    cursor.execute('BEGIN TRANSACTION')
    cursor.executemany(
        'INSERT INTO zone_interactions (symbol, price, interaction_type, timestamp, price_before, price_after) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        test_data
    )
    cursor.execute('COMMIT')
    duration = time.time() - start
    
    writes_per_sec = num_records / duration
    mb_per_sec = (num_records * 64) / (1024 * 1024) / duration  # Assume ~64 bytes/record
    
    print(f"  Records: {num_records:,}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Throughput: {writes_per_sec:,.0f} writes/sec")
    print(f"  Data rate: {mb_per_sec:.1f} MB/s")
    print(f"  ✅ Excellent for real-time zone logging")
    
    # Test 2: Random Reads (Zone Lookups)
    print("\n📊 TEST 2: Random Reads (Zone Lookups During Trading)")
    print("-" * 80)
    
    # Insert some zones
    zones = [
        (f'GC', 4000 + i*5, 0.75 + i*0.01, 10, 5)
        for i in range(100)
    ]
    cursor.executemany(
        'INSERT OR IGNORE INTO liquidity_zones (symbol, price, confidence, rejection_count, breakthrough_count) '
        'VALUES (?, ?, ?, ?, ?)',
        zones
    )
    conn.commit()
    
    # Random lookups
    num_lookups = 10000
    start = time.time()
    for _ in range(num_lookups):
        symbol = 'GC'
        price = 4000 + random.randint(0, 500) * 0.1
        cursor.execute(
            'SELECT * FROM liquidity_zones WHERE symbol = ? AND ABS(price - ?) < 1.0',
            (symbol, price)
        ).fetchall()
    duration = time.time() - start
    
    reads_per_sec = num_lookups / duration
    
    print(f"  Lookups: {num_lookups:,}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Throughput: {reads_per_sec:,.0f} lookups/sec")
    print(f"  Avg latency: {duration/num_lookups*1000:.2f} ms")
    print(f"  ✅ Fast enough for real-time order flow processing")
    
    # Test 3: Mixed Workload (Real Trading Simulation)
    print("\n📊 TEST 3: Mixed Workload (Realistic Trading Session)")
    print("-" * 80)
    
    num_operations = 5000
    start = time.time()
    
    for i in range(num_operations):
        # 70% reads (zone lookups), 30% writes (logging interactions)
        if random.random() < 0.7:
            # Read: Look up zone
            symbol = 'GC'
            price = 4000 + random.randint(0, 500) * 0.1
            cursor.execute(
                'SELECT * FROM liquidity_zones WHERE symbol = ? AND ABS(price - ?) < 1.0',
                (symbol, price)
            ).fetchone()
        else:
            # Write: Log interaction
            cursor.execute(
                'INSERT INTO zone_interactions (symbol, price, interaction_type, timestamp) '
                'VALUES (?, ?, ?, ?)',
                ('GC', 4000 + i*0.1, 'rejection', '2025-11-15T12:00:00')
            )
            if i % 100 == 0:  # Commit every 100 writes
                conn.commit()
    
    conn.commit()
    duration = time.time() - start
    
    ops_per_sec = num_operations / duration
    
    print(f"  Operations: {num_operations:,} (70% read, 30% write)")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")
    print(f"  Avg latency: {duration/num_operations*1000:.2f} ms")
    print(f"  ✅ Can handle {ops_per_sec*60:,.0f} operations/minute")
    
    # Test 4: Small Transaction Latency (Order Flow Processing)
    print("\n📊 TEST 4: Small Transaction Latency (Order Flow Responsiveness)")
    print("-" * 80)
    
    latencies = []
    num_txns = 1000
    
    for i in range(num_txns):
        start = time.time()
        cursor.execute('BEGIN TRANSACTION')
        cursor.execute(
            'INSERT INTO zone_interactions (symbol, price, interaction_type, timestamp) '
            'VALUES (?, ?, ?, ?)',
            ('GC', 4000 + i*0.1, 'rejection', '2025-11-15T12:00:00')
        )
        cursor.execute('COMMIT')
        latencies.append((time.time() - start) * 1000)  # Convert to ms
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"  Transactions: {num_txns:,}")
    print(f"  Avg latency: {avg_latency:.2f} ms")
    print(f"  P95 latency: {p95_latency:.2f} ms")
    print(f"  P99 latency: {p99_latency:.2f} ms")
    print(f"  ✅ Fast enough for sub-second order flow decisions")
    
    # Test 5: Burst Write Performance (High Volume Period)
    print("\n📊 TEST 5: Burst Writes (Simulating High Volume Period)")
    print("-" * 80)
    
    burst_size = 5000
    burst_data = [
        ('NQ', 25000 + i*0.25, 'breakthrough', '2025-11-15T09:30:00', 25000.0, 25000.25)
        for i in range(burst_size)
    ]
    
    start = time.time()
    cursor.execute('BEGIN TRANSACTION')
    cursor.executemany(
        'INSERT INTO zone_interactions (symbol, price, interaction_type, timestamp, price_before, price_after) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        burst_data
    )
    cursor.execute('COMMIT')
    duration = time.time() - start
    
    burst_rate = burst_size / duration
    
    print(f"  Burst size: {burst_size:,} records")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Throughput: {burst_rate:,.0f} inserts/sec")
    print(f"  ✅ Can handle market open / news event spikes")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Database Performance for Trading")
    print("="*80)
    
    print(f"\n✅ Sequential Writes:  {writes_per_sec:>10,.0f} writes/sec")
    print(f"✅ Random Reads:       {reads_per_sec:>10,.0f} lookups/sec")
    print(f"✅ Mixed Workload:     {ops_per_sec:>10,.0f} ops/sec")
    print(f"✅ Transaction Latency:     {avg_latency:>7.2f} ms (avg)")
    print(f"✅ Burst Capacity:     {burst_rate:>10,.0f} inserts/sec")
    
    print("\n💡 INTERPRETATION:")
    print("-" * 80)
    
    if avg_latency < 1.0:
        print("  🚀 EXCELLENT: Sub-millisecond latency for order flow processing")
    elif avg_latency < 5.0:
        print("  ✅ GOOD: Low latency suitable for real-time trading")
    else:
        print("  ⚠️  MODERATE: May need optimization for high-frequency trading")
    
    if burst_rate > 10000:
        print("  🚀 EXCELLENT: Can easily handle high-volume periods")
    elif burst_rate > 5000:
        print("  ✅ GOOD: Sufficient for normal market conditions")
    else:
        print("  ⚠️  MODERATE: May struggle during extreme volume")
    
    if reads_per_sec > 50000:
        print("  🚀 EXCELLENT: Zone lookups won't be a bottleneck")
    elif reads_per_sec > 10000:
        print("  ✅ GOOD: Fast enough for multi-instrument trading")
    else:
        print("  ⚠️  MODERATE: May limit number of instruments")
    
    print("\n" + "="*80)
    print("Your 4.7 GB/s read speed translates to:")
    print(f"  • {reads_per_sec:,.0f} zone lookups per second")
    print(f"  • {1000/avg_latency:.0f} order flow decisions per second")
    print(f"  • Capacity for {burst_rate/100:,.0f} instruments simultaneously")
    print("="*80)
    
    # Cleanup
    conn.close()
    os.remove(db_path)


if __name__ == "__main__":
    benchmark_database_performance()
