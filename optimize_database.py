"""
Optimal SQLite Configuration for 320 MB/s SSD

With your new USB-C 3.2 Gen 2 cable achieving 320 MB/s writes,
you can use more aggressive SQLite settings for maximum performance.
"""

import sqlite3
from contextlib import contextmanager


class OptimizedDatabase:
    """
    SQLite connection with optimized settings for fast SSD
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect with optimal PRAGMA settings"""
        self.conn = sqlite3.connect(self.db_path)
        
        # Apply optimizations
        cursor = self.conn.cursor()
        
        # ==================================================================
        # JOURNALING & SYNCHRONIZATION
        # ==================================================================
        
        # WAL mode: Much faster for concurrent read/write
        # Writes don't block reads, readers don't block writers
        cursor.execute("PRAGMA journal_mode = WAL")
        
        # Synchronous NORMAL: Safe for WAL mode, much faster than FULL
        # Still crash-safe, but doesn't wait for OS-level sync
        cursor.execute("PRAGMA synchronous = NORMAL")
        
        # ==================================================================
        # CACHE & MEMORY
        # ==================================================================
        
        # 64 MB cache (plenty of RAM on Raspberry Pi 5 with 8GB)
        # Keeps hot data in memory
        cursor.execute("PRAGMA cache_size = -65536")  # Negative = KB
        
        # Store temp tables in memory (faster)
        cursor.execute("PRAGMA temp_store = MEMORY")
        
        # ==================================================================
        # WRITE OPTIMIZATION
        # ==================================================================
        
        # Larger page size for SSD (4KB is optimal for most SSDs)
        # Note: Only effective when creating new database
        cursor.execute("PRAGMA page_size = 4096")
        
        # Auto-vacuum: Keep database file size reasonable
        # INCREMENTAL allows vacuuming during idle periods
        cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")
        
        # ==================================================================
        # PERFORMANCE TUNING
        # ==================================================================
        
        # Memory-mapped I/O: Use 256MB for mmap
        # Very fast reads for hot data
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256 MB
        
        # Larger WAL checkpoint threshold (32 MB instead of default 4MB)
        # Fewer checkpoints = better write performance
        cursor.execute("PRAGMA wal_autocheckpoint = 8000")  # Pages
        
        # ==================================================================
        # VERIFY SETTINGS
        # ==================================================================
        
        print("="*80)
        print("SQLite Optimization Settings Applied")
        print("="*80)
        
        settings = [
            "journal_mode",
            "synchronous", 
            "cache_size",
            "temp_store",
            "page_size",
            "auto_vacuum",
            "mmap_size",
            "wal_autocheckpoint"
        ]
        
        for setting in settings:
            result = cursor.execute(f"PRAGMA {setting}").fetchone()
            print(f"  {setting:25s}: {result[0]}")
        
        print("="*80)
        
        return self.conn
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()


# ==================================================================
# USAGE IN YOUR TRADING BOT
# ==================================================================

def optimize_existing_database(db_path: str):
    """
    Apply optimizations to existing database
    
    Run this once when you first install the new cable
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Optimizing database: {db_path}")
    
    # Switch to WAL mode (one-time, persists)
    cursor.execute("PRAGMA journal_mode = WAL")
    print("  ✅ Switched to WAL mode")
    
    # Set synchronous to NORMAL (persists)
    cursor.execute("PRAGMA synchronous = NORMAL")
    print("  ✅ Set synchronous = NORMAL")
    
    # Set auto_vacuum (persists)
    cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")
    print("  ✅ Set auto_vacuum = INCREMENTAL")
    
    # Vacuum to optimize file (run once)
    print("  🔄 Running VACUUM (this may take a minute)...")
    cursor.execute("VACUUM")
    print("  ✅ VACUUM complete")
    
    conn.close()
    print("✅ Database optimized!\n")


# ==================================================================
# INTEGRATION WITH LIQUIDITY ZONE TRACKER
# ==================================================================

def create_optimized_tracker():
    """
    Create liquidity zone tracker with optimized database
    """
    from liquidity_zones import LiquidityZoneTracker
    
    # First-time optimization
    optimize_existing_database('liquidity_zones.db')
    
    # Create tracker (it will use the optimized settings)
    tracker = LiquidityZoneTracker(db_path='liquidity_zones.db')
    
    return tracker


# ==================================================================
# BATCH INSERT OPTIMIZATION
# ==================================================================

@contextmanager
def optimized_batch_insert(conn):
    """
    Context manager for efficient batch inserts
    
    Usage:
        with optimized_batch_insert(conn):
            for record in records:
                cursor.execute("INSERT ...")
    """
    cursor = conn.cursor()
    
    # Begin explicit transaction
    cursor.execute("BEGIN TRANSACTION")
    
    try:
        yield cursor
        # Commit on success
        cursor.execute("COMMIT")
    except Exception as e:
        # Rollback on error
        cursor.execute("ROLLBACK")
        raise e


# ==================================================================
# EXAMPLE: HIGH-VOLUME DATA INGESTION
# ==================================================================

def ingest_high_volume_data(conn, records):
    """
    Example of efficiently ingesting high-volume market data
    
    With 320 MB/s SSD, can handle thousands of inserts per second
    """
    with optimized_batch_insert(conn) as cursor:
        # Batch insert all records in single transaction
        cursor.executemany(
            "INSERT INTO zone_interactions (symbol, price, interaction_type, timestamp) "
            "VALUES (?, ?, ?, ?)",
            records
        )
    
    print(f"✅ Inserted {len(records)} records")


# ==================================================================
# MONITORING PERFORMANCE
# ==================================================================

def show_database_stats(db_path: str):
    """Show current database statistics"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    
    # Page count
    page_count = cursor.execute("PRAGMA page_count").fetchone()[0]
    page_size = cursor.execute("PRAGMA page_size").fetchone()[0]
    db_size_mb = (page_count * page_size) / (1024 * 1024)
    
    print(f"  Database size: {db_size_mb:.2f} MB ({page_count:,} pages)")
    
    # WAL size
    wal_size = cursor.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    if wal_size:
        print(f"  WAL pages: {wal_size[1]:,}")
    
    # Cache stats
    cache_size = abs(cursor.execute("PRAGMA cache_size").fetchone()[0])
    cache_mb = (cache_size * page_size) / (1024 * 1024)
    print(f"  Cache size: {cache_mb:.2f} MB")
    
    # Record counts
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    
    print("\n  Record counts:")
    for (table,) in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    {table:30s}: {count:,}")
    
    print("="*80 + "\n")
    
    conn.close()


# ==================================================================
# BENCHMARK
# ==================================================================

def benchmark_write_performance(db_path: str, num_records: int = 10000):
    """
    Benchmark database write performance with your new SSD
    """
    import time
    
    conn = sqlite3.connect(db_path)
    
    # Create test table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_test (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            price REAL,
            timestamp TEXT,
            data TEXT
        )
    """)
    
    # Test data
    test_records = [
        (f'TEST{i%5}', 100.0 + i*0.01, '2025-11-15T12:00:00', 'benchmark_data')
        for i in range(num_records)
    ]
    
    print(f"\n🔬 Benchmarking {num_records:,} inserts...")
    
    # Test 1: Individual inserts (slow)
    conn.execute("DELETE FROM benchmark_test")
    start = time.time()
    for record in test_records[:1000]:  # Just 1000 for this test
        conn.execute(
            "INSERT INTO benchmark_test (symbol, price, timestamp, data) VALUES (?, ?, ?, ?)",
            record
        )
        conn.commit()
    duration1 = time.time() - start
    rate1 = 1000 / duration1
    print(f"  Individual commits: {rate1:,.0f} inserts/sec")
    
    # Test 2: Batch insert with transaction (fast)
    conn.execute("DELETE FROM benchmark_test")
    start = time.time()
    with optimized_batch_insert(conn) as cursor:
        cursor.executemany(
            "INSERT INTO benchmark_test (symbol, price, timestamp, data) VALUES (?, ?, ?, ?)",
            test_records
        )
    duration2 = time.time() - start
    rate2 = num_records / duration2
    print(f"  Batch transaction:  {rate2:,.0f} inserts/sec ({rate2/rate1:.1f}x faster)")
    
    # Cleanup
    conn.execute("DROP TABLE benchmark_test")
    conn.close()
    
    print(f"✅ With your 320 MB/s SSD, batch inserts are extremely fast!\n")


# ==================================================================
# MAIN: RUN OPTIMIZATIONS
# ==================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SQLite Optimization for 320 MB/s SSD")
    print("="*80 + "\n")
    
    # Optimize your existing database
    optimize_existing_database('liquidity_zones.db')
    
    # Show stats
    show_database_stats('liquidity_zones.db')
    
    # Run benchmark
    benchmark_write_performance('liquidity_zones.db')
    
    print("="*80)
    print("All optimizations applied!")
    print("Your database is now configured for maximum performance.")
    print("="*80 + "\n")
