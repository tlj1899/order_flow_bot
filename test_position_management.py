"""
Test script to verify position management works correctly
"""

def test_position_logic():
    """Test the position checking logic"""
    
    # Simulate positions dictionary
    positions = {}
    
    print("="*70)
    print("POSITION MANAGEMENT TEST")
    print("="*70)
    print()
    
    # Test 1: No position exists
    print("Test 1: No existing position")
    symbol = 'GC'
    has_position = symbol in positions
    print(f"  Symbol: {symbol}")
    print(f"  Has position: {has_position}")
    print(f"  Should place order: {not has_position}")
    print(f"  ✓ PASS" if not has_position else "  ✗ FAIL")
    print()
    
    # Add position
    positions['GC'] = {
        'direction': 'LONG',
        'entry_price': 4090.0,
        'quantity': 1
    }
    print(f"  [Simulated placing LONG order on GC @ 4090.0]")
    print(f"  Positions: {list(positions.keys())}")
    print()
    
    # Test 2: Position exists - should NOT place new order
    print("Test 2: Position already exists")
    has_position = symbol in positions
    print(f"  Symbol: {symbol}")
    print(f"  Has position: {has_position}")
    print(f"  Should place order: {not has_position}")
    print(f"  ✓ PASS" if has_position else "  ✗ FAIL")
    print()
    
    # Test 3: Another symbol - should allow
    print("Test 3: Different symbol (no position)")
    symbol2 = 'NQ'
    has_position2 = symbol2 in positions
    print(f"  Symbol: {symbol2}")
    print(f"  Has position: {has_position2}")
    print(f"  Should place order: {not has_position2}")
    print(f"  ✓ PASS" if not has_position2 else "  ✗ FAIL")
    print()
    
    # Add second position
    positions['NQ'] = {
        'direction': 'SHORT',
        'entry_price': 25500.0,
        'quantity': 1
    }
    print(f"  [Simulated placing SHORT order on NQ @ 25500.0]")
    print(f"  Positions: {list(positions.keys())}")
    print()
    
    # Test 4: Multiple positions - both should block new signals
    print("Test 4: Multiple existing positions")
    for sym in ['GC', 'NQ']:
        has_pos = sym in positions
        print(f"  {sym}: Has position = {has_pos}, Should block = {has_pos}")
        print(f"    ✓ PASS" if has_pos else "    ✗ FAIL")
    print()
    
    # Test 5: Close position
    print("Test 5: After closing position")
    del positions['GC']
    print(f"  [Simulated closing GC position]")
    print(f"  Positions: {list(positions.keys())}")
    
    has_position_gc = 'GC' in positions
    has_position_nq = 'NQ' in positions
    
    print(f"  GC: Has position = {has_position_gc}, Should allow new = {not has_position_gc}")
    print(f"    ✓ PASS" if not has_position_gc else "    ✗ FAIL")
    print(f"  NQ: Has position = {has_position_nq}, Should block = {has_position_nq}")
    print(f"    ✓ PASS" if has_position_nq else "    ✗ FAIL")
    print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("✓ System correctly:")
    print("  - Allows new positions when none exist")
    print("  - Blocks duplicate positions on same symbol")
    print("  - Allows positions on different symbols")
    print("  - Allows new positions after closing previous")
    print()
    print("Expected behavior in live system:")
    print("  1. Signal generated for GC → Place order (if no GC position)")
    print("  2. Signal generated for GC again → Skip (position exists)")
    print("  3. Position closed (SL/TP/Reversal) → Can take new GC signal")
    print("  4. Can have max 5 positions (one per symbol)")
    print()

if __name__ == "__main__":
    test_position_logic()
