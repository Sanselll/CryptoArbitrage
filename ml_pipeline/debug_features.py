"""
Debug script to compare UnifiedFeatureBuilder vs original portfolio.get_execution_state()
"""
import numpy as np
from models.rl.core.portfolio import Portfolio, Position
from common.features import UnifiedFeatureBuilder
from datetime import datetime, timezone
import pandas as pd

# Create a test position
now = datetime.now(timezone.utc)
pos = Position(
    opportunity_id="test_1",
    symbol="BTCUSDT",
    long_exchange="Binance",
    short_exchange="Bybit",
    entry_time=now,
    position_size_usd=1000.0,
    leverage=2.0,
    entry_long_price=50000.0,
    entry_short_price=50100.0,
    entry_apr=100.0,  # 100% APR
    long_funding_interval_hours=8,
    short_funding_interval_hours=8,
    slippage_pct=0.0,
    long_funding_rate=0.0001,
    short_funding_rate=-0.00005,
    long_next_funding_time=now,
    short_next_funding_time=now
)

# Simulate some P&L
pos.hours_held = 5.0
pos.unrealized_pnl_pct = 2.5  # 2.5% profit
pos.long_pnl_pct = 1.0  # 1% on long side
pos.short_pnl_pct = 1.5  # 1.5% on short side

# Create portfolio
portfolio = Portfolio(initial_capital=10000.0, max_positions=2, max_position_size_pct=50)
portfolio.positions.append(pos)

# Current prices (simulate 1% price increase)
price_data = {
    "BTCUSDT": {
        "long_price": 50500.0,  # +1%
        "short_price": 50600.0  # +1%
    }
}

# Current opportunities
current_opportunities = [
    {"symbol": "BTCUSDT", "fund_apr": 120.0},  # Current APR is 120%
    {"symbol": "ETHUSDT", "fund_apr": 80.0}
]

best_available_apr = 120.0

print("="*80)
print("ORIGINAL portfolio.get_execution_state()")
print("="*80)

original_features = portfolio.get_execution_state(
    exec_idx=0,
    price_data=price_data,
    best_available_apr=best_available_apr,
    current_opportunities=current_opportunities
)

print(f"Shape: {original_features.shape}")
for i, feat in enumerate(original_features, 1):
    print(f"{i:2d}. {feat:.8f}")

print("\n" + "="*80)
print("UNIFIED FEATURE BUILDER")
print("="*80)

# Build position dict for UnifiedFeatureBuilder
pos_dict = {
    'is_active': True,
    'symbol': pos.symbol,
    'position_size_usd': float(pos.position_size_usd),
    'position_age_hours': float(pos.hours_held),
    'leverage': float(pos.leverage),
    'entry_long_price': float(pos.entry_long_price),
    'entry_short_price': float(pos.entry_short_price),
    'current_long_price': 50500.0,
    'current_short_price': 50600.0,
    'unrealized_pnl_pct': float(pos.unrealized_pnl_pct),
    'long_pnl_pct': float(pos.long_pnl_pct),
    'short_pnl_pct': float(pos.short_pnl_pct),
    'long_funding_rate': float(pos.long_funding_rate),  # Use actual funding rates
    'short_funding_rate': float(pos.short_funding_rate),  # Use actual funding rates
    'long_funding_interval_hours': int(pos.long_funding_interval_hours),
    'short_funding_interval_hours': int(pos.short_funding_interval_hours),
    'entry_apr': float(pos.entry_apr),
    'current_position_apr': 120.0,
    'liquidation_distance': float(pos.get_liquidation_distance(50500.0, 50600.0)),
    'slippage_pct': float(pos.slippage_pct),
}

portfolio_dict = {
    'positions': [pos_dict],
    'total_capital': 10000.0,
    'capital_utilization': 20.0
}

builder = UnifiedFeatureBuilder(feature_scaler_path=None)
unified_features = builder.build_execution_features(portfolio_dict, best_available_apr=120.0)

# Get just the first position's features (first 17)
unified_features_slot0 = unified_features[:17]

print(f"Shape: {unified_features_slot0.shape}")
for i, feat in enumerate(unified_features_slot0, 1):
    print(f"{i:2d}. {feat:.8f}")

print("\n" + "="*80)
print("COMPARISON (Unified - Original)")
print("="*80)

diff = unified_features_slot0 - original_features
for i, (orig, uni, d) in enumerate(zip(original_features, unified_features_slot0, diff), 1):
    status = "✓" if abs(d) < 1e-6 else "✗"
    print(f"{status} {i:2d}. Original: {orig:12.8f}  Unified: {uni:12.8f}  Diff: {d:+12.8f}")
