"""
Diagnostic Script: Observation Difference Root Cause Analysis

Pinpoints exactly where and why the environment and server produce different observations.
"""

import numpy as np
import torch
from pathlib import Path

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from server.inference.rl_predictor import ModularRLPredictor


# Feature names for opportunity features (20 features per opportunity)
OPPORTUNITY_FEATURE_NAMES = [
    "long_funding_rate",
    "short_funding_rate",
    "long_funding_interval_hours / 8",
    "short_funding_interval_hours / 8",
    "fund_profit_8h",
    "fundProfit8h24hProj",
    "fundProfit8h3dProj",
    "fund_apr",
    "fundApr24hProj",
    "fundApr3dProj",
    "spread30SampleAvg",
    "priceSpread24hAvg",
    "priceSpread3dAvg",
    "spread_volatility_stddev",
    "log10(volume_24h)",  # Index 14 - THE CULPRIT
    "bidAskSpreadPercent",
    "log10(orderbookDepthUsd)",
    "estimatedProfitPercentage",
    "positionCostPercent",
    "net_funding_rate (short - long)",
]


def map_index_to_feature(index: int) -> str:
    """Map observation index to feature name."""
    if index < 5:
        return f"Config[{index}]"
    elif index < 15:
        return f"Portfolio[{index-5}]"
    elif index < 75:
        exec_idx = index - 15
        position_slot = exec_idx // 12
        feature_in_slot = exec_idx % 12
        return f"Execution[Pos {position_slot}, Feature {feature_in_slot}]"
    else:
        opp_idx = index - 75
        opportunity_slot = opp_idx // 20
        feature_in_slot = opp_idx % 20
        feature_name = OPPORTUNITY_FEATURE_NAMES[feature_in_slot]
        return f"Opportunity[Slot {opportunity_slot}, {feature_name}]"


def main():
    print("="*80)
    print("DIAGNOSTIC: OBSERVATION DIFFERENCE ROOT CAUSE ANALYSIS")
    print("="*80)

    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Trading config
    trading_config = TradingConfig(
        max_leverage=1.0,
        target_utilization=0.9,
        max_positions=3,
        stop_loss_threshold=-0.02,
        liquidation_buffer=0.15,
    )

    print("\n1. Initializing Environment...")
    env = FundingArbitrageEnv(
        data_path='data/rl_test.csv',
        initial_capital=10000.0,
        trading_config=trading_config,
        episode_length_days=7,
        step_hours=5/60,
        feature_scaler_path='trained_models/rl/feature_scaler.pkl',
        verbose=False,
    )
    print("✅ Environment initialized")

    print("\n2. Initializing Server Predictor...")
    server_predictor = ModularRLPredictor(
        model_path='checkpoints/best_model.pt',
        feature_scaler_path='trained_models/rl/feature_scaler.pkl',
        device='cpu',
    )
    print("✅ Server predictor initialized")

    print("\n3. Resetting environment with seed...")
    env_obs, info = env.reset(seed=seed)
    print(f"✅ Environment reset")

    # === DIAGNOSTIC ANALYSIS ===
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS - STEP 0")
    print("="*80)

    # Environment opportunities
    print(f"\n📊 ENVIRONMENT OPPORTUNITIES:")
    print(f"   Number of opportunities: {len(env.current_opportunities)}")
    if len(env.current_opportunities) > 0:
        print(f"   First opportunity: {env.current_opportunities[0]}")
    else:
        print(f"   (No opportunities available)")

    # Server opportunities (from environment)
    server_opportunities = []
    for opp in env.current_opportunities[:10]:
        server_opportunities.append(dict(opp))

    print(f"\n📊 SERVER OPPORTUNITIES (converted from environment):")
    print(f"   Number of opportunities: {len(server_opportunities)}")
    if len(server_opportunities) > 0:
        print(f"   First opportunity: {server_opportunities[0]}")

    # Build portfolio dict
    portfolio_dict = {
        'total_capital': env.portfolio.total_capital,
        'initial_capital': env.portfolio.initial_capital,
        'available_margin': env.portfolio.available_margin,
        'margin_utilization': env.portfolio.margin_utilization,
        'utilization': env.portfolio.capital_utilization,
        'total_pnl_pct': env.portfolio.total_pnl_pct,
        'max_drawdown': env.portfolio.max_drawdown_pct,
        'positions': [],
        'max_positions': env.portfolio.max_positions,
    }

    # Pad positions to 5 slots
    for _ in range(5):
        portfolio_dict['positions'].append({
            'symbol': '',
            'position_size_usd': 0.0,
            'unrealized_pnl_pct': 0.0,
            'hours_held': 0.0,
            'position_age_hours': 0.0,
            'is_active': False,
            'position_is_active': 0.0,
            'long_net_funding_usd': 0.0,
            'short_net_funding_usd': 0.0,
            'long_funding_rate': 0.0,
            'short_funding_rate': 0.0,
            'current_long_price': 0.0,
            'current_short_price': 0.0,
            'entry_long_price': 0.0,
            'entry_short_price': 0.0,
            'current_spread_pct': 0.0,
            'entry_spread_pct': 0.0,
            'long_pnl_pct': 0.0,
            'short_pnl_pct': 0.0,
            'liquidation_distance': 1.0,
            'entry_fees_paid_usd': 0.0,
        })

    trading_config_dict = {
        'max_leverage': env.current_config.max_leverage,
        'target_utilization': env.current_config.target_utilization,
        'max_positions': env.current_config.max_positions,
        'stop_loss_threshold': env.current_config.stop_loss_threshold,
        'liquidation_buffer': env.current_config.liquidation_buffer,
    }

    # Build server observation
    print(f"\n🔍 BUILDING SERVER OBSERVATION...")
    server_obs = server_predictor._build_observation(
        trading_config_dict,
        portfolio_dict,
        server_opportunities
    )

    # Compare observations
    print(f"\n📈 OBSERVATION COMPARISON:")
    print(f"   Environment obs shape: {env_obs.shape}")
    print(f"   Server obs shape: {server_obs.shape}")

    # Find differences
    obs_diff = np.abs(env_obs - server_obs)
    max_diff_idx = np.argmax(obs_diff)
    max_diff = obs_diff[max_diff_idx]

    print(f"\n⚠️  MAXIMUM DIFFERENCE:")
    print(f"   Index: {max_diff_idx}")
    print(f"   Feature: {map_index_to_feature(max_diff_idx)}")
    print(f"   Environment value: {env_obs[max_diff_idx]}")
    print(f"   Server value: {server_obs[max_diff_idx]}")
    print(f"   Difference: {max_diff}")

    # Show all differences > 0.001
    large_diffs = np.where(obs_diff > 0.001)[0]
    if len(large_diffs) > 0:
        print(f"\n🔍 ALL SIGNIFICANT DIFFERENCES (> 0.001):")
        for idx in large_diffs[:20]:  # Show first 20
            print(f"   [{idx:3d}] {map_index_to_feature(idx):60s} | "
                  f"Env={env_obs[idx]:12.6f} | Server={server_obs[idx]:12.6f} | Diff={obs_diff[idx]:.6f}")
        if len(large_diffs) > 20:
            print(f"   ... and {len(large_diffs) - 20} more")

    # === DEEP DIVE: OPPORTUNITY FEATURES ===
    print("\n" + "="*80)
    print("DEEP DIVE: OPPORTUNITY FEATURES EXTRACTION")
    print("="*80)

    # Extract opportunity features section
    opp_start_idx = 75
    opp_features_env = env_obs[opp_start_idx:opp_start_idx+200]
    opp_features_server = server_obs[opp_start_idx:opp_start_idx+200]

    print(f"\n🎯 FIRST OPPORTUNITY SLOT (Features 0-19):")
    for i in range(20):
        env_val = opp_features_env[i]
        server_val = opp_features_server[i]
        diff = abs(env_val - server_val)
        match_str = "✅" if diff < 0.001 else f"❌ DIFF={diff:.6f}"
        print(f"   [{i:2d}] {OPPORTUNITY_FEATURE_NAMES[i]:40s} | "
              f"Env={env_val:12.6f} | Server={server_val:12.6f} | {match_str}")

    # === INVESTIGATE FEATURE SCALER ===
    print("\n" + "="*80)
    print("FEATURE SCALER INVESTIGATION")
    print("="*80)

    if env.feature_scaler is not None:
        print(f"\n🔬 Environment feature scaler:")
        print(f"   Mean shape: {env.feature_scaler.mean_.shape}")
        print(f"   Scale shape: {env.feature_scaler.scale_.shape}")
        print(f"   First 5 means: {env.feature_scaler.mean_[:5]}")
        print(f"   First 5 scales: {env.feature_scaler.scale_[:5]}")
    else:
        print(f"\n⚠️  Environment feature scaler is None!")

    if server_predictor.feature_scaler is not None:
        print(f"\n🔬 Server feature scaler:")
        print(f"   Mean shape: {server_predictor.feature_scaler.mean_.shape}")
        print(f"   Scale shape: {server_predictor.feature_scaler.scale_.shape}")
        print(f"   First 5 means: {server_predictor.feature_scaler.mean_[:5]}")
        print(f"   First 5 scales: {server_predictor.feature_scaler.scale_[:5]}")

        # Check if scalers are identical
        if env.feature_scaler is not None:
            mean_match = np.allclose(env.feature_scaler.mean_, server_predictor.feature_scaler.mean_)
            scale_match = np.allclose(env.feature_scaler.scale_, server_predictor.feature_scaler.scale_)
            print(f"\n   Scaler means match: {mean_match}")
            print(f"   Scaler scales match: {scale_match}")
    else:
        print(f"\n⚠️  Server feature scaler is None!")

    # === RAW FEATURE EXTRACTION (BEFORE SCALING) ===
    print("\n" + "="*80)
    print("RAW FEATURE EXTRACTION (BEFORE SCALING)")
    print("="*80)

    # Manually extract first opportunity features WITHOUT scaling
    print(f"\n🔍 Extracting raw features for first opportunity...")
    print(f"   Number of opportunities: {len(env.current_opportunities)}")

    if len(env.current_opportunities) > 0:
        opp = env.current_opportunities[0]
        print(f"\n   Opportunity 0 data:")
        for key in ['symbol', 'long_funding_rate', 'short_funding_rate', 'volume_24h', 'fund_apr']:
            print(f"     {key}: {opp.get(key, 'N/A')}")

        # Extract raw feature 14 (volume_24h)
        raw_volume = float(opp.get('volume_24h', 1e6) or 1e6)
        log_volume = np.log10(max(raw_volume, 1e5))
        print(f"\n   Feature 14 (log10 volume_24h) calculation:")
        print(f"     Raw volume_24h: {raw_volume}")
        print(f"     max(volume, 1e5): {max(raw_volume, 1e5)}")
        print(f"     log10(volume): {log_volume}")
    else:
        print(f"\n   ❌ NO OPPORTUNITIES - Should all be zeros!")
        print(f"   But environment observation has non-zero values at index 89")
        print(f"   This suggests the environment is NOT properly padding empty slots with zeros")
        print(f"   OR the feature scaler is transforming zeros to non-zero values")

        # Test: What does the scaler do to a zero input?
        if env.feature_scaler is not None:
            print(f"\n🧪 TESTING: What does scaler do to zeros?")
            zero_input = np.zeros(20).reshape(1, 20)
            scaled_zeros = env.feature_scaler.transform(zero_input)
            print(f"   Input: {zero_input.flatten()[:5]}... (all zeros)")
            print(f"   Output: {scaled_zeros.flatten()[:5]}...")
            print(f"   Output[14]: {scaled_zeros.flatten()[14]} (should be environment's value at index 89)")

            if abs(scaled_zeros.flatten()[14] - env_obs[89]) < 0.0001:
                print(f"\n   🎯 FOUND IT! The scaler transforms zero to {scaled_zeros.flatten()[14]:.6f}")
                print(f"   This matches env_obs[89] = {env_obs[89]:.6f}")
                print(f"\n   ✅ ROOT CAUSE: The environment's feature scaler transforms zeros")
                print(f"                 But the server doesn't apply scaler to empty slots (correctly)")
            else:
                print(f"\n   ❌ Scaler output doesn't match - there's another issue")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
