import json

# Read all predictions from log
predictions = []
with open('/tmp/ml_decisions.log', 'r') as f:
    for line in f:
        predictions.append(json.loads(line))

print("=" * 80)
print("PRODUCTION ML FEATURE ANALYSIS")
print("=" * 80)

# Analyze first prediction (no positions)
print("\n1. FIRST PREDICTION (No positions yet)")
print("-" * 80)
pred1 = predictions[0]
req1 = pred1['request']
portfolio1 = req1['portfolio']
print(f"Capital: ${portfolio1['capital']:,.2f}")
print(f"Num positions: {portfolio1['num_positions']}")
print(f"Total P&L %: {portfolio1['total_pnl_pct']}")
print(f"Best opportunity APR: {req1['opportunities'][0]['fund_apr']:.2f}%")
print(f"Action: {pred1['response']['action']}")
print(f"Entering: {pred1['response']['opportunity_symbol']}")

# Analyze second prediction (5 minutes after entry)
print("\n2. SECOND PREDICTION (5 minutes after entry - EXIT decision)")
print("-" * 80)
pred2 = predictions[1]
req2 = pred2['request']
portfolio2 = req2['portfolio']
position = portfolio2['positions'][0]

print(f"\nPortfolio State:")
print(f"  Capital: ${portfolio2['capital']:,.2f}")
print(f"  Num positions: {portfolio2['num_positions']}")
print(f"  Total P&L %: {portfolio2['total_pnl_pct']:.4f}%")
print(f"  Avg position P&L %: {portfolio2['avg_position_pnl_pct']:.4f}%")

print(f"\nPosition Data (LSKUSDT):")
print(f"  Symbol: {position['symbol']}")
print(f"  Hours held: {position['position_age_hours']:.4f}h")
print(f"  Unrealized P&L %: {position['unrealized_pnl_pct']:.4f}%")
print(f"  Long P&L %: {position['long_pnl_pct']:.4f}%")
print(f"  Short P&L %: {position['short_pnl_pct']:.4f}%")

print(f"\nFunding Data:")
print(f"  Long net funding: ${position['long_net_funding_usd']}")
print(f"  Short net funding: ${position['short_net_funding_usd']}")
print(f"  Long funding rate: {position['long_funding_rate']}")
print(f"  Short funding rate: {position['short_funding_rate']}")

print(f"\nPhase 1 Exit Timing Features:")
print(f"  Entry APR: {position['entry_apr']:.2f}%")
print(f"  Peak P&L %: {position['peak_pnl_pct']:.4f}%")
print(f"  P&L history: {position['pnl_history']}")
print(f"  P&L history length: {len(position['pnl_history'])}")

print(f"\nPhase 2 APR Comparison Features:")
print(f"  Current position APR: {position['current_position_apr']:.2f}%")
print(f"  Best available APR: {position['best_available_apr']:.2f}%")
print(f"  APR advantage: {position['apr_advantage']:.2f}%")

print(f"\nLiquidation & Risk:")
print(f"  Liquidation distance: {position['liquidation_distance']:.4f}")
print(f"  Position size: ${position['position_size_usd']:.2f}")

print(f"\nML Decision:")
print(f"  Action: {pred2['response']['action']}")
print(f"  Confidence: {pred2['response']['confidence']:.2%}")
print(f"  Position index to exit: {pred2['response'].get('position_index', 'N/A')}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

# Calculate some key metrics
pnl_history = position['pnl_history']
if len(pnl_history) >= 2:
    pnl_velocity = (pnl_history[-1] - pnl_history[0]) / (len(pnl_history) - 1)
    print(f"1. P&L Velocity: {pnl_velocity:.6f}% per step")

peak_pnl = position['peak_pnl_pct']
current_pnl = position['unrealized_pnl_pct']
if peak_pnl > 0:
    drawdown_from_peak = (peak_pnl - current_pnl) / peak_pnl
    print(f"2. Peak Drawdown: {drawdown_from_peak:.2%} (from {peak_pnl:.4f}% to {current_pnl:.4f}%)")

apr_diff = position['apr_advantage']
print(f"3. APR Advantage: {apr_diff:.2f}% (Current {position['current_position_apr']:.2f}% vs Best {position['best_available_apr']:.2f}%)")

print(f"\n4. Position held for only {position['position_age_hours']:.4f} hours (~{position['position_age_hours']*60:.1f} minutes)")
print(f"   Yet model decided to EXIT after just 5 minutes")

print(f"\n5. Funding earned: ${position['long_net_funding_usd'] + position['short_net_funding_usd']}")
print(f"   Expected funding based on entry APR: ${position['entry_apr']/100 * position['position_size_usd']/365/24 * position['position_age_hours']:.4f}")
