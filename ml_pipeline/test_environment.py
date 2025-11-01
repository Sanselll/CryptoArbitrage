"""
Test the RL Environment with Random Actions

This verifies the environment works correctly before training RL agents.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from rl.environment import FundingArbitrageEnv


def test_random_agent(num_episodes: int = 3):
    """
    Run environment with random actions to test functionality.

    Args:
        num_episodes: Number of test episodes
    """
    # Initialize environment
    env = FundingArbitrageEnv(
        data_path='data/rl_opportunities.csv',
        initial_capital=10000.0,
        episode_length_days=2,  # Only have 3 days of data
        max_positions=3,
        max_opportunities_per_hour=5
    )

    print(f"\n{'='*80}")
    print(f"TESTING ENVIRONMENT WITH RANDOM AGENT")
    print(f"{'='*80}\n")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---\n")

        # Reset environment
        observation, info = env.reset(seed=episode)

        print(f"Episode Start: {info['episode_start']}")
        print(f"Episode End: {info['episode_end']}")
        print(f"Initial Portfolio Value: ${info['portfolio_value']:,.2f}\n")

        episode_reward = 0
        step = 0
        max_steps = 168  # 7 days * 24 hours

        done = False

        while not done and step < max_steps:
            # Random action
            action = env.action_space.sample()

            # Step
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            # Print every 24 hours
            if step % 24 == 0:
                print(f"Hour {step:3d}: "
                      f"Portfolio=${info['portfolio_value']:8,.2f} "
                      f"({info['total_pnl_pct']:+6.2f}%), "
                      f"Positions={info['num_positions']}, "
                      f"Utilization={info['capital_utilization']:5.1f}%")

            done = terminated or truncated

            # Show termination reason
            if terminated:
                print(f"\n⚠️  Episode TERMINATED at hour {step}")
                print(f"   Reason: Drawdown or capital loss")

            if truncated:
                print(f"\n✅ Episode COMPLETED ({step} hours)")

        # Final summary
        print(f"\n{'─'*60}")
        print(f"Episode Summary:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: ${episode_reward:,.2f}")
        print(f"  Final Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"  Final P&L: {info['total_pnl_pct']:+.2f}%")
        print(f"  Closed Positions: {len(env.portfolio.closed_positions)}")
        print(f"  Total Fees Paid: ${env.portfolio.total_fees_paid_usd:,.2f}")
        print(f"  Net Funding: ${env.portfolio.total_funding_net_usd:,.2f}")
        print(f"{'─'*60}\n")

        # Show detailed closed positions
        if len(env.portfolio.closed_positions) > 0:
            print("Closed Positions:")
            for i, pos in enumerate(env.portfolio.closed_positions[:5]):  # Show first 5
                breakdown = pos.get_breakdown()
                print(f"  {i+1}. {pos.symbol} ({pos.hours_held:.1f}h): "
                      f"P&L={breakdown['realized_pnl_pct']:+.2f}%, "
                      f"Funding={breakdown['net_funding_pct']:+.2f}%, "
                      f"Fees={breakdown['total_fees_pct']:-.2f}%")

            if len(env.portfolio.closed_positions) > 5:
                print(f"  ... and {len(env.portfolio.closed_positions) - 5} more")

    print(f"\n{'='*80}")
    print(f"ALL TESTS COMPLETED")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    test_random_agent(num_episodes=3)
