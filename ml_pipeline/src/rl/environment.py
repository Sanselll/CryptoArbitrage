"""
Gymnasium Environment for Funding Rate Arbitrage Trading

Simulates hour-by-hour trading with historical market data.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import timedelta

from .portfolio import Portfolio, Position
from data.price_history_loader import PriceHistoryLoader


class FundingArbitrageEnv(gym.Env):
    """
    RL Environment for funding rate arbitrage trading.

    Agent makes decisions every hour:
    - Enter new arbitrage executions
    - Hold existing positions
    - Exit positions

    Rewards based on hourly P&L changes.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 data_path: str,
                 price_history_path: str = None,
                 feature_scaler_path: str = None,
                 initial_capital: float = 10000.0,
                 max_positions: int = 3,
                 max_position_size_pct: float = 33.3,
                 episode_length_days: int = 3,
                 max_opportunities_per_hour: int = 5,
                 step_hours: int = 1,
                 max_position_loss_pct: float = -1.5,
                 pnl_reward_scale: float = 3.0,
                 hold_bonus: float = 0.0,
                 quality_entry_bonus: float = 0.5,
                 quality_entry_penalty: float = -0.5,
                 use_full_range_episodes: bool = False):
        """
        Initialize the environment.

        Args:
            data_path: Path to historical opportunities CSV
            price_history_path: Path to price history directory (parquet files)
            feature_scaler_path: Path to fitted StandardScaler pickle file (optional)
            initial_capital: Starting capital in USD
            max_positions: Max concurrent positions
            max_position_size_pct: Max % of capital per position side
            episode_length_days: Episode length in days (ignored if use_full_range_episodes=True)
            max_opportunities_per_hour: Max opportunities to show per timestep
            step_hours: Hours per timestep (default 1 = hourly)
            max_position_loss_pct: Stop-loss threshold (e.g., -1.5 = close if loss exceeds 1.5%)
            pnl_reward_scale: Scaling factor for P&L rewards (default 3.0 for absolute P&L optimization)
            hold_bonus: DEPRECATED - set to 0.0 to avoid inaction bias
            quality_entry_bonus: Bonus for entering high-quality opportunities (APR>150, spread<0.4%)
            quality_entry_penalty: Penalty for entering low-quality opportunities (APR<75 or spread>0.5%)
            use_full_range_episodes: If True, episodes span entire data range (data_start to data_end)
        """
        super().__init__()

        # Configuration
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size_pct = max_position_size_pct
        self.episode_length_hours = episode_length_days * 24
        self.max_opportunities = max_opportunities_per_hour
        self.step_hours = step_hours
        self.max_position_loss_pct = max_position_loss_pct  # Stop-loss threshold
        self.use_full_range_episodes = use_full_range_episodes  # Full-range episode mode

        # Reward shaping parameters
        self.pnl_reward_scale = pnl_reward_scale
        self.hold_bonus = hold_bonus
        self.quality_entry_bonus = quality_entry_bonus
        self.quality_entry_penalty = quality_entry_penalty

        # Load historical data
        self.data = self._load_data(data_path)
        self.data_start = self.data['entry_time'].min()
        self.data_end = self.data['entry_time'].max()

        # Initialize price history loader
        self.price_loader = None
        if price_history_path:
            self.price_loader = PriceHistoryLoader(price_history_path)
            print(f"   Price history loaded from: {price_history_path}")

        # Load feature scaler (for normalization)
        self.feature_scaler = None
        if feature_scaler_path and Path(feature_scaler_path).exists():
            with open(feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
            print(f"   Feature scaler loaded from: {feature_scaler_path}")
            print(f"   Features will be standardized (mean=0, std=1)")
        else:
            print(f"   WARNING: No feature scaler provided - using raw features (may cause training instability)")

        # Episode state
        self.portfolio: Optional[Portfolio] = None
        self.current_time: Optional[pd.Timestamp] = None
        self.episode_start: Optional[pd.Timestamp] = None
        self.episode_end: Optional[pd.Timestamp] = None
        self.step_count = 0

        # Current opportunities available to agent
        self.current_opportunities: List[Dict] = []

        # Action space: Discrete actions
        # 0 = Hold (do nothing)
        # 1-N = Enter opportunity 0 to N-1
        # N+1 to N+M = Exit position 0 to M-1
        self.action_space = spaces.Discrete(
            1 +  # Hold
            self.max_opportunities +  # Enter opportunities
            self.max_positions  # Exit positions
        )

        # Observation space: Portfolio state + opportunity features
        # Portfolio: [capital_ratio, utilization, num_positions, pnl, drawdown,
        #            pos1_pnl, pos1_hours, pos1_funding, pos2..., pos3...]
        portfolio_dim = 5 + (self.max_positions * 3)  # 5 + 9 = 14

        # Opportunities: For each opportunity, we have its features
        # 22 selected features per opportunity (19 base + 3 momentum features)
        opportunity_features = 22
        opportunities_dim = self.max_opportunities * opportunity_features

        total_dim = portfolio_dim + opportunities_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        print(f"✅ Environment initialized")
        print(f"   Data: {len(self.data):,} opportunities from {self.data_start} to {self.data_end}")
        if self.use_full_range_episodes:
            actual_hours = int((self.data_end - self.data_start).total_seconds() / 3600)
            print(f"   Episode length: Full range ({actual_hours}h)")
        else:
            print(f"   Episode length: {self.episode_length_hours}h ({episode_length_days} days)")
        print(f"   Action space: {self.action_space.n} actions")
        print(f"   Observation space: {self.observation_space.shape[0]} dimensions")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare historical opportunity data."""
        df = pd.read_csv(data_path, low_memory=False)

        # Convert timestamps with UTC timezone
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

        # Sort by entry time
        df = df.sort_values('entry_time').reset_index(drop=True)

        return df

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.

        Returns:
            observation, info
        """
        # IMPORTANT: Set seed BEFORE calling super().reset() to ensure proper seeding
        if seed is not None:
            np.random.seed(seed)
            # Also seed Python's random module for completeness
            import random
            random.seed(seed)

        super().reset(seed=seed)

        # Determine episode time range
        if self.use_full_range_episodes:
            # Full-range mode: episode spans entire data range
            self.episode_start = self.data_start
            self.episode_end = self.data_end
            # Calculate actual episode length in hours
            self.episode_length_hours = int((self.episode_end - self.episode_start).total_seconds() / 3600)
        else:
            # Sample random episode start time
            # Ensure we have enough data for full episode
            min_start = self.data_start
            max_start = self.data_end - timedelta(days=self.episode_length_hours / 24 + 1)

            # Random start time (with UTC timezone)
            # If seed was set, this will be deterministic
            start_timestamp = np.random.uniform(
                min_start.timestamp(),
                max_start.timestamp()
            )
            self.episode_start = pd.Timestamp(start_timestamp, unit='s', tz='UTC')
            self.episode_end = self.episode_start + timedelta(hours=self.episode_length_hours)

        self.current_time = self.episode_start

        # Reset portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_positions=self.max_positions,
            max_position_size_pct=self.max_position_size_pct
        )

        # Reset counters
        self.step_count = 0

        # Get initial opportunities
        self.current_opportunities = self._get_opportunities_at_time(self.current_time)

        # Get observation
        observation = self._get_observation()

        info = {
            'episode_start': self.episode_start,
            'episode_end': self.episode_end,
            'portfolio_value': self.portfolio.portfolio_value,
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one timestep.

        Args:
            action: Action index

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode and execute action
        reward, info = self._execute_action(action)

        # Advance time
        self.current_time += timedelta(hours=self.step_hours)
        self.step_count += 1

        # Update all open positions with new prices
        price_data = self._get_current_prices()
        pnl_change = self.portfolio.update_positions_hourly(self.current_time, price_data)

        # Check and close positions exceeding stop-loss threshold
        stop_loss_reward = self._check_stop_loss(price_data)
        reward += stop_loss_reward

        # REWARD STRUCTURE: Selectivity focus
        # The agent must learn that only 8.9% of opportunities are profitable
        # Being selective (doing nothing) is better than entering bad opportunities

        # Base P&L reward (ONLY signal - no double-counting)
        # This tracks unrealized P&L changes hour by hour
        # Exit fees will be penalized separately when positions close
        base_reward = pnl_change * self.pnl_reward_scale  # Tunable scaling factor

        # NO portfolio-level reward - it's cumulative and double-counts with base_reward
        # NO passivity penalty - waiting is free (hold action gets +0.1 bonus)
        # NO diversity bonus - quality over quantity
        reward += base_reward

        # Get new opportunities for next step
        self.current_opportunities = self._get_opportunities_at_time(self.current_time)

        # Check termination
        terminated = False
        truncated = False

        # Episode ends after specified hours
        if self.current_time >= self.episode_end:
            truncated = True
            # Close all remaining positions
            reward += self._close_all_positions()

        # Terminate if max drawdown exceeded (25%)
        if self.portfolio.max_drawdown_pct >= 25.0:
            terminated = True
            reward += self._close_all_positions()

        # Terminate if portfolio value too low
        if self.portfolio.portfolio_value < self.initial_capital * 0.5:
            terminated = True
            reward += self._close_all_positions()

        # Get observation
        observation = self._get_observation()

        # Info
        info.update({
            'step': self.step_count,
            'current_time': self.current_time,
            'portfolio_value': self.portfolio.portfolio_value,
            'total_pnl_pct': self.portfolio.total_pnl_pct,
            'num_positions': len(self.portfolio.positions),
            'capital_utilization': self.portfolio.capital_utilization,
        })

        # Add episode-level metrics for TensorBoard logging
        if terminated or truncated:
            total_trades = len(self.portfolio.closed_positions)
            winning_trades = len([p for p in self.portfolio.closed_positions if p.realized_pnl_pct > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            info['episode_pnl_pct'] = self.portfolio.total_pnl_pct
            info['episode_trades_count'] = total_trades
            info['episode_win_rate'] = win_rate
            info['episode_final_value'] = self.portfolio.portfolio_value

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Tuple[float, dict]:
        """
        Execute the agent's action.

        Returns:
            (reward, info)
        """
        reward = 0.0
        info = {'action_type': 'hold'}

        # Action 0: Hold (do nothing)
        if action == 0:
            info['action_type'] = 'hold'
            # NO HOLD BONUS: Removed to avoid bias toward inaction
            # Agent should learn to trade profitably without artificial hold incentive
            return reward, info

        # Actions 1 to max_opportunities: Enter opportunity
        elif 1 <= action <= self.max_opportunities:
            opp_idx = action - 1

            if opp_idx < len(self.current_opportunities):
                reward = self._enter_position(opp_idx)
                info['action_type'] = 'enter'
                info['opportunity_idx'] = opp_idx
            else:
                # Invalid action (opportunity doesn't exist)
                # Penalize to discourage wasting learning on invalid actions
                info['action_type'] = 'invalid_enter'
                reward = -0.5

        # Actions max_opportunities+1 to max_opportunities+max_positions: Exit position
        elif self.max_opportunities < action < self.action_space.n:
            pos_idx = action - self.max_opportunities - 1

            if pos_idx < len(self.portfolio.positions):
                reward = self._exit_position(pos_idx)
                info['action_type'] = 'exit'
                info['position_idx'] = pos_idx
            else:
                # Invalid action (position doesn't exist)
                # Penalize to discourage wasting learning on invalid actions
                info['action_type'] = 'invalid_exit'
                reward = -0.5

        return reward, info

    def _enter_position(self, opportunity_idx: int) -> float:
        """
        Enter a new arbitrage execution.

        Returns:
            Immediate reward (usually 0, but could penalize entry fees)
        """
        opp = self.current_opportunities[opportunity_idx]

        # Calculate position size (use max allowed)
        max_size_per_side = self.portfolio.total_capital * (self.max_position_size_pct / 100)
        position_size = min(
            max_size_per_side,
            self.portfolio.available_capital / 2  # Need 2x for long + short
        )

        # Create position
        position = Position(
            opportunity_id=opp['opportunity_id'],
            symbol=opp['symbol'],
            long_exchange=opp['long_exchange'],
            short_exchange=opp['short_exchange'],
            entry_time=self.current_time,
            entry_long_price=opp['entry_long_price'],
            entry_short_price=opp['entry_short_price'],
            position_size_usd=position_size,
            long_funding_rate=opp['long_funding_rate'],
            short_funding_rate=opp['short_funding_rate'],
            long_funding_interval_hours=opp['long_funding_interval_hours'],
            short_funding_interval_hours=opp['short_funding_interval_hours'],
            long_next_funding_time=pd.to_datetime(opp['long_next_funding_time']),
            short_next_funding_time=pd.to_datetime(opp['short_next_funding_time']),
        )

        # Try to open position
        success = self.portfolio.open_position(position)

        if success:
            # V3 FIX: IMMEDIATE ENTRY FEE PENALTY + NO QUALITY BONUS
            # Penalize entry fee immediately for better credit assignment
            # Agent learns: "I paid $0.66 to enter, better make it worth it!"
            # This encourages selectivity - only enter when expected P&L > fee
            entry_fee_penalty = -position.entry_fees_paid_usd * self.pnl_reward_scale

            # V3 FIX: NO QUALITY BONUS
            # Remove all quality-based bonuses/penalties (lines 390-416 in V2)
            # Let agent learn purely from actual P&L outcomes:
            # - Good symbols → positive P&L → agent learns to enter more
            # - Bad symbols → negative P&L → agent learns to avoid
            # - No forward-looking rewards, only backward-looking outcomes

            return entry_fee_penalty
        else:
            # Could not open (insufficient capital or position limit)
            # Penalty to discourage invalid attempts
            return -0.5

    def _exit_position(self, position_idx: int) -> float:
        """
        Exit an existing position.

        Returns:
            Exit bonus only (P&L already rewarded hourly, avoid double-counting)
        """
        position = self.portfolio.positions[position_idx]

        # Store P&L % before closing
        pnl_pct = position.unrealized_pnl_pct

        # Get current prices for exit
        prices = self._get_current_prices()

        if position.symbol in prices:
            symbol_prices = prices[position.symbol]

            # Close position (updates portfolio)
            # P&L was already rewarded hourly through pnl_change in step()
            realized_pnl = self.portfolio.close_position(
                position_idx,
                self.current_time,
                symbol_prices['long_price'],
                symbol_prices['short_price']
            )

            # V3 FIX: NO EXIT BONUS
            # Let agent learn optimal exit timing purely from cumulative P&L rewards
            # Holding winners longer accumulates more P&L → agent naturally learns to hold
            # Exiting early misses P&L accumulation → agent naturally learns not to exit early
            return 0.0
        else:
            # No price data available (shouldn't happen)
            return 0.0

    def _check_stop_loss(self, price_data: Dict[str, Dict[str, float]]) -> float:
        """
        Check open positions for stop-loss violations and auto-close them.

        Args:
            price_data: Current price data for all symbols

        Returns:
            Total reward from closed positions (usually negative)
        """
        total_reward = 0.0
        positions_to_close = []

        # Identify positions exceeding stop-loss threshold
        for idx, position in enumerate(self.portfolio.positions):
            if position.unrealized_pnl_pct < self.max_position_loss_pct:
                positions_to_close.append((idx, position.symbol, position.unrealized_pnl_pct))

        # Close positions in reverse order to avoid index shifting
        for idx, symbol, loss_pct in reversed(positions_to_close):
            self._exit_position(idx)
            # STOP-LOSS PENALTY: Penalize positions that hit stop-loss
            # Helps agent learn to avoid bad entries
            total_reward += -1.0

        return total_reward

    def _close_all_positions(self) -> float:
        """Close all open positions at episode end."""
        # Close in reverse order to avoid index issues
        for i in range(len(self.portfolio.positions) - 1, -1, -1):
            self._exit_position(i)
            # NO REWARD: P&L was already tracked hourly via base_reward

        return 0.0  # No additional reward

    def _get_opportunities_at_time(self, timestamp: pd.Timestamp) -> List[Dict]:
        """
        Get opportunities detected at the current time.

        Returns:
            List of opportunity dictionaries
        """
        # Get opportunities detected in current hour
        window_start = timestamp
        window_end = timestamp + timedelta(hours=self.step_hours)

        opps = self.data[
            (self.data['entry_time'] >= window_start) &
            (self.data['entry_time'] < window_end)
        ].head(self.max_opportunities)

        # Convert to list of dicts
        opportunities = []
        for _, row in opps.iterrows():
            opportunities.append({
                'opportunity_id': f"{row['entry_time']}_{row['symbol']}",
                'symbol': row['symbol'],
                'long_exchange': row['long_exchange'],
                'short_exchange': row['short_exchange'],
                'entry_long_price': row['entry_long_price'],
                'entry_short_price': row['entry_short_price'],
                'long_funding_rate': row['long_funding_rate'],
                'short_funding_rate': row['short_funding_rate'],
                'long_funding_interval_hours': int(row['long_funding_interval_hours']),
                'short_funding_interval_hours': int(row['short_funding_interval_hours']),
                'long_next_funding_time': row['long_next_funding_time'],
                'short_next_funding_time': row['short_next_funding_time'],

                # Features for observation (19 features: 14 original + 5 critical additions)
                'fund_profit_8h': row.get('fund_profit_8h', 0),
                'fundProfit8h24hProj': row.get('fundProfit8h24hProj', 0),
                'fundProfit8h3dProj': row.get('fundProfit8h3dProj', 0),
                'fund_apr': row.get('fund_apr', 0),
                'fundApr24hProj': row.get('fundApr24hProj', 0),
                'fundApr3dProj': row.get('fundApr3dProj', 0),
                'spread30SampleAvg': row.get('spread30SampleAvg', 0),
                'priceSpread24hAvg': row.get('priceSpread24hAvg', 0),
                'priceSpread3dAvg': row.get('priceSpread3dAvg', 0),
                'spread_volatility_stddev': row.get('spread_volatility_stddev', 0),

                # Critical additions for opportunity quality assessment
                'volume_24h': row.get('volume_24h', 1e6),  # Default to 1M to avoid log(0)
                'bidAskSpreadPercent': row.get('bidAskSpreadPercent', 0),
                'orderbookDepthUsd': row.get('orderbookDepthUsd', 1e4),  # Default to 10k
                'estimatedProfitPercentage': row.get('estimatedProfitPercentage', 0),
                'positionCostPercent': row.get('positionCostPercent', 0.2),  # Default 0.2%
            })

        return opportunities

    def _get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for all symbols with open positions.

        Uses historical price data if available, otherwise falls back to entry prices.

        Returns:
            Dict mapping symbol to {'long_price': float, 'short_price': float}
        """
        prices = {}

        for position in self.portfolio.positions:
            symbol = position.symbol

            # Try to get real historical prices
            if self.price_loader:
                # Get prices from both exchanges
                long_price = self.price_loader.get_price(
                    symbol=symbol,
                    timestamp=self.current_time,
                    exchange=position.long_exchange.lower(),
                    fallback=True
                )

                short_price = self.price_loader.get_price(
                    symbol=symbol,
                    timestamp=self.current_time,
                    exchange=position.short_exchange.lower(),
                    fallback=True
                )

                # Use real prices if available
                if long_price is not None and short_price is not None:
                    prices[symbol] = {
                        'long_price': long_price,
                        'short_price': short_price,
                    }
                    continue

            # Fallback: Use entry prices (no P&L from price movement)
            # This is a conservative fallback when historical data is missing
            prices[symbol] = {
                'long_price': position.entry_long_price,
                'short_price': position.entry_short_price,
            }

        return prices

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).

        Returns:
            Flattened numpy array of state features
        """
        # Portfolio state
        portfolio_features = self.portfolio.get_state_features()

        # Opportunity features (padded to max_opportunities)
        opportunity_features = []

        for i in range(self.max_opportunities):
            if i < len(self.current_opportunities):
                opp = self.current_opportunities[i]

                # Extract 22 RAW features per opportunity (NO manual scaling)
                # These will be standardized by the scaler if available
                opp_feats = [
                    # Funding rates (RAW - no scaling)
                    opp.get('long_funding_rate', 0),
                    opp.get('short_funding_rate', 0),
                    # Funding intervals (normalized to 8-hour base)
                    opp.get('long_funding_interval_hours', 8) / 8,
                    opp.get('short_funding_interval_hours', 8) / 8,
                    # Funding profit projections (RAW)
                    opp.get('fund_profit_8h', 0),
                    opp.get('fundProfit8h24hProj', 0),
                    opp.get('fundProfit8h3dProj', 0),
                    # APR projections (RAW)
                    opp.get('fund_apr', 0),
                    opp.get('fundApr24hProj', 0),
                    opp.get('fundApr3dProj', 0),
                    # Spread statistics (RAW)
                    opp.get('spread30SampleAvg', 0),
                    opp.get('priceSpread24hAvg', 0),
                    opp.get('priceSpread3dAvg', 0),
                    opp.get('spread_volatility_stddev', 0),

                    # Critical additions for opportunity quality (5 features)
                    np.log10(max(float(opp.get('volume_24h', 1e6) or 1e6), 1e5)),
                    float(opp.get('bidAskSpreadPercent', 0) or 0),
                    np.log10(max(float(opp.get('orderbookDepthUsd', 1e4) or 1e4), 1e3)),
                    float(opp.get('estimatedProfitPercentage', 0) or 0),
                    float(opp.get('positionCostPercent', 0.2) or 0.2),

                    # Momentum/Trend features (3 features)
                    opp.get('spread30SampleAvg', 0) - opp.get('priceSpread24hAvg', 0),
                    opp.get('fund_apr', 0) - opp.get('fundApr24hProj', 0),
                    opp.get('priceSpread24hAvg', 0) - opp.get('priceSpread3dAvg', 0),
                ]

                # Convert to float32 and ensure no NaN/inf
                opp_feats = [float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)) for x in opp_feats]
                opportunity_features.extend(opp_feats)
            else:
                # No opportunity at this index
                opportunity_features.extend([0.0] * 22)

        # Apply StandardScaler to opportunity features if available
        opportunity_features_array = np.array(opportunity_features, dtype=np.float32)

        if self.feature_scaler is not None:
            # Reshape to (max_opportunities, 22) for scaling
            opp_reshaped = opportunity_features_array.reshape(self.max_opportunities, 22)
            # Scale each opportunity's features
            opp_scaled = self.feature_scaler.transform(opp_reshaped)
            # Flatten back to 1D
            opportunity_features_array = opp_scaled.flatten()

        # Concatenate all features
        observation = np.concatenate([
            portfolio_features,
            opportunity_features_array
        ]).astype(np.float32)

        return observation

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Time: {self.current_time} (Step {self.step_count}/{self.episode_length_hours})")
            print(f"Portfolio Value: ${self.portfolio.portfolio_value:,.2f} ({self.portfolio.total_pnl_pct:+.2f}%)")
            print(f"Open Positions: {len(self.portfolio.positions)}/{self.max_positions}")
            print(f"Capital Utilization: {self.portfolio.capital_utilization:.1f}%")
            print(f"Max Drawdown: {self.portfolio.max_drawdown_pct:.2f}%")
            print(f"Available Opportunities: {len(self.current_opportunities)}")

            if len(self.portfolio.positions) > 0:
                print(f"\nOpen Positions:")
                for i, pos in enumerate(self.portfolio.positions):
                    breakdown = pos.get_breakdown()
                    print(f"  {i+1}. {pos.symbol}: {breakdown['unrealized_pnl_pct']:+.2f}% "
                          f"({pos.hours_held:.1f}h held)")

            print(f"{'='*60}\n")

    def close(self):
        """Clean up environment resources."""
        pass
