"""
Gymnasium Environment for Funding Rate Arbitrage Trading

Simulates hour-by-hour trading with historical market data.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pickle
import bisect
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import timedelta

from .portfolio import Portfolio, Position
from .config import TradingConfig
from .reward_config import RewardConfig
from common.data.price_history_loader import PriceHistoryLoader


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
                 trading_config: Optional[TradingConfig] = None,
                 sample_random_config: bool = False,
                 max_position_size_pct: float = 33.3,
                 episode_length_days: int = 3,
                 max_opportunities_per_hour: int = 10,
                 step_hours: int = 1,
                 reward_config: Optional[RewardConfig] = None,
                 pnl_reward_scale: float = None,  # Deprecated: use reward_config
                 use_full_range_episodes: bool = False,
                 fixed_position_size_usd: float = None,
                 force_zero_total_pnl_pct: bool = False,
                 verbose: bool = True):
        """
        Initialize the environment.

        Args:
            data_path: Path to historical opportunities CSV
            price_history_path: Path to price history directory (parquet files)
            feature_scaler_path: Path to fitted StandardScaler pickle file (optional)
            initial_capital: Starting capital in USD
            trading_config: TradingConfig instance (if None, uses default moderate config)
            sample_random_config: If True, sample random config each episode (for training diversity)
            max_position_size_pct: Max % of capital per position side
            episode_length_days: Episode length in days (ignored if use_full_range_episodes=True)
            max_opportunities_per_hour: Max opportunities to show per timestep (default 10)
            step_hours: Hours per timestep (default 1 = hourly)
            pnl_reward_scale: Scaling factor for P&L rewards (default 3.0)
            use_full_range_episodes: If True, episodes span entire data range (data_start to data_end)
            fixed_position_size_usd: If set, use this fixed size per side instead of capital-based sizing
        """
        super().__init__()

        # Trading configuration
        if trading_config is None:
            # Default to moderate config if not provided
            self.trading_config = TradingConfig.get_moderate()
        else:
            self.trading_config = trading_config

        self.sample_random_config = sample_random_config  # Whether to sample new config each episode
        self.current_config = self.trading_config  # Active config for current episode

        # Configuration
        self.verbose = verbose
        self.initial_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.episode_length_hours = episode_length_days * 24
        self.max_opportunities = max_opportunities_per_hour
        self.step_hours = step_hours
        self.use_full_range_episodes = use_full_range_episodes
        self.fixed_position_size_usd = fixed_position_size_usd
        self.force_zero_total_pnl_pct = force_zero_total_pnl_pct  # For testing: simulates production bug

        # Reward configuration (Pure RL-v2 approach)
        if reward_config is None:
            reward_config = RewardConfig()  # Use defaults

        # Note: pnl_reward_scale parameter is deprecated (kept for signature compatibility)
        # Use reward_config parameter instead
        self.reward_config = reward_config

        # Store individual scales for easy access (Pure RL-v2 approach + opportunity cost)
        self.funding_reward_scale = reward_config.funding_reward_scale
        self.price_reward_scale = reward_config.price_reward_scale
        self.liquidation_penalty_scale = reward_config.liquidation_penalty_scale
        self.opportunity_cost_scale = reward_config.opportunity_cost_scale

        # Load historical data
        self.data = self._load_data(data_path)
        self.data_start = self.data['entry_time'].min()
        self.data_end = self.data['entry_time'].max()

        # Initialize price history loader
        self.price_loader = None
        if price_history_path:
            self.price_loader = PriceHistoryLoader(price_history_path)
            if self.verbose:
                print(f"   Price history loaded from: {price_history_path}")

        # Load feature scaler (for normalization)
        self.feature_scaler = None
        if feature_scaler_path and Path(feature_scaler_path).exists():
            with open(feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
            if self.verbose:
                print(f"   Feature scaler loaded from: {feature_scaler_path}")
                print(f"   Features will be standardized (mean=0, std=1)")
        elif self.verbose:
            print(f"   WARNING: No feature scaler provided - using raw features (may cause training instability)")

        # Episode state
        self.portfolio: Optional[Portfolio] = None
        self.current_time: Optional[pd.Timestamp] = None
        self.episode_start: Optional[pd.Timestamp] = None
        self.episode_end: Optional[pd.Timestamp] = None
        self.step_count = 0

        # Track previous funding total for separating funding vs price P&L
        self.previous_funding_total_usd = 0.0

        # Current opportunities available to agent
        self.current_opportunities: List[Dict] = []

        # Action space: 36 discrete actions
        # 0 = HOLD
        # 1-10 = ENTER_OPP_0-9_SMALL (10% of max allowed size)
        # 11-20 = ENTER_OPP_0-9_MEDIUM (20% of max allowed size)
        # 21-30 = ENTER_OPP_0-9_LARGE (30% of max allowed size)
        # 31-35 = EXIT_POS_0-4
        self.action_space = spaces.Discrete(36)

        # Observation space dimensions (V3 Refactoring: 301→203 dims)
        # Config: 5 dims (unchanged)
        # Portfolio: 3 dims (was 6: removed 3 historical metrics)
        # Executions: 5 slots × 17 features = 85 dims (was 100: removed 8, added 6)
        # Opportunities: 10 slots × 11 features = 110 dims (was 190: removed 9 market quality, added 1)
        config_dim = 5
        portfolio_dim = 3  # V3: 6→3 (removed avg_position_pnl_pct, total_pnl_pct, max_drawdown_pct)
        executions_dim = 5 * 17  # V3: 85 (was 100)
        opportunities_dim = 10 * 11  # V3: 110 (was 190)
        total_dim = config_dim + portfolio_dim + executions_dim + opportunities_dim  # 203

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        if self.verbose:
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
        """
        Load and prepare historical opportunity data.

        PERFORMANCE OPTIMIZATION: Pre-converts DataFrame to list of dicts
        to avoid expensive to_dict() calls in _get_opportunities_at_time().
        """
        df = pd.read_csv(data_path, low_memory=False)

        # Convert timestamps with UTC timezone
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

        # Sort by entry time
        df = df.sort_values('entry_time').reset_index(drop=True)

        # PERFORMANCE: Pre-convert entire dataset to list of dicts ONCE
        print(f"   Pre-converting {len(df):,} opportunities to dict format...")

        self.data_records = df.to_dict('records')
        self.data_timestamps = df['entry_time'].tolist()

        # Ensure int types for intervals
        for record in self.data_records:
            record['long_funding_interval_hours'] = int(record['long_funding_interval_hours'])
            record['short_funding_interval_hours'] = int(record['short_funding_interval_hours'])

        print(f"   ✓ Pre-conversion complete ({len(self.data_records):,} records)")

        return df

    def _calculate_composite_score(self, opp: Dict) -> float:
        """
        Calculate composite quality score for opportunity ranking.

        Combines APR projection with quality indicators.

        Args:
            opp: Opportunity dictionary

        Returns:
            Composite score (higher is better)
        """
        # Use 24h APR projection if available, otherwise current APR
        apr = opp.get('fund_apr_24h_proj', opp.get('fund_apr', 0))

        # Quality multiplier based on volume and spread
        volume_score = 1.0 if opp.get('volume_24h', 0) > 1_000_000 else 0.5
        spread_score = 1.0 if opp.get('bidAskSpreadPercent', 1.0) < 0.1 else 0.5

        quality_multiplier = volume_score * spread_score

        return apr * quality_multiplier

    def _select_top_opportunities(self, all_opps: List[Dict], n: int = 10) -> List[Dict]:
        """
        Select top N opportunities by raw APR (MATCHES PRODUCTION BEHAVIOR).

        Production (AgentBackgroundService.cs:257) sorts by raw FundApr:
            .OrderByDescending(o => o.FundApr).Take(7)

        Training MUST match this to see the same APR distribution!
        Previously used composite score (APR × quality), which filtered out
        high-APR low-quality opportunities that production actually sees.

        Args:
            all_opps: List of all available opportunities
            n: Number of opportunities to select

        Returns:
            Top N opportunities by highest APR (or all if fewer than N available)
        """
        if len(all_opps) == 0:
            return []

        # Sort by raw APR only (matches production exactly)
        # Use fund_apr directly, not composite score
        sorted_opps = sorted(all_opps, key=lambda opp: opp.get('fund_apr', 0), reverse=True)

        # Return top N
        return sorted_opps[:n]

    def _get_action_mask(self) -> np.ndarray:
        """
        Get boolean mask of valid actions for current state.

        Returns:
            Boolean array of shape (36,) where True = valid action

        Action indices:
            0: HOLD (always valid)
            1-10: ENTER_OPP_0-9_SMALL
            11-20: ENTER_OPP_0-9_MEDIUM
            21-30: ENTER_OPP_0-9_LARGE
            31-35: EXIT_POS_0-4
        """
        mask = np.zeros(36, dtype=bool)

        # HOLD is always valid
        mask[0] = True

        # ENTER actions: valid if opportunity exists AND we can open position AND symbol not already held
        num_positions = len(self.portfolio.positions)
        max_positions = self.current_config.max_positions
        has_capacity = num_positions < max_positions

        # Get set of symbols we already have positions in
        existing_symbols = {pos.symbol for pos in self.portfolio.positions}

        if has_capacity:
            for i in range(10):
                if i < len(self.current_opportunities):
                    opp = self.current_opportunities[i]
                    opp_symbol = opp.get('symbol', '')

                    # Only allow ENTER if we don't already have this symbol
                    if opp_symbol not in existing_symbols:
                        mask[1 + i] = True      # SMALL
                        mask[11 + i] = True     # MEDIUM
                        mask[21 + i] = True     # LARGE

        # EXIT actions: valid if position exists
        for i in range(5):
            if i < num_positions:
                mask[31 + i] = True

        return mask

    def _calculate_position_size(self, action: int) -> float:
        """
        Calculate position size based on action and current config.

        Args:
            action: Action index (1-30 for ENTER actions)

        Returns:
            Position size in USD (per side)
        """
        # Determine size multiplier based on action
        if 1 <= action <= 10:
            # SMALL: 10% of max allowed
            size_multiplier = 0.10
        elif 11 <= action <= 20:
            # MEDIUM: 20% of max allowed
            size_multiplier = 0.20
        elif 21 <= action <= 30:
            # LARGE: 30% of max allowed
            size_multiplier = 0.30
        else:
            raise ValueError(f"Invalid action for position sizing: {action}")

        # Calculate max allowed size based on config
        # max_allowed = (available_margin × target_utilization) / max_positions
        available_margin = self.portfolio.available_margin
        target_util = self.current_config.target_utilization
        max_positions = self.current_config.max_positions
        leverage = self.current_config.max_leverage

        # With leverage, we can control more capital with less margin
        # max_allowed_size = (available_margin × leverage × target_util) / max_positions
        max_allowed_size = (available_margin * leverage * target_util) / max_positions

        # Apply size multiplier
        position_size = max_allowed_size * size_multiplier

        # Ensure we don't exceed available margin
        margin_needed = (position_size * 2) / leverage
        if margin_needed > available_margin:
            # Scale down to fit available margin
            position_size = (available_margin * leverage) / 2

        return position_size

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed for reproducibility
            options: Optional dict that may contain 'trading_config' to override

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

        # Sample or set trading config for this episode
        if options and 'trading_config' in options:
            # Override with provided config
            self.current_config = options['trading_config']
        elif self.sample_random_config:
            # Sample random config for training diversity
            self.current_config = TradingConfig.sample_random(seed=seed)
            if self.verbose:
                print(f"   Sampled config: {self.current_config}")
        else:
            # Use default config
            self.current_config = self.trading_config

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
            max_positions=self.current_config.max_positions,
            max_position_size_pct=self.max_position_size_pct
        )

        # Reset counters
        self.step_count = 0
        self.previous_funding_total_usd = 0.0

        # Track episode capital for reward normalization (constant throughout episode)
        # This prevents reward inflation when positions close
        self.episode_capital_for_normalization = self.initial_capital

        # PERFORMANCE: Reset price and funding rate caches (per-timestep caching)
        self._cached_prices = None
        self._cached_prices_time = None
        self._cached_funding_rates = None
        self._cached_funding_rates_time = None

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

        # CRITICAL FIX: Update opportunities IMMEDIATELY after advancing time
        # This ensures we have current market data for the new time step
        self.current_opportunities = self._get_opportunities_at_time(self.current_time)

        # Update all open positions with new prices and funding rates
        # CRITICAL FIX: Get current funding rates from market data
        price_data = self._get_current_prices()
        funding_rates = self._get_current_funding_rates()
        pnl_change = self.portfolio.update_positions_hourly(
            self.current_time,
            price_data,
            funding_rates  # Pass current funding rates!
        )

        # Check and close positions exceeding stop-loss threshold
        stop_loss_reward = self._check_stop_loss(price_data)
        reward += stop_loss_reward

        # REWARD STRUCTURE: Pure RL-v2 Approach
        # Component 1: Funding P&L reward (1x) - Primary signal
        # Component 2: Price P&L reward (1x) - Secondary signal
        # Component 3: Liquidation risk penalty - Safety critical
        # Component 4: Opportunity cost penalty - DISABLED (causes overtrading due to step accumulation)

        # Initialize reward breakdown for logging
        reward_breakdown = {
            'funding_reward': 0.0,
            'price_reward': 0.0,
            'liquidation_penalty': 0.0,
            'opportunity_cost_penalty': 0.0,
        }

        # Component 1: P&L Reward normalized by episode capital (FIXED)
        # Use constant episode capital instead of current position sizes to prevent reward inflation
        # Get current funding total
        current_funding_total = self.portfolio.get_total_funding_usd()

        # Separate funding vs price P&L
        funding_pnl_change = current_funding_total - self.previous_funding_total_usd
        price_pnl_change = pnl_change - funding_pnl_change

        # Update tracking
        self.previous_funding_total_usd = current_funding_total

        # Normalize by CONSTANT episode capital (not dynamic position sizes)
        # This ensures reward scale is consistent regardless of how many positions are open
        funding_pnl_pct = (funding_pnl_change / self.episode_capital_for_normalization) * 100
        price_pnl_pct = (price_pnl_change / self.episode_capital_for_normalization) * 100

        funding_reward = funding_pnl_pct * self.funding_reward_scale
        price_reward = price_pnl_pct * self.price_reward_scale

        reward += funding_reward + price_reward
        reward_breakdown['funding_reward'] = funding_reward
        reward_breakdown['price_reward'] = price_reward

        # Component 2: Liquidation Risk Penalty (Safety Critical)
        min_liq_distance = self.portfolio.get_min_liquidation_distance(price_data)
        liquidation_buffer = self.current_config.liquidation_buffer  # 0.15 default

        if min_liq_distance < liquidation_buffer:
            # Steep penalty for approaching liquidation
            # Example: 0.10 distance with 0.15 buffer → (0.15-0.10) * 10 = -0.5
            liquidation_penalty = -(liquidation_buffer - min_liq_distance) * self.liquidation_penalty_scale
            reward += liquidation_penalty
            reward_breakdown['liquidation_penalty'] = liquidation_penalty

        # Component 3: Opportunity Cost Penalty (Incentivizes rotation to better opportunities)
        # Applied when holding positions with SIGNIFICANTLY lower APR than best available
        # Uses a threshold to avoid constant churning on minor differences
        if self.opportunity_cost_scale > 0.0 and len(self.portfolio.positions) > 0:
            # Find best available APR from current opportunities
            best_available_apr = 0.0
            if len(self.current_opportunities) > 0:
                best_available_apr = max(opp.get('fund_apr', 0.0) for opp in self.current_opportunities)

            # Calculate opportunity cost for each position
            opportunity_cost_total = 0.0
            apr_threshold = 10.0  # Only penalize if APR gap > 10% (e.g., 25% vs 35%+)

            for pos in self.portfolio.positions:
                # Look up current APR for this symbol from opportunities
                current_pos_apr = 0.0
                for opp in self.current_opportunities:
                    if opp['symbol'] == pos.symbol:
                        current_pos_apr = opp.get('fund_apr', 0.0)
                        break

                apr_gap = best_available_apr - current_pos_apr

                # Only apply penalty if gap exceeds threshold (significantly better opportunity exists)
                if apr_gap > apr_threshold:
                    # Penalty proportional to APR gap beyond threshold and capital in this position
                    position_capital_ratio = (pos.position_size_usd * 2) / self.episode_capital_for_normalization
                    opportunity_cost = -(apr_gap - apr_threshold) * position_capital_ratio * self.opportunity_cost_scale
                    opportunity_cost_total += opportunity_cost

            reward += opportunity_cost_total
            reward_breakdown['opportunity_cost_penalty'] = opportunity_cost_total

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

        # V3: Update velocity tracking for next step (DISABLED - velocity features set to 0)
        # This stores current values as "previous" for velocity calculations
        # self.portfolio.update_velocity_tracking(price_data)

        # Info
        info.update({
            'step': self.step_count,
            'current_time': self.current_time,
            'portfolio_value': self.portfolio.portfolio_value,
            'total_pnl_pct': self.portfolio.total_pnl_pct,
            'num_positions': len(self.portfolio.positions),
            'capital_utilization': self.portfolio.capital_utilization,
            'reward_breakdown': reward_breakdown,  # For reward component analysis
        })

        # Add episode-level metrics for TensorBoard logging
        if terminated or truncated:
            total_trades = len(self.portfolio.closed_positions)
            winning_trades = len([p for p in self.portfolio.closed_positions if p.realized_pnl_pct > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            # CAPITAL-INDEPENDENT METRIC: Use execution-based P&L, not portfolio impact
            info['episode_pnl_pct'] = self.portfolio.get_execution_avg_pnl_pct()
            info['episode_trades_count'] = total_trades
            info['episode_win_rate'] = win_rate
            info['episode_final_value'] = self.portfolio.portfolio_value

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Tuple[float, dict]:
        """
        Execute the agent's action.

        36 actions:
        - 0: HOLD
        - 1-10: ENTER_OPP_0-9_SMALL
        - 11-20: ENTER_OPP_0-9_MEDIUM
        - 21-30: ENTER_OPP_0-9_LARGE
        - 31-35: EXIT_POS_0-4

        Returns:
            (reward, info)
        """
        reward = 0.0
        info = {'action_type': 'hold'}

        if action == 0:
            # HOLD - Pure RL-v2: No opportunity cost penalty
            # Agent learns to take good opportunities through experience:
            # - Taking good opportunities → accumulates P&L → higher cumulative reward
            # - Missing good opportunities → misses P&L → lower cumulative reward
            # This natural feedback teaches opportunity selection without artificial penalties
            info['action_type'] = 'hold'
            return reward, info

        elif 1 <= action <= 30:
            # ENTER actions
            # Decode opportunity index and size
            if 1 <= action <= 10:
                opp_idx = action - 1
                size_type = 'small'
            elif 11 <= action <= 20:
                opp_idx = action - 11
                size_type = 'medium'
            else:  # 21-30
                opp_idx = action - 21
                size_type = 'large'

            # Validate opportunity exists
            if opp_idx < len(self.current_opportunities):
                reward = self._enter_position(opp_idx, action=action)
                info['action_type'] = f'enter_{size_type}'
                info['opportunity_idx'] = opp_idx
                info['size_type'] = size_type
            else:
                # Invalid: opportunity doesn't exist (masked, shouldn't happen)
                info['action_type'] = 'invalid_enter'
                reward = 0.0  # No penalty - rely on action masking

        elif 31 <= action <= 35:
            # EXIT actions
            pos_idx = action - 31

            # Validate position exists
            if pos_idx < len(self.portfolio.positions):
                reward = self._exit_position(pos_idx)
                info['action_type'] = 'exit'
                info['position_idx'] = pos_idx
            else:
                # Invalid: position doesn't exist (masked, shouldn't happen)
                info['action_type'] = 'invalid_exit'
                reward = 0.0  # No penalty - rely on action masking

        else:
            # Should never happen (action out of range)
            info['action_type'] = 'invalid'
            reward = 0.0  # No penalty

        return reward, info

    def _enter_position(self, opportunity_idx: int, action: Optional[int] = None) -> float:
        """
        Enter a new arbitrage execution.

        Args:
            opportunity_idx: Index of opportunity to enter
            action: Action index (for full mode dynamic sizing)

        Returns:
            Immediate reward (entry fee penalty)
        """
        opp = self.current_opportunities[opportunity_idx]

        # Determine leverage from config
        leverage = self.current_config.max_leverage

        # Calculate position size
        if self.fixed_position_size_usd is not None:
            # Fixed size mode (capital-independent training)
            position_size = self.fixed_position_size_usd
        elif action is not None:
            # Dynamic sizing based on action
            position_size = self._calculate_position_size(action)
        else:
            # Legacy: capital-based sizing
            max_size_per_side = self.portfolio.total_capital * (self.max_position_size_pct / 100)
            # With leverage, we can control more with less capital
            available_for_position = self.portfolio.available_margin * leverage / 2
            position_size = min(max_size_per_side, available_for_position)

        # CRITICAL FIX: Fetch ACTUAL market prices at entry time from price history
        # CSV prices may be stale, averaged, or from a different source
        # We must use the same price source for entry AND exit for accurate P&L
        entry_long_price = opp['entry_long_price']  # Default from CSV
        entry_short_price = opp['entry_short_price']

        # CRITICAL FIX: Also fetch ACTUAL funding rates from price history
        # CSV rates are frozen at detection time, but rates change hourly
        entry_long_rate = opp['long_funding_rate']  # Default from CSV
        entry_short_rate = opp['short_funding_rate']

        if self.price_loader:
            # Get real market prices at current_time
            actual_long_price = self.price_loader.get_price(
                symbol=opp['symbol'],
                timestamp=self.current_time,
                exchange=opp['long_exchange'].lower(),
                fallback=True
            )
            actual_short_price = self.price_loader.get_price(
                symbol=opp['symbol'],
                timestamp=self.current_time,
                exchange=opp['short_exchange'].lower(),
                fallback=True
            )

            # Get real funding rates at current_time
            actual_long_rate = self.price_loader.get_funding_rate(
                symbol=opp['symbol'],
                timestamp=self.current_time,
                exchange=opp['long_exchange'].lower(),
                fallback=True
            )
            actual_short_rate = self.price_loader.get_funding_rate(
                symbol=opp['symbol'],
                timestamp=self.current_time,
                exchange=opp['short_exchange'].lower(),
                fallback=True
            )

            # Use actual prices if available, otherwise fall back to CSV
            if actual_long_price is not None:
                entry_long_price = actual_long_price
            if actual_short_price is not None:
                entry_short_price = actual_short_price

            # Use actual funding rates if available, otherwise fall back to CSV
            if actual_long_rate is not None:
                entry_long_rate = actual_long_rate
            if actual_short_rate is not None:
                entry_short_rate = actual_short_rate

        # Create position
        position = Position(
            opportunity_id=opp['opportunity_id'],
            symbol=opp['symbol'],
            long_exchange=opp['long_exchange'],
            short_exchange=opp['short_exchange'],
            entry_time=self.current_time,
            entry_long_price=entry_long_price,  # Use actual market price
            entry_short_price=entry_short_price,  # Use actual market price
            position_size_usd=position_size,
            leverage=leverage,  # Use config leverage
            long_funding_rate=entry_long_rate,  # Use actual market rate
            short_funding_rate=entry_short_rate,  # Use actual market rate
            long_funding_interval_hours=opp['long_funding_interval_hours'],
            short_funding_interval_hours=opp['short_funding_interval_hours'],
            long_next_funding_time=pd.to_datetime(opp['long_next_funding_time']),
            short_next_funding_time=pd.to_datetime(opp['short_next_funding_time']),
            entry_apr=opp.get('fund_apr', 0.0),  # Use APR directly from opportunity
        )

        # Try to open position
        success = self.portfolio.open_position(position)

        if success:
            # Pure RL-v2: No entry penalty
            # Entry fees are naturally deducted from P&L, which reduces cumulative reward
            # Agent learns selectivity through outcomes, not artificial penalties
            return 0.0
        else:
            # Could not open (insufficient capital or position limit - masked, shouldn't happen)
            # No penalty - rely on action masking to prevent invalid actions
            return 0.0

    def _exit_position(self, position_idx: int) -> float:
        """
        Exit an existing position.

        RL-v2 Philosophy: NO EXIT REWARD
        Agent learns optimal exit timing purely from cumulative P&L rewards.
        - Holding winners longer accumulates more P&L → agent naturally learns to hold
        - Exiting early misses P&L accumulation → agent naturally learns not to exit early
        - Holding losers reduces P&L → agent naturally learns to exit losers

        Returns:
            Reward for exiting (includes bonus for exiting negative funding positions)
        """
        position = self.portfolio.positions[position_idx]

        # Calculate estimated funding BEFORE closing
        # Use estimated rates (with fallback chain) instead of actual rates
        estimated_rates = self._get_current_funding_rates(for_pnl=False)
        symbol_rates = estimated_rates.get(position.symbol, {})
        estimated_long_rate = symbol_rates.get('long_rate', position.long_funding_rate)
        estimated_short_rate = symbol_rates.get('short_rate', position.short_funding_rate)

        # Calculate 8h funding using estimated rates
        long_payments_8h = 8.0 / position.long_funding_interval_hours
        short_payments_8h = 8.0 / position.short_funding_interval_hours
        long_funding_8h = -estimated_long_rate * long_payments_8h
        short_funding_8h = estimated_short_rate * short_payments_8h
        estimated_funding_8h_pct = (long_funding_8h + short_funding_8h) * 100

        # Get current prices for exit
        prices = self._get_current_prices()

        if position.symbol in prices:
            symbol_prices = prices[position.symbol]

            # Close position (updates portfolio)
            realized_pnl = self.portfolio.close_position(
                position_idx,
                self.current_time,
                symbol_prices['long_price'],
                symbol_prices['short_price']
            )

            # Negative funding exit reward: Bonus for exiting positions losing money via funding
            # When estimated_funding_8h_pct < 0, position is paying funding (bad!)
            # Reward agent for recognizing and exiting these positions
            exit_reward = 0.0
            if estimated_funding_8h_pct < 0:
                # Positive reward for exiting negative funding position
                # Scale by magnitude of negative funding and config scale
                exit_reward = abs(estimated_funding_8h_pct) * self.reward_config.negative_funding_exit_reward_scale

            return exit_reward
        else:
            # No price data available (shouldn't happen)
            return 0.0

    def _check_stop_loss(self, price_data: Dict[str, Dict[str, float]]) -> float:
        """
        Check open positions for stop-loss violations and auto-close them.

        Uses stop_loss_threshold from config.

        Args:
            price_data: Current price data for all symbols

        Returns:
            Total reward from closed positions (stop-loss penalty)
        """
        total_reward = 0.0
        positions_to_close = []

        # Get stop-loss threshold from config
        stop_loss_threshold = self.current_config.stop_loss_threshold

        # Identify positions exceeding stop-loss threshold
        for idx, position in enumerate(self.portfolio.positions):
            if position.unrealized_pnl_pct / 100 < stop_loss_threshold:  # Convert to decimal
                positions_to_close.append((idx, position.symbol, position.unrealized_pnl_pct))

        # Close positions in reverse order to avoid index shifting
        for idx, symbol, loss_pct in reversed(positions_to_close):
            self._exit_position(idx)
            # Pure RL-v2: No stop-loss penalty
            # The -2% loss is already negative P&L reward (naturally punishing)
            # Agent learns risk management through outcomes, not artificial penalties

        return total_reward  # Always 0.0 now

    def _close_all_positions(self) -> float:
        """Close all open positions at episode end."""
        # Close in reverse order to avoid index issues
        for i in range(len(self.portfolio.positions) - 1, -1, -1):
            self._exit_position(i)
            # NO REWARD: P&L was already tracked hourly via base_reward

        return 0.0  # No additional reward

    def _get_opportunities_at_time(self, timestamp: pd.Timestamp) -> List[Dict]:
        """
        PERFORMANCE OPTIMIZED: Uses pre-converted data_records with O(log n) binary search
        instead of O(n) DataFrame filtering + expensive to_dict() conversion.

        Before: 13.9 seconds per 3 episodes (38% of total time!)
        After: <0.5 seconds per 3 episodes (~30x speedup)

        In full mode, selects top 10 by composite score.
        In simple mode, selects top 1.

        Returns:
            List of opportunity dictionaries
        """
        # Get ALL opportunities detected in current hour
        window_start = timestamp
        window_end = timestamp + timedelta(hours=self.step_hours)

        # FAST: O(log n) binary search on pre-sorted timestamps
        start_idx = bisect.bisect_left(self.data_timestamps, window_start)
        end_idx = bisect.bisect_left(self.data_timestamps, window_end, lo=start_idx)

        # Collect opportunities in time window
        all_opportunities = []
        for i in range(start_idx, end_idx):
            record = self.data_records[i]
            opp = record.copy()
            opp['opportunity_id'] = f"{record['entry_time']}_{record['symbol']}"
            all_opportunities.append(opp)

        # Select top N opportunities by composite score
        top_opportunities = self._select_top_opportunities(all_opportunities, n=self.max_opportunities)

        return top_opportunities

    def _get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for all symbols with open positions.

        PERFORMANCE: Caches prices per timestep (called 3-5 times per step).

        Uses historical price data if available, otherwise falls back to entry prices.

        Returns:
            Dict mapping symbol to {'long_price': float, 'short_price': float}
        """
        # Return cached prices if already computed for this timestep
        if self._cached_prices_time == self.current_time and self._cached_prices is not None:
            return self._cached_prices

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

        # Cache for this timestep
        self._cached_prices = prices
        self._cached_prices_time = self.current_time

        return prices

    def _get_current_funding_rates(self, for_pnl: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Get current funding rates for all symbols with open positions.

        PERFORMANCE: Caches rates per timestep (called 2-3 times per step).

        CRITICAL FIX: Funding rates change hourly and must be updated from market data!
        Rates are fetched from price_history, NOT from static opportunity CSV.

        Args:
            for_pnl: If True (default), use ONLY parquet data for P&L calculation (0 if missing).
                    If False, use fallback chain for feature estimation: parquet → CSV → last known.

        Returns:
            Dict mapping symbol to {'long_rate': float, 'short_rate': float}
        """
        # Return cached rates if already computed for this timestep and mode
        # Note: We cache separately for P&L vs estimation modes
        cache_key = (self.current_time, for_pnl)
        if hasattr(self, '_cached_funding_rates_dict') and cache_key in self._cached_funding_rates_dict:
            return self._cached_funding_rates_dict[cache_key]

        funding_rates = {}

        for position in self.portfolio.positions:
            symbol = position.symbol
            long_exchange = position.long_exchange
            short_exchange = position.short_exchange

            # CRITICAL FIX: Fetch ACTUAL hourly funding rates from price history
            # NOT from opportunity CSV which has frozen rates at detection time
            if self.price_loader:
                long_rate = self.price_loader.get_funding_rate(
                    symbol=symbol,
                    timestamp=self.current_time,
                    exchange=long_exchange.lower(),
                    fallback=True
                )
                short_rate = self.price_loader.get_funding_rate(
                    symbol=symbol,
                    timestamp=self.current_time,
                    exchange=short_exchange.lower(),
                    fallback=True
                )

                if for_pnl:
                    # P&L calculation: Use parquet data, but 0.0 means "no payment this hour"
                    # Funding rates persist between payments - use last known rate from position
                    if long_rate is None or long_rate == 0.0:
                        long_rate = position.long_funding_rate  # Keep current rate
                    if short_rate is None or short_rate == 0.0:
                        short_rate = position.short_funding_rate  # Keep current rate

                    funding_rates[symbol] = {
                        'long_rate': long_rate,
                        'short_rate': short_rate,
                    }
                else:
                    # Feature estimation: Use fallback chain for better estimates
                    # parquet → opportunity CSV → last known position rate
                    # Check for None OR 0.0 (parquet returns 0.0 when no funding payment)
                    if long_rate is None or long_rate == 0.0 or short_rate is None or short_rate == 0.0:
                        # Fallback to CSV rates from opportunities
                        csv_rates = self._get_funding_rates_from_opportunities(symbol)
                        if long_rate is None or long_rate == 0.0:
                            long_rate = csv_rates.get('long_rate', position.long_funding_rate)
                        if short_rate is None or short_rate == 0.0:
                            short_rate = csv_rates.get('short_rate', position.short_funding_rate)

                    funding_rates[symbol] = {
                        'long_rate': long_rate,
                        'short_rate': short_rate,
                    }
            else:
                # No price_loader available: Keep current rates
                funding_rates[symbol] = {
                    'long_rate': position.long_funding_rate,
                    'short_rate': position.short_funding_rate,
                }

        # Cache for this timestep and mode
        if not hasattr(self, '_cached_funding_rates_dict'):
            self._cached_funding_rates_dict = {}
        self._cached_funding_rates_dict[cache_key] = funding_rates

        return funding_rates

    def _get_funding_rates_from_opportunities(self, symbol: str) -> Dict[str, float]:
        """
        Get funding rates from current opportunities CSV data (fallback source).

        This is used when parquet files have missing/0 funding rates and we need
        a better estimate for feature calculation (not for P&L).

        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Dict with 'long_rate' and 'short_rate' keys (0.0 if not found)
        """
        for opp in self.current_opportunities:
            if opp['symbol'] == symbol:
                return {
                    'long_rate': opp.get('long_funding_rate', 0.0),
                    'short_rate': opp.get('short_funding_rate', 0.0),
                }
        # Symbol not found in opportunities
        return {'long_rate': 0.0, 'short_rate': 0.0}

    def get_raw_state_for_ml_api(self) -> dict:
        """
        Get the EXACT raw state the environment used for the last observation.

        CRITICAL: This returns the CACHED raw data dict that was built during
        the most recent _get_observation() call. This ensures 100% consistency
        between direct mode and ML API mode - both use the exact same data.

        Returns:
            Dict with trading_config, portfolio, opportunities (cached from last observation)
        """
        # Return cached raw data from the last _get_observation() call
        # This ensures we use the EXACT same data that was used to build the observation
        if hasattr(self, '_cached_raw_data'):
            return self._cached_raw_data

        # Fallback: if no cache exists (e.g., before first step), build it now
        # This should only happen in edge cases
        price_data = self._get_current_prices()

        # Build trading config
        config_array = self.current_config.to_array()
        trading_config = {
            'max_leverage': float(config_array[0]),
            'target_utilization': float(config_array[1]),
            'max_positions': int(config_array[2]),
            'stop_loss_threshold': float(config_array[3]),
            'liquidation_buffer': float(config_array[4]),
        }

        # Build positions list
        positions = []
        for pos in self.portfolio.positions:
            symbol_prices = price_data.get(pos.symbol, {})
            current_long_price = symbol_prices.get('long_price', pos.entry_long_price)
            current_short_price = symbol_prices.get('short_price', pos.entry_short_price)

            current_position_apr = 0.0
            for opp in self.current_opportunities:
                if opp['symbol'] == pos.symbol:
                    current_position_apr = opp.get('fund_apr', 0.0)
                    break

            positions.append({
                'is_active': True,
                'symbol': pos.symbol,
                'position_size_usd': float(pos.position_size_usd),
                'position_age_hours': float(pos.hours_held),
                'leverage': float(pos.leverage),
                'entry_long_price': float(pos.entry_long_price),
                'entry_short_price': float(pos.entry_short_price),
                'current_long_price': float(current_long_price),
                'current_short_price': float(current_short_price),
                'unrealized_pnl_pct': float(pos.unrealized_pnl_pct),
                'long_pnl_pct': float(pos.long_pnl_pct),
                'short_pnl_pct': float(pos.short_pnl_pct),
                'long_funding_rate': float(pos.long_funding_rate),
                'short_funding_rate': float(pos.short_funding_rate),
                'long_funding_interval_hours': int(pos.long_funding_interval_hours),
                'short_funding_interval_hours': int(pos.short_funding_interval_hours),
                'entry_apr': float(pos.entry_apr),
                'current_position_apr': float(current_position_apr),
                'liquidation_distance': float(pos.get_liquidation_distance(current_long_price, current_short_price)),
                'slippage_pct': float(pos.slippage_pct),
            })

        portfolio = {
            'positions': positions,
            'total_capital': float(self.portfolio.total_capital),
            'capital_utilization': float(self.portfolio.capital_utilization),
        }

        return {
            'trading_config': trading_config,
            'portfolio': portfolio,
            'opportunities': self.current_opportunities,
        }

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state) using UnifiedFeatureBuilder.

        203 dimensions (V3 Refactoring):
          config (5) + portfolio (3) + executions (85) + opportunities (110)

        V3 Changes:
        - Portfolio: 6→3 (removed historical metrics)
        - Executions: 100→85 (5×20→5×17)
        - Opportunities: 190→110 (10×19→10×11)

        Returns:
            Flattened numpy array of state features
        """
        # Import UnifiedFeatureBuilder
        from common.features import UnifiedFeatureBuilder

        # Initialize if not already done
        if not hasattr(self, 'feature_builder'):
            self.feature_builder = UnifiedFeatureBuilder(feature_scaler_path=None)
            self.feature_builder.feature_scaler = self.feature_scaler

        # Get current prices
        price_data = self._get_current_prices()

        # Build trading config
        config_array = self.current_config.to_array()
        trading_config = {
            'max_leverage': float(config_array[0]),
            'target_utilization': float(config_array[1]),
            'max_positions': int(config_array[2]),
            'stop_loss_threshold': float(config_array[3]),
            'liquidation_buffer': float(config_array[4]),
        }

        # Get estimated funding rates (with fallback chain for features)
        estimated_funding_rates = self._get_current_funding_rates(for_pnl=False)

        # Build positions list - convert Position objects to dicts with ALL needed attributes
        positions = []
        for pos in self.portfolio.positions:
            # Get current prices
            symbol_prices = price_data.get(pos.symbol, {})
            current_long_price = symbol_prices.get('long_price', pos.entry_long_price)
            current_short_price = symbol_prices.get('short_price', pos.entry_short_price)

            # Find current APR for this symbol
            current_position_apr = 0.0
            for opp in self.current_opportunities:
                if opp['symbol'] == pos.symbol:
                    current_position_apr = opp.get('fund_apr', 0.0)
                    break

            # Get estimated rates for this position (with fallback to CSV if parquet=0)
            estimated_rates = estimated_funding_rates.get(pos.symbol, {})
            estimated_long_rate = estimated_rates.get('long_rate', pos.long_funding_rate)
            estimated_short_rate = estimated_rates.get('short_rate', pos.short_funding_rate)

            positions.append({
                'is_active': True,
                'symbol': pos.symbol,
                'position_size_usd': float(pos.position_size_usd),
                'position_age_hours': float(pos.hours_held),
                'leverage': float(pos.leverage),
                'entry_long_price': float(pos.entry_long_price),
                'entry_short_price': float(pos.entry_short_price),
                'current_long_price': float(current_long_price),
                'current_short_price': float(current_short_price),
                'unrealized_pnl_pct': float(pos.unrealized_pnl_pct),
                'long_pnl_pct': float(pos.long_pnl_pct),  # Direct from Position object
                'short_pnl_pct': float(pos.short_pnl_pct),  # Direct from Position object
                'long_funding_rate': float(pos.long_funding_rate),  # Actual rate (for P&L)
                'short_funding_rate': float(pos.short_funding_rate),  # Actual rate (for P&L)
                'estimated_long_funding_rate': float(estimated_long_rate),  # Estimated (for features)
                'estimated_short_funding_rate': float(estimated_short_rate),  # Estimated (for features)
                'long_funding_interval_hours': int(pos.long_funding_interval_hours),
                'short_funding_interval_hours': int(pos.short_funding_interval_hours),
                'entry_apr': float(pos.entry_apr),
                'current_position_apr': float(current_position_apr),
                'liquidation_distance': float(pos.get_liquidation_distance(current_long_price, current_short_price)),
                'slippage_pct': float(pos.slippage_pct),
            })

        # Build portfolio state
        portfolio_dict = {
            'positions': positions,
            'total_capital': float(self.portfolio.total_capital),
            'capital_utilization': float(self.portfolio.capital_utilization),
        }

        # Calculate best available APR
        best_available_apr = 0.0
        if len(self.current_opportunities) > 0:
            best_available_apr = max(opp.get('fund_apr', 0.0) for opp in self.current_opportunities)

        # Build raw data dict
        raw_data = {
            'trading_config': trading_config,
            'portfolio': portfolio_dict,
            'opportunities': self.current_opportunities
        }

        # CRITICAL: Cache raw_data for ML API mode (deep copy to avoid mutation)
        # This ensures get_raw_state_for_ml_api() returns the EXACT same data
        # that was used to build this observation, even if opportunities list changes
        import copy
        self._cached_raw_data = copy.deepcopy(raw_data)

        # Use UnifiedFeatureBuilder (with optional logging)
        log_file = getattr(self, 'feature_log_file', None)
        observation = self.feature_builder.build_observation_from_raw_data(raw_data, log_file=log_file)

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
