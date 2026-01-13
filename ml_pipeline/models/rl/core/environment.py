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
from enum import Enum

from .portfolio import Portfolio, Position
from .config import TradingConfig
from .reward_config import RewardConfig
from common.data.price_history_loader import PriceHistoryLoader
from common.features.feature_config import DIMS


class ExitType(Enum):
    """
    Type of position exit for reward differentiation.

    AGENT: Agent chose to exit (action 31-35) - Gets full price P&L reward
    STOP_LOSS: Triggered by stop-loss - Gets price P&L reward (penalty is natural)
    EPISODE_END: Forced at episode boundary - NO price P&L reward (incomplete trade)
    """
    AGENT = "agent"
    STOP_LOSS = "stop_loss"
    EPISODE_END = "episode"


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
                 episode_length_hours_min: int = None,  # For random episode lengths
                 episode_length_hours_max: int = None,  # For random episode lengths
                 max_opportunities_per_hour: int = 5,  # V8: reduced from 10
                 step_hours: int = 1,
                 reward_config: Optional[RewardConfig] = None,
                 pnl_reward_scale: float = None,  # Deprecated: use reward_config
                 use_full_range_episodes: bool = False,
                 fixed_position_size_usd: float = None,
                 force_zero_total_pnl_pct: bool = False,
                 disable_early_termination: bool = False,
                 mask_enter_actions: bool = True,
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
            max_opportunities_per_hour: Max opportunities to show per timestep (V8: default 5)
            step_hours: Hours per timestep (default 1 = hourly)
            pnl_reward_scale: Scaling factor for P&L rewards (default 3.0)
            use_full_range_episodes: If True, episodes span entire data range (data_start to data_end)
            fixed_position_size_usd: If set, use this fixed size per side instead of capital-based sizing
            mask_enter_actions: If True (default), apply APR/timing filters to ENTER actions.
                               Set False for training to let model learn entry criteria.
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
        self.disable_early_termination = disable_early_termination  # For testing: disable 25% drawdown termination
        self.mask_enter_actions = mask_enter_actions  # For training: disable to let model learn entry criteria

        # Random episode length support (to prevent episode-length-dependent learning)
        # If min/max not set, use fixed episode length
        self.episode_length_hours_min = episode_length_hours_min
        self.episode_length_hours_max = episode_length_hours_max
        self.use_random_episode_length = (episode_length_hours_min is not None and
                                          episode_length_hours_max is not None)

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
        self.negative_funding_exit_reward_scale = reward_config.negative_funding_exit_reward_scale
        self.trade_diversity_bonus = reward_config.trade_diversity_bonus
        self.inactivity_penalty_hours = reward_config.inactivity_penalty_hours
        self.inactivity_penalty_scale = reward_config.inactivity_penalty_scale

        # V7: New reward parameters for APR direction flip detection
        self.negative_apr_penalty_scale = reward_config.negative_apr_penalty_scale
        self.apr_flip_exit_bonus_scale = reward_config.apr_flip_exit_bonus_scale
        self.opportunity_cost_threshold = reward_config.opportunity_cost_threshold

        # V8: New reward parameters for profit protection and entry quality
        self.peak_drawdown_threshold = reward_config.peak_drawdown_threshold
        self.peak_drawdown_penalty_scale = reward_config.peak_drawdown_penalty_scale
        self.entry_quality_penalty_scale = reward_config.entry_quality_penalty_scale
        self.apr_decline_threshold = reward_config.apr_decline_threshold
        self.apr_decline_penalty_scale = reward_config.apr_decline_penalty_scale

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

        # Action space: 17 discrete actions (V9: single position)
        # 0 = HOLD
        # 1-5 = ENTER_OPP_0-4_SMALL (10% of max allowed size)
        # 6-10 = ENTER_OPP_0-4_MEDIUM (20% of max allowed size)
        # 11-15 = ENTER_OPP_0-4_LARGE (30% of max allowed size)
        # 16 = EXIT_POS_0
        self.action_space = spaces.Discrete(DIMS.TOTAL_ACTIONS)  # V9: 17 actions

        # Observation space dimensions (V10: 91 dims)
        # Config: 5 dims
        # Portfolio: 2 dims (min_liq_distance, time_to_next_funding_norm)
        # Executions: 1 slot × 19 features = 19 dims
        # Opportunities: 5 slots × 13 features = 65 dims (V10: +time_to_profitable_funding)
        # Uses DIMS from feature_config.py as single source of truth
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(DIMS.TOTAL,),
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

    def is_in_funding_recalc_window(self) -> bool:
        """
        Check if current_time is in funding rate recalculation window.

        Funding rates are unreliable during:
        - 5 minutes before hour end (XX:55-XX:59)
        - 10 minutes after hour start (XX:00-XX:10)

        During this window, predictions should be skipped (force HOLD)
        but P&L calculations should still continue.

        Returns:
            True if in skip window, False otherwise
        """
        if self.current_time is None:
            return False
        minute = self.current_time.minute
        return minute >= 55 or minute <= 10

    def _select_top_opportunities(self, all_opps: List[Dict], n: int = 10) -> List[Dict]:
        """
        Select top N opportunities by raw APR, ensuring position opportunities are always included.

        CRITICAL: Position opportunities MUST be included so current_position_apr lookup works.
        If a position's symbol is not in top N by APR, we include it anyway by removing
        the lowest-APR opportunities to make room.

        Production (AgentBackgroundService.cs) matches this behavior:
        - Deduplicates by symbol (keeps highest APR per symbol)
        - Then selects top N

        Args:
            all_opps: List of all available opportunities
            n: Number of opportunities to select

        Returns:
            Top N opportunities with position opportunities guaranteed to be included
        """
        if len(all_opps) == 0:
            return []

        # STEP 1: Deduplicate by symbol - keep only the highest APR per symbol
        # This matches production backend behavior
        best_by_symbol: Dict[str, Dict] = {}
        for opp in all_opps:
            symbol = opp.get('symbol', '')
            if not symbol:
                continue
            current_apr = opp.get('fund_apr', 0)
            if symbol not in best_by_symbol or current_apr > best_by_symbol[symbol].get('fund_apr', 0):
                best_by_symbol[symbol] = opp

        deduplicated_opps = list(best_by_symbol.values())

        # Get symbols from open positions
        position_symbols = {pos.symbol for pos in self.portfolio.positions}

        # Separate opportunities: those matching positions vs others
        position_opps = [opp for opp in deduplicated_opps if opp.get('symbol') in position_symbols]
        other_opps = [opp for opp in deduplicated_opps if opp.get('symbol') not in position_symbols]

        # Sort others by APR (descending)
        other_opps.sort(key=lambda opp: opp.get('fund_apr', 0), reverse=True)

        # Build final list: position opps first, then fill remaining slots
        remaining_slots = max(0, n - len(position_opps))
        result = position_opps + other_opps[:remaining_slots]

        return result

    def _get_action_mask(self) -> np.ndarray:
        """
        Get boolean mask of valid actions for current state (V10: 17 actions).

        Delegates to UnifiedFeatureBuilder.get_action_mask() to ensure consistency
        between direct mode and API mode.

        ENTER masking criteria (V11):
        - HARD: Must have position capacity
        - HARD: Must not have existing position for same symbol
        - SOFT (mask_enter_actions=True): fund_apr >= MIN_APR (2500)
        - SOFT (mask_enter_actions=True): fund_apr <= MAX_APR (15000)
        - SOFT (mask_enter_actions=True): time_to_funding <= MAX_MINUTES (30)
        - SOFT (mask_enter_actions=True): liquidity in ALLOWED_LIQUIDITY

        Returns:
            Boolean array of shape (17,) where True = valid action

        Action indices:
            0: HOLD (always valid)
            1-5: ENTER_OPP_0-4_SMALL
            6-10: ENTER_OPP_0-4_MEDIUM
            11-15: ENTER_OPP_0-4_LARGE
            16: EXIT_POS_0
        """
        # Masking parameters
        MIN_APR = 2000.0
        MAX_APR = 15000.0
        MAX_MINUTES_TO_FUNDING = 30.0
        ALLOWED_LIQUIDITY = [0.0]  # Good (0) only, block Medium (1) and Low (2)

        # Prepare opportunities with has_existing_position flag
        existing_symbols = {pos.symbol for pos in self.portfolio.positions}
        opportunities_with_flags = []
        for opp in self.current_opportunities:
            opp_copy = opp.copy()
            opp_copy['has_existing_position'] = opp.get('symbol', '') in existing_symbols
            opportunities_with_flags.append(opp_copy)

        # Delegate to UnifiedFeatureBuilder for consistent masking with API mode
        return self.feature_builder.get_action_mask(
            opportunities=opportunities_with_flags,
            num_positions=len(self.portfolio.positions),
            max_positions=self.current_config.max_positions,
            current_time=self.current_time,
            max_minutes_to_funding=MAX_MINUTES_TO_FUNDING,
            min_apr=MIN_APR,
            max_apr=MAX_APR,
            mask_enter=self.mask_enter_actions,
            allowed_liquidity=ALLOWED_LIQUIDITY
        )

    def _calc_minutes_to_profitable_funding(self, opp: Dict) -> float:
        """
        Calculate minutes until next funding payment on the profitable side (V10).

        For funding arbitrage:
        - Long side profitable when long_funding_rate < 0 (we receive funding)
        - Short side profitable when short_funding_rate > 0 (we receive funding)

        Uses opportunity's entry_time as reference (for historical data).

        Returns:
            Minutes to next profitable funding, or 999999 if none
        """
        from datetime import timezone

        # Get reference time: opportunity's entry_time or current_time
        ref_time = opp.get('entry_time')
        if ref_time is None:
            ref_time = self.current_time
        if ref_time is None:
            return 999999.0

        # Convert to pandas Timestamp if needed
        if not isinstance(ref_time, pd.Timestamp):
            ref_time = pd.to_datetime(ref_time)

        # Get funding rates
        long_rate = opp.get('long_funding_rate', 0.0)
        short_rate = opp.get('short_funding_rate', 0.0)

        # Get next funding times
        long_next = opp.get('long_next_funding_time')
        short_next = opp.get('short_next_funding_time')

        profitable_minutes = []

        # Long is profitable if rate < 0 (we receive)
        if long_rate < 0 and long_next is not None:
            if not isinstance(long_next, pd.Timestamp):
                try:
                    long_next = pd.to_datetime(long_next)
                except:
                    long_next = None
            if long_next is not None:
                minutes = (long_next - ref_time).total_seconds() / 60
                if minutes > 0:
                    profitable_minutes.append(minutes)

        # Short is profitable if rate > 0 (we receive)
        if short_rate > 0 and short_next is not None:
            if not isinstance(short_next, pd.Timestamp):
                try:
                    short_next = pd.to_datetime(short_next)
                except:
                    short_next = None
            if short_next is not None:
                minutes = (short_next - ref_time).total_seconds() / 60
                if minutes > 0:
                    profitable_minutes.append(minutes)

        if not profitable_minutes:
            return 999999.0  # No profitable funding

        return min(profitable_minutes)

    def _calculate_position_size(self, action: int) -> float:
        """
        Calculate position size based on action and current config (V8).

        IMPORTANT: This formula must match production backend (AgentBackgroundService.cs:474):
            positionSizeUsd = availableCapital * sizeMultiplier

        Args:
            action: Action index (1-15 for ENTER actions, V8)

        Returns:
            Position size in USD (per side)
        """
        # Determine size multiplier based on action (V8: updated ranges)
        if 1 <= action <= 5:
            # SMALL: 10% of available capital
            size_multiplier = 0.10
        elif 6 <= action <= 10:
            # MEDIUM: 20% of available capital
            size_multiplier = 0.20
        elif 11 <= action <= 15:
            # LARGE: 30% of available capital
            size_multiplier = 0.30
        else:
            raise ValueError(f"Invalid action for position sizing: {action}")

        # Get available capital and config
        available_capital = self.portfolio.available_margin  # Same as total_capital after margin deduction
        target_util = self.current_config.target_utilization
        leverage = self.current_config.max_leverage

        # Match production formula: positionSize = availableCapital × sizeMultiplier
        position_size = available_capital * size_multiplier

        # Apply min/max limits (matching production)
        min_position_size = 10.0  # $10 minimum
        max_position_size = available_capital * target_util  # Don't exceed target utilization
        position_size = max(min_position_size, min(position_size, max_position_size))

        # Ensure we don't exceed available margin after leverage
        margin_needed = (position_size * 2) / leverage
        if margin_needed > available_capital:
            # Scale down to fit available margin
            position_size = (available_capital * leverage) / 2

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
            actual_episode_length = int((self.episode_end - self.episode_start).total_seconds() / 3600)
        else:
            # Random episode length (to prevent episode-length-dependent learning)
            if self.use_random_episode_length:
                actual_episode_length = np.random.randint(
                    self.episode_length_hours_min,
                    self.episode_length_hours_max + 1
                )
            else:
                actual_episode_length = self.episode_length_hours

            # Sample random episode start time
            # Ensure we have enough data for full episode
            min_start = self.data_start
            max_start = self.data_end - timedelta(days=actual_episode_length / 24 + 1)

            # Random start time (with UTC timezone)
            # If seed was set, this will be deterministic
            start_timestamp = np.random.uniform(
                min_start.timestamp(),
                max_start.timestamp()
            )
            self.episode_start = pd.Timestamp(start_timestamp, unit='s', tz='UTC')
            self.episode_end = self.episode_start + timedelta(hours=actual_episode_length)

        # Store current episode length for reference
        self.current_episode_length_hours = actual_episode_length

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

        # Track realized funding from closed positions (fixes P&L tracking bug)
        # Without this, funding P&L goes negative when positions close
        self.realized_funding_total_usd = 0.0

        # Track exit statistics for monitoring (forced vs agent closes)
        self.agent_closes = 0
        self.forced_closes = 0
        self.stop_loss_closes = 0

        # V6.1: Track rotation for rotation bonus
        # Stores APR of last exited position to reward ENTER with higher APR
        self._last_exited_apr = None
        self._rotation_count = 0

        # Track episode capital for reward normalization (constant throughout episode)
        # This prevents reward inflation when positions close
        self.episode_capital_for_normalization = self.initial_capital

        # PERFORMANCE: Reset price and funding rate caches (per-timestep caching)
        self._cached_prices = None
        self._cached_prices_time = None
        self._cached_funding_rates = None
        self._cached_funding_rates_time = None

        # Reset velocity tracking in feature builder (prevents stale values between episodes)
        if hasattr(self, 'feature_builder') and self.feature_builder is not None:
            self.feature_builder.reset_velocity_tracking()

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

        # REWARD STRUCTURE: Episode-Independent Learning
        # Key change: Price P&L only rewarded on agent EXIT decisions (not hourly)
        # This prevents episode-length-dependent behavior where agent learns "hold until episode ends"
        #
        # Component 1: Funding P&L reward - Hourly (actual cash flow)
        # Component 2: Price P&L reward - ONLY on agent close (realized P&L)
        # Component 3: Liquidation risk penalty - Safety critical
        # Component 4: Opportunity cost penalty - DISABLED

        # Initialize reward breakdown for logging
        reward_breakdown = {
            'funding_reward': 0.0,
            'price_reward': 0.0,  # Only set on exit now
            'liquidation_penalty': 0.0,
            'opportunity_cost_penalty': 0.0,
            'inactivity_penalty': 0.0,  # V6.1: Penalty for holding too long
            'negative_apr_penalty': 0.0,  # V7: Penalty for holding negative APR positions
            'peak_drawdown_penalty': 0.0,  # V8: Penalty for giving back profits
            'apr_decline_penalty': 0.0,  # V8: Penalty for holding declining APR positions
        }

        # Component 1: Funding P&L Reward (hourly - actual cash flow)
        # FIX: Include realized funding from closed positions to prevent negative spikes
        open_funding_total = self.portfolio.get_total_funding_usd()
        total_funding = open_funding_total + self.realized_funding_total_usd

        # Calculate funding change since last step
        funding_pnl_change = total_funding - self.previous_funding_total_usd

        # Update tracking
        self.previous_funding_total_usd = total_funding

        # Normalize by CONSTANT episode capital
        funding_pnl_pct = (funding_pnl_change / self.episode_capital_for_normalization) * 100
        funding_reward = funding_pnl_pct * self.funding_reward_scale

        # Only add funding reward (NOT price reward - that's on exit only)
        reward += funding_reward
        reward_breakdown['funding_reward'] = funding_reward
        # Note: price_reward stays 0.0 here - it's given in _exit_position()

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
        # Applied when holding positions with lower APR than best available
        # V7: Configurable threshold (default 50%, lowered from 100%)
        if self.opportunity_cost_scale > 0.0 and len(self.portfolio.positions) > 0:
            # Find best available APR from current opportunities
            best_available_apr = 0.0
            if len(self.current_opportunities) > 0:
                best_available_apr = max(opp.get('fund_apr', 0.0) for opp in self.current_opportunities)

            # Calculate opportunity cost for each position
            opportunity_cost_total = 0.0
            apr_threshold = self.opportunity_cost_threshold  # V7: Configurable threshold

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
                    # V6.1: Increased penalty using /500 (was /1000) for stronger rotation signal
                    # Example: 500% gap → -0.03 * 1.0 * capital_ratio ≈ -0.03 per step
                    position_capital_ratio = (pos.position_size_usd * 2) / self.episode_capital_for_normalization
                    opportunity_cost = -(apr_gap / 500) * position_capital_ratio * self.opportunity_cost_scale
                    opportunity_cost_total += opportunity_cost

            reward += opportunity_cost_total
            reward_breakdown['opportunity_cost_penalty'] = opportunity_cost_total

        # Component 4: Inactivity penalty - penalize holding positions too long
        # V6.1: Encourages rotation and exploration
        if self.inactivity_penalty_scale > 0 and len(self.portfolio.positions) > 0:
            inactivity_penalty_total = 0.0
            for pos in self.portfolio.positions:
                hours_held = pos.hours_held
                if hours_held > self.inactivity_penalty_hours:
                    # Penalty increases with time beyond threshold
                    excess_hours = hours_held - self.inactivity_penalty_hours
                    inactivity_penalty = -excess_hours * self.inactivity_penalty_scale
                    inactivity_penalty_total += inactivity_penalty
            reward += inactivity_penalty_total
            reward_breakdown['inactivity_penalty'] = inactivity_penalty_total

        # Component 5 (V7): Negative APR penalty - penalize holding positions with negative APR
        # This is CRITICAL for learning to exit when funding rate flips direction
        # Example: AIAUSDT entry +687% → current -105% should trigger exit, not HOLD
        if self.negative_apr_penalty_scale > 0 and len(self.portfolio.positions) > 0:
            negative_apr_penalty_total = 0.0
            for pos in self.portfolio.positions:
                # Look up current APR for this position
                current_pos_apr = 0.0
                for opp in self.current_opportunities:
                    if opp['symbol'] == pos.symbol:
                        current_pos_apr = opp.get('fund_apr', 0.0)
                        break

                # Apply penalty only for negative APR (funding working against us)
                if current_pos_apr < 0:
                    # Penalty proportional to how negative the APR is
                    # Example: -100% APR → penalty = 100/1000 * 0.02 = 0.002 per step
                    # Over 24h (24 steps): 0.002 * 24 = 0.048 cumulative penalty
                    penalty = (abs(current_pos_apr) / 1000.0) * self.negative_apr_penalty_scale
                    negative_apr_penalty_total -= penalty

            reward += negative_apr_penalty_total
            reward_breakdown['negative_apr_penalty'] = negative_apr_penalty_total

        # Component 6 (V8): Peak Drawdown Penalty - Protect accumulated profits
        # Penalizes watching profits evaporate (e.g., PIPPINUSDT: 5.4% → 0.99%)
        # This teaches the model to exit when giving back too much profit
        if self.peak_drawdown_penalty_scale > 0 and len(self.portfolio.positions) > 0:
            peak_drawdown_penalty_total = 0.0
            for pos in self.portfolio.positions:
                # Only apply if position was profitable at some point
                if pos.peak_pnl_pct > 0:
                    drawdown_ratio = pos.get_peak_drawdown()  # 0-1 ratio
                    if drawdown_ratio >= self.peak_drawdown_threshold:
                        # Penalty proportional to drawdown severity
                        # Example: 50% drawdown with 30% threshold → 0.2 * 0.5 = 0.1 penalty per step
                        excess_drawdown = drawdown_ratio - self.peak_drawdown_threshold
                        penalty = excess_drawdown * self.peak_drawdown_penalty_scale
                        peak_drawdown_penalty_total -= penalty

            reward += peak_drawdown_penalty_total
            reward_breakdown['peak_drawdown_penalty'] = peak_drawdown_penalty_total

        # Component 7 (V8): APR Decline Penalty - Proactive exit signal
        # Penalizes holding positions with significantly declining APR (before it goes negative)
        # This catches PIPPINUSDT case: 11961% → 6000% = 50% decline (should exit)
        if self.apr_decline_penalty_scale > 0 and len(self.portfolio.positions) > 0:
            apr_decline_penalty_total = 0.0
            for pos in self.portfolio.positions:
                # Look up current APR for this position
                current_pos_apr = 0.0
                for opp in self.current_opportunities:
                    if opp['symbol'] == pos.symbol:
                        current_pos_apr = opp.get('fund_apr', 0.0)
                        break

                # Calculate APR decline from entry (only for positions that were positive)
                entry_apr = pos.entry_apr
                if entry_apr > 0 and current_pos_apr > 0:  # Both still positive
                    decline_ratio = 1 - (current_pos_apr / entry_apr)
                    if decline_ratio >= self.apr_decline_threshold:
                        # Penalty proportional to decline severity
                        # Example: 60% decline with 50% threshold → 0.1 * 0.015 = 0.0015 per step
                        excess_decline = decline_ratio - self.apr_decline_threshold
                        penalty = excess_decline * self.apr_decline_penalty_scale
                        apr_decline_penalty_total -= penalty

            reward += apr_decline_penalty_total
            reward_breakdown['apr_decline_penalty'] = apr_decline_penalty_total

        # Check termination
        terminated = False
        truncated = False

        # Episode ends after specified hours
        if self.current_time >= self.episode_end:
            truncated = True
            # Close all remaining positions
            reward += self._close_all_positions()

        # Terminate if max drawdown exceeded (25%) - skip if disabled for testing
        if not self.disable_early_termination and self.portfolio.max_drawdown_pct >= 25.0:
            terminated = True
            reward += self._close_all_positions()

        # Terminate if portfolio value too low - skip if disabled for testing
        if not self.disable_early_termination and self.portfolio.portfolio_value < self.initial_capital * 0.5:
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

            # Exit statistics for monitoring (episode-independent learning)
            info['agent_closes'] = self.agent_closes
            info['forced_closes'] = self.forced_closes
            info['stop_loss_closes'] = self.stop_loss_closes
            info['rotation_count'] = self._rotation_count  # V6.1: Track successful rotations
            info['episode_length_hours'] = self.current_episode_length_hours

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Tuple[float, dict]:
        """
        Execute the agent's action (V9: 17 actions).

        17 actions:
        - 0: HOLD
        - 1-5: ENTER_OPP_0-4_SMALL
        - 6-10: ENTER_OPP_0-4_MEDIUM
        - 11-15: ENTER_OPP_0-4_LARGE
        - 16: EXIT_POS_0

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

        elif 1 <= action <= DIMS.ACTION_ENTER_END:  # 1-15
            # ENTER actions (V8: updated ranges)
            # Decode opportunity index and size
            if 1 <= action <= 5:
                opp_idx = action - 1
                size_type = 'small'
            elif 6 <= action <= 10:
                opp_idx = action - 6
                size_type = 'medium'
            else:  # 11-15
                opp_idx = action - 11
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

        elif DIMS.ACTION_EXIT_START <= action <= DIMS.ACTION_EXIT_END:  # 16-17
            # EXIT actions (V8: updated range)
            pos_idx = action - DIMS.ACTION_EXIT_START

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
            # CRITICAL FIX: Check for both None AND 0.0 since price_loader returns 0.0
            # when funding rate data is not available in parquet files
            if actual_long_rate is not None and actual_long_rate != 0.0:
                entry_long_rate = actual_long_rate
            if actual_short_rate is not None and actual_short_rate != 0.0:
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
            # V7.1: Realistic round-trip fee penalty at entry
            # Previous 0.25% was too conservative, causing model to under-trade (2 trades/episode)
            # Real fees: ~0.04% per side × 2 sides × 2 (entry+exit) = 0.16%
            # Reduced to 0.12% to encourage more trading while still preventing overtrading
            round_trip_fee_pct = 0.12  # Realistic fee estimate
            # Normalize by episode capital and apply price reward scale for consistency
            entry_fee_penalty = -(round_trip_fee_pct / 100) * (position_size * 2) / self.episode_capital_for_normalization * 100 * self.price_reward_scale

            # V7.1: Re-enable rotation bonus with higher threshold
            # Only reward rotation when new position has SIGNIFICANTLY better APR (>200% improvement)
            # This encourages switching to much better opportunities without causing overtrading
            rotation_bonus = 0.0
            new_apr = opp.get('fund_apr', 0.0)

            if self._last_exited_apr is not None:
                apr_improvement = new_apr - self._last_exited_apr

                # Only reward if new APR is >200% better (e.g., 100% → 350%)
                if apr_improvement > 200:
                    # Moderate bonus: scale by improvement magnitude, capped at 1.0
                    # Example: exited 100% APR, entered 500% APR → min(400/400, 1.0) * 0.5 = 0.5 bonus
                    rotation_bonus = min(apr_improvement / 400.0, 1.0) * 0.5

                # Clear the tracker after processing
                self._last_exited_apr = None

            # V8: Entry quality penalty - Discourage low-quality entries
            # Quality indicators:
            # 1. APR consistency: current APR vs 24h/3d projections
            # 2. Spread stability: low spread volatility
            entry_quality_penalty = 0.0

            if self.entry_quality_penalty_scale > 0:
                current_apr = opp.get('fund_apr', 0)
                apr_24h_proj = opp.get('fund_apr_24h_proj', 0)
                apr_3d_proj = opp.get('fund_apr_3d_proj', 0)
                spread_volatility = opp.get('spread_volatility_stddev', 0)

                # Quality check 1: APR projection disagreement
                # High current APR but negative projections = unreliable/temporary spike
                if current_apr > 500 and (apr_24h_proj < 0 or apr_3d_proj < 0):
                    # Penalty proportional to disagreement severity
                    # Example: current 1000%, 24h proj -200% → disagreement = 1200
                    disagreement = abs(current_apr - min(apr_24h_proj, apr_3d_proj))
                    entry_quality_penalty -= (disagreement / 1000) * self.entry_quality_penalty_scale

                # Quality check 2: High spread volatility = risky entry
                # Spread volatility > 15% indicates unstable arbitrage conditions
                if spread_volatility > 0.15:
                    entry_quality_penalty -= spread_volatility * self.entry_quality_penalty_scale

            # Return entry fee penalty + rotation bonus + quality penalty
            return entry_fee_penalty + rotation_bonus + entry_quality_penalty
        else:
            # Could not open (insufficient capital or position limit - masked, shouldn't happen)
            # No penalty - rely on action masking to prevent invalid actions
            return 0.0

    def _exit_position(self, position_idx: int, exit_type: ExitType = ExitType.AGENT) -> float:
        """
        Exit an existing position with exit-type-dependent reward.

        Episode-Independent Learning:
        - Price P&L reward ONLY for AGENT and STOP_LOSS exits (realized P&L)
        - EPISODE_END exits get NO price P&L reward (incomplete trade)
        - This prevents episode-length-dependent behavior

        Args:
            position_idx: Index of position to close
            exit_type: Type of exit (AGENT, STOP_LOSS, or EPISODE_END)

        Returns:
            Reward for exiting (price P&L for AGENT/STOP_LOSS, 0 for EPISODE_END)
        """
        position = self.portfolio.positions[position_idx]

        # Track position's funding BEFORE closing (for realized funding tracking)
        position_funding = position.long_net_funding_usd + position.short_net_funding_usd

        # Get current prices for exit
        prices = self._get_current_prices()

        if position.symbol in prices:
            symbol_prices = prices[position.symbol]

            # Close position (updates portfolio)
            # realized_pnl includes ALL fees (entry + exit) - this is the source of truth
            realized_pnl = self.portfolio.close_position(
                position_idx,
                self.current_time,
                symbol_prices['long_price'],
                symbol_prices['short_price']
            )

            # Track realized funding from closed position (fixes funding P&L tracking)
            self.realized_funding_total_usd += position_funding

            # Price P&L = realized - funding (funding already rewarded hourly, don't double-count)
            # This CORRECTLY includes ALL trading fees (entry + exit)
            position_price_pnl = realized_pnl - position_funding

            # Calculate exit reward based on exit type
            exit_reward = 0.0

            if exit_type == ExitType.AGENT:
                # Agent decided to close - reward realized price P&L
                # This is the KEY change: price P&L only rewarded on agent decision
                price_pnl_pct = (position_price_pnl / self.episode_capital_for_normalization) * 100
                exit_reward = price_pnl_pct * self.price_reward_scale
                self.agent_closes += 1

                # Look up current fund_apr and best available APR (needed for rotation tracking)
                current_fund_apr = 0.0
                best_available_apr = 0.0
                for opp in self.current_opportunities:
                    if opp.get('symbol') == position.symbol:
                        current_fund_apr = opp.get('fund_apr', 0.0)
                    best_available_apr = max(best_available_apr, opp.get('fund_apr', 0.0))

                # BONUS: Reward for exiting positions with NEGATIVE funding ONLY
                # V7 SIMPLIFIED: Removed "inferior position" bonus - it caused overtrading!
                # The model was getting rewarded for every exit because there's usually
                # a better opportunity available. This led to 380 trades/episode.
                # Now ONLY reward exiting when funding rate is actually negative.
                if self.negative_funding_exit_reward_scale > 0 and current_fund_apr < 0:
                    # Bonus proportional to how negative the APR is
                    # Example: -100% APR → 100/100 * 2.0 = 2.0 bonus
                    negative_funding_bonus = abs(current_fund_apr) / 100.0 * self.negative_funding_exit_reward_scale
                    exit_reward += negative_funding_bonus

                # V7: APR FLIP EXIT BONUS - Extra reward for exiting when APR direction flipped
                # This is CRITICAL for the AIAUSDT case: +687% entry → -105% current
                # The model must learn that an APR flip is a strong EXIT signal
                if self.apr_flip_exit_bonus_scale > 0:
                    entry_apr = position.entry_apr  # APR at position entry
                    # Check for direction flip: entered positive, now negative
                    if entry_apr > 0 and current_fund_apr < 0:
                        # Bonus proportional to severity of the flip
                        # Example: +687% entry → -105% current = 792% swing
                        apr_swing = entry_apr - current_fund_apr  # Always positive when flip occurs
                        flip_bonus = min(apr_swing / 500.0, 2.0) * self.apr_flip_exit_bonus_scale
                        exit_reward += flip_bonus

                # V6.1: Store exited position's APR for rotation tracking
                self._last_exited_apr = current_fund_apr

                # V6.1: Trade diversity bonus - reward completing trades to encourage activity
                if self.trade_diversity_bonus > 0:
                    exit_reward += self.trade_diversity_bonus

            elif exit_type == ExitType.STOP_LOSS:
                # Stop-loss triggered - still reward price P&L (natural penalty)
                price_pnl_pct = (position_price_pnl / self.episode_capital_for_normalization) * 100
                exit_reward = price_pnl_pct * self.price_reward_scale
                self.stop_loss_closes += 1

            elif exit_type == ExitType.EPISODE_END:
                # Forced close at episode end - NO price P&L reward
                # This teaches agent: "You should have closed before episode end"
                exit_reward = 0.0
                self.forced_closes += 1

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
            # Pass STOP_LOSS exit type - gets price P&L reward (natural penalty)
            reward = self._exit_position(idx, exit_type=ExitType.STOP_LOSS)
            total_reward += reward

        return total_reward

    def _close_all_positions(self) -> float:
        """
        Close all open positions at episode end.

        Uses EPISODE_END exit type - NO price P&L reward for forced closes.
        This is key to preventing episode-length-dependent learning.
        """
        total_reward = 0.0

        # Close in reverse order to avoid index issues
        for i in range(len(self.portfolio.positions) - 1, -1, -1):
            # Pass EPISODE_END - these are interrupted trades, NO price P&L reward
            reward = self._exit_position(i, exit_type=ExitType.EPISODE_END)
            total_reward += reward  # Will be 0.0 for EPISODE_END

        return total_reward  # Always 0.0 for episode-end closes

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

                # CRITICAL FIX: Parquet only has funding rates at payment timestamps (0 otherwise)
                # Position's current rate should reflect the CURRENT MARKET rate from opportunities CSV
                # Fallback chain: parquet (if non-zero) → CSV opportunities → position's stored rate
                if long_rate is None or long_rate == 0.0 or short_rate is None or short_rate == 0.0:
                    # Fallback to CSV rates from current opportunities
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
                'slippage_pct': float(pos.slippage_pct),
                # Raw funding and fees (Python calculates P&L)
                'long_funding_earned_usd': float(pos.long_net_funding_usd),
                'short_funding_earned_usd': float(pos.short_net_funding_usd),
                # Fees: entry + estimated exit to match production (UserDataCollector sends entry + exit)
                'long_fees_usd': float((pos.entry_fees_paid_usd + pos.estimated_exit_fees_usd) / 2),
                'short_fees_usd': float((pos.entry_fees_paid_usd + pos.estimated_exit_fees_usd) / 2),
                'long_funding_rate': float(pos.long_funding_rate),
                'short_funding_rate': float(pos.short_funding_rate),
                'long_funding_interval_hours': int(pos.long_funding_interval_hours),
                'short_funding_interval_hours': int(pos.short_funding_interval_hours),
                'entry_apr': float(pos.entry_apr),
                'current_position_apr': float(current_position_apr),
                'liquidation_distance': float(pos.get_liquidation_distance(current_long_price, current_short_price)),
            })

        portfolio = {
            'positions': positions,
            'total_capital': float(self.portfolio.total_capital),
            'capital_utilization': float(self.portfolio.capital_utilization),
        }

        # Add has_existing_position flag to opportunities (for action mask consistency)
        existing_symbols = {pos.symbol for pos in self.portfolio.positions}
        opportunities_with_flags = []
        for opp in self.current_opportunities:
            opp_copy = opp.copy()
            opp_copy['has_existing_position'] = opp.get('symbol', '') in existing_symbols
            opportunities_with_flags.append(opp_copy)

        return {
            'trading_config': trading_config,
            'portfolio': portfolio,
            'opportunities': opportunities_with_flags,
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
                'slippage_pct': float(pos.slippage_pct),
                # Raw funding and fees (Python calculates P&L from these + prices)
                'long_funding_earned_usd': float(pos.long_net_funding_usd),
                'short_funding_earned_usd': float(pos.short_net_funding_usd),
                # Fees: entry + estimated exit to match production (UserDataCollector sends entry + exit)
                'long_fees_usd': float((pos.entry_fees_paid_usd + pos.estimated_exit_fees_usd) / 2),
                'short_fees_usd': float((pos.entry_fees_paid_usd + pos.estimated_exit_fees_usd) / 2),
                'long_funding_rate': float(pos.long_funding_rate),
                'short_funding_rate': float(pos.short_funding_rate),
                'estimated_long_funding_rate': float(estimated_long_rate),
                'estimated_short_funding_rate': float(estimated_short_rate),
                'long_funding_interval_hours': int(pos.long_funding_interval_hours),
                'short_funding_interval_hours': int(pos.short_funding_interval_hours),
                'entry_apr': float(pos.entry_apr),
                'current_position_apr': float(current_position_apr),
                'liquidation_distance': float(pos.get_liquidation_distance(current_long_price, current_short_price)),
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

        # Add has_existing_position flag to opportunities (for action mask consistency)
        # This flag must be set for UnifiedFeatureBuilder.get_action_mask() to work correctly
        existing_symbols = {pos.symbol for pos in self.portfolio.positions}
        opportunities_with_flags = []
        for opp in self.current_opportunities:
            opp_copy = opp.copy()
            opp_copy['has_existing_position'] = opp.get('symbol', '') in existing_symbols
            opportunities_with_flags.append(opp_copy)

        # Build raw data dict
        raw_data = {
            'trading_config': trading_config,
            'portfolio': portfolio_dict,
            'opportunities': opportunities_with_flags,
            'current_time': self.current_time,  # V6: For time_to_next_funding feature
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
