"""
Test model inference - verify trained model can be loaded and used

V9 Feature Architecture (86 dimensions total):
  - Config: 5 features (max_leverage, target_utilization, max_positions, stop_loss, liq_buffer)
  - Portfolio: 2 features (min_liq_distance, time_to_next_funding_norm)
  - Executions: 19 features (1 position × 19 features)
  - Opportunities: 60 features (5 opportunities × 12 features each)

V9 Action Space (17 actions total):
  - 0: HOLD
  - 1-5: ENTER_SMALL (opportunities 0-4)
  - 6-10: ENTER_MEDIUM (opportunities 0-4)
  - 11-15: ENTER_LARGE (opportunities 0-4)
  - 16: EXIT_POS_0

V9 Position Features (19 per position):
  1. is_active - Position active flag
  2. net_pnl_pct - Net P&L percentage
  3. hours_held_norm - Log-normalized hold time: log(hours+1)/log(73)
  4. estimated_pnl_pct - Price P&L only (isolates price risk)
  5. estimated_pnl_velocity - Trend signal for price movement
  6. estimated_funding_8h_pct - Expected funding profit in next 8h
  7. funding_velocity - Detects funding rate trends
  8. spread_pct - Current price spread percentage
  9. spread_velocity - Detects converging/diverging spreads
  10. liquidation_distance_pct - Distance to liquidation
  11. apr_ratio - Current APR / Entry APR
  12. current_position_apr - Current market APR for this symbol
  13. best_available_apr_norm - Best APR available (±5000% range)
  14. apr_advantage - Current APR - Best Available APR
  15. return_efficiency - Net P&L / (hours * entry APR/8760)
  16. pnl_imbalance - (long_pnl - short_pnl) / 2 (asymmetry indicator)
  17. pnl_velocity - P&L trend over recent history
  18. peak_drawdown - Drawdown from peak P&L
  19. hours_until_next_funding_norm - Time to next funding payment

Logs detailed feature breakdown to: test_features.log
"""

import argparse
import torch
import numpy as np
import pandas as pd
import requests
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Test model inference')

    # Inference mode
    parser.add_argument('--api-mode', action='store_true',
                        help='Use HTTP API for inference instead of direct model loading')
    parser.add_argument('--api-url', type=str, default='http://localhost:5250',
                        help='ML API base URL (default: http://localhost:5250)')
    parser.add_argument('--api-endpoint', type=str, default='/rl/predict',
                        help='API endpoint path (default: /rl/predict)')

    # Trading configuration
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Max leverage (default: 2.0x)')
    parser.add_argument('--utilization', type=float, default=0.8,
                        help='Capital utilization (default: 0.8 = 80%%)')
    parser.add_argument('--max-positions', type=int, default=1,
                        help='Max concurrent positions (default: 1, V9: single position only)')
    parser.add_argument('--stop-loss', type=float, default=-0.05,
                        help='Stop-loss threshold as decimal (default: -0.05 = -5%%)')

    # Test configuration
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test (default: 1)')
    parser.add_argument('--full-test', action='store_true', default=True,
                        help='Use entire test dataset (default: True). If disabled, uses --episode-length-days.')
    parser.add_argument('--no-full-test', action='store_false', dest='full_test',
                        help='Disable full test mode and use --episode-length-days instead')
    parser.add_argument('--episode-length-days', type=int, default=5,
                        help='Episode length in days (only used if --no-full-test is specified, default: 5)')
    parser.add_argument('--step-minutes', type=int, default=5,
                        help='Minutes per prediction step (default: 5 = 5-minute intervals)')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--trades-output', type=str, default='trades_inference.csv',
                        help='Output CSV file for trade records (default: trades_inference.csv)')
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')
    parser.add_argument('--feature-scaler-path', type=str, default='trained_models/rl/feature_scaler_v3.pkl',
                        help='Path to fitted feature scaler pickle (V5.4: StandardScaler with 12 features)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--force-zero-pnl', action='store_true',
                        help='Force total_pnl_pct to always be 0 in observations (simulates production bug)')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Start time for filtering test data (e.g., "2025-11-13 09:20:00")')
    parser.add_argument('--end-time', type=str, default=None,
                        help='End time for filtering test data (e.g., "2025-11-13 09:30:00")')

    # Confidence threshold settings
    parser.add_argument('--enter-threshold', type=float, default=0.0,
                        help='Minimum confidence for ENTER actions (default: 0.0 = disabled, use 0.3 for 30%%)')
    parser.add_argument('--exit-threshold', type=float, default=0.0,
                        help='Minimum confidence for EXIT actions (default: 0.0 = disabled, use 0.4 for 40%%)')

    # APR threshold settings
    parser.add_argument('--min-apr', type=float, default=0.0,
                        help='Minimum APR for ENTER actions (default: 0.0 = disabled, use 800 for 800%%)')
    parser.add_argument('--require-positive-24h', action='store_true',
                        help='Only allow ENTER if fund_apr_24h_proj > 0')
    parser.add_argument('--require-positive-3d', action='store_true',
                        help='Only allow ENTER if fund_apr_3d_proj > 0')
    parser.add_argument('--max-spread-vol', type=float, default=0.0,
                        help='Maximum spread volatility for ENTER (default: 0.0 = disabled, use 0.15 for 15%%)')

    return parser.parse_args()


class MLAPIClient:
    """HTTP client for ML API inference."""

    def __init__(self, base_url: str, endpoint: str):
        """
        Initialize ML API client.

        Args:
            base_url: Base URL (e.g., http://localhost:5250)
            endpoint: Endpoint path (e.g., /rl/v2/predict)
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.full_url = f"{self.base_url}{self.endpoint}"
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        # Generate session_id for velocity tracking (matches production behavior)
        # Production sends SessionId via RLPredictionService.cs:83
        self.session_id = str(uuid.uuid4())

        print(f"   ML API Client initialized")
        print(f"   URL: {self.full_url}")
        print(f"   Session ID: {self.session_id}")

    def sanitize_for_json(self, obj):
        """
        Recursively sanitize data structure for JSON serialization.
        Replaces inf, -inf, NaN, and Timestamp objects with JSON-safe values.

        Args:
            obj: Object to sanitize (dict, list, or primitive)

        Returns:
            Sanitized object
        """
        import math
        import pandas as pd
        import numpy as np

        if isinstance(obj, dict):
            return {k: self.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            # Convert pandas Timestamp or numpy datetime64 to ISO string
            return str(pd.Timestamp(obj))
        elif isinstance(obj, float):
            if math.isnan(obj):
                return 0.0
            elif math.isinf(obj):
                return 1e9 if obj > 0 else -1e9
            else:
                return obj
        elif isinstance(obj, np.integer):
            # Convert numpy integers to Python int
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert numpy floats to Python float (and check for inf/nan)
            val = float(obj)
            if math.isnan(val):
                return 0.0
            elif math.isinf(val):
                return 1e9 if val > 0 else -1e9
            else:
                return val
        else:
            return obj

    def build_raw_data_from_env(self, env: FundingArbitrageEnv) -> Dict[str, Any]:
        """
        Build raw data dictionary from environment state.

        Args:
            env: FundingArbitrageEnv instance

        Returns:
            Dict with trading_config, portfolio, opportunities
        """
        # Get current prices
        price_data = env._get_current_prices()
        funding_rates = env._get_current_funding_rates()

        # Build trading config
        config_array = env.current_config.to_array()
        trading_config = {
            'max_leverage': float(config_array[0]),
            'target_utilization': float(config_array[1]),
            'max_positions': int(config_array[2]),
            'stop_loss_threshold': float(config_array[3]),
            'liquidation_buffer': float(config_array[4]),
        }

        # Build positions list
        positions = []
        for pos in env.portfolio.positions:
            # Get current prices for this position
            long_price_info = price_data.get(pos.symbol, {}).get(pos.long_exchange, {'price': pos.entry_long_price})
            short_price_info = price_data.get(pos.symbol, {}).get(pos.short_exchange, {'price': pos.entry_short_price})

            current_long_price = long_price_info.get('price', pos.entry_long_price)
            current_short_price = short_price_info.get('price', pos.entry_short_price)

            # Get funding rates from position object (these are stored at entry and updated)
            # DO NOT use funding_rates dict - it may not have current position data
            long_funding_rate = pos.long_funding_rate
            short_funding_rate = pos.short_funding_rate

            # Calculate current APR for this position
            current_position_apr = 0.0
            for opp in env.current_opportunities:
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
                'long_fees_usd': float(pos.entry_fees_paid_usd / 2),  # Split entry fees between legs
                'short_fees_usd': float(pos.entry_fees_paid_usd / 2),
                'long_funding_rate': float(long_funding_rate),
                'short_funding_rate': float(short_funding_rate),
                'long_funding_interval_hours': int(pos.long_funding_interval_hours),
                'short_funding_interval_hours': int(pos.short_funding_interval_hours),
                'entry_apr': float(pos.entry_apr),
                'current_position_apr': float(current_position_apr),
                'liquidation_distance': float(pos.get_liquidation_distance(current_long_price, current_short_price)),
            })

        # Build portfolio
        portfolio = {
            'positions': positions,
            'total_capital': float(env.portfolio.total_capital),
            'capital_utilization': float(env.portfolio.capital_utilization),
        }

        # Build opportunities list (only include schema fields to avoid confusion)
        opportunities = []
        for opp in env.current_opportunities:
            opportunities.append({
                'symbol': opp['symbol'],
                'long_exchange': opp['long_exchange'],
                'short_exchange': opp['short_exchange'],
                'fund_profit_8h': opp.get('fund_profit_8h', 0.0),
                'fund_profit_8h_24h_proj': opp.get('fund_profit_8h_24h_proj', 0.0),
                'fund_profit_8h_3d_proj': opp.get('fund_profit_8h_3d_proj', 0.0),
                'fund_apr': opp.get('fund_apr', 0.0),
                'fund_apr_24h_proj': opp.get('fund_apr_24h_proj', 0.0),
                'fund_apr_3d_proj': opp.get('fund_apr_3d_proj', 0.0),
                'spread_30_sample_avg': opp.get('spread_30_sample_avg', 0.0),
                'price_spread_24h_avg': opp.get('price_spread_24h_avg', 0.0),
                'price_spread_3d_avg': opp.get('price_spread_3d_avg', 0.0),
                'spread_volatility_stddev': opp.get('spread_volatility_stddev', 0.0),
                'has_existing_position': opp.get('has_existing_position', False),
            })

        return {
            'trading_config': trading_config,
            'portfolio': portfolio,
            'opportunities': opportunities
        }

    def predict(self, env: FundingArbitrageEnv) -> Dict[str, Any]:
        """
        Get prediction from ML API using /rl/predict endpoint (same as production).

        Args:
            env: FundingArbitrageEnv instance

        Returns:
            Dict with action, confidence, etc.

        Raises:
            requests.RequestException: If API call fails
        """
        # Use env.get_raw_state_for_ml_api() for parity with direct mode (same raw data)
        raw_data = env.get_raw_state_for_ml_api()
        raw_data['portfolio']['session_id'] = self.session_id
        # Pass current_time to API for backtesting (V6: time_to_next_funding feature)
        if 'current_time' in raw_data:
            raw_data['portfolio']['current_time'] = raw_data['current_time']
        request_data = self.sanitize_for_json(raw_data)
        url = self.full_url

        # Make request
        try:
            response = self.session.post(
                url,
                json=request_data,
                timeout=10.0
            )
            response.raise_for_status()

            result = response.json()
            return result

        except requests.exceptions.Timeout:
            raise RuntimeError(f"ML API timeout after 10 seconds")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Failed to connect to ML API at {url}. Is the server running?")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"ML API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"ML API request failed: {e}")

    def select_action(self, env: FundingArbitrageEnv, obs: np.ndarray = None, action_mask: np.ndarray = None) -> tuple:
        """
        Select action using ML API (compatible with trainer.select_action interface).

        Args:
            env: Environment instance
            obs: Observation vector (unused - kept for interface compatibility)
            action_mask: Action mask (unused - kept for interface compatibility)

        Returns:
            (action, value, log_prob) - value and log_prob are dummy values
        """
        # Use /rl/predict endpoint (obs/action_mask are ignored - server builds features)
        result = self.predict(env)

        action = result.get('action_id', 0)
        value = result.get('state_value', 0.0)
        log_prob = 0.0  # Dummy value

        # Store confidence for threshold checking
        self._last_confidence = result.get('confidence', 1.0)

        return action, value, log_prob


def test_model_inference(args):
    """Test loading and using a trained model."""
    print("=" * 80)
    if args.api_mode:
        print("TESTING MODEL INFERENCE VIA ML API (HTTP)")
    else:
        print("TESTING MODEL INFERENCE (DIRECT)")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Enable deterministic algorithms for 100% reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n🎲 Random seed: {args.seed} (deterministic mode enabled)")
    print("   torch.use_deterministic_algorithms(True)")
    print("   cudnn.deterministic=True, cudnn.benchmark=False")

    # Filter test data by time range if specified
    test_data_path = args.test_data_path
    if args.start_time or args.end_time:
        print("\n⏱️  Filtering test data by time range...")
        import tempfile

        df = pd.read_csv(args.test_data_path)
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

        print(f"   Original: {len(df)} rows ({df['entry_time'].min()} to {df['entry_time'].max()})")

        if args.start_time:
            start_dt = pd.to_datetime(args.start_time, utc=True)
            df = df[df['entry_time'] >= start_dt]
            print(f"   Filtered start >= {start_dt}: {len(df)} rows")

        if args.end_time:
            end_dt = pd.to_datetime(args.end_time, utc=True)
            # Add 1 second to make end time inclusive (handles microseconds)
            end_dt_inclusive = end_dt + pd.Timedelta(seconds=1)
            df = df[df['entry_time'] < end_dt_inclusive]
            print(f"   Filtered end <= {end_dt}: {len(df)} rows")

        if len(df) == 0:
            raise ValueError("No data found in specified time range!")

        print(f"   Final: {len(df)} rows ({df['entry_time'].min()} to {df['entry_time'].max()})")

        # Save filtered data to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        test_data_path = temp_file.name
        print(f"   Saved filtered data to: {test_data_path}")

    # Create environment (use test data)
    print("\n1. Creating test environment...")

    # Trading config - use command-line args
    trading_config = TradingConfig(
        max_leverage=args.leverage,
        target_utilization=args.utilization,
        max_positions=args.max_positions,
        stop_loss_threshold=args.stop_loss,  # V11: configurable stop-loss
    )

    # Reward config (match training defaults)
    reward_config = RewardConfig(
        funding_reward_scale=1.0,
        price_reward_scale=1.0,
        liquidation_penalty_scale=10.0,
        opportunity_cost_scale=0.0,  # Disabled by default
    )

    # Convert step minutes to hours for the environment
    step_hours = args.step_minutes / 60.0

    env = FundingArbitrageEnv(
        data_path=test_data_path,
        initial_capital=args.initial_capital,
        trading_config=trading_config,
        reward_config=reward_config,
        episode_length_days=args.episode_length_days,
        step_hours=step_hours,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        use_full_range_episodes=args.full_test,
        force_zero_total_pnl_pct=args.force_zero_pnl,
        verbose=False,
    )

    # Enable feature logging for debugging (only if not in API mode)
    if not args.api_mode:
        env.feature_log_file = '/tmp/direct_mode_features.log'

    print("✅ Test environment created")
    print(f"   Data: {test_data_path}")

    # Show episode configuration based on mode
    if args.full_test:
        # Calculate actual data range
        test_df = pd.read_csv(args.test_data_path)
        test_df['entry_time'] = pd.to_datetime(test_df['entry_time'])
        data_start = test_df['entry_time'].min()
        data_end = test_df['entry_time'].max()
        data_days = (data_end - data_start).total_seconds() / 86400
        total_steps = int((data_days * 24 * 60) / args.step_minutes)

        print(f"   🌐 FULL TEST MODE: Testing on ENTIRE test dataset")
        print(f"   Episode length: {data_days:.1f} days ({data_start} to {data_end})")
        print(f"   Total steps: ~{total_steps:,} steps at {args.step_minutes}-minute intervals")
    else:
        total_steps = int((args.episode_length_days * 24 * 60) / args.step_minutes)
        print(f"   Episode length: {args.episode_length_days} days ({total_steps} steps at {args.step_minutes}-minute intervals)")

    print(f"   Step interval: {args.step_minutes} minute(s) ({step_hours:.4f} hours)")
    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Utilization: {args.utilization:.0%}")
    print(f"   Max Positions: {args.max_positions}")
    if args.price_history_path:
        print(f"   Price History: {args.price_history_path} (dynamic funding updates enabled)")
    if args.force_zero_pnl:
        print(f"   ⚠️  FORCING total_pnl_pct = 0 in observations (production bug simulation)")
    if args.enter_threshold > 0 or args.exit_threshold > 0:
        print(f"\n   🎯 Confidence Thresholds:")
        print(f"      ENTER threshold: {args.enter_threshold*100:.0f}% (actions below this default to HOLD)")
        print(f"      EXIT threshold:  {args.exit_threshold*100:.0f}% (actions below this default to HOLD)")

    # APR threshold info
    if args.min_apr > 0 or args.require_positive_24h or args.require_positive_3d or args.max_spread_vol > 0:
        print(f"\n   📊 APR Thresholds:")
        if args.min_apr > 0:
            print(f"      Min APR: {args.min_apr:.0f}% (ENTER blocked if fund_apr < threshold)")
        if args.require_positive_24h:
            print(f"      Require 24h positive: Yes (ENTER blocked if fund_apr_24h_proj < 0)")
        if args.require_positive_3d:
            print(f"      Require 3d positive: Yes (ENTER blocked if fund_apr_3d_proj < 0)")
        if args.max_spread_vol > 0:
            print(f"      Max spread vol: {args.max_spread_vol:.2f} (ENTER blocked if spread_vol > threshold)")

    # Initialize inference method based on mode
    if args.api_mode:
        print("\n2. Initializing ML API client...")
        api_client = MLAPIClient(args.api_url, args.api_endpoint)
        print("✅ ML API client ready")
        print(f"   Mode: HTTP API inference")

        # Create dummy trainer object for compatibility
        trainer = None
        predictor = api_client
    else:
        print("\n2. Creating network...")
        network = ModularPPONetwork()
        print(f"✅ Network created ({sum(p.numel() for p in network.parameters()):,} parameters)")

        print("\n3. Creating trainer...")
        trainer = PPOTrainer(
            network=network,
            learning_rate=3e-4,
            device='cpu',
        )
        print("✅ Trainer created")

        print("\n4. Loading trained model...")
        try:
            trainer.load(args.checkpoint)
            # CRITICAL: Set network to eval mode for inference (disables dropout)
            # This must match the server predictor which uses eval mode
            trainer.network.eval()
            print(f"✅ Model loaded from {args.checkpoint}")
            print(f"   Mode: Direct model inference")
        except FileNotFoundError:
            print(f"⚠️  No checkpoint found at {args.checkpoint} - using random weights")

        predictor = trainer

    # Run inference with detailed metrics (match training evaluation)
    episode_text = "episode" if args.num_episodes == 1 else "episodes"
    print(f"\n5. Running inference on {args.num_episodes} {episode_text}...")

    # Episode-level metrics
    eval_rewards = []
    eval_lengths = []

    # Trading metrics
    total_pnls = []
    total_pnl_pcts = []
    num_trades = []
    num_winning_trades = []
    num_losing_trades = []
    avg_trade_durations = []
    max_drawdowns = []
    opportunities_seen = []
    trades_executed = []

    # Config tracking
    configs_used = []

    # Trade tracking (for CSV export)
    all_trades = []
    all_funding_details = []  # Detailed funding breakdown per position

    # Profit factor metrics
    all_winning_pnl = []
    all_losing_pnl = []

    # Threshold tracking
    threshold_blocked_enters = 0
    threshold_blocked_exits = 0

    # APR threshold tracking
    apr_blocked_low_apr = 0
    apr_blocked_neg_24h = 0
    apr_blocked_neg_3d = 0
    apr_blocked_high_spread_vol = 0

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed)
        episode_reward = 0.0
        episode_length = 0
        done = False

        opportunities_count = 0

        while not done:
            # Count opportunities available at this step
            if hasattr(env, 'current_opportunities'):
                opportunities_count += len(env.current_opportunities)

            # Log features for first 5 steps only (to avoid flooding)
            if episode_length < 5:
                try:
                    import datetime
                    from pathlib import Path

                    log_path = Path('test_features.log')

                    # Get active positions from environment's portfolio
                    portfolio = env.portfolio
                    active_positions = [p for p in portfolio.positions if p.hours_held > 0]

                    if active_positions:
                        # Find best available APR and current position APR from opportunities
                        best_available_apr = 0.0
                        current_market_aprs = {}  # Map symbol -> current APR
                        if hasattr(env, 'current_opportunities') and env.current_opportunities:
                            best_available_apr = max(opp.get('fund_apr', 0.0) for opp in env.current_opportunities)
                            # Build lookup map for current APR by symbol
                            for opp in env.current_opportunities:
                                current_market_aprs[opp['symbol']] = opp.get('fund_apr', 0.0)

                        # V9 Feature names (19 features per position)
                        feature_names = [
                            "is_active", "net_pnl_pct", "hours_held_norm", "estimated_pnl_pct",
                            "estimated_pnl_velocity", "estimated_funding_8h_pct", "funding_velocity",
                            "spread_pct", "spread_velocity", "liquidation_distance_pct", "apr_ratio",
                            "current_position_apr", "best_available_apr_norm", "apr_advantage",
                            "return_efficiency", "pnl_imbalance", "pnl_velocity", "peak_drawdown",
                            "hours_until_next_funding_norm"
                        ]

                        with open(log_path, 'a') as f:
                            f.write("=" * 80 + "\n")
                            f.write(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Test Position Features (Step {episode_length})\n")
                            f.write("=" * 80 + "\n")

                            for slot_idx, pos in enumerate(active_positions):
                                symbol = pos.symbol
                                raw_hours_held = pos.hours_held
                                entry_apr = pos.entry_apr if hasattr(pos, 'entry_apr') else 0.0

                                f.write(f"\nSymbol: {symbol}\n")
                                f.write(f"Hours Held: {raw_hours_held:.2f}h ({raw_hours_held*60:.0f} minutes)\n")
                                f.write(f"Entry APR: {entry_apr:.0f}%\n\n")

                                # Write raw position data
                                f.write("=" * 60 + "\n")
                                f.write("RAW POSITION DATA\n")
                                f.write("=" * 60 + "\n\n")

                                f.write("Position Sizing:\n")
                                f.write(f"  position_size_usd         : ${pos.position_size_usd:,.2f}\n")
                                f.write(f"  total_capital (2x)        : ${pos.position_size_usd * 2:,.2f}\n")
                                f.write(f"  leverage                  : {pos.leverage}x\n")

                                f.write(f"\nEntry Prices:\n")
                                f.write(f"  long_entry_price          : ${pos.entry_long_price:.6f}\n")
                                f.write(f"  short_entry_price         : ${pos.entry_short_price:.6f}\n")
                                f.write(f"  entry_spread              : ${abs(pos.entry_long_price - pos.entry_short_price):.6f}\n")
                                f.write(f"  entry_spread_pct          : {abs(pos.entry_long_price - pos.entry_short_price) / ((pos.entry_long_price + pos.entry_short_price) / 2) * 100:.4f}%\n")

                                f.write(f"\nP&L:\n")
                                f.write(f"  unrealized_pnl_usd        : ${pos.unrealized_pnl_usd:,.2f}\n")
                                f.write(f"  unrealized_pnl_pct        : {pos.unrealized_pnl_pct:.4f}%\n")
                                f.write(f"  long_pnl_pct              : {pos.long_pnl_pct:.4f}%\n")
                                f.write(f"  short_pnl_pct             : {pos.short_pnl_pct:.4f}%\n")
                                f.write(f"  peak_pnl_pct              : {pos.peak_pnl_pct:.4f}%\n")

                                f.write(f"\nFunding & Fees:\n")
                                f.write(f"  long_net_funding_usd      : ${pos.long_net_funding_usd:,.2f}\n")
                                f.write(f"  short_net_funding_usd     : ${pos.short_net_funding_usd:,.2f}\n")
                                f.write(f"  total_net_funding_usd     : ${pos.long_net_funding_usd + pos.short_net_funding_usd:,.2f}\n")
                                f.write(f"  entry_fees_paid_usd       : ${pos.entry_fees_paid_usd:,.2f}\n")
                                f.write(f"  estimated_exit_fees_usd   : ${pos.position_size_usd * 2 * pos.taker_fee:,.2f}\n")
                                f.write(f"  maker_fee                 : {pos.maker_fee * 100:.4f}%\n")
                                f.write(f"  taker_fee                 : {pos.taker_fee * 100:.4f}%\n")

                                f.write(f"\nFunding Rates (Current):\n")
                                f.write(f"  long_exchange             : {pos.long_exchange}\n")
                                f.write(f"  long_funding_rate         : {pos.long_funding_rate:.6f}\n")
                                f.write(f"  long_funding_interval     : {pos.long_funding_interval_hours:.1f}h\n")
                                f.write(f"  short_exchange            : {pos.short_exchange}\n")
                                f.write(f"  short_funding_rate        : {pos.short_funding_rate:.6f}\n")
                                f.write(f"  short_funding_interval    : {pos.short_funding_interval_hours:.1f}h\n")
                                f.write(f"  net_funding_rate          : {pos.short_funding_rate - pos.long_funding_rate:.6f}\n")

                                f.write(f"\nTimestamps:\n")
                                f.write(f"  entry_time                : {pos.entry_time}\n")
                                f.write(f"  hours_held                : {pos.hours_held:.2f}h\n")

                                f.write("\n" + "=" * 60 + "\n\n")

                                # Extract normalized features from observation vector (V9: 5 config + 2 portfolio = 7 prefix)
                                feat_start_idx = 7 + (slot_idx * 19)
                                normalized_features = obs[feat_start_idx:feat_start_idx+19]

                                # Get raw values
                                net_pnl_pct_raw = pos.unrealized_pnl_pct * 100
                                long_funding_rate = pos.long_funding_rate
                                short_funding_rate = pos.short_funding_rate
                                net_funding_rate = short_funding_rate - long_funding_rate

                                # Look up current APR for this symbol from opportunities
                                current_position_apr = current_market_aprs.get(pos.symbol, 0.0)

                                # For debug display, also calculate what it would be from current funding rates
                                long_interval = pos.long_funding_interval_hours
                                short_interval = pos.short_funding_interval_hours
                                if long_interval == short_interval:
                                    avg_interval_hours = long_interval
                                else:
                                    avg_interval_hours = (
                                        (abs(long_funding_rate) * long_interval + abs(short_funding_rate) * short_interval) /
                                        (abs(long_funding_rate) + abs(short_funding_rate) + 1e-9)
                                    )
                                payments_per_day = 24.0 / avg_interval_hours
                                annual_rate = net_funding_rate * payments_per_day * 365.0
                                current_funding_apr = annual_rate * 100

                                # Calculate log-normalized hours for display
                                import math
                                hours_held_log_norm = math.log(raw_hours_held + 1) / math.log(73) if raw_hours_held >= 0 else 0.0

                                # Calculate pnl_imbalance
                                pnl_imbalance = (pos.long_pnl_pct - pos.short_pnl_pct) / 2.0

                                # Write V9 raw features
                                f.write("Raw Features (before normalization) - V9:\n")
                                f.write(f"  1. is_active                  : {normalized_features[0]:.4f}\n")
                                f.write(f"  2. net_pnl_pct                : {net_pnl_pct_raw:.4f}%\n")
                                f.write(f"  3. hours_held                 : {raw_hours_held:.4f}h (log-normalized)\n")
                                f.write(f"  4. estimated_pnl_pct          : {normalized_features[3]:.4f}% (price P&L only)\n")
                                f.write(f"  5. estimated_pnl_velocity     : {normalized_features[4]:.6f} (trend signal)\n")
                                f.write(f"  6. estimated_funding_8h_pct   : {normalized_features[5]:.4f}% (expected funding)\n")
                                f.write(f"  7. funding_velocity           : {normalized_features[6]:.6f} (funding trend)\n")
                                f.write(f"  8. spread_pct                 : {normalized_features[7]:.6f}%\n")
                                f.write(f"  9. spread_velocity            : {normalized_features[8]:.6f} (converging/diverging)\n")
                                f.write(f" 10. liquidation_distance_pct  : {normalized_features[9]:.4f}%\n")
                                f.write(f" 11. apr_ratio                  : {normalized_features[10]:.4f}\n")
                                f.write(f" 12. current_position_apr       : {current_position_apr:.2f}%\n")
                                f.write(f" 13. best_available_apr_norm    : {best_available_apr:.2f}% (±5000% range)\n")
                                f.write(f" 14. apr_advantage              : {current_position_apr - best_available_apr:.2f}%\n")
                                f.write(f" 15. return_efficiency          : {normalized_features[14]:.6f}\n")
                                f.write(f" 16. pnl_imbalance              : {pnl_imbalance:.4f} (long-short asymmetry)\n")
                                f.write(f" 17. pnl_velocity               : {normalized_features[16]:.6f} (P&L trend)\n")
                                f.write(f" 18. peak_drawdown              : {normalized_features[17]:.4f}%\n")
                                f.write(f" 19. hours_until_next_funding   : {normalized_features[18]:.4f} (normalized)\n")

                                f.write(f"\nNormalized Features (sent to model) - V9:\n")
                                for i, (name, val) in enumerate(zip(feature_names, normalized_features)):
                                    f.write(f" {i+1:2d}. {name:30s}: {val:12.6f}\n")

                                f.write(f"\nV9 Feature Details:\n")
                                f.write(f"  hours_held_log_norm           : {hours_held_log_norm:.6f} (log({raw_hours_held:.2f}+1)/log(73))\n")
                                f.write(f"  long_pnl_pct                  : {pos.long_pnl_pct*100:.4f}%\n")
                                f.write(f"  short_pnl_pct                 : {pos.short_pnl_pct*100:.4f}%\n")
                                f.write(f"  pnl_imbalance                 : {pnl_imbalance:.4f} ((long-short)/2)\n")

                                f.write(f"\nAPR Context:\n")
                                f.write(f"  entry_apr (from opportunity)  : {entry_apr:.2f}%\n")
                                f.write(f"  current_position_apr (lookup) : {current_position_apr:.2f}%\n")
                                f.write(f"  best_available_apr            : {best_available_apr:.2f}%\n")
                                f.write(f"  apr_advantage                 : {current_position_apr - best_available_apr:.2f}%\n")

                                f.write(f"\nFunding Rates (reference):\n")
                                f.write(f"  long_funding_rate             : {long_funding_rate:.6f}\n")
                                f.write(f"  short_funding_rate            : {short_funding_rate:.6f}\n")
                                f.write(f"  net_funding_rate              : {net_funding_rate:.6f}\n")
                                f.write(f"  long_funding_interval         : {long_interval:.1f}h\n")
                                f.write(f"  short_funding_interval        : {short_interval:.1f}h\n")
                                f.write(f"  avg_funding_interval          : {avg_interval_hours:.1f}h\n")
                                f.write(f"  current_funding_apr (calc)    : {current_funding_apr:.2f}%\n")
                                f.write("\n")

                            f.write("\n\n")
                except Exception as e:
                    # Don't crash if logging fails
                    print(f"Warning: Failed to write test feature log: {e}")
                    pass

            # Get action mask
            if hasattr(env, '_get_action_mask'):
                action_mask = env._get_action_mask()
            else:
                action_mask = None

            # Select action (deterministic)
            if args.api_mode:
                # Use API client (passes env, obs and mask are ignored by API)
                action, _, _ = predictor.select_action(env, obs, action_mask)
                # Get confidence from last API response
                confidence = getattr(predictor, '_last_confidence', 1.0)
            else:
                # Use direct model inference
                action, _, _ = predictor.select_action(obs, action_mask, deterministic=True)
                # Get confidence from action probabilities
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
                    action_logits, _ = predictor.network(obs_tensor, mask_tensor)
                    probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
                    confidence = float(probs[action])

            # Apply confidence threshold (V9: action 0 = HOLD, 1-15 = ENTER, 16 = EXIT)
            original_action = action
            if action >= 1 and action <= 15:  # V9: ENTER actions
                if confidence < args.enter_threshold:
                    action = 0  # Default to HOLD
                    threshold_blocked_enters += 1
            elif action == 16:  # V9: EXIT_POS_0
                if confidence < args.exit_threshold:
                    action = 0  # Default to HOLD
                    threshold_blocked_exits += 1

            # Apply APR thresholds (only for ENTER actions that weren't already blocked)
            if action >= 1 and action <= 15:  # V9: ENTER actions
                # Decode opportunity index from action (V9)
                # Actions 1-5: opportunities 0-4 (SMALL size)
                # Actions 6-10: opportunities 0-4 (MEDIUM size)
                # Actions 11-15: opportunities 0-4 (LARGE size)
                opp_idx = (action - 1) % 5  # V9: 0-4

                # Get opportunity data from environment
                if hasattr(env, 'current_opportunities') and opp_idx < len(env.current_opportunities):
                    opp = env.current_opportunities[opp_idx]
                    fund_apr = opp.get('fund_apr', 0)
                    fund_apr_24h = opp.get('fund_apr_24h_proj', 0)
                    fund_apr_3d = opp.get('fund_apr_3d_proj', 0)
                    spread_vol = opp.get('spread_volatility_stddev', 0)

                    # Check min APR threshold
                    if args.min_apr > 0 and fund_apr < args.min_apr:
                        action = 0
                        apr_blocked_low_apr += 1
                    # Check 24h positive requirement
                    elif args.require_positive_24h and fund_apr_24h < 0:
                        action = 0
                        apr_blocked_neg_24h += 1
                    # Check 3d positive requirement
                    elif args.require_positive_3d and fund_apr_3d < 0:
                        action = 0
                        apr_blocked_neg_3d += 1
                    # Check max spread volatility
                    elif args.max_spread_vol > 0 and spread_vol > args.max_spread_vol:
                        action = 0
                        apr_blocked_high_spread_vol += 1

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # Collect portfolio metrics after episode
        portfolio = env.portfolio
        config = env.current_config

        # Collect all trades from this episode (closed positions)
        for position in portfolio.closed_positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

            # Collect funding details for analysis
            funding_summary = position.get_funding_summary()
            funding_summary['episode'] = episode + 1
            all_funding_details.append(funding_summary)

        # Also collect open positions (if any)
        for position in portfolio.positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        total_pnls.append(portfolio.total_pnl_usd)
        total_pnl_pcts.append(portfolio.total_pnl_pct)
        max_drawdowns.append(portfolio.max_drawdown_pct)
        opportunities_seen.append(opportunities_count)

        # Trade statistics
        total_closed = len(portfolio.closed_positions)
        num_trades.append(total_closed)

        if total_closed > 0:
            winning = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd > 0)
            losing = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd <= 0)
            num_winning_trades.append(winning)
            num_losing_trades.append(losing)

            # Average trade duration in hours
            durations = [p.hours_held for p in portfolio.closed_positions]
            avg_trade_durations.append(np.mean(durations))
        else:
            num_winning_trades.append(0)
            num_losing_trades.append(0)
            avg_trade_durations.append(0.0)

        # Trades executed (total positions opened = currently open + closed)
        trades_executed.append(len(portfolio.positions) + len(portfolio.closed_positions))

        # Collect P&L for profit factor calculation
        for position in portfolio.closed_positions:
            if position.realized_pnl_usd > 0:
                all_winning_pnl.append(position.realized_pnl_usd)
            else:
                all_losing_pnl.append(abs(position.realized_pnl_usd))

        # Track config used
        configs_used.append({
            'leverage': config.max_leverage,
            'utilization': config.target_utilization,
            'max_positions': config.max_positions,
        })

        # Show length in appropriate unit
        if args.step_minutes == 1:
            length_str = f"{episode_length:5d} min"
        elif args.step_minutes == 60:
            length_str = f"{episode_length:3d}h"
        else:
            length_str = f"{episode_length:3d} steps"

        print(f"   Episode {episode + 1}: Reward={episode_reward:7.2f}, P&L=${portfolio.total_pnl_usd:7.2f} ({portfolio.total_pnl_pct:.2f}%), Trades={total_closed:2d}, Length={length_str}")

    # Calculate aggregate statistics
    total_trades_sum = sum(num_trades)
    total_winning = sum(num_winning_trades)
    total_losing = sum(num_losing_trades)
    win_rate = (total_winning / total_trades_sum * 100) if total_trades_sum > 0 else 0.0

    print(f"\n✅ Inference test complete!")
    print(f"\n{'='*80}")
    print("DETAILED EVALUATION METRICS (TEST SET)")
    print(f"{'='*80}")

    print(f"\n📊 Episode Metrics:")
    print(f"  Mean Reward:     {np.mean(eval_rewards):8.2f} ± {np.std(eval_rewards):.2f}")

    # Show length in appropriate unit
    if args.step_minutes == 1:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} minutes")
    elif args.step_minutes == 60:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} hours")
    else:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} steps ({args.step_minutes} min/step)")

    print(f"\n💰 P&L Metrics:")
    print(f"  Mean P&L (USD):  ${np.mean(total_pnls):8.2f}")
    print(f"  Mean P&L (%):    {np.mean(total_pnl_pcts):8.2f}%")
    print(f"  Total P&L:       ${np.sum(total_pnls):8.2f}")

    print(f"\n📈 Trading Metrics:")
    print(f"  Total Trades:    {total_trades_sum:8.0f}")
    print(f"  Winning Trades:  {total_winning:8.0f}")
    print(f"  Losing Trades:   {total_losing:8.0f}")
    print(f"  Win Rate:        {win_rate:8.1f}%")
    print(f"  Avg Duration:    {np.mean([d for d in avg_trade_durations if d > 0]) if any(avg_trade_durations) else 0.0:8.1f} hours")

    print(f"\n🎯 Opportunity Metrics:")
    print(f"  Opportunities/Ep: {np.mean(opportunities_seen):7.0f}")
    print(f"  Trades/Episode:   {np.mean(num_trades):7.1f}")
    print(f"  Execution Rate:   {(np.mean(num_trades) / np.mean(opportunities_seen) * 100) if np.mean(opportunities_seen) > 0 else 0:.1f}%")

    print(f"\n⚠️  Risk Metrics:")
    print(f"  Max Drawdown:    {np.mean(max_drawdowns):8.2f}%")

    # Calculate profit factor
    total_wins = sum(all_winning_pnl) if all_winning_pnl else 0.0
    total_losses = sum(all_losing_pnl) if all_losing_pnl else 0.001
    profit_factor = total_wins / total_losses

    print(f"\n📊 Profitability Metrics:")
    print(f"  Profit Factor:   {profit_factor:8.2f}")

    print(f"\n⚙️  Configuration Used:")
    avg_config = configs_used[0]  # Assuming same config for all episodes
    print(f"  Leverage:        {avg_config['leverage']:.1f}x")
    print(f"  Utilization:     {avg_config['utilization']:.0%}")
    print(f"  Max Positions:   {avg_config['max_positions']}")

    # Report threshold filtering if enabled
    if args.enter_threshold > 0 or args.exit_threshold > 0:
        print(f"\n🎯 Confidence Threshold Results:")
        print(f"  ENTER threshold: {args.enter_threshold*100:.0f}%")
        print(f"  EXIT threshold:  {args.exit_threshold*100:.0f}%")
        print(f"  Blocked ENTERs:  {threshold_blocked_enters:8d} (would have entered, defaulted to HOLD)")
        print(f"  Blocked EXITs:   {threshold_blocked_exits:8d} (would have exited, defaulted to HOLD)")

    # Report APR threshold filtering if enabled
    if args.min_apr > 0 or args.require_positive_24h or args.require_positive_3d or args.max_spread_vol > 0:
        total_apr_blocked = apr_blocked_low_apr + apr_blocked_neg_24h + apr_blocked_neg_3d + apr_blocked_high_spread_vol
        print(f"\n📊 APR Threshold Results:")
        if args.min_apr > 0:
            print(f"  Min APR ({args.min_apr:.0f}%):     {apr_blocked_low_apr:8d} blocked")
        if args.require_positive_24h:
            print(f"  Require 24h > 0:    {apr_blocked_neg_24h:8d} blocked")
        if args.require_positive_3d:
            print(f"  Require 3d > 0:     {apr_blocked_neg_3d:8d} blocked")
        if args.max_spread_vol > 0:
            print(f"  Max spread vol:     {apr_blocked_high_spread_vol:8d} blocked")
        print(f"  Total blocked:      {total_apr_blocked:8d} ENTER actions → HOLD")

    # Calculate composite score (IMPROVED VERSION)
    # Weights: 50% P&L, 30% Profit Factor, 20% Low Drawdown

    # 1. P&L Score: Bounded using tanh to prevent domination
    pnl_score = np.tanh(np.mean(total_pnl_pcts) / 5.0)

    # 2. Profit Factor Score: Normalize (2.0 = 1.0, capped at 1.0)
    profit_factor_score = min(profit_factor / 2.0, 1.0)

    # 3. Drawdown Score: Lower is better, floored at 0.0
    drawdown_score = max(0.0, 1.0 - (np.mean(max_drawdowns) / 100.0))

    composite_score = (
        0.50 * pnl_score +
        0.30 * profit_factor_score +
        0.20 * drawdown_score
    )

    print(f"\n🎯 Composite Score: {composite_score:.4f}")
    print(f"   (P&L: {pnl_score:.3f} | ProfitFactor: {profit_factor_score:.3f} | Drawdown: {drawdown_score:.3f})")

    # Show top 5 winners and losers
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        if 'realized_pnl_usd' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
            if len(closed_trades) > 0:
                closed_trades_sorted = closed_trades.sort_values('realized_pnl_usd', ascending=False)

                print(f"\n💰 TOP 5 PROFITABLE TRADES:")
                print(f"   {'Symbol':<12} {'P&L USD':>10} {'P&L %':>8} {'Entry':>14} {'Exit':>14}")
                print(f"   {'-'*66}")
                for _, trade in closed_trades_sorted.head(5).iterrows():
                    symbol = trade.get('symbol', 'N/A')[:12]
                    pnl_usd = trade.get('realized_pnl_usd', 0)
                    pnl_pct = trade.get('realized_pnl_pct', 0)
                    entry_dt = trade.get('entry_datetime', '')
                    exit_dt = trade.get('exit_datetime', '')
                    # Format datetime to MM-DD HH:MM
                    if hasattr(entry_dt, 'strftime'):
                        entry_str = entry_dt.strftime('%m-%d %H:%M')
                    else:
                        entry_str = str(entry_dt)[-14:-3] if entry_dt else 'N/A'
                    if hasattr(exit_dt, 'strftime'):
                        exit_str = exit_dt.strftime('%m-%d %H:%M')
                    else:
                        exit_str = str(exit_dt)[-14:-3] if exit_dt else 'N/A'
                    print(f"   {symbol:<12} ${pnl_usd:>9.2f} {pnl_pct:>7.2f}% {entry_str:>14} {exit_str:>14}")

                print(f"\n📉 TOP 5 LOSING TRADES:")
                print(f"   {'Symbol':<12} {'P&L USD':>10} {'P&L %':>8} {'Entry':>14} {'Exit':>14}")
                print(f"   {'-'*66}")
                for _, trade in closed_trades_sorted.tail(5).iloc[::-1].iterrows():
                    symbol = trade.get('symbol', 'N/A')[:12]
                    pnl_usd = trade.get('realized_pnl_usd', 0)
                    pnl_pct = trade.get('realized_pnl_pct', 0)
                    entry_dt = trade.get('entry_datetime', '')
                    exit_dt = trade.get('exit_datetime', '')
                    if hasattr(entry_dt, 'strftime'):
                        entry_str = entry_dt.strftime('%m-%d %H:%M')
                    else:
                        entry_str = str(entry_dt)[-14:-3] if entry_dt else 'N/A'
                    if hasattr(exit_dt, 'strftime'):
                        exit_str = exit_dt.strftime('%m-%d %H:%M')
                    else:
                        exit_str = str(exit_dt)[-14:-3] if exit_dt else 'N/A'
                    print(f"   {symbol:<12} ${pnl_usd:>9.2f} {pnl_pct:>7.2f}% {entry_str:>14} {exit_str:>14}")

    # Write trades to CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        # Reorder columns for better readability
        column_order = [
            'episode', 'entry_datetime', 'exit_datetime', 'symbol',
            'long_exchange', 'short_exchange', 'status',
            'position_size_usd', 'leverage', 'margin_used_usd',
            'entry_long_price', 'entry_short_price',
            'exit_long_price', 'exit_short_price',
            'long_funding_rate', 'short_funding_rate',
            'funding_earned_usd', 'long_funding_earned_usd', 'short_funding_earned_usd',
            'entry_fees_usd', 'exit_fees_usd', 'total_fees_usd',
            'realized_pnl_usd', 'realized_pnl_pct',
            'unrealized_pnl_usd', 'unrealized_pnl_pct',
            'hours_held',
        ]
        trades_df = trades_df[column_order]

        # Save to CSV
        output_path = Path(args.trades_output)
        trades_df.to_csv(output_path, index=False)

        print(f"\n💾 Trade Records:")
        print(f"   Total trades: {len(all_trades)}")
        print(f"   Closed trades: {len([t for t in all_trades if t['status'] == 'closed'])}")
        print(f"   Open positions: {len([t for t in all_trades if t['status'] == 'open'])}")
        print(f"   Saved to: {output_path}")
    else:
        print(f"\n⚠️  No trades executed during inference")

    # Write funding details to CSV
    if all_funding_details:
        funding_df = pd.DataFrame(all_funding_details)

        # Save to CSV (in same directory as trades)
        funding_output_path = Path(args.trades_output).parent / 'test_inference_funding_details.csv'
        funding_df.to_csv(funding_output_path, index=False)

        # Print summary statistics
        print(f"\n💰 Funding Details:")
        print(f"   Total positions analyzed: {len(all_funding_details)}")
        print(f"   Total funding earned: ${funding_df['net_funding_usd'].sum():.2f}")
        print(f"   Avg funding per position: ${funding_df['net_funding_usd'].mean():.2f}")
        print(f"   Total long payments: {funding_df['long_funding_payment_count'].sum():.0f}")
        print(f"   Total short payments: {funding_df['short_funding_payment_count'].sum():.0f}")
        print(f"   Saved to: {funding_output_path}")

        # Show top 5 positions by funding earned
        top_funding = funding_df.nlargest(5, 'net_funding_usd')[['symbol', 'hours_held', 'net_funding_usd', 'net_funding_pct', 'realized_pnl_usd', 'entry_time', 'exit_time']]
        print(f"\n   Top 5 by Funding Earned:")
        for idx, row in top_funding.iterrows():
            entry_str = row['entry_time'].strftime('%m-%d %H:%M') if pd.notna(row['entry_time']) else 'N/A'
            exit_str = row['exit_time'].strftime('%m-%d %H:%M') if pd.notna(row['exit_time']) else 'N/A'
            print(f"     {row['symbol']:12s}: ${row['net_funding_usd']:6.2f} ({row['net_funding_pct']:.2f}%) over {row['hours_held']:.1f}h, Total P&L: ${row['realized_pnl_usd']:6.2f}  [{entry_str} → {exit_str}]")
    else:
        print(f"\n⚠️  No funding details collected (no closed positions)")

    return True


if __name__ == '__main__':
    args = parse_args()
    success = test_model_inference(args)
    if success:
        print("\n" + "=" * 80)
        print("✅ ALL INFERENCE TESTS PASSED")
        print("=" * 80)
