"""
Backtesting Framework

Walk-forward backtesting to validate ML model performance on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..scoring.opportunity_scorer import OpportunityScorer
from .metrics import PerformanceMetrics


@dataclass
class BacktestPosition:
    """
    Represents a simulated trading position in the backtest.
    """
    entry_time: datetime
    exit_time: datetime
    symbol: str
    strategy: str
    predicted_profit: float
    predicted_duration: float
    success_probability: float
    composite_score: float
    actual_profit: float
    actual_duration: float
    was_profitable: bool
    fees_paid: float


@dataclass
class BacktestResult:
    """
    Results from a backtest run.
    """
    positions: List[BacktestPosition]
    metrics: Dict[str, float]
    equity_curve: pd.Series
    start_date: datetime
    end_date: datetime
    model_name: str


class Backtester:
    """
    Backtest trading strategies using historical opportunity data.

    Simulates: "If we had used this model to pick opportunities, what would returns be?"
    """

    def __init__(
        self,
        scorer: OpportunityScorer,
        initial_capital: float = 10000.0
    ):
        """
        Initialize backtester.

        Args:
            scorer: OpportunityScorer instance with trained models
            initial_capital: Starting capital for backtest
        """
        self.scorer = scorer
        self.initial_capital = initial_capital
        self.capital = initial_capital

    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        selection_interval_hours: int = 24,
        max_concurrent_positions: int = 1,
        model_name: str = "Model"
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Strategy:
        1. Every `selection_interval_hours`, look at all available opportunities
        2. Score them using the OpportunityScorer
        3. Select the top opportunity
        4. Simulate holding it until exit
        5. Track results

        Args:
            historical_data: DataFrame with simulated execution data
            start_date: Start date for backtest (default: earliest data)
            end_date: End date for backtest (default: latest data)
            selection_interval_hours: How often to select new opportunities (hours)
            max_concurrent_positions: Maximum number of concurrent positions (default: 1)
            model_name: Name of the model being tested

        Returns:
            BacktestResult with positions and performance metrics
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {model_name}")
        print(f"{'='*80}\n")

        # Filter by date range
        if start_date:
            historical_data = historical_data[historical_data['entry_time'] >= start_date]
        if end_date:
            historical_data = historical_data[historical_data['entry_time'] <= end_date]

        if len(historical_data) == 0:
            raise ValueError("No data available for the specified date range")

        actual_start = historical_data['entry_time'].min()
        actual_end = historical_data['entry_time'].max()

        print(f"Period: {actual_start} to {actual_end}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Selection Interval: Every {selection_interval_hours} hours")
        print(f"Max Concurrent Positions: {max_concurrent_positions}\n")

        # Reset capital
        self.capital = self.initial_capital

        # Track positions and equity
        positions: List[BacktestPosition] = []
        equity_history: List[Tuple[datetime, float]] = [(actual_start, self.capital)]

        # Walk forward through time
        current_time = actual_start
        active_positions: List[BacktestPosition] = []

        while current_time <= actual_end:
            # Close any positions that have reached their exit time
            for pos in active_positions[:]:  # Copy list to safely remove during iteration
                if current_time >= pos.exit_time:
                    # Realize P&L
                    self.capital += pos.actual_profit
                    active_positions.remove(pos)
                    equity_history.append((current_time, self.capital))

            # Check if we can open new positions
            if len(active_positions) < max_concurrent_positions:
                # Get all opportunities available at this timestamp
                available_opps = historical_data[
                    (historical_data['entry_time'] >= current_time) &
                    (historical_data['entry_time'] < current_time + timedelta(hours=selection_interval_hours))
                ]

                if len(available_opps) > 0:
                    # Score and select best opportunity
                    try:
                        best_opp, score = self.scorer.select_best_opportunity(
                            available_opps,
                            apply_filters=True
                        )

                        # Create position
                        position = BacktestPosition(
                            entry_time=best_opp['entry_time'],
                            exit_time=best_opp['exit_time'],
                            symbol=best_opp.get('symbol', 'N/A'),
                            strategy=best_opp.get('strategy', 'N/A'),
                            predicted_profit=best_opp['predicted_profit'],
                            predicted_duration=best_opp['predicted_duration'],
                            success_probability=best_opp['success_probability'],
                            composite_score=score,
                            actual_profit=best_opp['target_profit_usd'],
                            actual_duration=best_opp['target_hold_hours'],
                            was_profitable=best_opp['target_was_profitable'],
                            fees_paid=best_opp.get('total_fees_usd', 0)
                        )

                        active_positions.append(position)
                        positions.append(position)

                    except ValueError as e:
                        # No opportunities passed filters
                        pass

            # Move forward by selection interval
            current_time += timedelta(hours=selection_interval_hours)

        # Close any remaining active positions at end of backtest
        for pos in active_positions:
            self.capital += pos.actual_profit
            equity_history.append((actual_end, self.capital))

        # Create equity curve
        equity_df = pd.DataFrame(equity_history, columns=['timestamp', 'equity'])
        equity_df = equity_df.set_index('timestamp')
        equity_curve = equity_df['equity']

        # Calculate metrics
        metrics = PerformanceMetrics.calculate_all(positions, self.initial_capital, equity_curve)

        # Print summary
        self._print_summary(positions, metrics)

        return BacktestResult(
            positions=positions,
            metrics=metrics,
            equity_curve=equity_curve,
            start_date=actual_start,
            end_date=actual_end,
            model_name=model_name
        )

    def _print_summary(self, positions: List[BacktestPosition], metrics: Dict[str, float]) -> None:
        """Print backtest summary."""
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}\n")

        print(f"Total Trades: {metrics['num_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate_pct']:.1f}%)")
        print(f"Losing Trades: {metrics['losing_trades']}")

        print(f"\n💰 Returns:")
        print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"   Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"   Total Profit: ${metrics['total_profit_usd']:,.2f}")

        print(f"\n📊 Per-Trade Metrics:")
        print(f"   Avg Profit per Trade: {metrics['avg_profit_per_trade_pct']:.2f}%")
        print(f"   Avg Winning Trade: {metrics['avg_winning_trade_pct']:.2f}%")
        print(f"   Avg Losing Trade: {metrics['avg_losing_trade_pct']:.2f}%")

        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\n⏱️  Duration:")
        print(f"   Avg Hold Time: {metrics['avg_hold_hours']:.1f} hours")

        print(f"\n{'='*80}\n")


class WalkForwardValidator:
    """
    Perform walk-forward validation: train on period X, test on period Y, repeat.
    """

    def __init__(
        self,
        train_window_months: int = 4,
        test_window_months: int = 1
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_window_months: Training window size in months
            test_window_months: Testing window size in months
        """
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months

    def validate(
        self,
        historical_data: pd.DataFrame,
        train_model_fn: callable,
        scorer_class: type
    ) -> List[BacktestResult]:
        """
        Perform walk-forward validation.

        Args:
            historical_data: Full historical dataset
            train_model_fn: Function to train model: train_model_fn(train_data) -> ModelEnsemble
            scorer_class: OpportunityScorer class

        Returns:
            List of BacktestResult for each test period
        """
        results = []

        # Get date range
        start_date = historical_data['entry_time'].min()
        end_date = historical_data['entry_time'].max()

        current_train_start = start_date

        while current_train_start < end_date:
            # Define train and test periods
            train_end = current_train_start + pd.DateOffset(months=self.train_window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_window_months)

            if test_end > end_date:
                break

            print(f"\n{'='*80}")
            print(f"WALK-FORWARD FOLD")
            print(f"Train: {current_train_start} to {train_end}")
            print(f"Test: {test_start} to {test_end}")
            print(f"{'='*80}\n")

            # Split data
            train_data = historical_data[
                (historical_data['entry_time'] >= current_train_start) &
                (historical_data['entry_time'] < train_end)
            ]

            test_data = historical_data[
                (historical_data['entry_time'] >= test_start) &
                (historical_data['entry_time'] < test_end)
            ]

            # Train model
            print("Training model...")
            model_ensemble = train_model_fn(train_data)

            # Create scorer
            scorer = scorer_class(model_ensemble)

            # Backtest on test period
            backtester = Backtester(scorer)
            result = backtester.run_backtest(
                test_data,
                start_date=test_start,
                end_date=test_end,
                model_name=f"WF-{current_train_start.strftime('%Y%m')}"
            )

            results.append(result)

            # Move to next fold
            current_train_start = test_start

        # Aggregate results
        self._print_aggregate_results(results)

        return results

    def _print_aggregate_results(self, results: List[BacktestResult]) -> None:
        """Print aggregated walk-forward results."""
        print(f"\n{'='*80}")
        print("WALK-FORWARD VALIDATION - AGGREGATE RESULTS")
        print(f"{'='*80}\n")

        avg_return = np.mean([r.metrics['total_return_pct'] for r in results])
        avg_win_rate = np.mean([r.metrics['win_rate_pct'] for r in results])
        avg_sharpe = np.mean([r.metrics['sharpe_ratio'] for r in results])
        avg_max_dd = np.mean([r.metrics['max_drawdown_pct'] for r in results])

        print(f"Number of Folds: {len(results)}")
        print(f"\nAverage Metrics Across Folds:")
        print(f"   Avg Return: {avg_return:.2f}%")
        print(f"   Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"   Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"   Avg Max Drawdown: {avg_max_dd:.2f}%")

        print(f"\n{'='*80}\n")
