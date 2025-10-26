"""
Performance Metrics

Calculate trading performance metrics for backtesting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class BacktestPosition:
    """Position data structure (imported from backtester)."""
    entry_time: any
    exit_time: any
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


class PerformanceMetrics:
    """
    Calculate performance metrics for backtest results.
    """

    @staticmethod
    def calculate_all(
        positions: List[BacktestPosition],
        initial_capital: float,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            positions: List of backtest positions
            initial_capital: Starting capital
            equity_curve: Equity over time

        Returns:
            Dictionary with all metrics
        """
        if len(positions) == 0:
            return PerformanceMetrics._empty_metrics(initial_capital)

        # Extract profit/loss data
        profits = [p.actual_profit for p in positions]
        profit_pcts = [(p.actual_profit / initial_capital) * 100 for p in positions]

        winning_trades = [p for p in positions if p.was_profitable]
        losing_trades = [p for p in positions if not p.was_profitable]

        # Basic metrics
        num_trades = len(positions)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        win_rate = (winning_count / num_trades * 100) if num_trades > 0 else 0

        # Return metrics
        total_profit = sum(profits)
        total_return_pct = (total_profit / initial_capital) * 100
        final_capital = initial_capital + total_profit

        avg_profit_pct = np.mean(profit_pcts)

        avg_winning_pct = np.mean([p.actual_profit / initial_capital * 100 for p in winning_trades]) if winning_trades else 0
        avg_losing_pct = np.mean([p.actual_profit / initial_capital * 100 for p in losing_trades]) if losing_trades else 0

        # Risk metrics
        sharpe_ratio = PerformanceMetrics._calculate_sharpe(profit_pcts)
        max_drawdown_pct = PerformanceMetrics._calculate_max_drawdown(equity_curve, initial_capital)

        # Profit factor
        gross_profit = sum([p.actual_profit for p in winning_trades])
        gross_loss = abs(sum([p.actual_profit for p in losing_trades]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Duration metrics
        avg_hold_hours = np.mean([p.actual_duration for p in positions])

        # Prediction accuracy
        prediction_errors = [abs(p.predicted_profit - (p.actual_profit / initial_capital * 100)) for p in positions]
        avg_prediction_error = np.mean(prediction_errors)

        return {
            # Trade counts
            'num_trades': num_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate_pct': win_rate,

            # Returns
            'total_return_pct': total_return_pct,
            'total_profit_usd': total_profit,
            'final_capital': final_capital,

            # Per-trade metrics
            'avg_profit_per_trade_pct': avg_profit_pct,
            'avg_winning_trade_pct': avg_winning_pct,
            'avg_losing_trade_pct': avg_losing_pct,

            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,

            # Duration
            'avg_hold_hours': avg_hold_hours,

            # Prediction accuracy
            'avg_prediction_error_pct': avg_prediction_error
        }

    @staticmethod
    def _calculate_sharpe(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: List of returns (%)
            risk_free_rate: Risk-free rate (annual %)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)

        # Annualize (assuming returns are per trade, adjust based on frequency)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return 0.0

        # Simple Sharpe (not annualized - would need to know trade frequency)
        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series, initial_capital: float) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity over time
            initial_capital: Starting capital

        Returns:
            Maximum drawdown percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max * 100

        # Get maximum drawdown (most negative value)
        max_drawdown = drawdown.min()

        return abs(max_drawdown)

    @staticmethod
    def _empty_metrics(initial_capital: float) -> Dict[str, float]:
        """Return empty metrics when no trades."""
        return {
            'num_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0.0,
            'total_return_pct': 0.0,
            'total_profit_usd': 0.0,
            'final_capital': initial_capital,
            'avg_profit_per_trade_pct': 0.0,
            'avg_winning_trade_pct': 0.0,
            'avg_losing_trade_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'avg_hold_hours': 0.0,
            'avg_prediction_error_pct': 0.0
        }


class MetricsComparison:
    """
    Compare metrics across multiple models/strategies.
    """

    @staticmethod
    def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare metrics from multiple models.

        Args:
            results_dict: Dict of {model_name: metrics_dict}

        Returns:
            DataFrame with comparison
        """
        df = pd.DataFrame(results_dict).T

        # Sort by Sharpe ratio (best first)
        df = df.sort_values('sharpe_ratio', ascending=False)

        return df

    @staticmethod
    def print_comparison(comparison_df: pd.DataFrame) -> None:
        """
        Print formatted comparison table.

        Args:
            comparison_df: Comparison DataFrame
        """
        print("\n" + "=" * 120)
        print("MODEL COMPARISON")
        print("=" * 120)

        print(f"\n{'Model':<20} {'Return %':<12} {'Win Rate %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Trades':<10}")
        print("-" * 120)

        for idx, row in comparison_df.iterrows():
            print(f"{idx:<20} {row['total_return_pct']:>10.2f}% {row['win_rate_pct']:>10.1f}% "
                  f"{row['sharpe_ratio']:>9.2f} {row['max_drawdown_pct']:>10.2f}% {row['num_trades']:>9.0f}")

        print("=" * 120)

        # Highlight best model
        best_model = comparison_df.iloc[0].name
        print(f"\n✅ Best Model (by Sharpe Ratio): {best_model}")
        print(f"   Sharpe: {comparison_df.iloc[0]['sharpe_ratio']:.2f}")
        print(f"   Return: {comparison_df.iloc[0]['total_return_pct']:.2f}%")
        print(f"   Win Rate: {comparison_df.iloc[0]['win_rate_pct']:.1f}%\n")
