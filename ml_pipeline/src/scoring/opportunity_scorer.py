"""
Opportunity Scorer

Composite scoring system to rank multiple opportunities and select the best one.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

from ..models.base_model import ModelEnsemble


class OpportunityScorer:
    """
    Score and rank arbitrage opportunities using ML predictions.

    Combines multiple factors into a composite score:
    - Predicted profit %
    - Success probability
    - Risk-adjusted return
    - Hold duration (prefer shorter holds)
    """

    def __init__(
        self,
        model_ensemble: ModelEnsemble,
        config_path: Optional[Path] = None
    ):
        """
        Initialize scorer.

        Args:
            model_ensemble: Trained model ensemble (profit, success, duration)
            config_path: Path to scoring configuration YAML
        """
        self.ensemble = model_ensemble
        self.config = self._load_config(config_path)

        # Extract weights from config
        weights = self.config.get('weights', {})
        self.weight_profit = weights.get('predicted_profit', 0.40)
        self.weight_success = weights.get('success_probability', 0.30)
        self.weight_risk_adjusted = weights.get('risk_adjusted_return', 0.20)
        self.weight_duration = weights.get('hold_duration', 0.10)

        # Normalization ranges
        norm = self.config.get('normalization', {})
        self.profit_range = norm.get('profit_range', {'min': -5.0, 'max': 20.0})
        self.duration_range = norm.get('duration_range', {'min': 1.0, 'max': 168.0})

        # Filtering thresholds
        filtering = self.config.get('filtering', {})
        self.min_predicted_profit = filtering.get('min_predicted_profit', 0.1)
        self.min_success_probability = filtering.get('min_success_probability', 0.5)
        self.max_hold_duration = filtering.get('max_hold_duration', 168)

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load scoring configuration from YAML."""
        if config_path is None:
            # Use default config
            return {
                'weights': {
                    'predicted_profit': 0.40,
                    'success_probability': 0.30,
                    'risk_adjusted_return': 0.20,
                    'hold_duration': 0.10
                }
            }

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def score_single_opportunity(
        self,
        features: pd.Series,
        return_components: bool = False
    ) -> float:
        """
        Score a single opportunity.

        Args:
            features: Feature vector for the opportunity
            return_components: If True, return dict with score breakdown

        Returns:
            Composite score (0-100) or dict with breakdown
        """
        # Get predictions from all three models
        X = features.to_frame().T

        profit_pred = self.ensemble.profit_model.predict(X)[0]
        success_prob = self.ensemble.success_model.predict_proba(X)[0, 1]
        duration_pred = self.ensemble.duration_model.predict(X)[0]

        # Calculate risk metrics
        spread_volatility = features.get('spread_volatility_cv', 0)

        # === COMPONENT SCORES ===

        # 1. Profit Score (normalized to 0-100)
        profit_score = self._normalize(
            profit_pred,
            self.profit_range['min'],
            self.profit_range['max']
        ) * 100

        # 2. Success Score (probability is already 0-1, scale to 0-100)
        success_score = success_prob * 100

        # 3. Risk-Adjusted Return Score
        if spread_volatility > 0:
            risk_adjusted_profit = profit_pred / (1 + spread_volatility)
        else:
            risk_adjusted_profit = profit_pred

        risk_adjusted_score = self._normalize(
            risk_adjusted_profit,
            self.profit_range['min'] / 2,  # Risk-adjusted will be lower
            self.profit_range['max'] / 2
        ) * 100

        # 4. Duration Score (prefer shorter holds)
        # Inverse: shorter duration = higher score
        duration_score = self._normalize_inverse(
            duration_pred,
            self.duration_range['min'],
            self.duration_range['max']
        ) * 100

        # === COMPOSITE SCORE ===
        composite_score = (
            profit_score * self.weight_profit +
            success_score * self.weight_success +
            risk_adjusted_score * self.weight_risk_adjusted +
            duration_score * self.weight_duration
        )

        # Clip to 0-100 range
        composite_score = np.clip(composite_score, 0, 100)

        if return_components:
            return {
                'composite_score': composite_score,
                'profit_score': profit_score,
                'success_score': success_score,
                'risk_adjusted_score': risk_adjusted_score,
                'duration_score': duration_score,
                'predicted_profit': profit_pred,
                'success_probability': success_prob,
                'predicted_duration': duration_pred
            }

        return composite_score

    def score_opportunities(
        self,
        opportunities_df: pd.DataFrame,
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Score multiple opportunities and return sorted by score.

        Args:
            opportunities_df: DataFrame with opportunity features
            return_components: If True, include score breakdown columns

        Returns:
            DataFrame with scores, sorted by composite_score (descending)
        """
        scored_opportunities = []

        for idx, row in opportunities_df.iterrows():
            score_result = self.score_single_opportunity(row, return_components=True)

            scored_opp = row.to_dict()
            scored_opp.update(score_result)
            scored_opportunities.append(scored_opp)

        scored_df = pd.DataFrame(scored_opportunities)

        # Sort by composite score (best first)
        scored_df = scored_df.sort_values('composite_score', ascending=False).reset_index(drop=True)

        # Add rank
        scored_df['rank'] = range(1, len(scored_df) + 1)

        if not return_components:
            # Keep only essential columns
            keep_cols = [col for col in scored_df.columns if col not in [
                'profit_score', 'success_score', 'risk_adjusted_score', 'duration_score'
            ]]
            scored_df = scored_df[keep_cols]

        return scored_df

    def select_best_opportunity(
        self,
        opportunities_df: pd.DataFrame,
        apply_filters: bool = True
    ) -> Tuple[pd.Series, float]:
        """
        Select the single best opportunity from a list.

        Args:
            opportunities_df: DataFrame with opportunity features
            apply_filters: Whether to apply minimum thresholds

        Returns:
            (best_opportunity, score)
        """
        # Score all opportunities
        scored_df = self.score_opportunities(opportunities_df, return_components=True)

        if apply_filters:
            # Apply filtering thresholds
            scored_df = scored_df[
                (scored_df['predicted_profit'] >= self.min_predicted_profit) &
                (scored_df['success_probability'] >= self.min_success_probability) &
                (scored_df['predicted_duration'] <= self.max_hold_duration)
            ]

            if len(scored_df) == 0:
                raise ValueError("No opportunities passed filtering criteria")

        # Select top opportunity
        best_opp = scored_df.iloc[0]
        best_score = best_opp['composite_score']

        return best_opp, best_score

    def rank_and_explain(
        self,
        opportunities_df: pd.DataFrame,
        top_n: int = 5
    ) -> None:
        """
        Rank opportunities and print explanation.

        Args:
            opportunities_df: DataFrame with opportunity features
            top_n: Number of top opportunities to display
        """
        scored_df = self.score_opportunities(opportunities_df, return_components=True)

        print("\n" + "=" * 80)
        print(f"OPPORTUNITY RANKING (Top {top_n})")
        print("=" * 80)

        for idx, row in scored_df.head(top_n).iterrows():
            print(f"\n#{row['rank']} - Score: {row['composite_score']:.1f}/100")
            print(f"   Symbol: {row.get('symbol', 'N/A')}")
            print(f"   Strategy: {row.get('strategy', 'N/A')}")
            print(f"\n   Predictions:")
            print(f"     Profit: {row['predicted_profit']:.2f}%")
            print(f"     Success Prob: {row['success_probability']:.1%}")
            print(f"     Hold Duration: {row['predicted_duration']:.1f} hours")
            print(f"\n   Score Breakdown:")
            print(f"     Profit Score:        {row['profit_score']:.1f} (weight: {self.weight_profit:.0%})")
            print(f"     Success Score:       {row['success_score']:.1f} (weight: {self.weight_success:.0%})")
            print(f"     Risk-Adjusted Score: {row['risk_adjusted_score']:.1f} (weight: {self.weight_risk_adjusted:.0%})")
            print(f"     Duration Score:      {row['duration_score']:.1f} (weight: {self.weight_duration:.0%})")
            print("-" * 80)

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    @staticmethod
    def _normalize_inverse(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range (inverse: lower value = higher score)."""
        if max_val == min_val:
            return 0.5

        normalized = 1 - (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)


class ScoringReport:
    """
    Generate scoring reports and visualizations.
    """

    @staticmethod
    def generate_report(
        scored_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> None:
        """
        Generate a detailed scoring report.

        Args:
            scored_df: DataFrame with scores
            output_path: Path to save report (optional)
        """
        report = []

        report.append("=" * 80)
        report.append("OPPORTUNITY SCORING REPORT")
        report.append("=" * 80)

        report.append(f"\nTotal Opportunities: {len(scored_df)}")
        report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n--- SCORE DISTRIBUTION ---")
        report.append(f"Mean Score: {scored_df['composite_score'].mean():.1f}")
        report.append(f"Median Score: {scored_df['composite_score'].median():.1f}")
        report.append(f"Std Dev: {scored_df['composite_score'].std():.1f}")
        report.append(f"Min Score: {scored_df['composite_score'].min():.1f}")
        report.append(f"Max Score: {scored_df['composite_score'].max():.1f}")

        report.append("\n--- TOP 10 OPPORTUNITIES ---")
        for idx, row in scored_df.head(10).iterrows():
            report.append(f"\n#{row['rank']}: {row.get('symbol', 'N/A')} - Score: {row['composite_score']:.1f}")
            report.append(f"   Profit: {row['predicted_profit']:.2f}% | Success: {row['success_probability']:.1%} | Duration: {row['predicted_duration']:.1f}h")

        report_text = "\n".join(report)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(report_text)

            print(f"\n✅ Report saved to {output_path}")
        else:
            print(report_text)
