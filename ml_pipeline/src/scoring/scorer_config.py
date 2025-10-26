"""
Scorer Configuration Utilities

Helper functions for managing scoring configuration.
"""

from pathlib import Path
from typing import Dict
import yaml


def load_scoring_config(config_path: Path) -> Dict:
    """
    Load scoring configuration from YAML file.

    Args:
        config_path: Path to scoring config YAML

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate weights sum to 1.0
    weights = config.get('weights', {})
    total_weight = sum(weights.values())

    if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
        print(f"⚠️  Warning: Weights sum to {total_weight}, not 1.0. Normalizing...")

        # Normalize weights
        for key in weights:
            weights[key] = weights[key] / total_weight

        config['weights'] = weights

    return config


def save_scoring_config(config: Dict, output_path: Path) -> None:
    """
    Save scoring configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Scoring config saved to {output_path}")


def create_default_config() -> Dict:
    """
    Create default scoring configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'weights': {
            'predicted_profit': 0.40,
            'success_probability': 0.30,
            'risk_adjusted_return': 0.20,
            'hold_duration': 0.10
        },
        'normalization': {
            'min_score': 0,
            'max_score': 100,
            'profit_range': {
                'min': -5.0,
                'max': 20.0
            },
            'success_probability_range': {
                'min': 0.0,
                'max': 1.0
            },
            'duration_range': {
                'min': 1.0,
                'max': 168.0
            }
        },
        'risk_adjustment': {
            'method': 'sharpe',
            'volatility_penalty_factor': 1.0,
            'min_spread_volatility_cv': 0.0,
            'max_spread_volatility_cv': 1.0
        },
        'filtering': {
            'min_predicted_profit': 0.1,
            'min_success_probability': 0.5,
            'max_hold_duration': 168,
            'min_volume_24h': 100000,
            'required_liquidity_status': ['Good', 'Medium']
        },
        'duration_preference': {
            'optimal_duration_hours': 24,
            'duration_penalty_factor': 0.1
        }
    }
