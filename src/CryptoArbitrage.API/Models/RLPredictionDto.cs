using System.Text.Json.Serialization;

namespace CryptoArbitrage.API.Models;

/// <summary>
/// RL (Reinforcement Learning) prediction for an opportunity or position
/// Provides action probabilities (ENTER/EXIT) and confidence levels
/// </summary>
public class RLPredictionDto
{
    /// <summary>
    /// Probability of taking the ENTER action (for opportunities) or EXIT action (for positions)
    /// Range: 0.0 to 1.0
    /// </summary>
    public float ActionProbability { get; set; }

    /// <summary>
    /// Probability of taking the HOLD action (doing nothing)
    /// Range: 0.0 to 1.0
    /// </summary>
    public float HoldProbability { get; set; }

    /// <summary>
    /// Confidence level of the prediction
    /// Values: "HIGH", "MEDIUM", "LOW"
    /// Based on probability level and action distribution entropy
    /// </summary>
    public string Confidence { get; set; } = "LOW";

    /// <summary>
    /// State value estimate from the RL model
    /// Higher values indicate better expected future rewards
    /// </summary>
    public float StateValue { get; set; }

    /// <summary>
    /// Model version used for prediction (e.g., "pbt_20251101_083701")
    /// </summary>
    public string ModelVersion { get; set; } = string.Empty;
}

/// <summary>
/// Trading configuration for RL model (5 features matching training environment)
/// Maps to config.py:TradingConfig.to_array()
/// </summary>
public class RLTradingConfig
{
    public decimal MaxLeverage { get; set; } = 2.0m;  // Feature 1: 1-5x leverage
    public decimal TargetUtilization { get; set; } = 0.8m;  // Feature 2: 50-100% (0.5-1.0)
    public int MaxPositions { get; set; } = 2;  // Feature 3: 1-3 concurrent executions
    public decimal StopLossThreshold { get; set; } = -0.02m;  // Feature 4: Stop loss % (e.g., -2%)
    public decimal LiquidationBuffer { get; set; } = 0.15m;  // Feature 5: Safety buffer from liquidation (15%)
}

/// <summary>
/// Portfolio state for RL model evaluation (10 features matching training environment)
/// Maps to environment.py:_get_observation() portfolio section
/// Provides context about current trading status
/// </summary>
public class RLPortfolioState
{
    // Core capital metrics (features 1-2)
    public decimal Capital { get; set; } = 10000m;
    public decimal InitialCapital { get; set; } = 10000m;

    // Position and utilization metrics (features 3-4)
    public int NumPositions { get; set; } = 0;
    public float MarginUtilization { get; set; } = 0.0f;  // Percentage of capital locked as margin (Feature #3)

    // P&L and drawdown metrics (features 5-7)
    public float AvgPositionPnlPct { get; set; } = 0.0f;  // Average P&L across executions
    public float TotalPnlPct { get; set; } = 0.0f;  // Total portfolio P&L %
    public float Drawdown { get; set; } = 0.0f;  // Max drawdown %

    // Progress and risk metrics (features 8-10)
    public float EpisodeProgress { get; set; } = 0.5f;  // Progress through trading session (0-1)
    public float MinLiquidationDistance { get; set; } = 1.0f;  // Closest position to liquidation (0-1)
    public float CapitalUtilization { get; set; } = 0.0f;  // Total notional value of positions / capital % (Feature #10)

    // Position details (up to 5 slots × 12 features each = 60 features)
    public List<RLPositionState> Positions { get; set; } = new();
}

/// <summary>
/// Position state for RL model evaluation (12 features per execution/position)
/// Maps to rl_predictor.py:_build_execution_features()
/// NOTE: Each "position" represents one EXECUTION (long + short pair), not individual legs
/// ML predictor calculates most features from raw data fields below
/// </summary>
public class RLPositionState
{
    // ===== FIELDS USED DIRECTLY BY ML PREDICTOR =====
    // These are read directly from the position dict by rl_predictor.py

    // P&L metrics (used directly by ML)
    public float UnrealizedPnlPct { get; set; }  // Net P&L % - read directly at line 219
    public float LongPnlPct { get; set; }  // Long side P&L % - read directly at line 261
    public float ShortPnlPct { get; set; }  // Short side P&L % - read directly at line 264

    // Risk metric (used directly by ML)
    public float LiquidationDistance { get; set; }  // Distance to liquidation - read directly at line 267

    // ===== RAW DATA FIELDS (ML Calculates Features From These) =====
    // rl_predictor.py uses these to calculate: hours_held, net_funding_ratio, net_funding_rate,
    // current_spread_pct, entry_spread_pct, value_ratio, funding_efficiency

    // Time metrics
    public float PositionAgeHours { get; set; }  // Raw hours held (ML normalizes by 72h at line 222)

    // Funding amounts (raw USD values)
    public float LongNetFundingUsd { get; set; }  // Long side funding USD (line 226)
    public float ShortNetFundingUsd { get; set; }  // Short side funding USD (line 227)

    // Funding rates (individual exchange rates)
    public float ShortFundingRate { get; set; }  // Short exchange rate (line 232)
    public float LongFundingRate { get; set; }  // Long exchange rate (line 233)

    // Price data (raw prices for both legs)
    public float CurrentLongPrice { get; set; }  // Current long price (line 236)
    public float CurrentShortPrice { get; set; }  // Current short price (line 237)
    public float EntryLongPrice { get; set; }  // Entry long price (line 245)
    public float EntryShortPrice { get; set; }  // Entry short price (line 246)

    // Position sizing
    public float PositionSizeUsd { get; set; }  // Position size per side in USD (line 225)
    public float EntryFeesPaidUsd { get; set; }  // Entry fees paid (line 257)

    // ===== DEPRECATED FIELDS (kept for backward compatibility) =====
    // Backend still populates these but ML predictor ignores them
    public bool IsActive { get; set; } = true;
    public float HoursHeld { get; set; }
    public float NetFundingRatio { get; set; }
    public float FundingRate { get; set; }
    public float CurrentSpreadPct { get; set; }
    public float EntrySpreadPct { get; set; }
    public float ValueToCapitalRatio { get; set; }
    public float FundingEfficiency { get; set; }
    public float LiquidationDistancePct { get; set; }
    public float PnlPct { get; set; }
    public float TotalNetFundingUsd { get; set; }
    public float PositionIsActive { get; set; }
}
