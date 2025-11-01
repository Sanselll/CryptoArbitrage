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
/// Portfolio state for RL model evaluation
/// Provides context about current trading status
/// </summary>
public class RLPortfolioState
{
    public decimal Capital { get; set; } = 10000m;
    public decimal InitialCapital { get; set; } = 10000m;
    public int NumPositions { get; set; } = 0;
    public float Utilization { get; set; } = 0.0f;
    public float TotalPnlPct { get; set; } = 0.0f;
    public float Drawdown { get; set; } = 0.0f;
    public List<RLPositionState> Positions { get; set; } = new();
}

/// <summary>
/// Position state for RL model evaluation
/// </summary>
public class RLPositionState
{
    public float PnlPct { get; set; }
    public float HoursHeld { get; set; }
    public float FundingRate { get; set; }
}
