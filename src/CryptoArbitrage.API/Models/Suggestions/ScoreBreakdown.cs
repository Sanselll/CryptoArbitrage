using System.Text.Json.Serialization;

namespace CryptoArbitrage.API.Models.Suggestions;

/// <summary>
/// Detailed breakdown of suggestion confidence scores
/// </summary>
public class ScoreBreakdown
{
    /// <summary>
    /// Funding rate quality score (0-100)
    /// Evaluates funding rate consistency, historical alignment, and reversal risk
    /// </summary>
    [JsonPropertyName("fundingQuality")]
    public decimal FundingQualityScore { get; set; }

    /// <summary>
    /// Profit potential score (0-100)
    /// Evaluates APR projections, break-even time, and estimated returns
    /// </summary>
    [JsonPropertyName("profitPotential")]
    public decimal ProfitPotentialScore { get; set; }

    /// <summary>
    /// Spread efficiency score (0-100)
    /// Evaluates current vs historical spread and mean reversion potential
    /// </summary>
    [JsonPropertyName("spreadEfficiency")]
    public decimal SpreadEfficiencyScore { get; set; }

    /// <summary>
    /// Market quality score (0-100)
    /// Evaluates liquidity, volume, bid-ask spread, and execution risk
    /// </summary>
    [JsonPropertyName("marketQuality")]
    public decimal MarketQualityScore { get; set; }

    /// <summary>
    /// Overall confidence score (weighted average of all scores)
    /// </summary>
    [JsonPropertyName("overallConfidence")]
    public decimal OverallConfidence { get; set; }

    /// <summary>
    /// Time efficiency score (0-100)
    /// Evaluates profit rate relative to holding period (faster = better)
    /// </summary>
    [JsonPropertyName("timeEfficiency")]
    public decimal TimeEfficiencyScore { get; set; }

    /// <summary>
    /// Overall risk score (0-100, lower = more risky)
    /// Combines spread volatility, funding reversal, and liquidation risks
    /// </summary>
    [JsonPropertyName("riskScore")]
    public decimal RiskScore { get; set; }

    /// <summary>
    /// Execution safety score (0-100)
    /// Evaluates if profit target adequately covers execution costs
    /// </summary>
    [JsonPropertyName("executionSafety")]
    public decimal ExecutionSafetyScore { get; set; }

    /// <summary>
    /// Estimated total execution cost as percentage
    /// Includes bid-ask spreads, slippage, and position costs
    /// </summary>
    [JsonPropertyName("executionCostPercent")]
    public decimal ExecutionCostPercent { get; set; }

    /// <summary>
    /// Expected profit after all execution costs
    /// </summary>
    [JsonPropertyName("profitAfterCosts")]
    public decimal ProfitAfterCosts { get; set; }

    /// <summary>
    /// Detailed explanation of scoring factors
    /// </summary>
    [JsonPropertyName("scoringFactors")]
    public List<string> ScoringFactors { get; set; } = new();
}
