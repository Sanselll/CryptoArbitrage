namespace CryptoArbitrage.API.Models.Suggestions;

/// <summary>
/// AI-driven suggestion for trading an arbitrage opportunity
/// </summary>
public class OpportunitySuggestion
{
    /// <summary>
    /// Overall confidence score (0-100)
    /// </summary>
    public decimal ConfidenceScore { get; set; }

    /// <summary>
    /// Recommended strategy type for this opportunity
    /// </summary>
    public RecommendedStrategyType RecommendedStrategy { get; set; }

    /// <summary>
    /// Entry recommendation level
    /// </summary>
    public EntryRecommendation EntryRecommendation { get; set; }

    /// <summary>
    /// Suggested position size in USD
    /// </summary>
    public decimal SuggestedPositionSizeUsd { get; set; }

    /// <summary>
    /// Suggested leverage (1x-20x)
    /// </summary>
    public decimal SuggestedLeverage { get; set; }

    /// <summary>
    /// Suggested holding period in hours
    /// </summary>
    public decimal SuggestedHoldingPeriodHours { get; set; }

    /// <summary>
    /// Profit target as percentage (e.g., 0.4 = 0.4% profit target)
    /// </summary>
    public decimal ProfitTargetPercent { get; set; }

    /// <summary>
    /// Maximum holding time in hours before forced exit
    /// </summary>
    public decimal MaxHoldingHours { get; set; }

    /// <summary>
    /// Detailed score breakdown
    /// </summary>
    public ScoreBreakdown ScoreBreakdown { get; set; } = new();

    /// <summary>
    /// Human-readable reasoning for the suggestion
    /// </summary>
    public string Reasoning { get; set; } = string.Empty;

    /// <summary>
    /// Risk warnings or concerns
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Timestamp when suggestion was generated
    /// </summary>
    public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Recommended strategy type
/// </summary>
public enum RecommendedStrategyType
{
    /// <summary>
    /// Focus on funding rate arbitrage only
    /// </summary>
    FundingOnly,

    /// <summary>
    /// Focus on price spread arbitrage only
    /// </summary>
    SpreadOnly,

    /// <summary>
    /// Combine both funding and spread strategies
    /// </summary>
    Hybrid
}

/// <summary>
/// Entry recommendation level based on confidence
/// </summary>
public enum EntryRecommendation
{
    /// <summary>
    /// Skip this opportunity (confidence &lt; 40)
    /// </summary>
    Skip,

    /// <summary>
    /// Hold and monitor (confidence 40-59)
    /// </summary>
    Hold,

    /// <summary>
    /// Consider entry with caution (confidence 60-79)
    /// </summary>
    Buy,

    /// <summary>
    /// Strong recommendation to enter (confidence ≥ 80)
    /// </summary>
    StrongBuy
}
