namespace CryptoArbitrage.API.Models.Suggestions;

/// <summary>
/// Signal indicating exit condition has been triggered for a position
/// </summary>
public class ExitSignal
{
    /// <summary>
    /// Type of exit condition that triggered
    /// </summary>
    public ExitConditionType ConditionType { get; set; }

    /// <summary>
    /// Whether this condition has been triggered
    /// </summary>
    public bool IsTriggered { get; set; }

    /// <summary>
    /// Confidence level that exit should occur (0-100)
    /// </summary>
    public decimal Confidence { get; set; }

    /// <summary>
    /// Urgency level of the exit signal
    /// </summary>
    public ExitUrgency Urgency { get; set; }

    /// <summary>
    /// Recommended action to take
    /// </summary>
    public string RecommendedAction { get; set; } = string.Empty;

    /// <summary>
    /// Detailed message explaining why exit is recommended
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Current value that triggered the signal (e.g., current funding rate)
    /// </summary>
    public decimal? CurrentValue { get; set; }

    /// <summary>
    /// Entry/threshold value for comparison
    /// </summary>
    public decimal? ThresholdValue { get; set; }

    /// <summary>
    /// Timestamp when signal was generated
    /// </summary>
    public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Type of exit condition
/// </summary>
public enum ExitConditionType
{
    /// <summary>
    /// Profit target has been reached
    /// </summary>
    ProfitTarget,

    /// <summary>
    /// Funding rate has reversed or deteriorated significantly
    /// </summary>
    FundingReversal,

    /// <summary>
    /// Maximum holding time has been exceeded
    /// </summary>
    TimeLimit,

    /// <summary>
    /// Market conditions have degraded (liquidity, volume, spread)
    /// </summary>
    MarketDegradation
}

/// <summary>
/// Urgency level for exit recommendation
/// </summary>
public enum ExitUrgency
{
    /// <summary>
    /// Low urgency - informational
    /// </summary>
    Low,

    /// <summary>
    /// Medium urgency - should consider exiting soon
    /// </summary>
    Medium,

    /// <summary>
    /// High urgency - exit recommended immediately
    /// </summary>
    High,

    /// <summary>
    /// Critical - exit as soon as possible to prevent losses
    /// </summary>
    Critical
}
