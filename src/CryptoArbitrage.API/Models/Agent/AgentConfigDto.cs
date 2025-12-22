using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Models.Agent;

/// <summary>
/// Agent configuration data transfer object
/// </summary>
public class AgentConfigDto
{
    /// <summary>Maximum leverage per position (1-5x)</summary>
    [Range(1.0, 5.0)]
    public decimal MaxLeverage { get; set; } = 1.0m;

    /// <summary>Target capital utilization (0.5-1.0 = 50%-100%)</summary>
    [Range(0.5, 1.0)]
    public decimal TargetUtilization { get; set; } = 0.9m;

    /// <summary>Maximum concurrent executions (V9: single position only). Each execution opens 2 positions (long + short hedge).</summary>
    [Range(1, 1)]
    public int MaxPositions { get; set; } = 1;

    public int PredictionIntervalSeconds { get; set; } = 60;
}

/// <summary>
/// Agent status response DTO
/// </summary>
public class AgentStatusDto
{
    public string Status { get; set; } = "stopped";
    public DateTime? StartedAt { get; set; }
    public DateTime? PausedAt { get; set; }
    public int? DurationSeconds { get; set; }
    public string? ErrorMessage { get; set; }
    public int TotalPredictions { get; set; }
    public AgentConfigDto? Config { get; set; }
    public AgentStatsDto? Stats { get; set; }
}

/// <summary>
/// Agent statistics DTO
/// </summary>
public class AgentStatsDto
{
    public int TotalDecisions { get; set; }
    public int HoldDecisions { get; set; }
    public int EnterDecisions { get; set; }
    public int ExitDecisions { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal TotalPnLUsd { get; set; }
    public decimal TotalPnLPct { get; set; }
    public decimal TodayPnLUsd { get; set; }
    public decimal TodayPnLPct { get; set; }
    public int ActivePositions { get; set; }
}

/// <summary>
/// Agent decision log item DTO
/// </summary>
public class AgentDecisionDto
{
    public DateTime Timestamp { get; set; }
    public string Action { get; set; } = "";
    public string? OpportunitySymbol { get; set; }
    public string? Confidence { get; set; }
    public double? EnterProbability { get; set; }
    public double? ExitProbability { get; set; }
    public double? StateValue { get; set; }
    public int NumOpportunities { get; set; }
    public int NumPositions { get; set; }
    public string? Reasoning { get; set; }
}
