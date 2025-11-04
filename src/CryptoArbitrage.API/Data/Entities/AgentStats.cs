using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

/// <summary>
/// Agent performance statistics.
/// Stores cumulative statistics for the current or most recent agent session.
/// </summary>
public class AgentStats
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    // Link to session (nullable for current/live stats)
    public int? AgentSessionId { get; set; }
    public AgentSession? AgentSession { get; set; }

    // Decision statistics
    public int TotalDecisions { get; set; }
    public int HoldDecisions { get; set; }
    public int EnterDecisions { get; set; }
    public int ExitDecisions { get; set; }

    // Trading statistics
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }  // Percentage (0-100)

    // P&L statistics
    public decimal TotalPnLUsd { get; set; }
    public decimal TotalPnLPct { get; set; }
    public decimal TodayPnLUsd { get; set; }
    public decimal TodayPnLPct { get; set; }
    public decimal MaxDrawdownPct { get; set; }

    // Position statistics
    public int ActivePositions { get; set; }
    public int MaxActivePositions { get; set; }  // Peak concurrent positions
    public decimal AveragePositionDurationHours { get; set; }

    // Timestamps
    public DateTime StatsPeriodStart { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
