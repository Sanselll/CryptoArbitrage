using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

/// <summary>
/// Agent session tracking.
/// Records when the agent was started, stopped, and its status.
/// </summary>
public class AgentSession
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    // Link to configuration
    public int AgentConfigurationId { get; set; }
    public AgentConfiguration AgentConfiguration { get; set; } = null!;

    // Session tracking
    public AgentStatus Status { get; set; } = AgentStatus.Stopped;
    public DateTime? StartedAt { get; set; }
    public DateTime? PausedAt { get; set; }
    public DateTime? StoppedAt { get; set; }

    // Error tracking
    public string? ErrorMessage { get; set; }

    // Statistics (snapshot at session end)
    public int TotalPredictions { get; set; }
    public int TotalTrades { get; set; }
    public decimal? FinalPnLUsd { get; set; }
    public decimal? FinalPnLPct { get; set; }

    // Timestamps
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Agent status enum
/// </summary>
public enum AgentStatus
{
    Stopped = 0,
    Running = 1,
    Paused = 2,
    Error = 3
}
