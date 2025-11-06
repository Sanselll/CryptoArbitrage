using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

/// <summary>
/// Agent session tracking.
/// Records when the agent was started, stopped, and its status.
/// </summary>
public class AgentSession
{
    public Guid Id { get; set; }

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

    // Session-only statistics (live-updated during session)
    // Decision counts
    public int HoldDecisions { get; set; }
    public int EnterDecisions { get; set; }
    public int ExitDecisions { get; set; }

    // Trade outcomes
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }

    // P&L tracking
    public decimal SessionPnLUsd { get; set; }
    public decimal SessionPnLPct { get; set; }

    // Position tracking
    public int ActivePositions { get; set; }
    public int MaxActivePositions { get; set; }

    // Navigation properties
    public ICollection<Position> Positions { get; set; } = new List<Position>();

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
