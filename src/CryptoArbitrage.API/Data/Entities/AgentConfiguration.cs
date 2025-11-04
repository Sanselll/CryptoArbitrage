using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

/// <summary>
/// Agent trading configuration for a user.
/// Stores the parameters that control the autonomous trading agent.
/// </summary>
public class AgentConfiguration
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    // Trading configuration parameters
    public decimal MaxLeverage { get; set; } = 1.0m;  // 1-5x
    public decimal TargetUtilization { get; set; } = 0.9m;  // 50-100% (0.5-1.0)
    public int MaxPositions { get; set; } = 3;  // 1-3 concurrent positions

    // Prediction settings
    public int PredictionIntervalSeconds { get; set; } = 60;  // Seconds between predictions

    // Timestamps
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    // Optional notes
    public string? Notes { get; set; }
}
