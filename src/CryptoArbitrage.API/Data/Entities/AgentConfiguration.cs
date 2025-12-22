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
    public decimal MaxLeverage { get; set; } = 2.0m;  // 1-5x (trained with 2.0x)
    public decimal TargetUtilization { get; set; } = 0.8m;  // 50-100% (0.5-1.0) (trained with 0.8)
    public int MaxPositions { get; set; } = 1;  // V9: single position only

    // Prediction settings
    public int PredictionIntervalSeconds { get; set; } = 300;  // 5 minutes to match training environment (300 seconds)

    // Timestamps
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    // Optional notes
    public string? Notes { get; set; }
}
