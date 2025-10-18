using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

public class Execution
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    public string Symbol { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;
    public DateTime? StoppedAt { get; set; }
    public ExecutionState State { get; set; } = ExecutionState.Running;
    public decimal FundingEarned { get; set; } = 0;
    public decimal PositionSizeUsd { get; set; }

    // Order/position IDs for reference (SpotOrderId is nullable for futures-only cross-exchange)
    public string? SpotOrderId { get; set; }
    public string? PerpOrderId { get; set; }
}
