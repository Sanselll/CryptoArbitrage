namespace CryptoArbitrage.API.Data.Entities;

public class Execution
{
    public int Id { get; set; }
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
