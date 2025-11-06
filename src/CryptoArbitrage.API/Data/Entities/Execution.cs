using System.ComponentModel.DataAnnotations;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Data.Entities;

public class Execution
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    public string Symbol { get; set; } = string.Empty;

    // NOTE: Exchange field is kept for backward compatibility
    // For SpotPerpetual: Exchange is the single exchange name
    // For CrossExchange: Exchange is "{LongExchange}/{ShortExchange}" format (legacy)
    public string Exchange { get; set; } = string.Empty;

    // Strategy metadata (added for ML context)
    public ArbitrageStrategy Strategy { get; set; } = ArbitrageStrategy.CrossExchange;
    public StrategySubType SubType { get; set; } = StrategySubType.CrossExchangeFuturesFutures;

    // Exchange details for cross-exchange strategies (provides clarity)
    public string? LongExchange { get; set; }  // Exchange where long position is opened
    public string? ShortExchange { get; set; } // Exchange where short position is opened

    public DateTime StartedAt { get; set; } = DateTime.UtcNow;
    public DateTime? StoppedAt { get; set; }
    public ExecutionState State { get; set; } = ExecutionState.Running;
    public decimal FundingEarned { get; set; } = 0;
    public decimal PositionSizeUsd { get; set; }

    // Order/position IDs for reference (SpotOrderId is nullable for futures-only cross-exchange)
    public string? SpotOrderId { get; set; }
    public string? PerpOrderId { get; set; }
}
