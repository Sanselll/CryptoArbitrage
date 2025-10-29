using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

public class Position
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    // Link to Execution (nullable for historical positions)
    public int? ExecutionId { get; set; }
    public Execution? Execution { get; set; }

    // Position identification
    public string Symbol { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;
    public PositionType Type { get; set; }  // Perpetual or Spot
    public PositionSide Side { get; set; }  // Long or Short
    public PositionStatus Status { get; set; } = PositionStatus.Open;

    // Position details
    public decimal EntryPrice { get; set; }
    public decimal? ExitPrice { get; set; }
    public decimal Quantity { get; set; }
    public decimal Leverage { get; set; } = 1m;  // Spot positions default to 1x
    public decimal InitialMargin { get; set; }

    // P&L tracking
    public decimal RealizedPnL { get; set; }
    public decimal UnrealizedPnL { get; set; }

    // Fee tracking: Calculated on-the-fly from PositionTransaction table (no storage needed)

    // Order/Position references
    public string? OrderId { get; set; }  // Opening order ID (Spot buy or Perpetual short/long)
    public string? CloseOrderId { get; set; }  // Closing order ID
    public string? ExchangePositionId { get; set; }  // Exchange's position ID for perpetual positions (used to close positions)

    // Transaction reconciliation
    public ReconciliationStatus ReconciliationStatus { get; set; } = ReconciliationStatus.Preliminary;
    public DateTime? ReconciliationCompletedAt { get; set; }

    // Timestamps
    public DateTime OpenedAt { get; set; } = DateTime.UtcNow;
    public DateTime? ClosedAt { get; set; }

    // Optional notes
    public string? Notes { get; set; }

    // Navigation properties
    public ICollection<PositionTransaction> Transactions { get; set; } = new List<PositionTransaction>();
}
