namespace CryptoArbitrage.API.Data.Entities;

public class Position
{
    public int Id { get; set; }

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

    // Funding fee tracking
    public decimal TotalFundingFeePaid { get; set; }
    public decimal TotalFundingFeeReceived { get; set; }
    public decimal NetFundingFee => TotalFundingFeeReceived - TotalFundingFeePaid;

    // Order/Position references
    public string? OrderId { get; set; }  // Spot order ID or Perpetual position ID

    // Timestamps
    public DateTime OpenedAt { get; set; } = DateTime.UtcNow;
    public DateTime? ClosedAt { get; set; }

    // Optional notes
    public string? Notes { get; set; }
}
