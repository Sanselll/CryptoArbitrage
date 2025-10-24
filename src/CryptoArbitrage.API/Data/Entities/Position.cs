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

    // Funding fee tracking
    public decimal TotalFundingFeePaid { get; set; }
    public decimal TotalFundingFeeReceived { get; set; }
    public decimal NetFundingFee => TotalFundingFeeReceived - TotalFundingFeePaid;

    // Order/Position references
    public string? OrderId { get; set; }  // Spot order ID or Perpetual order ID
    public string? ExchangePositionId { get; set; }  // Exchange's position ID for perpetual positions (used to close positions)

    // Timestamps
    public DateTime OpenedAt { get; set; } = DateTime.UtcNow;
    public DateTime? ClosedAt { get; set; }

    // Optional notes
    public string? Notes { get; set; }

    // Entry snapshot for algorithmic suggester (exit strategy monitoring)
    /// <summary>
    /// Funding rate at position entry (for monitoring reversals)
    /// </summary>
    public decimal? EntryFundingRate { get; set; }

    /// <summary>
    /// Price spread at entry (for cross-exchange strategies)
    /// </summary>
    public decimal? EntrySpread { get; set; }

    /// <summary>
    /// Spot price at entry (for spot-perpetual strategies)
    /// </summary>
    public decimal? EntrySpotPrice { get; set; }

    /// <summary>
    /// Perpetual price at entry
    /// </summary>
    public decimal? EntryPerpPrice { get; set; }

    /// <summary>
    /// Profit target percentage for exit signal (e.g., 0.4 = 0.4%)
    /// </summary>
    public decimal? ProfitTargetPercent { get; set; }

    /// <summary>
    /// Maximum holding hours before suggesting exit
    /// </summary>
    public decimal? MaxHoldingHours { get; set; }

    /// <summary>
    /// Confidence score from suggestion engine (0-100)
    /// </summary>
    public decimal? EntryConfidenceScore { get; set; }

    /// <summary>
    /// Recommended strategy at entry (FundingOnly, SpreadOnly, Hybrid)
    /// </summary>
    public string? RecommendedStrategy { get; set; }
}
