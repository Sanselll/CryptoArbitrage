using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Models;

public class PositionDto
{
    public int Id { get; set; }
    public int? ExecutionId { get; set; }
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public PositionType Type { get; set; }
    public PositionSide Side { get; set; }
    public PositionStatus Status { get; set; }
    public decimal EntryPrice { get; set; }
    public decimal? ExitPrice { get; set; }
    public decimal Quantity { get; set; }
    public decimal Leverage { get; set; }
    public decimal InitialMargin { get; set; }
    public decimal RealizedPnL { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal TotalFundingFeePaid { get; set; }
    public decimal TotalFundingFeeReceived { get; set; }
    public decimal NetFundingFee => TotalFundingFeeReceived - TotalFundingFeePaid;
    public DateTime OpenedAt { get; set; }
    public DateTime? ClosedAt { get; set; }
    public int? ActiveOpportunityId { get; set; }

    /// <summary>
    /// Exit signals for this position (only populated for open positions)
    /// </summary>
    public List<ExitSignal>? ExitSignals { get; set; }
}
