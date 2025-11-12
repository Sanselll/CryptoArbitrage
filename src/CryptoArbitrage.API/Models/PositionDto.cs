using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public class PositionDto
{
    public int Id { get; set; }
    public int ExecutionId { get; set; }  // Required - every position belongs to an execution
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

    // P&L Breakdown
    public decimal FundingEarnedUsd { get; set; }
    public decimal TradingFeesUsd { get; set; }
    public decimal PricePnLUsd { get; set; }
    public decimal RealizedPnLUsd { get; set; }
    public decimal RealizedPnLPct { get; set; }

    public decimal UnrealizedPnL { get; set; }
    public decimal TotalFundingFeePaid { get; set; }
    public decimal TotalFundingFeeReceived { get; set; }
    public decimal NetFundingFee => TotalFundingFeeReceived - TotalFundingFeePaid;
    public decimal TradingFeePaid { get; set; }
    public ReconciliationStatus ReconciliationStatus { get; set; }
    public DateTime? ReconciliationCompletedAt { get; set; }
    public DateTime OpenedAt { get; set; }
    public DateTime? ClosedAt { get; set; }
    public int? ActiveOpportunityId { get; set; }

    // ML features (Phase 1 exit timing)
    public decimal EntryApr { get; set; }  // APR at entry time (for apr_ratio feature)
    public decimal PeakPnlPct { get; set; }  // Peak P&L percentage (for drawdown feature)
    public string? PnlHistoryJson { get; set; }  // Hourly P&L snapshots (for velocity feature)
}
