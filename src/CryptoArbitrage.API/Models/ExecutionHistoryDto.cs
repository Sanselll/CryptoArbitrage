using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

/// <summary>
/// Position details within an execution history entry
/// </summary>
public class ExecutionHistoryPositionDto
{
    public int Id { get; set; }
    public string Exchange { get; set; } = string.Empty;
    public PositionType Type { get; set; }
    public PositionSide Side { get; set; }
    public decimal EntryPrice { get; set; }
    public decimal ExitPrice { get; set; }
    public decimal Quantity { get; set; }
    public decimal Leverage { get; set; }
    public decimal PricePnL { get; set; }
    public decimal FundingEarned { get; set; }
    public decimal TradingFees { get; set; }
    public decimal RealizedPnL { get; set; }
}

/// <summary>
/// Execution history entry with position details
/// </summary>
public class ExecutionHistoryDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;
    public string? LongExchange { get; set; }
    public string? ShortExchange { get; set; }
    public StrategySubType Strategy { get; set; }
    public decimal PositionSizeUsd { get; set; }

    // Aggregated totals for the execution
    public decimal TotalPricePnL { get; set; }
    public decimal TotalFundingEarned { get; set; }
    public decimal TotalTradingFees { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal TotalPnLPct { get; set; }

    public DateTime StartedAt { get; set; }
    public DateTime ClosedAt { get; set; }
    public double DurationSeconds { get; set; }
    public ExecutionState State { get; set; }

    // Individual positions for this execution
    public List<ExecutionHistoryPositionDto> Positions { get; set; } = new();
}
