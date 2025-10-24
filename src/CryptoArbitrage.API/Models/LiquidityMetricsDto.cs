namespace CryptoArbitrage.API.Models;

public enum LiquidityStatus
{
    Good = 0,
    Medium = 1,
    Low = 2
}

public class LiquidityMetricsDto
{
    public decimal BidAskSpreadPercent { get; set; }
    public decimal OrderbookDepthUsd { get; set; }
    public LiquidityStatus Status { get; set; }
    public string? WarningMessage { get; set; }
}
