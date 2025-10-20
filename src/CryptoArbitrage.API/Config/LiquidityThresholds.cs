namespace CryptoArbitrage.API.Config;

public class LiquidityThresholds
{
    public decimal MinVolume24hUsd { get; set; } = 500_000m;
    public decimal MaxBidAskSpreadPercent { get; set; } = 0.5m;
    public decimal MinOrderbookDepthUsd { get; set; } = 25_000m;
}
