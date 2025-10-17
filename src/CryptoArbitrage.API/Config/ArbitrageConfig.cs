namespace CryptoArbitrage.API.Config;

public class ArbitrageConfig
{
    public decimal MinSpreadPercentage { get; set; } = 0.1m; // 0.1% minimum spread
    public decimal MaxPositionSizeUsd { get; set; } = 10000m;
    public decimal MinPositionSizeUsd { get; set; } = 100m;
    public decimal MaxLeverage { get; set; } = 5m;
    public decimal MaxTotalExposure { get; set; } = 50000m;
    public List<string> WatchedSymbols { get; set; } = new();
    public List<string> EnabledExchanges { get; set; } = new();
    public bool AutoExecute { get; set; } = false;
    public int DataRefreshIntervalSeconds { get; set; } = 5;

    // Dynamic symbol discovery settings
    public bool AutoDiscoverSymbols { get; set; } = true;
    public decimal MinDailyVolumeUsd { get; set; } = 10_000_000m; // $10M minimum daily volume
    public int MaxSymbolCount { get; set; } = 50; // Track top 50 symbols by volume
    public decimal MinAbsFundingRate { get; set; } = 0.0001m; // 0.01% = 3.65% APR minimum
    public int SymbolRefreshIntervalHours { get; set; } = 24; // Refresh symbol list daily
}
