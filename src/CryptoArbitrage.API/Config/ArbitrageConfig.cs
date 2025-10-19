using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Config;

public class ExchangeConfig
{
    public string Name { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public string ApiSecret { get; set; } = string.Empty;
    public bool IsEnabled { get; set; } = true;
    public bool UseDemoTrading { get; set; } = false;
}

public class ArbitrageConfig
{
    public decimal MinSpreadPercentage { get; set; } = 0.1m; // 0.1% minimum spread
    public decimal MaxPositionSizeUsd { get; set; } = 10000m;
    public decimal MinPositionSizeUsd { get; set; } = 100m;
    public decimal MaxLeverage { get; set; } = 5m;
    public decimal MaxTotalExposure { get; set; } = 50000m;
    public List<string> WatchedSymbols { get; set; } = new();
    public bool AutoExecute { get; set; } = false;
    public int DataRefreshIntervalSeconds { get; set; } = 5;

    // Dynamic symbol discovery settings
    public bool AutoDiscoverSymbols { get; set; } = true;
    public decimal MinDailyVolumeUsd { get; set; } = 10_000_000m; // $10M minimum daily volume
    public int MaxSymbolCount { get; set; } = 50; // Track top 50 symbols by volume
    public decimal MinAbsFundingRate { get; set; } = 0.0001m; // 0.01% = 3.65% APR minimum
    public decimal MinHighPriorityFundingRate { get; set; } = 0.01m; // 1% = 365% APR - always include these
    public double SymbolRefreshIntervalHours { get; set; } = 24; // Refresh symbol list daily

    // Strategy configuration (enable/disable each strategy type)
    public bool EnableSpotPerpetualSameExchange { get; set; } = true;       // Buy spot + short perp on same exchange
    public bool EnableCrossExchangeFuturesFutures { get; set; } = true;     // Long perp on one exchange + short perp on another
    public bool EnableCrossExchangeSpotFutures { get; set; } = true;        // Buy spot on one exchange + short perp on another

    // Exchange configurations (moved from database)
    public List<ExchangeConfig> Exchanges { get; set; } = new();

    // Helper method to check if a strategy is enabled
    public bool IsStrategyEnabled(StrategySubType strategy)
    {
        return strategy switch
        {
            StrategySubType.SpotPerpetualSameExchange => EnableSpotPerpetualSameExchange,
            StrategySubType.CrossExchangeFuturesFutures => EnableCrossExchangeFuturesFutures,
            StrategySubType.CrossExchangeSpotFutures => EnableCrossExchangeSpotFutures,
            _ => false
        };
    }
}
