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
    // === TIMING CONFIGURATION ===
    // Data collection: How often to fetch data from exchanges and detect opportunities
    public int OpportunityCollectionIntervalSeconds { get; set; } = 60;

    // SignalR streaming: How often to broadcast cached data to UI
    public int SignalRBroadcastIntervalSeconds { get; set; } = 1;

    // === LIQUIDITY CONFIGURATION ===
    // Minimum 24h trading volume in USD for a symbol to be considered liquid
    public decimal MinVolume24hUsd { get; set; } = 500_000m;

    // Maximum acceptable bid/ask spread percentage for good liquidity
    public decimal MaxBidAskSpreadPercent { get; set; } = 0.5m;

    // Minimum orderbook depth in USD for reliable execution
    public decimal MinOrderbookDepthUsd { get; set; } = 25_000m;

    // Max parallel liquidity requests to avoid rate limits
    public int MaxConcurrentLiquidityRequests { get; set; } = 10;

    // === TRADING PARAMETERS ===
    public decimal MinSpreadPercentage { get; set; } = 0.1m; // 0.1% minimum spread
    public decimal MaxPositionSizeUsd { get; set; } = 10000m;
    public decimal MinPositionSizeUsd { get; set; } = 100m;
    public decimal MaxLeverage { get; set; } = 5m;
    public decimal MaxTotalExposure { get; set; } = 50000m;
    public List<string> WatchedSymbols { get; set; } = new();
    public bool AutoExecute { get; set; } = false;

    // === SYMBOL DISCOVERY SETTINGS ===
    public bool AutoDiscoverSymbols { get; set; } = true;
    public decimal MinDailyVolumeUsd { get; set; } = 10_000_000m; // $10M minimum daily volume
    public int MaxSymbolCount { get; set; } = 50; // Track top 50 symbols by volume
    public decimal MinAbsFundingRate { get; set; } = 0.0001m; // 0.01% = 3.65% APR minimum
    public decimal MinHighPriorityFundingRate { get; set; } = 0.01m; // 1% = 365% APR - always include these

    // === STRATEGY CONFIGURATION ===
    public bool EnableSpotPerpetualSameExchange { get; set; } = true;       // Buy spot + short perp on same exchange
    public bool EnableCrossExchangeFuturesFutures { get; set; } = true;     // Long perp on one exchange + short perp on another
    public bool EnableCrossExchangeSpotFutures { get; set; } = true;        // Buy spot on one exchange + short perp on another

    // === EXCHANGE CONFIGURATIONS ===
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
