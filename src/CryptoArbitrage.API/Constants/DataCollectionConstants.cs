namespace CryptoArbitrage.API.Constants;

/// <summary>
/// Constants for data collection, caching, and event bus
/// </summary>
public static class DataCollectionConstants
{
    /// <summary>
    /// Cache key constants for data repositories
    /// </summary>
    public static class CacheKeys
    {
        /// <summary>
        /// Prefix for funding rate cache keys: "funding:"
        /// Full pattern: funding:{exchangeName}:{symbol}
        /// </summary>
        public const string FundingRatePrefix = "funding";

        /// <summary>
        /// Key for market data snapshot in cache
        /// </summary>
        public const string MarketDataSnapshot = "market_data_snapshot";

        /// <summary>
        /// Prefix for liquidity metric cache keys: "liquidity:"
        /// Full pattern: liquidity:{exchangeName}:{symbol}
        /// </summary>
        public const string LiquidityMetricPrefix = "liquidity";

        /// <summary>
        /// Builds a funding rate cache key
        /// </summary>
        public static string BuildFundingRateKey(string exchangeName, string symbol)
            => $"{FundingRatePrefix}:{exchangeName}:{symbol}";

        /// <summary>
        /// Builds a liquidity metric cache key
        /// </summary>
        public static string BuildLiquidityMetricKey(string exchangeName, string symbol)
            => $"{LiquidityMetricPrefix}:{exchangeName}:{symbol}";

        /// <summary>
        /// Pattern to match all funding rate keys
        /// </summary>
        public static string FundingRatePattern => $"{FundingRatePrefix}:*";

        /// <summary>
        /// Pattern to match all liquidity metric keys
        /// </summary>
        public static string LiquidityMetricPattern => $"{LiquidityMetricPrefix}:*";
    }

    /// <summary>
    /// Event type constants for the data collection event bus
    /// </summary>
    public static class EventTypes
    {
        /// <summary>
        /// Published when funding rates are collected from exchanges
        /// </summary>
        public const string FundingRatesCollected = "FundingRatesCollected";

        /// <summary>
        /// Published when market prices (spot and perp) are collected from exchanges
        /// </summary>
        public const string MarketPricesCollected = "MarketPricesCollected";

        /// <summary>
        /// Published when user data (balances and positions) is collected
        /// </summary>
        public const string UserDataCollected = "UserDataCollected";

        /// <summary>
        /// Published when arbitrage opportunities are detected
        /// </summary>
        public const string OpportunitiesDetected = "OpportunitiesDetected";

        /// <summary>
        /// Published when opportunities are enriched with additional data
        /// </summary>
        public const string OpportunitiesEnriched = "OpportunitiesEnriched";

        /// <summary>
        /// Published when liquidity metrics are collected from exchanges
        /// </summary>
        public const string LiquidityMetricsCollected = "LiquidityMetricsCollected";
    }
}
