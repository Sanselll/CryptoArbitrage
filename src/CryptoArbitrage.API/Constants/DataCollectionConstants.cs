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
        /// Prefix for price history cache keys: "price_history:"
        /// Full pattern: price_history:{exchangeName}:{symbol}
        /// </summary>
        public const string PriceHistoryPrefix = "price_history";

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
        /// Builds a price history cache key
        /// </summary>
        public static string BuildPriceHistoryKey(string exchangeName, string symbol)
            => $"{PriceHistoryPrefix}:{exchangeName}:{symbol}";

        /// <summary>
        /// Builds a general cache key (for historical prices, etc.)
        /// </summary>
        public static string BuildKey(string exchangeName, string symbol)
            => $"{exchangeName}:{symbol}";

        /// <summary>
        /// Pattern to match all funding rate keys
        /// </summary>
        public static string FundingRatePattern => $"{FundingRatePrefix}:*";

        /// <summary>
        /// Pattern to match all liquidity metric keys
        /// </summary>
        public static string LiquidityMetricPattern => $"{LiquidityMetricPrefix}:*";

        /// <summary>
        /// Pattern to match all price history keys
        /// </summary>
        public static string PriceHistoryPattern => $"{PriceHistoryPrefix}:*";
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
        /// Published when historical funding rate averages are collected and updated
        /// </summary>
        public const string FundingRateHistoryCollected = "FundingRateHistoryCollected";

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

        /// <summary>
        /// Published when open orders are collected for users
        /// </summary>
        public const string OpenOrdersCollected = "OpenOrdersCollected";

        /// <summary>
        /// Published when order history is collected for users
        /// </summary>
        public const string OrderHistoryCollected = "OrderHistoryCollected";

        /// <summary>
        /// Published when trade history is collected for users
        /// </summary>
        public const string TradeHistoryCollected = "TradeHistoryCollected";

        /// <summary>
        /// Published when transaction history is collected for users
        /// </summary>
        public const string TransactionHistoryCollected = "TransactionHistoryCollected";

        /// <summary>
        /// Published when historical price data (KLines) is collected for spread projections
        /// </summary>
        public const string HistoricalPriceCollected = "HistoricalPriceCollected";
    }
}
