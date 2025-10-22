using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Services.DataCollection.Configuration;

/// <summary>
/// Base configuration for all data collectors
/// </summary>
public class CollectorConfiguration
{
    /// <summary>
    /// Whether this collector is enabled
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Collection interval in seconds
    /// </summary>
    public int CollectionIntervalSeconds { get; set; } = 60;

    /// <summary>
    /// Maximum number of parallel fetches across exchanges
    /// </summary>
    public int MaxParallelFetches { get; set; } = 5;

    /// <summary>
    /// Storage strategy to use
    /// </summary>
    public StorageStrategy StorageStrategy { get; set; } = StorageStrategy.MemoryOnly;

    /// <summary>
    /// Time-to-live for cached data (null = use default)
    /// </summary>
    public int? CacheTtlMinutes { get; set; }

    /// <summary>
    /// Whether to retry on failure
    /// </summary>
    public bool RetryOnFailure { get; set; } = true;

    /// <summary>
    /// Maximum retry attempts
    /// </summary>
    public int MaxRetryAttempts { get; set; } = 3;

    /// <summary>
    /// Initial retry delay in milliseconds
    /// </summary>
    public int InitialRetryDelayMs { get; set; } = 1000;
}

/// <summary>
/// Configuration for funding rate collector
/// </summary>
public class FundingRateCollectorConfiguration : CollectorConfiguration
{
    public FundingRateCollectorConfiguration()
    {
        CollectionIntervalSeconds = 60;
        StorageStrategy = StorageStrategy.Dual; // Memory + Database
        CacheTtlMinutes = 60;
    }

    /// <summary>
    /// Whether to fetch historical rates on startup
    /// </summary>
    public bool FetchHistoryOnStartup { get; set; } = false;

    /// <summary>
    /// Number of historical data points to fetch
    /// </summary>
    public int HistoricalDataPoints { get; set; } = 100;
}

/// <summary>
/// Configuration for funding rate history collector (historical data collection)
/// </summary>
public class FundingRateHistoryCollectorConfiguration : CollectorConfiguration
{
    public FundingRateHistoryCollectorConfiguration()
    {
        CollectionIntervalSeconds = 3600; // Hourly
        StorageStrategy = StorageStrategy.DatabaseOnly;
    }

    /// <summary>
    /// How many days of history to collect
    /// </summary>
    public int HistoryDays { get; set; } = 30;
}

/// <summary>
/// Configuration for market price collector (spot + perpetual prices)
/// </summary>
public class MarketPriceCollectorConfiguration : CollectorConfiguration
{
    public MarketPriceCollectorConfiguration()
    {
        CollectionIntervalSeconds = 60;
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 5;
    }
}

/// <summary>
/// Configuration for volume collector (24h volumes)
/// </summary>
public class VolumeCollectorConfiguration : CollectorConfiguration
{
    public VolumeCollectorConfiguration()
    {
        CollectionIntervalSeconds = 60;
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 10;
    }
}

/// <summary>
/// Configuration for user data collector (balances + positions)
/// </summary>
public class UserDataCollectorConfiguration : CollectorConfiguration
{
    public UserDataCollectorConfiguration()
    {
        CollectionIntervalSeconds = 30;
        StorageStrategy = StorageStrategy.DatabaseOnly;
        MaxParallelFetches = 3; // Limit parallel user data fetches
    }

    /// <summary>
    /// Whether to broadcast position updates via SignalR
    /// </summary>
    public bool BroadcastPositions { get; set; } = true;

    /// <summary>
    /// Whether to broadcast balance updates via SignalR
    /// </summary>
    public bool BroadcastBalances { get; set; } = true;
}

/// <summary>
/// Configuration for liquidity metrics collector (bid-ask spread, orderbook depth)
/// </summary>
public class LiquidityCollectorConfiguration : CollectorConfiguration
{
    public LiquidityCollectorConfiguration()
    {
        CollectionIntervalSeconds = 120; // Every 2 minutes
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 10;
        MaxParallelFetches = 10; // Limit concurrent API requests
    }
}

/// <summary>
/// Configuration for open orders collector
/// </summary>
public class OpenOrdersCollectorConfiguration : CollectorConfiguration
{
    public OpenOrdersCollectorConfiguration()
    {
        CollectionIntervalSeconds = 30; // Every 30 seconds
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 5;
        MaxParallelFetches = 3;
    }
}

/// <summary>
/// Configuration for order history collector
/// </summary>
public class OrderHistoryCollectorConfiguration : CollectorConfiguration
{
    public OrderHistoryCollectorConfiguration()
    {
        CollectionIntervalSeconds = 300; // Every 5 minutes
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 10;
        MaxParallelFetches = 3;
    }

    /// <summary>
    /// Number of days to fetch history for
    /// </summary>
    public int HistoryDays { get; set; } = 7;
}

/// <summary>
/// Configuration for trade history collector
/// </summary>
public class TradeHistoryCollectorConfiguration : CollectorConfiguration
{
    public TradeHistoryCollectorConfiguration()
    {
        CollectionIntervalSeconds = 60; // Every minute
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 10;
        MaxParallelFetches = 3;
    }

    /// <summary>
    /// Number of days to fetch history for
    /// </summary>
    public int HistoryDays { get; set; } = 7;
}

/// <summary>
/// Configuration for transaction history collector
/// </summary>
public class TransactionHistoryCollectorConfiguration : CollectorConfiguration
{
    public TransactionHistoryCollectorConfiguration()
    {
        CollectionIntervalSeconds = 300; // Every 5 minutes
        StorageStrategy = StorageStrategy.MemoryOnly;
        CacheTtlMinutes = 15;
        MaxParallelFetches = 3;
    }

    /// <summary>
    /// Number of days to fetch history for
    /// </summary>
    public int HistoryDays { get; set; } = 7;
}

/// <summary>
/// Master configuration for all data collectors
/// </summary>
public class DataCollectionConfiguration
{
    public FundingRateCollectorConfiguration FundingRate { get; set; } = new();
    public FundingRateHistoryCollectorConfiguration FundingRateHistory { get; set; } = new();
    public MarketPriceCollectorConfiguration MarketPrice { get; set; } = new();
    public VolumeCollectorConfiguration Volume { get; set; } = new();
    public UserDataCollectorConfiguration UserData { get; set; } = new();
    public LiquidityCollectorConfiguration Liquidity { get; set; } = new();
    public OpenOrdersCollectorConfiguration OpenOrders { get; set; } = new();
    public OrderHistoryCollectorConfiguration OrderHistory { get; set; } = new();
    public TradeHistoryCollectorConfiguration TradeHistory { get; set; } = new();
    public TransactionHistoryCollectorConfiguration TransactionHistory { get; set; } = new();
}
