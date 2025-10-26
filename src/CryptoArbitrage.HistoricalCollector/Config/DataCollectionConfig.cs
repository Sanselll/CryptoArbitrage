namespace CryptoArbitrage.HistoricalCollector.Config;

/// <summary>
/// Configuration for historical data collection rate limiting and concurrency
/// </summary>
public class DataCollectionConfig
{
    public ExchangeConfig Binance { get; set; } = new();
    public ExchangeConfig Bybit { get; set; } = new();

    public ExchangeConfig GetConfigForExchange(string exchange)
    {
        return exchange switch
        {
            "Binance" => Binance,
            "Bybit" => Bybit,
            _ => new ExchangeConfig() // Default config
        };
    }
}

public class ExchangeConfig
{
    /// <summary>
    /// Maximum number of concurrent requests to the exchange API
    /// Binance: Recommended 1-2 (strict rate limits)
    /// Bybit: Recommended 5-10 (more lenient)
    /// </summary>
    public int MaxConcurrentRequests { get; set; } = 1;

    /// <summary>
    /// Delay in milliseconds between API requests to avoid rate limiting
    /// Binance: 250ms (20 req/s limit = 1200/min)
    /// Bybit: 100ms (24 req/5s limit = 120/5s)
    /// </summary>
    public int RateLimitDelayMs { get; set; } = 250;
}
