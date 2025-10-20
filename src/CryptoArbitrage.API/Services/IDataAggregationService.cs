using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service responsible for fetching and caching market data (funding rates, prices, volumes)
/// </summary>
public interface IDataAggregationService
{
    /// <summary>
    /// Fetch and cache all market data (funding rates, spot prices, perp prices, volumes)
    /// </summary>
    Task<MarketDataSnapshot> FetchAndCacheMarketDataAsync(
        List<string> symbols,
        Dictionary<string, IExchangeConnector> exchangeConnectors,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get cached funding rates for all exchanges
    /// </summary>
    Dictionary<string, List<FundingRateDto>> GetCachedFundingRates();

    /// <summary>
    /// Get cached spot prices for all exchanges
    /// </summary>
    Dictionary<string, Dictionary<string, SpotPriceDto>> GetCachedSpotPrices();

    /// <summary>
    /// Get cached perpetual prices for all exchanges
    /// </summary>
    Dictionary<string, Dictionary<string, decimal>> GetCachedPerpPrices();
}

/// <summary>
/// Snapshot of all market data fetched in one collection cycle
/// </summary>
public class MarketDataSnapshot
{
    public Dictionary<string, List<FundingRateDto>> FundingRates { get; set; } = new();
    public Dictionary<string, Dictionary<string, SpotPriceDto>> SpotPrices { get; set; } = new();
    public Dictionary<string, Dictionary<string, decimal>> PerpPrices { get; set; } = new();
    public DateTime FetchedAt { get; set; }
}
