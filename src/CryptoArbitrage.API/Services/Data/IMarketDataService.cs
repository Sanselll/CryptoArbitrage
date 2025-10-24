using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;

namespace CryptoArbitrage.API.Services.Data;

/// <summary>
/// Service for reading cached market data (funding rates, spot prices, perpetual prices)
/// </summary>
public interface IMarketDataService
{
    /// <summary>
    /// Get all cached funding rates for all exchanges
    /// </summary>
    Task<Dictionary<string, List<FundingRateDto>>> GetFundingRatesAsync();

    /// <summary>
    /// Get all cached spot prices for all exchanges
    /// </summary>
    Task<Dictionary<string, Dictionary<string, PriceDto>>> GetSpotPricesAsync();

    /// <summary>
    /// Get all cached perpetual prices for all exchanges
    /// </summary>
    Task<Dictionary<string, Dictionary<string, PriceDto>>> GetPerpetualPricesAsync();

    /// <summary>
    /// Get complete market data snapshot (all data in one call)
    /// </summary>
    Task<MarketDataSnapshot?> GetMarketDataSnapshotAsync();
}
