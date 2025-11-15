using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;

namespace CryptoArbitrage.API.Services.Arbitrage.Detection;

/// <summary>
/// Service responsible for detecting arbitrage opportunities from market data
/// </summary>
public interface IOpportunityDetectionService
{
    /// <summary>
    /// Detect all types of arbitrage opportunities from the provided market data
    /// </summary>
    /// <param name="snapshot">Market data snapshot</param>
    /// <param name="positionKeys">Optional: Active position keys (format: "SYMBOL|LongExchange|ShortExchange").
    /// If provided, opportunities for these positions will always be included.</param>
    Task<List<ArbitrageOpportunityDto>> DetectOpportunitiesAsync(
        MarketDataSnapshot snapshot,
        HashSet<string>? positionKeys = null);
}
