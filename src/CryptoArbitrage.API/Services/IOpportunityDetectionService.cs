using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service responsible for detecting arbitrage opportunities from market data
/// </summary>
public interface IOpportunityDetectionService
{
    /// <summary>
    /// Detect all types of arbitrage opportunities from the provided market data
    /// </summary>
    Task<List<ArbitrageOpportunityDto>> DetectOpportunitiesAsync(MarketDataSnapshot snapshot);
}
