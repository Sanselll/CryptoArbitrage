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
    Task<List<ArbitrageOpportunityDto>> DetectOpportunitiesAsync(MarketDataSnapshot snapshot);
}
