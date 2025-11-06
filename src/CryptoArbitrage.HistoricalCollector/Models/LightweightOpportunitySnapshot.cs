using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.HistoricalCollector.Models;

/// <summary>
/// Lightweight snapshot containing only timestamp and opportunities for storage.
/// Excludes price data, funding rates, and other metadata to reduce file size.
/// </summary>
public class LightweightOpportunitySnapshot
{
    public DateTime Timestamp { get; set; }
    public List<ArbitrageOpportunityDto> Opportunities { get; set; } = new();
}
