using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Snapshot of market prices from all exchanges
/// </summary>
public class MarketDataSnapshot
{
    public Dictionary<string, List<FundingRateDto>> FundingRates { get; set; } = new();
    public Dictionary<string, Dictionary<string, PriceDto>> SpotPrices { get; set; } = new();
    public Dictionary<string, Dictionary<string, PriceDto>> PerpPrices { get; set; } = new();
    public DateTime FetchedAt { get; set; }
}
