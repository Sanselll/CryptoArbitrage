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

    /// <summary>
    /// Historical perpetual prices (last 30 samples) for cross-exchange spread analysis
    /// Structure: [Exchange][Symbol] -> PriceHistoryDto
    /// </summary>
    public Dictionary<string, Dictionary<string, PriceHistoryDto>> PerpPriceHistory { get; set; } = new();

    public DateTime FetchedAt { get; set; }
}
