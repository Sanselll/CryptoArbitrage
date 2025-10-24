namespace CryptoArbitrage.API.Models;

/// <summary>
/// In-memory cache entry for funding rates with previous rate tracking
/// </summary>
public class FundingRateCacheEntry
{
    public FundingRateDto CurrentRate { get; set; } = null!;
    public FundingRateDto? PreviousRate { get; set; }
    public DateTime LastUpdated { get; set; }
}
