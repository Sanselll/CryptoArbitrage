using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.HistoricalCollector.Models;

/// <summary>
/// Represents a complete market snapshot at a specific point in time
/// Used for historical reconstruction and ML training data generation
/// </summary>
public class HistoricalMarketSnapshot
{
    public DateTime Timestamp { get; set; }

    // All detected opportunities at this moment
    public List<ArbitrageOpportunityDto> Opportunities { get; set; } = new();

    // Market prices for all tracked symbols
    // Exchange -> Symbol -> Price
    public Dictionary<string, Dictionary<string, PriceDto>> PerpPrices { get; set; } = new();
    public Dictionary<string, Dictionary<string, PriceDto>> SpotPrices { get; set; } = new();

    // Funding rates
    // Exchange -> List of rates
    public Dictionary<string, List<FundingRateDto>> FundingRates { get; set; } = new();

    // Liquidity metrics (only available in live mode, null for historical)
    public Dictionary<string, LiquidityMetricsDto>? Liquidity { get; set; }

    // Market context
    public decimal BtcPrice { get; set; }
    public decimal BtcVolume24h { get; set; }
    public string MarketRegime { get; set; } = "Unknown";

    // Metadata
    public int OpportunitiesCount => Opportunities.Count;
    public bool IsHistorical => Liquidity == null;
}
