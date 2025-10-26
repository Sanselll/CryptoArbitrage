using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.HistoricalCollector.Models;

/// <summary>
/// Container for persisted raw market data
/// Saved to data/raw/ folder for reusability
/// </summary>
public class RawMarketData
{
    public DateTime CollectionDate { get; set; }
    public List<string> Exchanges { get; set; } = new();
    public List<string> Symbols { get; set; } = new();
    public Dictionary<string, List<FundingRateDto>> FundingRates { get; set; } = new();
    public Dictionary<string, Dictionary<string, List<PriceDto>>> PriceKlines { get; set; } = new();
    public Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? LiquidityMetrics { get; set; }
    public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Metadata about collected raw data
/// </summary>
public class DataCollectionManifest
{
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public List<string> Exchanges { get; set; } = new();
    public List<string> Symbols { get; set; } = new();
    public int TotalFundingRates { get; set; }
    public int TotalPriceKlines { get; set; }
    public bool HasLiquidityData { get; set; }
    public DateTime CollectedAt { get; set; }
    public string DataPath { get; set; } = string.Empty;
}
