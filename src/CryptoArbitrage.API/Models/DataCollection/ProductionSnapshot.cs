namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Complete snapshot of market state for ML training data collection
/// </summary>
public class ProductionSnapshot
{
    /// <summary>
    /// UTC timestamp when this snapshot was collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Collection interval in minutes (typically 5)
    /// </summary>
    public int IntervalMinutes { get; set; } = 5;

    /// <summary>
    /// Market data for all tracked exchange-symbol pairs (prices, funding rates, volume)
    /// Each entry represents one exchange's data for one symbol
    /// </summary>
    public List<SymbolMarketData> MarketData { get; set; } = new();

    /// <summary>
    /// All detected arbitrage opportunities at this timestamp
    /// </summary>
    public List<ArbitrageOpportunityDto> Opportunities { get; set; } = new();

    /// <summary>
    /// Collection metadata
    /// </summary>
    public CollectionMetadata Metadata { get; set; } = new();
}

/// <summary>
/// Market data for a single exchange-symbol pair
/// </summary>
public class SymbolMarketData
{
    public string Symbol { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;

    /// <summary>
    /// Perpetual price
    /// </summary>
    public decimal? Price { get; set; }

    /// <summary>
    /// Funding rate (set to 0 when no payment occurs in this interval)
    /// </summary>
    public decimal FundingRate { get; set; }

    /// <summary>
    /// Annualized funding rate
    /// </summary>
    public decimal AnnualizedFundingRate { get; set; }

    /// <summary>
    /// 24-hour trading volume in USD
    /// </summary>
    public decimal? Volume24h { get; set; }

    /// <summary>
    /// Bid-ask spread percentage
    /// </summary>
    public decimal? BidAskSpreadPct { get; set; }
}


/// <summary>
/// Metadata about the collection process
/// </summary>
public class CollectionMetadata
{
    /// <summary>
    /// Total number of symbols in this snapshot
    /// </summary>
    public int TotalSymbols { get; set; }

    /// <summary>
    /// Total number of opportunities detected
    /// </summary>
    public int TotalOpportunities { get; set; }

    /// <summary>
    /// Collection duration in milliseconds
    /// </summary>
    public long CollectionDurationMs { get; set; }

    /// <summary>
    /// Source of collection (e.g., "production", "test")
    /// </summary>
    public string Source { get; set; } = "production";
}

/// <summary>
/// Daily manifest file that tracks all snapshots for a given date
/// </summary>
public class DailyManifest
{
    public string Date { get; set; } = string.Empty;
    public List<SnapshotInfo> Snapshots { get; set; } = new();
    public int TotalSnapshots { get; set; }
    public int SymbolsTracked { get; set; }
}

/// <summary>
/// Information about a single snapshot file
/// </summary>
public class SnapshotInfo
{
    public string Time { get; set; } = string.Empty;  // HH:mm format
    public string File { get; set; } = string.Empty;   // Filename (e.g., "10-05.json")
    public int Opportunities { get; set; }
}
