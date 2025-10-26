namespace CryptoArbitrage.API.Config;

/// <summary>
/// Configuration for opportunity snapshot dumping to disk
/// Captures enriched opportunities in the same format as HistoricalCollector
/// </summary>
public class OpportunityDumpConfig
{
    /// <summary>
    /// Enable/disable opportunity dumping
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Base path for saving dumps (e.g., "Data/backend_dumps")
    /// Creates subfolders by date: Data/backend_dumps/2025-10-25/
    /// </summary>
    public string BasePath { get; set; } = "Data/backend_dumps";

    /// <summary>
    /// Minimum number of opportunities required to trigger a dump
    /// Set to 0 to dump even when no opportunities are detected
    /// </summary>
    public int MinOpportunitiesThreshold { get; set; } = 1;
}
