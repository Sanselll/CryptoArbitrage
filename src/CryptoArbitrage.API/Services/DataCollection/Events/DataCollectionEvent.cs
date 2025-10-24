namespace CryptoArbitrage.API.Services.DataCollection.Events;

/// <summary>
/// Event emitted after data collection completes.
/// Generic type T is the data type collected (e.g., FundingRateDto, MarketDataSnapshot, UserDataSnapshot).
/// </summary>
public class DataCollectionEvent<T>
{
    /// <summary>
    /// Type of event (e.g., "FundingRatesCollected", "MarketPricesCollected", "UserDataCollected")
    /// </summary>
    public string EventType { get; set; } = string.Empty;

    /// <summary>
    /// The collected data. For batch collections, this is typically a Dictionary<string, T>.
    /// For single item collections, this can be a single T or List<T>.
    /// </summary>
    public T? Data { get; set; }

    /// <summary>
    /// When the collection completed
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// How long the collection took
    /// </summary>
    public TimeSpan CollectionDuration { get; set; }

    /// <summary>
    /// Whether the collection was successful
    /// </summary>
    public bool Success { get; set; } = true;

    /// <summary>
    /// Number of items collected (useful for logging)
    /// </summary>
    public int ItemCount { get; set; }
}
