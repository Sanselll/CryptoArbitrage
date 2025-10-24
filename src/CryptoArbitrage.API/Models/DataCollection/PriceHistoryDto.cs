namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Tracks historical prices for a specific exchange/symbol pair (limited to last 30 samples)
/// </summary>
public class PriceHistoryDto
{
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;

    /// <summary>
    /// Price history limited to last 30 samples (most recent last)
    /// </summary>
    public List<decimal> PriceHistory { get; set; } = new();

    /// <summary>
    /// Timestamps corresponding to each price sample (most recent last)
    /// </summary>
    public List<DateTime> Timestamps { get; set; } = new();

    /// <summary>
    /// Maximum number of samples to keep in history
    /// </summary>
    public const int MaxSamples = 30;

    /// <summary>
    /// Appends a new price to history, maintaining the 30-sample limit
    /// </summary>
    public void AppendPrice(decimal price, DateTime timestamp)
    {
        PriceHistory.Add(price);
        Timestamps.Add(timestamp);

        // Remove oldest samples if we exceed the limit
        while (PriceHistory.Count > MaxSamples)
        {
            PriceHistory.RemoveAt(0);
            Timestamps.RemoveAt(0);
        }
    }

    /// <summary>
    /// Gets the number of samples currently in history
    /// </summary>
    public int SampleCount => PriceHistory.Count;

    /// <summary>
    /// Checks if we have a full 30-sample history
    /// </summary>
    public bool IsFullHistory => PriceHistory.Count >= MaxSamples;
}
