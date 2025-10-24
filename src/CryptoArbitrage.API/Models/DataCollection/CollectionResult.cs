namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Result of a data collection operation
/// </summary>
public class CollectionResult<T> where T : class
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
    public IDictionary<string, T> Data { get; set; } = new Dictionary<string, T>();
    public int ItemsCollected => Data.Count;
    public TimeSpan Duration { get; set; }
}
