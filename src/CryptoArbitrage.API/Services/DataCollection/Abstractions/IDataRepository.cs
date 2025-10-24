namespace CryptoArbitrage.API.Services.DataCollection.Abstractions;

/// <summary>
/// Storage strategy for data repositories
/// </summary>
public enum StorageStrategy
{
    /// <summary>In-memory cache only (volatile, fast)</summary>
    MemoryOnly,

    /// <summary>Database only (persistent, slower)</summary>
    DatabaseOnly,

    /// <summary>Both memory cache and database (hot cache with persistence)</summary>
    Dual
}

/// <summary>
/// Generic data repository interface for storing and retrieving collected data
/// </summary>
/// <typeparam name="T">The type of data being stored</typeparam>
public interface IDataRepository<T> where T : class
{
    /// <summary>
    /// Storage strategy used by this repository
    /// </summary>
    StorageStrategy Strategy { get; }

    /// <summary>
    /// Store a single item
    /// </summary>
    Task StoreAsync(string key, T value, TimeSpan? ttl = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Store multiple items
    /// </summary>
    Task StoreBatchAsync(IDictionary<string, T> items, TimeSpan? ttl = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieve a single item by key
    /// </summary>
    Task<T?> GetAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieve multiple items by keys
    /// </summary>
    Task<IDictionary<string, T>> GetBatchAsync(IEnumerable<string> keys, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get all items matching a pattern (e.g., "funding:BTCUSDT:*")
    /// </summary>
    Task<IDictionary<string, T>> GetByPatternAsync(string pattern, CancellationToken cancellationToken = default);

    /// <summary>
    /// Delete a single item
    /// </summary>
    Task DeleteAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Delete all items matching a pattern
    /// </summary>
    Task DeleteByPatternAsync(string pattern, CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if an item exists
    /// </summary>
    Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Clear all data from the repository
    /// </summary>
    Task ClearAsync(CancellationToken cancellationToken = default);
}
