using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.HistoricalCollector.Infrastructure;

/// <summary>
/// Mock repository for testing - returns empty data
/// Used to satisfy OpportunityDetectionService dependencies without Redis
/// </summary>
public class MockDataRepository<T> : IDataRepository<T> where T : class
{
    public StorageStrategy Strategy => StorageStrategy.MemoryOnly;

    public Task StoreAsync(string key, T value, TimeSpan? ttl = null, CancellationToken cancellationToken = default)
        => Task.CompletedTask;

    public Task StoreBatchAsync(IDictionary<string, T> items, TimeSpan? ttl = null, CancellationToken cancellationToken = default)
        => Task.CompletedTask;

    public Task<T?> GetAsync(string key, CancellationToken cancellationToken = default)
        => Task.FromResult<T?>(default);

    public Task<IDictionary<string, T>> GetBatchAsync(IEnumerable<string> keys, CancellationToken cancellationToken = default)
        => Task.FromResult<IDictionary<string, T>>(new Dictionary<string, T>());

    public Task<IDictionary<string, T>> GetByPatternAsync(string pattern, CancellationToken cancellationToken = default)
        => Task.FromResult<IDictionary<string, T>>(new Dictionary<string, T>());

    public Task DeleteAsync(string key, CancellationToken cancellationToken = default)
        => Task.CompletedTask;

    public Task DeleteByPatternAsync(string pattern, CancellationToken cancellationToken = default)
        => Task.CompletedTask;

    public Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default)
        => Task.FromResult(false);

    public Task ClearAsync(CancellationToken cancellationToken = default)
        => Task.CompletedTask;
}
