using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Base repository implementation for memory cache storage with optional pattern matching support
/// </summary>
public abstract class BaseMemoryCacheRepository<T> : IDataRepository<T> where T : class
{
    protected readonly IMemoryCache _cache;
    protected readonly ILogger _logger;
    private readonly HashSet<string> _keys = new(); // Track keys for pattern matching
    private readonly object _keysLock = new();
    private readonly bool _enablePatternMatching;

    public StorageStrategy Strategy => StorageStrategy.MemoryOnly;

    protected BaseMemoryCacheRepository(
        IMemoryCache cache,
        ILogger logger,
        bool enablePatternMatching = true)
    {
        _cache = cache;
        _logger = logger;
        _enablePatternMatching = enablePatternMatching;
    }

    public virtual Task StoreAsync(string key, T value, TimeSpan? ttl = null, CancellationToken cancellationToken = default)
    {
        _cache.Set(key, value, ttl ?? GetDefaultTtl());

        if (_enablePatternMatching)
        {
            lock (_keysLock)
            {
                _keys.Add(key);
            }
        }

        return Task.CompletedTask;
    }

    public virtual Task StoreBatchAsync(IDictionary<string, T> items, TimeSpan? ttl = null, CancellationToken cancellationToken = default)
    {
        foreach (var (key, value) in items)
        {
            _cache.Set(key, value, ttl ?? GetDefaultTtl());

            if (_enablePatternMatching)
            {
                lock (_keysLock)
                {
                    _keys.Add(key);
                }
            }
        }

        _logger.LogDebug("Stored {Count} items in memory cache", items.Count);
        return Task.CompletedTask;
    }

    public virtual Task<T?> GetAsync(string key, CancellationToken cancellationToken = default)
    {
        if (_cache.TryGetValue<T>(key, out var cached))
        {
            return Task.FromResult<T?>(cached);
        }

        return Task.FromResult<T?>(null);
    }

    public virtual Task<IDictionary<string, T>> GetBatchAsync(IEnumerable<string> keys, CancellationToken cancellationToken = default)
    {
        IDictionary<string, T> result = new Dictionary<string, T>();

        foreach (var key in keys)
        {
            if (_cache.TryGetValue<T>(key, out var cached) && cached != null)
            {
                result[key] = cached;
            }
        }

        return Task.FromResult(result);
    }

    public virtual Task<IDictionary<string, T>> GetByPatternAsync(string pattern, CancellationToken cancellationToken = default)
    {
        IDictionary<string, T> result = new Dictionary<string, T>();

        if (!_enablePatternMatching)
        {
            _logger.LogWarning("Pattern matching not enabled for this repository: {Pattern}", pattern);
            return Task.FromResult(result);
        }

        // Simple pattern matching (supports * wildcard)
        var regexPattern = "^" + Regex.Escape(pattern).Replace("\\*", ".*") + "$";
        var regex = new Regex(regexPattern);

        lock (_keysLock)
        {
            foreach (var key in _keys.Where(k => regex.IsMatch(k)))
            {
                if (_cache.TryGetValue<T>(key, out var cached) && cached != null)
                {
                    result[key] = cached;
                }
            }
        }

        return Task.FromResult(result);
    }

    public virtual Task DeleteAsync(string key, CancellationToken cancellationToken = default)
    {
        _cache.Remove(key);

        if (_enablePatternMatching)
        {
            lock (_keysLock)
            {
                _keys.Remove(key);
            }
        }

        return Task.CompletedTask;
    }

    public virtual Task DeleteByPatternAsync(string pattern, CancellationToken cancellationToken = default)
    {
        if (!_enablePatternMatching)
        {
            _logger.LogWarning("Pattern deletion not enabled for this repository: {Pattern}", pattern);
            return Task.CompletedTask;
        }

        // Simple pattern matching (supports * wildcard)
        var regexPattern = "^" + Regex.Escape(pattern).Replace("\\*", ".*") + "$";
        var regex = new Regex(regexPattern);

        lock (_keysLock)
        {
            var keysToDelete = _keys.Where(k => regex.IsMatch(k)).ToList();

            foreach (var key in keysToDelete)
            {
                _cache.Remove(key);
                _keys.Remove(key);
            }

            if (keysToDelete.Count > 0)
            {
                _logger.LogDebug("Deleted {Count} items matching pattern: {Pattern}",
                    keysToDelete.Count, pattern);
            }
        }

        return Task.CompletedTask;
    }

    public virtual Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_cache.TryGetValue(key, out _));
    }

    public virtual Task ClearAsync(CancellationToken cancellationToken = default)
    {
        if (!_enablePatternMatching)
        {
            _logger.LogWarning("Clear all not fully supported without pattern matching enabled");
            return Task.CompletedTask;
        }

        lock (_keysLock)
        {
            foreach (var key in _keys)
            {
                _cache.Remove(key);
            }
            _keys.Clear();
        }

        _logger.LogInformation("Cleared all items from memory cache");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Get default TTL for cached items - override in derived classes
    /// </summary>
    protected abstract TimeSpan GetDefaultTtl();
}
