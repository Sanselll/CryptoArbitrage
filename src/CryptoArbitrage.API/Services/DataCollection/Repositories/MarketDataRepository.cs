using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Repository for market data (spot/perpetual prices and volumes) with memory-only storage
/// </summary>
public class MarketDataRepository : BaseMemoryCacheRepository<MarketDataSnapshot>
{
    private const string SnapshotKey = "market_data_snapshot";

    public MarketDataRepository(
        IMemoryCache cache,
        ILogger<MarketDataRepository> logger)
        : base(cache, logger, enablePatternMatching: false)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromMinutes(5);

    /// <summary>
    /// Get the current market data snapshot (convenience method)
    /// </summary>
    public Task<MarketDataSnapshot?> GetCurrentSnapshotAsync(CancellationToken cancellationToken = default)
    {
        return GetAsync(SnapshotKey, cancellationToken);
    }

    /// <summary>
    /// Store the current market data snapshot (convenience method)
    /// </summary>
    public Task StoreCurrentSnapshotAsync(MarketDataSnapshot snapshot, TimeSpan? ttl = null, CancellationToken cancellationToken = default)
    {
        return StoreAsync(SnapshotKey, snapshot, ttl, cancellationToken);
    }
}
