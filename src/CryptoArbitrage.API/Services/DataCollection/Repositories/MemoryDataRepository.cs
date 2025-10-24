using CryptoArbitrage.API.Services.DataCollection.Repositories;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Generic memory repository for storing any type of data with a 5-minute TTL
/// </summary>
public class MemoryDataRepository<T> : BaseMemoryCacheRepository<T> where T : class
{
    public MemoryDataRepository(
        IMemoryCache cache,
        ILogger<MemoryDataRepository<T>> logger)
        : base(cache, logger, enablePatternMatching: true)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromMinutes(5);
}
