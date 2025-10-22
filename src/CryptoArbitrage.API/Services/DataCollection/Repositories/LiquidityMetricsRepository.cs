using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Repository for liquidity metrics with memory-only storage
/// </summary>
public class LiquidityMetricsRepository : BaseMemoryCacheRepository<LiquidityMetricsDto>
{
    public LiquidityMetricsRepository(
        IMemoryCache cache,
        ILogger<LiquidityMetricsRepository> logger)
        : base(cache, logger, enablePatternMatching: true)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromMinutes(10);
}
