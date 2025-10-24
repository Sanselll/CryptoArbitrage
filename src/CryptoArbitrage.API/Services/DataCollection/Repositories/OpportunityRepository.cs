using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Repository for arbitrage opportunities with memory-only storage
/// </summary>
public class OpportunityRepository : BaseMemoryCacheRepository<ArbitrageOpportunityDto>
{
    public OpportunityRepository(
        IMemoryCache cache,
        ILogger<OpportunityRepository> logger)
        : base(cache, logger, enablePatternMatching: true)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromMinutes(5);
}
