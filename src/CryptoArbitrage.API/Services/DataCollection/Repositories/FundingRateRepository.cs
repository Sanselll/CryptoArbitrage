using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Repository for funding rates with memory-only storage
/// </summary>
public class FundingRateRepository : BaseMemoryCacheRepository<FundingRateDto>
{
    public FundingRateRepository(
        IMemoryCache cache,
        ILogger<FundingRateRepository> logger)
        : base(cache, logger, enablePatternMatching: true)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromHours(1);

    /// <summary>
    /// Get all funding rates (convenience method)
    /// </summary>
    public Task<Dictionary<string, FundingRateDto>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        // Use pattern matching to get all
        return GetByPatternAsync("*", cancellationToken)
            .ContinueWith(t => new Dictionary<string, FundingRateDto>(t.Result), cancellationToken);
    }
}
