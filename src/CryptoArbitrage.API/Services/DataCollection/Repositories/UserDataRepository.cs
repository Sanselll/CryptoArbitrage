using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Repositories;

/// <summary>
/// Repository for user data (balances and positions) with memory-only storage
/// Note: Position persistence is handled separately by ArbitrageExecutionService
/// This repository is primarily for caching current state for SignalR broadcasting
/// Pattern matching enabled to support balance aggregation for AI suggestions
/// </summary>
public class UserDataRepository : BaseMemoryCacheRepository<UserDataSnapshot>
{
    public UserDataRepository(
        IMemoryCache cache,
        ILogger<UserDataRepository> logger)
        : base(cache, logger, enablePatternMatching: true)
    {
    }

    protected override TimeSpan GetDefaultTtl() => TimeSpan.FromMinutes(10);

    /// <summary>
    /// Get user data snapshot by user ID and exchange (convenience method)
    /// </summary>
    public Task<UserDataSnapshot?> GetUserDataAsync(int userId, string exchange, CancellationToken cancellationToken = default)
    {
        var key = $"userdata:{userId}:{exchange}";
        return GetAsync(key, cancellationToken);
    }

    /// <summary>
    /// Get all user data snapshots for a specific user across all exchanges
    /// </summary>
    public async Task<List<UserDataSnapshot>> GetUserDataAllExchangesAsync(int userId, CancellationToken cancellationToken = default)
    {
        // Note: This is limited since pattern matching is disabled for this repository
        // In production, consider using Redis or maintaining a separate key registry
        _logger.LogWarning("GetUserDataAllExchangesAsync has limited functionality without pattern matching");
        return new List<UserDataSnapshot>();
    }
}
