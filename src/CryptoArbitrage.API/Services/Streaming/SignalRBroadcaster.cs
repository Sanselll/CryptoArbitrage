using System.Text.Json;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.Streaming;

/// <summary>
/// Subscribes to data collection events and broadcasts to SignalR clients.
/// PURE broadcaster: Only handles broadcasting, no collection or aggregation.
/// Uses hash-based change detection to avoid duplicate broadcasts.
/// </summary>
public class SignalRBroadcaster : IHostedService
{
    private readonly ILogger<SignalRBroadcaster> _logger;
    private readonly IDataCollectionEventBus _eventBus;
    private readonly ISignalRStreamingService _signalRStreamingService;

    // Hash tracking for change detection
    private int _lastFundingRatesHash = 0;
    private int _lastOpportunitiesHash = 0;
    private readonly Dictionary<string, int> _lastUserBalancesHash = new();
    private readonly Dictionary<string, int> _lastUserPositionsHash = new();

    private readonly object _hashLock = new();

    public SignalRBroadcaster(
        ILogger<SignalRBroadcaster> logger,
        IDataCollectionEventBus eventBus,
        ISignalRStreamingService signalRStreamingService)
    {
        _logger = logger;
        _eventBus = eventBus;
        _signalRStreamingService = signalRStreamingService;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Subscribe to events
        _eventBus.Subscribe<IDictionary<string, FundingRateDto>>(
            DataCollectionConstants.EventTypes.FundingRatesCollected,
            OnFundingRatesCollectedAsync);

        _eventBus.Subscribe<List<ArbitrageOpportunityDto>>(
            DataCollectionConstants.EventTypes.OpportunitiesEnriched,
            OnOpportunitiesEnrichedAsync);

        _eventBus.Subscribe<IDictionary<string, UserDataSnapshot>>(
            DataCollectionConstants.EventTypes.UserDataCollected,
            OnUserDataCollectedAsync);

        _logger.LogInformation("SignalRBroadcaster started and subscribed to events");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("SignalRBroadcaster stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Called when funding rates are collected
    /// </summary>
    private async Task OnFundingRatesCollectedAsync(DataCollectionEvent<IDictionary<string, FundingRateDto>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No funding rates to broadcast");
            return;
        }

        try
        {
            // Convert dictionary to list
            var fundingRates = @event.Data.Values.ToList();

            // Check for changes using hash
            var newHash = ComputeHash(fundingRates);

            lock (_hashLock)
            {
                if (newHash == _lastFundingRatesHash)
                {
                    _logger.LogDebug("Funding rates unchanged, skipping broadcast");
                    return;
                }

                _lastFundingRatesHash = newHash;
            }

            // Broadcast to all clients
            await _signalRStreamingService.BroadcastFundingRatesAsync(fundingRates);

            _logger.LogInformation("Broadcasted {Count} funding rates to clients", fundingRates.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting funding rates");
        }
    }

    /// <summary>
    /// Called when opportunities are enriched
    /// </summary>
    private async Task OnOpportunitiesEnrichedAsync(DataCollectionEvent<List<ArbitrageOpportunityDto>> @event)
    {
        if (@event.Data == null)
        {
            _logger.LogDebug("No opportunities to broadcast");
            return;
        }

        try
        {
            var opportunities = @event.Data;

            // Check for changes using hash
            var newHash = ComputeHash(opportunities);

            lock (_hashLock)
            {
                if (newHash == _lastOpportunitiesHash)
                {
                    _logger.LogDebug("Opportunities unchanged, skipping broadcast");
                    return;
                }

                _lastOpportunitiesHash = newHash;
            }

            // Broadcast to all clients
            await _signalRStreamingService.BroadcastOpportunitiesAsync(opportunities);

            _logger.LogInformation("Broadcasted {Count} opportunities to clients", opportunities.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting opportunities");
        }
    }

    /// <summary>
    /// Called when user data is collected
    /// </summary>
    private async Task OnUserDataCollectedAsync(DataCollectionEvent<IDictionary<string, UserDataSnapshot>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No user data to broadcast");
            return;
        }

        try
        {
            // Group snapshots by userId
            var userGroups = @event.Data.Values.GroupBy(s => s.UserId);

            foreach (var userGroup in userGroups)
            {
                var userId = userGroup.Key;

                // Extract balances and positions for this user
                var balances = userGroup
                    .Where(s => s.Balance != null)
                    .Select(s => s.Balance!)
                    .ToList();

                var positions = userGroup
                    .SelectMany(s => s.Positions)
                    .ToList();

                // Broadcast balances if changed
                if (balances.Any())
                {
                    var balancesHash = ComputeHash(balances);
                    bool shouldBroadcastBalances = false;

                    lock (_hashLock)
                    {
                        if (!_lastUserBalancesHash.TryGetValue(userId, out var lastHash) || lastHash != balancesHash)
                        {
                            _lastUserBalancesHash[userId] = balancesHash;
                            shouldBroadcastBalances = true;
                        }
                    }

                    if (shouldBroadcastBalances)
                    {
                        // Broadcast balances to this specific user
                        await _signalRStreamingService.BroadcastBalancesToUserAsync(userId, balances);

                        _logger.LogInformation(
                            "Broadcasted {Count} balances to user {UserId}",
                            balances.Count,
                            userId);
                    }
                    else
                    {
                        _logger.LogDebug("Balances unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }

                // Broadcast positions if changed
                if (positions.Any())
                {
                    var positionsHash = ComputeHash(positions);
                    bool shouldBroadcastPositions = false;

                    lock (_hashLock)
                    {
                        if (!_lastUserPositionsHash.TryGetValue(userId, out var lastHash) || lastHash != positionsHash)
                        {
                            _lastUserPositionsHash[userId] = positionsHash;
                            shouldBroadcastPositions = true;
                        }
                    }

                    if (shouldBroadcastPositions)
                    {
                        // Broadcast positions to this specific user
                        await _signalRStreamingService.BroadcastPositionsToUserAsync(userId, positions);

                        _logger.LogInformation(
                            "Broadcasted {Count} positions to user {UserId}",
                            positions.Count,
                            userId);
                    }
                    else
                    {
                        _logger.LogDebug("Positions unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting user data");
        }
    }

    /// <summary>
    /// Computes a hash code for an object to detect changes
    /// </summary>
    private int ComputeHash<T>(T obj)
    {
        if (obj == null)
            return 0;

        try
        {
            // Serialize to JSON and compute hash
            var json = JsonSerializer.Serialize(obj);
            return json.GetHashCode();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error computing hash, using default");
            return obj.GetHashCode();
        }
    }
}
