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
    private readonly Dictionary<string, int> _lastUserOpenOrdersHash = new();
    private readonly Dictionary<string, int> _lastUserOrderHistoryHash = new();
    private readonly Dictionary<string, int> _lastUserTradeHistoryHash = new();
    private readonly Dictionary<string, int> _lastUserTransactionHistoryHash = new();

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

        _eventBus.Subscribe<IDictionary<string, List<OrderDto>>>(
            DataCollectionConstants.EventTypes.OpenOrdersCollected,
            OnOpenOrdersCollectedAsync);

        _eventBus.Subscribe<IDictionary<string, List<OrderDto>>>(
            DataCollectionConstants.EventTypes.OrderHistoryCollected,
            OnOrderHistoryCollectedAsync);

        _eventBus.Subscribe<IDictionary<string, List<TradeDto>>>(
            DataCollectionConstants.EventTypes.TradeHistoryCollected,
            OnTradeHistoryCollectedAsync);

        _eventBus.Subscribe<IDictionary<string, List<TransactionDto>>>(
            DataCollectionConstants.EventTypes.TransactionHistoryCollected,
            OnTransactionHistoryCollectedAsync);

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
    /// Called when open orders are collected
    /// </summary>
    private async Task OnOpenOrdersCollectedAsync(DataCollectionEvent<IDictionary<string, List<OrderDto>>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No open orders to broadcast");
            return;
        }

        try
        {
            // Each key is "openorders:{userId}:{exchange}"
            foreach (var kvp in @event.Data)
            {
                // Parse userId from key (format: openorders:userId:exchange)
                var parts = kvp.Key.Split(':');
                if (parts.Length >= 2)
                {
                    var userId = parts[1];
                    var orders = kvp.Value;

                    // Check for changes using hash
                    var ordersHash = ComputeHash(orders);
                    bool shouldBroadcast = false;

                    lock (_hashLock)
                    {
                        var cacheKey = $"{userId}_openorders";
                        if (!_lastUserOpenOrdersHash.TryGetValue(cacheKey, out var lastHash) || lastHash != ordersHash)
                        {
                            _lastUserOpenOrdersHash[cacheKey] = ordersHash;
                            shouldBroadcast = true;
                        }
                    }

                    if (shouldBroadcast)
                    {
                        await _signalRStreamingService.BroadcastOpenOrdersToUserAsync(userId, orders);
                        _logger.LogInformation("Broadcasted {Count} open orders to user {UserId}", orders.Count, userId);
                    }
                    else
                    {
                        _logger.LogDebug("Open orders unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting open orders");
        }
    }

    /// <summary>
    /// Called when order history is collected
    /// </summary>
    private async Task OnOrderHistoryCollectedAsync(DataCollectionEvent<IDictionary<string, List<OrderDto>>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No order history to broadcast");
            return;
        }

        try
        {
            // Each key is "orderhistory:{userId}:{exchange}"
            foreach (var kvp in @event.Data)
            {
                // Parse userId from key
                var parts = kvp.Key.Split(':');
                if (parts.Length >= 2)
                {
                    var userId = parts[1];
                    var orders = kvp.Value;

                    // Check for changes using hash
                    var ordersHash = ComputeHash(orders);
                    bool shouldBroadcast = false;

                    lock (_hashLock)
                    {
                        var cacheKey = $"{userId}_orderhistory";
                        if (!_lastUserOrderHistoryHash.TryGetValue(cacheKey, out var lastHash) || lastHash != ordersHash)
                        {
                            _lastUserOrderHistoryHash[cacheKey] = ordersHash;
                            shouldBroadcast = true;
                        }
                    }

                    if (shouldBroadcast)
                    {
                        await _signalRStreamingService.BroadcastOrderHistoryToUserAsync(userId, orders);
                        _logger.LogInformation("Broadcasted {Count} historical orders to user {UserId}", orders.Count, userId);
                    }
                    else
                    {
                        _logger.LogDebug("Order history unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting order history");
        }
    }

    /// <summary>
    /// Called when trade history is collected
    /// </summary>
    private async Task OnTradeHistoryCollectedAsync(DataCollectionEvent<IDictionary<string, List<TradeDto>>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No trade history to broadcast");
            return;
        }

        try
        {
            // Each key is "tradehistory:{userId}:{exchange}"
            foreach (var kvp in @event.Data)
            {
                // Parse userId from key
                var parts = kvp.Key.Split(':');
                if (parts.Length >= 2)
                {
                    var userId = parts[1];
                    var trades = kvp.Value;

                    // Check for changes using hash
                    var tradesHash = ComputeHash(trades);
                    bool shouldBroadcast = false;

                    lock (_hashLock)
                    {
                        var cacheKey = $"{userId}_tradehistory";
                        if (!_lastUserTradeHistoryHash.TryGetValue(cacheKey, out var lastHash) || lastHash != tradesHash)
                        {
                            _lastUserTradeHistoryHash[cacheKey] = tradesHash;
                            shouldBroadcast = true;
                        }
                    }

                    if (shouldBroadcast)
                    {
                        await _signalRStreamingService.BroadcastTradeHistoryToUserAsync(userId, trades);
                        _logger.LogInformation("Broadcasted {Count} trades to user {UserId}", trades.Count, userId);
                    }
                    else
                    {
                        _logger.LogDebug("Trade history unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting trade history");
        }
    }

    /// <summary>
    /// Called when transaction history is collected
    /// </summary>
    private async Task OnTransactionHistoryCollectedAsync(DataCollectionEvent<IDictionary<string, List<TransactionDto>>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No transaction history to broadcast");
            return;
        }

        try
        {
            // Each key is "transactionhistory:{userId}:{exchange}"
            foreach (var kvp in @event.Data)
            {
                // Parse userId from key
                var parts = kvp.Key.Split(':');
                if (parts.Length >= 2)
                {
                    var userId = parts[1];
                    var transactions = kvp.Value;

                    // Check for changes using hash
                    var transactionsHash = ComputeHash(transactions);
                    bool shouldBroadcast = false;

                    lock (_hashLock)
                    {
                        var cacheKey = $"{userId}_transactionhistory";
                        if (!_lastUserTransactionHistoryHash.TryGetValue(cacheKey, out var lastHash) || lastHash != transactionsHash)
                        {
                            _lastUserTransactionHistoryHash[cacheKey] = transactionsHash;
                            shouldBroadcast = true;
                        }
                    }

                    if (shouldBroadcast)
                    {
                        await _signalRStreamingService.BroadcastTransactionHistoryToUserAsync(userId, transactions);
                        _logger.LogInformation("Broadcasted {Count} transactions to user {UserId}", transactions.Count, userId);
                    }
                    else
                    {
                        _logger.LogDebug("Transaction history unchanged for user {UserId}, skipping broadcast", userId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting transaction history");
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
