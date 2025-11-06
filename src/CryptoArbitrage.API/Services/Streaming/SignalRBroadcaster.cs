using System.Text.Json;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Events;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.ML;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Data;
using Microsoft.EntityFrameworkCore;

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
    private readonly RLPredictionService _rlPredictionService;
    private readonly IServiceScopeFactory _serviceScopeFactory;

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
        ISignalRStreamingService signalRStreamingService,
        RLPredictionService rlPredictionService,
        IServiceScopeFactory serviceScopeFactory)
    {
        _logger = logger;
        _eventBus = eventBus;
        _signalRStreamingService = signalRStreamingService;
        _rlPredictionService = rlPredictionService;
        _serviceScopeFactory = serviceScopeFactory;
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
                        // Enrich open positions with RL predictions
                        var openPositions = positions.Where(p => p.Status == PositionStatus.Open).ToList();

                        if (openPositions.Any())
                        {
                            try
                            {
                                _logger.LogDebug("Enriching {Count} open positions with RL predictions for user {UserId}",
                                    openPositions.Count, userId);

                                // Fetch current opportunities from cache
                                List<ArbitrageOpportunityDto>? currentOpportunities = null;
                                try
                                {
                                    using var scope = _serviceScopeFactory.CreateScope();
                                    var opportunityRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<ArbitrageOpportunityDto>>();

                                    var opportunitiesDict = await opportunityRepository.GetByPatternAsync("opportunity:*");
                                    currentOpportunities = opportunitiesDict.Values.ToList();

                                    _logger.LogDebug("Fetched {Count} current opportunities for RL prediction", currentOpportunities.Count);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogWarning(ex, "Failed to fetch opportunities for RL prediction, will proceed without them");
                                }

                                var portfolioState = await BuildPortfolioStateAsync(userId);

                                var rlPredictions = await _rlPredictionService.EvaluatePositionsAsync(
                                    openPositions,
                                    portfolioState,
                                    currentOpportunities
                                );

                                // RL prediction enrichment removed - fields no longer exist in PositionDto

                                _logger.LogInformation("Successfully enriched {Count} positions with RL predictions for user {UserId}",
                                    rlPredictions.Count, userId);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogError(ex, "Failed to enrich positions with RL predictions for user {UserId}", userId);
                            }
                        }

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

    /// <summary>
    /// Build portfolio state for RL prediction (for a specific user)
    /// </summary>
    private async Task<RLPortfolioState> BuildPortfolioStateAsync(string userId)
    {
        try
        {
            using var scope = _serviceScopeFactory.CreateScope();
            var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            var openPositions = await dbContext.Positions
                .Include(p => p.Transactions)
                .Where(p => p.UserId == userId && p.Status == PositionStatus.Open)
                .ToListAsync();

            var initialCapital = 10000m;  // TODO: Get from user settings
            var currentCapital = initialCapital;  // TODO: Calculate actual current capital

            return new RLPortfolioState
            {
                Capital = currentCapital,
                InitialCapital = initialCapital,
                NumPositions = openPositions.Count,
                Utilization = (float)(openPositions.Sum(p => p.InitialMargin) / currentCapital),
                TotalPnlPct = (float)(openPositions.Sum(p => p.UnrealizedPnL) / initialCapital * 100),
                Drawdown = 0.0f,  // TODO: Calculate drawdown
                Positions = openPositions.Take(3).Select(p =>
                {
                    var hoursHeld = (DateTime.UtcNow - p.OpenedAt).TotalHours;

                    // Calculate net funding fee from transactions
                    var fundingReceived = p.Transactions
                        .Where(t => t.TransactionType == TransactionType.FundingFee && t.Amount > 0)
                        .Sum(t => t.Amount);
                    var fundingPaid = p.Transactions
                        .Where(t => t.TransactionType == TransactionType.FundingFee && t.Amount < 0)
                        .Sum(t => Math.Abs(t.Amount));
                    var netFunding = fundingReceived - fundingPaid;

                    var fundingRate = hoursHeld > 0
                        ? (float)(netFunding / p.InitialMargin / (decimal)hoursHeld * 8 * 100)
                        : 0.0f;

                    return new RLPositionState
                    {
                        PnlPct = (float)(p.UnrealizedPnL / p.InitialMargin * 100),
                        HoursHeld = (float)hoursHeld,
                        FundingRate = fundingRate
                    };
                }).ToList()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to build portfolio state for RL predictions for user {UserId}", userId);

            // Return default state if error
            return new RLPortfolioState
            {
                Capital = 10000m,
                InitialCapital = 10000m,
                NumPositions = 0,
                Utilization = 0f,
                TotalPnlPct = 0f,
                Drawdown = 0f,
                Positions = new List<RLPositionState>()
            };
        }
    }
}
