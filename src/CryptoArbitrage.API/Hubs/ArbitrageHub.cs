using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using System.Security.Claims;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Hubs;

/// <summary>
/// SignalR hub for real-time broadcasting of arbitrage data.
/// CRITICAL: Hub requires authentication - clients must provide JWT token.
/// Uses user-specific groups to isolate data between users.
/// Funding rates are broadcast globally (shared data).
/// Positions, balances, P&L are broadcast to user-specific groups only.
/// </summary>
[Authorize]
public class ArbitrageHub : Hub
{
    private readonly ILogger<ArbitrageHub> _logger;
    private readonly IDataRepository<FundingRateDto> _fundingRateRepository;
    private readonly IDataRepository<UserDataSnapshot> _userDataRepository;
    private readonly IDataRepository<ArbitrageOpportunityDto> _opportunityRepository;
    private readonly IDataRepository<List<OrderDto>> _orderRepository;
    private readonly IDataRepository<List<TradeDto>> _tradeRepository;
    private readonly IDataRepository<List<TransactionDto>> _transactionRepository;

    public ArbitrageHub(
        ILogger<ArbitrageHub> logger,
        IDataRepository<FundingRateDto> fundingRateRepository,
        IDataRepository<UserDataSnapshot> userDataRepository,
        IDataRepository<ArbitrageOpportunityDto> opportunityRepository,
        IDataRepository<List<OrderDto>> orderRepository,
        IDataRepository<List<TradeDto>> tradeRepository,
        IDataRepository<List<TransactionDto>> transactionRepository)
    {
        _logger = logger;
        _fundingRateRepository = fundingRateRepository;
        _userDataRepository = userDataRepository;
        _opportunityRepository = opportunityRepository;
        _orderRepository = orderRepository;
        _tradeRepository = tradeRepository;
        _transactionRepository = transactionRepository;
    }

    /// <summary>
    /// Called when a client connects to the hub.
    /// Adds the connection to a user-specific group for targeted broadcasting.
    /// Broadcasts initial cached data (funding rates, balances, positions, opportunities) asynchronously.
    /// </summary>
    public override async Task OnConnectedAsync()
    {
        // Get userId from authenticated JWT token
        var userId = Context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        var email = Context.User?.FindFirst(ClaimTypes.Email)?.Value;

        if (string.IsNullOrEmpty(userId))
        {
            _logger.LogWarning("Connection attempt without valid user ID - aborting connection");
            Context.Abort();
            return;
        }

        // Add connection to user-specific group
        await Groups.AddToGroupAsync(Context.ConnectionId, $"user_{userId}");

        // Broadcast initial cached data immediately (don't use Task.Run - Hub will be disposed)
        await BroadcastInitialDataAsync(userId);

        await base.OnConnectedAsync();
    }

    /// <summary>
    /// Broadcasts initial cached data when a user connects
    /// </summary>
    private async Task BroadcastInitialDataAsync(string userId)
    {
        try
        {
            var startTime = DateTime.UtcNow;
            _logger.LogDebug("Broadcasting initial cached data to user {UserId}", userId);

            // Fetch all data in parallel for better performance
            var fundingRatesTask = _fundingRateRepository.GetByPatternAsync("funding:*");
            var userDataTask = _userDataRepository.GetByPatternAsync($"user:{userId}:*");
            var opportunitiesTask = _opportunityRepository.GetByPatternAsync("opportunity:*");
            var openOrdersTask = _orderRepository.GetByPatternAsync($"openorders:{userId}:*");
            var orderHistoryTask = _orderRepository.GetByPatternAsync($"orderhistory:{userId}:*");
            var tradeHistoryTask = _tradeRepository.GetByPatternAsync($"tradehistory:{userId}:*");
            var transactionHistoryTask = _transactionRepository.GetByPatternAsync($"transactionhistory:{userId}:*");

            await Task.WhenAll(
                fundingRatesTask, userDataTask, opportunitiesTask,
                openOrdersTask, orderHistoryTask, tradeHistoryTask, transactionHistoryTask
            );

            var fetchTime = (DateTime.UtcNow - startTime).TotalMilliseconds;
            _logger.LogDebug("Fetched all cached data in {Milliseconds}ms for user {UserId}", fetchTime, userId);

            // Broadcast funding rates
            var fundingRatesDict = await fundingRatesTask;
            if (fundingRatesDict.Any())
            {
                var fundingRates = fundingRatesDict.Values.ToList();
                await Clients.Caller.SendAsync("ReceiveFundingRates", fundingRates);
                _logger.LogDebug("Sent {Count} cached funding rates to user {UserId}", fundingRates.Count, userId);
            }

            // Broadcast user data (balances and positions)
            var userDataDict = await userDataTask;
            if (userDataDict.Any())
            {
                var userSnapshots = userDataDict.Values.ToList();

                var balances = userSnapshots
                    .Where(s => s.Balance != null)
                    .Select(s => s.Balance!)
                    .ToList();

                if (balances.Any())
                {
                    await Clients.Caller.SendAsync("ReceiveBalances", balances);
                    _logger.LogDebug("Sent {Count} cached balances to user {UserId}", balances.Count, userId);
                }

                var positions = userSnapshots
                    .SelectMany(s => s.Positions)
                    .ToList();

                if (positions.Any())
                {
                    await Clients.Caller.SendAsync("ReceivePositions", positions);
                    _logger.LogDebug("Sent {Count} cached positions to user {UserId}", positions.Count, userId);
                }
            }

            // Broadcast opportunities
            var opportunitiesDict = await opportunitiesTask;
            if (opportunitiesDict.Any())
            {
                var opportunities = opportunitiesDict.Values.ToList();
                await Clients.Caller.SendAsync("ReceiveOpportunities", opportunities);
                _logger.LogDebug("Sent {Count} cached opportunities to user {UserId}", opportunities.Count, userId);
            }

            // Broadcast open orders
            var openOrdersDict = await openOrdersTask;
            if (openOrdersDict.Any())
            {
                var openOrders = openOrdersDict.Values.SelectMany(list => list).ToList();
                await Clients.Caller.SendAsync("ReceiveOpenOrders", openOrders);
                _logger.LogDebug("Sent {Count} cached open orders to user {UserId}", openOrders.Count, userId);
            }

            // Broadcast order history
            var orderHistoryDict = await orderHistoryTask;
            if (orderHistoryDict.Any())
            {
                var orderHistory = orderHistoryDict.Values.SelectMany(list => list).ToList();
                await Clients.Caller.SendAsync("ReceiveOrderHistory", orderHistory);
                _logger.LogDebug("Sent {Count} cached order history to user {UserId}", orderHistory.Count, userId);
            }

            // Broadcast trade history
            var tradeHistoryDict = await tradeHistoryTask;
            if (tradeHistoryDict.Any())
            {
                var tradeHistory = tradeHistoryDict.Values.SelectMany(list => list).ToList();
                await Clients.Caller.SendAsync("ReceiveTradeHistory", tradeHistory);
                _logger.LogDebug("Sent {Count} cached trade history to user {UserId}", tradeHistory.Count, userId);
            }

            // Broadcast transaction history
            var transactionHistoryDict = await transactionHistoryTask;
            if (transactionHistoryDict.Any())
            {
                var transactionHistory = transactionHistoryDict.Values.SelectMany(list => list).ToList();
                await Clients.Caller.SendAsync("ReceiveTransactionHistory", transactionHistory);
                _logger.LogDebug("Sent {Count} cached transaction history to user {UserId}", transactionHistory.Count, userId);
            }

            var totalTime = (DateTime.UtcNow - startTime).TotalMilliseconds;
            _logger.LogInformation("Initial data broadcast completed in {Milliseconds}ms for user {UserId}", totalTime, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting initial data to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Called when a client disconnects from the hub.
    /// Removes the connection from user-specific group.
    /// </summary>
    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        var userId = Context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (!string.IsNullOrEmpty(userId))
        {
            await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"user_{userId}");
            _logger.LogInformation("User {UserId} disconnected from SignalR (ConnectionId: {ConnectionId})",
                userId, Context.ConnectionId);
        }

        await base.OnDisconnectedAsync(exception);
    }

    /// <summary>
    /// Broadcasts funding rates to ALL connected clients (shared global data).
    /// Called by ArbitrageEngineService for every user to see current funding rates.
    /// </summary>
    public async Task SendFundingRates(List<FundingRateDto> fundingRates)
    {
        await Clients.All.SendAsync("ReceiveFundingRates", fundingRates);
    }

    /// <summary>
    /// Broadcasts positions to a SPECIFIC USER ONLY.
    /// Called by ArbitrageEngineService - each user sees only their own positions.
    /// </summary>
    public async Task SendPositions(string userId, List<PositionDto> positions)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceivePositions", positions);
    }

    /// <summary>
    /// Broadcasts opportunities to a SPECIFIC USER ONLY.
    /// Called when opportunities are detected for a user with enabled API keys.
    /// </summary>
    public async Task SendOpportunities(string userId, List<ArbitrageOpportunityDto> opportunities)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceiveOpportunities", opportunities);
    }

    /// <summary>
    /// Broadcasts account balances to a SPECIFIC USER ONLY.
    /// Each user sees only their own exchange balances.
    /// </summary>
    public async Task SendBalances(string userId, List<AccountBalanceDto> balances)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceiveBalances", balances);
    }

    /// <summary>
    /// Broadcasts dashboard data to a SPECIFIC USER ONLY.
    /// </summary>
    public async Task SendDashboardData(string userId, DashboardDataDto data)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceiveDashboardData", data);
    }

    /// <summary>
    /// Broadcasts P&L update to a SPECIFIC USER ONLY.
    /// Each user sees only their own P&L metrics.
    /// </summary>
    public async Task SendPnLUpdate(string userId, decimal totalPnL, decimal todayPnL)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceivePnLUpdate",
            new { totalPnL, todayPnL, timestamp = DateTime.UtcNow });
    }

    /// <summary>
    /// Broadcasts alert to a SPECIFIC USER ONLY.
    /// Alerts are user-specific (e.g., execution alerts, API errors).
    /// </summary>
    public async Task SendAlert(string userId, string message, string severity)
    {
        // CRITICAL: Only send to the specific user's group
        await Clients.Group($"user_{userId}").SendAsync("ReceiveAlert",
            new { message, severity, timestamp = DateTime.UtcNow });
    }

    // ============================================================================
    // AGENT BROADCAST METHODS
    // ============================================================================

    /// <summary>
    /// Broadcasts agent status update to a SPECIFIC USER ONLY.
    /// Called when agent starts, stops, pauses, resumes, or errors.
    /// </summary>
    public async Task SendAgentStatus(string userId, string status, int? durationSeconds, bool isRunning, string? errorMessage = null)
    {
        await Clients.Group($"user_{userId}").SendAsync("ReceiveAgentStatus",
            new
            {
                status,
                durationSeconds,
                isRunning,
                errorMessage,
                timestamp = DateTime.UtcNow
            });
    }

    /// <summary>
    /// Broadcasts agent statistics update to a SPECIFIC USER ONLY.
    /// Called after each prediction cycle with updated P&L and trade stats.
    /// </summary>
    public async Task SendAgentStats(string userId, object stats)
    {
        await Clients.Group($"user_{userId}").SendAsync("ReceiveAgentStats", stats);
    }

    /// <summary>
    /// Broadcasts agent decision to a SPECIFIC USER ONLY.
    /// Called after each prediction cycle with the action taken.
    /// </summary>
    public async Task SendAgentDecision(string userId, object decision)
    {
        await Clients.Group($"user_{userId}").SendAsync("ReceiveAgentDecision", decision);
    }

    /// <summary>
    /// Broadcasts agent error to a SPECIFIC USER ONLY.
    /// Called when agent encounters an error during operation.
    /// </summary>
    public async Task SendAgentError(string userId, string error)
    {
        await Clients.Group($"user_{userId}").SendAsync("ReceiveAgentError",
            new { error, timestamp = DateTime.UtcNow });
    }
}
