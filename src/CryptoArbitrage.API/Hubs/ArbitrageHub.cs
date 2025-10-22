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

    public ArbitrageHub(
        ILogger<ArbitrageHub> logger,
        IDataRepository<FundingRateDto> fundingRateRepository,
        IDataRepository<UserDataSnapshot> userDataRepository,
        IDataRepository<ArbitrageOpportunityDto> opportunityRepository)
    {
        _logger = logger;
        _fundingRateRepository = fundingRateRepository;
        _userDataRepository = userDataRepository;
        _opportunityRepository = opportunityRepository;
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

        _logger.LogInformation(
            "User {UserId} ({Email}) connected to SignalR (ConnectionId: {ConnectionId})",
            userId, email, Context.ConnectionId);

        // Broadcast initial cached data in background to avoid blocking connection
        _ = Task.Run(async () =>
        {
            try
            {
                await BroadcastInitialDataAsync(userId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in background broadcast for user {UserId}", userId);
            }
        });

        await base.OnConnectedAsync();
    }

    /// <summary>
    /// Broadcasts initial cached data when a user connects
    /// </summary>
    private async Task BroadcastInitialDataAsync(string userId)
    {
        try
        {
            _logger.LogDebug("Broadcasting initial cached data to user {UserId}", userId);

            // Broadcast cached funding rates (global data)
            var fundingRatesDict = await _fundingRateRepository.GetByPatternAsync("funding:*");
            if (fundingRatesDict.Any())
            {
                var fundingRates = fundingRatesDict.Values.ToList();
                await Clients.Caller.SendAsync("ReceiveFundingRates", fundingRates);
                _logger.LogDebug("Sent {Count} cached funding rates to user {UserId}", fundingRates.Count, userId);
            }

            // Broadcast cached user data (user-specific)
            var userDataDict = await _userDataRepository.GetByPatternAsync($"user:{userId}:*");
            if (userDataDict.Any())
            {
                var userSnapshots = userDataDict.Values.ToList();

                // Extract and send balances
                var balances = userSnapshots
                    .Where(s => s.Balance != null)
                    .Select(s => s.Balance!)
                    .ToList();

                if (balances.Any())
                {
                    await Clients.Caller.SendAsync("ReceiveBalances", balances);
                    _logger.LogDebug("Sent {Count} cached balances to user {UserId}", balances.Count, userId);
                }

                // Extract and send positions
                var positions = userSnapshots
                    .SelectMany(s => s.Positions)
                    .ToList();

                if (positions.Any())
                {
                    await Clients.Caller.SendAsync("ReceivePositions", positions);
                    _logger.LogDebug("Sent {Count} cached positions to user {UserId}", positions.Count, userId);
                }
            }

            // Broadcast cached opportunities (global or user-specific)
            var opportunitiesDict = await _opportunityRepository.GetByPatternAsync("opportunity:*");
            if (opportunitiesDict.Any())
            {
                var opportunities = opportunitiesDict.Values.ToList();
                await Clients.Caller.SendAsync("ReceiveOpportunities", opportunities);
                _logger.LogDebug("Sent {Count} cached opportunities to user {UserId}", opportunities.Count, userId);
            }

            _logger.LogInformation("Initial data broadcast completed for user {UserId}", userId);
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
}
