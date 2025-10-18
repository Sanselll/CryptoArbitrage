using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using System.Security.Claims;
using CryptoArbitrage.API.Models;

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

    public ArbitrageHub(ILogger<ArbitrageHub> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Called when a client connects to the hub.
    /// Adds the connection to a user-specific group for targeted broadcasting.
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

        await base.OnConnectedAsync();
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
