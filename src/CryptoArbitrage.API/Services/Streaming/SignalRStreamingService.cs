using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Models;
using Microsoft.AspNetCore.SignalR;

namespace CryptoArbitrage.API.Services.Streaming;

/// <summary>
/// Service responsible for streaming data to clients via SignalR
/// </summary>
public class SignalRStreamingService : ISignalRStreamingService
{
    private readonly IHubContext<ArbitrageHub> _hubContext;
    private readonly ILogger<SignalRStreamingService> _logger;

    // Track last broadcast time per user to prevent race conditions with immediate broadcasts
    private readonly Dictionary<string, DateTime> _lastUserBroadcast = new();
    private readonly object _broadcastTimestampLock = new();

    public SignalRStreamingService(
        IHubContext<ArbitrageHub> hubContext,
        ILogger<SignalRStreamingService> logger)
    {
        _hubContext = hubContext;
        _logger = logger;
    }

    /// <summary>
    /// Broadcast funding rates to all connected clients
    /// </summary>
    public async Task BroadcastFundingRatesAsync(List<FundingRateDto> rates, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.All.SendAsync("ReceiveFundingRates", rates, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} funding rates to all clients", rates.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting funding rates");
        }
    }

    /// <summary>
    /// Broadcast opportunities to all connected clients
    /// </summary>
    public async Task BroadcastOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.All.SendAsync("ReceiveOpportunities", opportunities, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} opportunities to all clients", opportunities.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting opportunities");
        }
    }

    /// <summary>
    /// Broadcast positions to a specific user
    /// </summary>
    public async Task BroadcastPositionsToUserAsync(string userId, List<PositionDto> positions, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceivePositions", positions, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} positions to user {UserId}", positions.Count, userId);

            // Record broadcast timestamp to enable debouncing in refresh loops
            lock (_broadcastTimestampLock)
            {
                _lastUserBroadcast[userId] = DateTime.UtcNow;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting positions to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Check if a user broadcast happened recently (within last N seconds)
    /// Used for debouncing refresh loops to prevent race conditions
    /// </summary>
    public bool ShouldSkipUserRefresh(string userId, double debounceSeconds = 2.0)
    {
        lock (_broadcastTimestampLock)
        {
            if (_lastUserBroadcast.TryGetValue(userId, out var lastBroadcast))
            {
                var timeSince = DateTime.UtcNow - lastBroadcast;
                return timeSince.TotalSeconds < debounceSeconds;
            }
            return false;
        }
    }

    /// <summary>
    /// Broadcast balances to a specific user
    /// </summary>
    public async Task BroadcastBalancesToUserAsync(string userId, List<AccountBalanceDto> balances, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveBalances", balances, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} balances to user {UserId}", balances.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting balances to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast P&L update to a specific user
    /// </summary>
    public async Task BroadcastPnLToUserAsync(string userId, decimal totalPnL, decimal todayPnL, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceivePnLUpdate", new { TotalPnL = totalPnL, TodayPnL = todayPnL }, cancellationToken);
            _logger.LogDebug("Broadcasted P&L to user {UserId}: Total={Total}, Today={Today}", userId, totalPnL, todayPnL);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting P&L to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Send a notification to a specific user
    /// </summary>
    public async Task SendNotificationAsync(string userId, NotificationDto notification, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveNotification", notification, cancellationToken);
            _logger.LogDebug("Sent notification to user {UserId}: {Type} - {Title}", userId, notification.Type, notification.Title);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending notification to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast open orders to a specific user
    /// </summary>
    public async Task BroadcastOpenOrdersToUserAsync(string userId, List<OrderDto> orders, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveOpenOrders", orders, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} open orders to user {UserId}", orders.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting open orders to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast order history to a specific user
    /// </summary>
    public async Task BroadcastOrderHistoryToUserAsync(string userId, List<OrderDto> orders, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveOrderHistory", orders, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} historical orders to user {UserId}", orders.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting order history to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast trade history to a specific user
    /// </summary>
    public async Task BroadcastTradeHistoryToUserAsync(string userId, List<TradeDto> trades, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveTradeHistory", trades, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} trades to user {UserId}", trades.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting trade history to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast transaction history to a specific user
    /// </summary>
    public async Task BroadcastTransactionHistoryToUserAsync(string userId, List<TransactionDto> transactions, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveTransactionHistory", transactions, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} transactions to user {UserId}", transactions.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting transaction history to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast agent status to a specific user
    /// </summary>
    public async Task BroadcastAgentStatusAsync(string userId, string status, int durationSeconds, bool hasData, string? errorMessage, CancellationToken cancellationToken = default)
    {
        try
        {
            var statusUpdate = new
            {
                status,
                durationSeconds,
                hasData,
                errorMessage
            };

            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveAgentStatus", statusUpdate, cancellationToken);
            _logger.LogDebug("Broadcasted agent status to user {UserId}: {Status}", userId, status);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting agent status to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast agent stats to a specific user
    /// </summary>
    public async Task BroadcastAgentStatsAsync(string userId, object stats, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveAgentStats", stats, cancellationToken);
            _logger.LogDebug("Broadcasted agent stats to user {UserId}", userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting agent stats to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast agent decision to a specific user
    /// </summary>
    public async Task BroadcastAgentDecisionAsync(string userId, object decision, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveAgentDecision", decision, cancellationToken);
            _logger.LogDebug("Broadcasted agent decision to user {UserId}", userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting agent decision to user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast execution history to a specific user
    /// </summary>
    public async Task BroadcastExecutionHistoryToUserAsync(string userId, List<ExecutionHistoryDto> history, CancellationToken cancellationToken = default)
    {
        try
        {
            await _hubContext.Clients.Group($"user_{userId}").SendAsync("ReceiveExecutionHistory", history, cancellationToken);
            _logger.LogDebug("Broadcasted {Count} execution history entries to user {UserId}", history.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting execution history to user {UserId}", userId);
        }
    }
}
