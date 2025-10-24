using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Services.Streaming;

/// <summary>
/// Service responsible for streaming data to clients via SignalR
/// </summary>
public interface ISignalRStreamingService
{
    /// <summary>
    /// Broadcast funding rates to all connected clients
    /// </summary>
    Task BroadcastFundingRatesAsync(List<FundingRateDto> rates, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast opportunities to all connected clients
    /// </summary>
    Task BroadcastOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast positions to a specific user
    /// </summary>
    Task BroadcastPositionsToUserAsync(string userId, List<PositionDto> positions, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast balances to a specific user
    /// </summary>
    Task BroadcastBalancesToUserAsync(string userId, List<AccountBalanceDto> balances, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast P&L update to a specific user
    /// </summary>
    Task BroadcastPnLToUserAsync(string userId, decimal totalPnL, decimal todayPnL, CancellationToken cancellationToken = default);

    /// <summary>
    /// Send a notification to a specific user
    /// </summary>
    Task SendNotificationAsync(string userId, NotificationDto notification, CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if a user broadcast happened recently (for debouncing refresh loops)
    /// </summary>
    bool ShouldSkipUserRefresh(string userId, double debounceSeconds = 2.0);

    /// <summary>
    /// Broadcast open orders to a specific user
    /// </summary>
    Task BroadcastOpenOrdersToUserAsync(string userId, List<OrderDto> orders, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast order history to a specific user
    /// </summary>
    Task BroadcastOrderHistoryToUserAsync(string userId, List<OrderDto> orders, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast trade history to a specific user
    /// </summary>
    Task BroadcastTradeHistoryToUserAsync(string userId, List<TradeDto> trades, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast transaction history to a specific user
    /// </summary>
    Task BroadcastTransactionHistoryToUserAsync(string userId, List<TransactionDto> transactions, CancellationToken cancellationToken = default);

    /// <summary>
    /// Broadcast exit signal to a specific user for a position
    /// </summary>
    Task BroadcastExitSignalToUserAsync(string userId, int positionId, ExitSignal signal, CancellationToken cancellationToken = default);
}
