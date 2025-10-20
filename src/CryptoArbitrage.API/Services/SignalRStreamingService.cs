using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Models;
using Microsoft.AspNetCore.SignalR;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service responsible for streaming data to clients via SignalR
/// </summary>
public class SignalRStreamingService : ISignalRStreamingService
{
    private readonly IHubContext<ArbitrageHub> _hubContext;
    private readonly ILogger<SignalRStreamingService> _logger;

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
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting positions to user {UserId}", userId);
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
}
