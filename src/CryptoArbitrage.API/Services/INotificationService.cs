using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services;

public interface INotificationService
{
    /// <summary>
    /// Check if any active executions have negative funding and send notifications
    /// </summary>
    Task CheckNegativeFundingForUserAsync(string userId);

    /// <summary>
    /// Send a notification for execution state change
    /// </summary>
    Task NotifyExecutionStateChangeAsync(string userId, int executionId, string symbol, string exchange, ExecutionState oldState, ExecutionState newState);

    /// <summary>
    /// Send a notification for exchange connectivity issues
    /// </summary>
    Task NotifyExchangeConnectivityAsync(string userId, string exchange, bool isConnected, string? errorMessage = null);

    /// <summary>
    /// Send a notification for large arbitrage opportunity detected
    /// </summary>
    Task NotifyLargeOpportunityAsync(string userId, string symbol, decimal annualizedSpread, decimal estimatedProfit);

    /// <summary>
    /// Send a notification for liquidation risk
    /// </summary>
    Task NotifyLiquidationRiskAsync(string userId, string symbol, string exchange, decimal marginPercent);
}
