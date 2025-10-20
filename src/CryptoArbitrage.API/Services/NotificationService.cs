using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace CryptoArbitrage.API.Services;

public class NotificationService : INotificationService
{
    private readonly ArbitrageDbContext _context;
    private readonly ISignalRStreamingService _signalRService;
    private readonly ILogger<NotificationService> _logger;
    private readonly NotificationSettings _settings;

    // Track which executions have already triggered negative funding notifications to avoid spam
    private readonly Dictionary<int, DateTime> _negativeFundingNotificationsSent = new();
    private readonly TimeSpan _notificationCooldown = TimeSpan.FromMinutes(15); // Don't spam same notification

    public NotificationService(
        ArbitrageDbContext context,
        ISignalRStreamingService signalRService,
        ILogger<NotificationService> logger,
        IOptions<NotificationSettings> settings)
    {
        _context = context;
        _signalRService = signalRService;
        _logger = logger;
        _settings = settings.Value;
    }

    public async Task CheckNegativeFundingForUserAsync(string userId)
    {
        try
        {
            // Get all running executions for this user
            var activeExecutions = await _context.Executions
                .Where(e => e.UserId == userId && e.State == ExecutionState.Running)
                .ToListAsync();

            foreach (var execution in activeExecutions)
            {
                // Get all open positions for this execution
                var positions = await _context.Positions
                    .Where(p => p.ExecutionId == execution.Id && p.Status == PositionStatus.Open)
                    .ToListAsync();

                if (!positions.Any())
                    continue;

                // Calculate net funding fee (TotalReceived - TotalPaid)
                var netFunding = positions.Sum(p => p.NetFundingFee);

                // If negative funding and not recently notified, send alert
                if (netFunding < 0)
                {
                    // Check cooldown
                    if (_negativeFundingNotificationsSent.TryGetValue(execution.Id, out var lastNotification))
                    {
                        if (DateTime.UtcNow - lastNotification < _notificationCooldown)
                            continue; // Skip, too soon since last notification
                    }

                    // Create notification
                    var notification = new NotificationDto
                    {
                        Type = NotificationType.NegativeFunding,
                        Severity = NotificationSeverity.Warning,
                        Title = "Execution Losing Money on Funding",
                        Message = $"{execution.Symbol} on {execution.Exchange}: Net funding ${netFunding:F2} (negative). Positions are paying more in funding fees than receiving.",
                        Data = new
                        {
                            executionId = execution.Id,
                            symbol = execution.Symbol,
                            exchange = execution.Exchange,
                            netFunding = netFunding,
                            positionsCount = positions.Count
                        },
                        AutoClose = false, // Persistent notification for critical issue
                        AutoCloseDelay = null
                    };

                    await _signalRService.SendNotificationAsync(userId, notification);
                    _negativeFundingNotificationsSent[execution.Id] = DateTime.UtcNow;

                    _logger.LogInformation(
                        "Sent negative funding notification for execution {ExecutionId} ({Symbol} on {Exchange}): {NetFunding}",
                        execution.Id, execution.Symbol, execution.Exchange, netFunding);
                }
                else
                {
                    // If funding is now positive, remove from tracking
                    _negativeFundingNotificationsSent.Remove(execution.Id);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking negative funding for user {UserId}", userId);
        }
    }

    public async Task NotifyExecutionStateChangeAsync(
        string userId,
        int executionId,
        string symbol,
        string exchange,
        ExecutionState oldState,
        ExecutionState newState)
    {
        try
        {
            var (severity, autoClose) = newState switch
            {
                ExecutionState.Running => (NotificationSeverity.Success, true),
                ExecutionState.Stopped => (NotificationSeverity.Info, true),
                ExecutionState.Failed => (NotificationSeverity.Error, false),
                _ => (NotificationSeverity.Info, true)
            };

            var notification = new NotificationDto
            {
                Type = NotificationType.ExecutionStateChange,
                Severity = severity,
                Title = $"Execution {newState}",
                Message = $"{symbol} on {exchange} execution changed from {oldState} to {newState}",
                Data = new
                {
                    executionId,
                    symbol,
                    exchange,
                    oldState = oldState.ToString(),
                    newState = newState.ToString()
                },
                AutoClose = autoClose,
                AutoCloseDelay = autoClose ? 5000 : null
            };

            await _signalRService.SendNotificationAsync(userId, notification);

            _logger.LogInformation(
                "Sent execution state change notification: {Symbol} on {Exchange} changed from {OldState} to {NewState}",
                symbol, exchange, oldState, newState);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending execution state change notification");
        }
    }

    public async Task NotifyExchangeConnectivityAsync(
        string userId,
        string exchange,
        bool isConnected,
        string? errorMessage = null)
    {
        try
        {
            var notification = new NotificationDto
            {
                Type = NotificationType.ExchangeConnectivity,
                Severity = isConnected ? NotificationSeverity.Success : NotificationSeverity.Error,
                Title = isConnected ? "Exchange Reconnected" : "Exchange Disconnected",
                Message = isConnected
                    ? $"{exchange} API connection restored"
                    : $"{exchange} API disconnected{(errorMessage != null ? $": {errorMessage}" : "")}",
                Data = new
                {
                    exchange,
                    isConnected,
                    errorMessage
                },
                AutoClose = isConnected,
                AutoCloseDelay = isConnected ? 5000 : null
            };

            await _signalRService.SendNotificationAsync(userId, notification);

            _logger.LogInformation(
                "Sent exchange connectivity notification: {Exchange} {Status}",
                exchange, isConnected ? "connected" : "disconnected");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending exchange connectivity notification");
        }
    }

    public async Task NotifyLargeOpportunityAsync(
        string userId,
        string symbol,
        decimal annualizedSpread,
        decimal estimatedProfit)
    {
        try
        {
            // Only notify if spread exceeds configured threshold
            if (annualizedSpread < _settings.LargeOpportunityThresholdPercent)
                return;

            var notification = new NotificationDto
            {
                Type = NotificationType.OpportunityDetected,
                Severity = NotificationSeverity.Success,
                Title = "Large Arbitrage Opportunity",
                Message = $"{symbol}: {annualizedSpread:F2}% annualized spread detected! Estimated profit: ${estimatedProfit:F2}",
                Data = new
                {
                    symbol,
                    annualizedSpread,
                    estimatedProfit
                },
                AutoClose = true,
                AutoCloseDelay = 8000 // 8 seconds for important opportunities
            };

            await _signalRService.SendNotificationAsync(userId, notification);

            _logger.LogInformation(
                "Sent large opportunity notification: {Symbol} with {Spread}% spread",
                symbol, annualizedSpread);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending large opportunity notification");
        }
    }

    public async Task NotifyLiquidationRiskAsync(
        string userId,
        string symbol,
        string exchange,
        decimal marginPercent)
    {
        try
        {
            var notification = new NotificationDto
            {
                Type = NotificationType.LiquidationRisk,
                Severity = NotificationSeverity.Error,
                Title = "Liquidation Risk Alert",
                Message = $"{symbol} on {exchange} position at risk - only {marginPercent:F1}% from liquidation price!",
                Data = new
                {
                    symbol,
                    exchange,
                    marginPercent
                },
                AutoClose = false, // Critical alert, must be manually dismissed
                AutoCloseDelay = null
            };

            await _signalRService.SendNotificationAsync(userId, notification);

            _logger.LogWarning(
                "Sent liquidation risk notification: {Symbol} on {Exchange} at {MarginPercent}% from liquidation",
                symbol, exchange, marginPercent);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending liquidation risk notification");
        }
    }
}

public class NotificationSettings
{
    public decimal LargeOpportunityThresholdPercent { get; set; } = 10.0m; // 10% annualized
    public decimal LiquidationRiskMarginPercent { get; set; } = 20.0m; // Alert when within 20% of liquidation
    public int CheckIntervalSeconds { get; set; } = 30;
}
