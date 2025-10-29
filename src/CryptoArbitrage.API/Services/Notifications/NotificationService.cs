using System.Collections.Concurrent;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.Streaming;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;

namespace CryptoArbitrage.API.Services.Notifications;

public class NotificationService : INotificationService
{
    private readonly ArbitrageDbContext _context;
    private readonly ISignalRStreamingService _signalRService;
    private readonly ILogger<NotificationService> _logger;
    private readonly NotificationSettings _settings;
    private readonly IMemoryCache _cache;
    private const string CACHE_KEY_PREFIX_FUNDING = "funding_";
    private const string CACHE_KEY_PREFIX_NOTIFICATION_COOLDOWN = "notification_cooldown_";

    private readonly TimeSpan _notificationCooldown = TimeSpan.FromMinutes(15); // Don't spam same notification

    // Static dictionary to ensure singleton locks per execution ID across all service instances
    private static readonly ConcurrentDictionary<int, SemaphoreSlim> _executionLocks = new();

    public NotificationService(
        ArbitrageDbContext context,
        ISignalRStreamingService signalRService,
        ILogger<NotificationService> logger,
        IOptions<NotificationSettings> settings,
        IMemoryCache cache)
    {
        _context = context;
        _signalRService = signalRService;
        _logger = logger;
        _settings = settings.Value;
        _cache = cache;
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

                // Calculate historical net funding fee from PositionTransactions
                // Net funding = (received - paid) for each position
                var historicalNetFunding = 0m;

                // Calculate estimated funding for next settlement
                decimal estimatedFunding = 0;

                // Determine strategy type based on positions
                var perpPositions = positions.Where(p => p.Type == PositionType.Perpetual).ToList();
                var spotPositions = positions.Where(p => p.Type == PositionType.Spot).ToList();

                if (perpPositions.Count == 2)
                {
                    // Cross-Fut: Two perpetual positions (long and short)
                    var longPerp = perpPositions.FirstOrDefault(p => p.Side == PositionSide.Long);
                    var shortPerp = perpPositions.FirstOrDefault(p => p.Side == PositionSide.Short);

                    if (longPerp != null && shortPerp != null)
                    {
                        // Get funding rates from cache
                        var longCacheKey = $"{CACHE_KEY_PREFIX_FUNDING}{longPerp.Exchange}:{execution.Symbol}";
                        var shortCacheKey = $"{CACHE_KEY_PREFIX_FUNDING}{shortPerp.Exchange}:{execution.Symbol}";

                        decimal longRate = 0;
                        decimal shortRate = 0;

                        if (_cache.TryGetValue<FundingRateCacheEntry>(longCacheKey, out var longEntry))
                        {
                            longRate = longEntry.CurrentRate.Rate;
                        }

                        if (_cache.TryGetValue<FundingRateCacheEntry>(shortCacheKey, out var shortEntry))
                        {
                            shortRate = shortEntry.CurrentRate.Rate;
                        }

                        var longValue = longPerp.Quantity * longPerp.EntryPrice;
                        var shortValue = shortPerp.Quantity * shortPerp.EntryPrice;

                        // Long: negative rate = receive (positive), positive rate = pay (negative)
                        // Short: negative rate = pay (negative), positive rate = receive (positive)
                        estimatedFunding = -longRate * longValue + shortRate * shortValue;
                    }
                }
                else if (perpPositions.Count == 1)
                {
                    // Spot-Perp or Cross-Spot: One perpetual position
                    var perpPosition = perpPositions.First();

                    // Get funding rate from cache
                    var cacheKey = $"{CACHE_KEY_PREFIX_FUNDING}{execution.Exchange}:{execution.Symbol}";
                    decimal rate = 0;

                    if (_cache.TryGetValue<FundingRateCacheEntry>(cacheKey, out var entry))
                    {
                        rate = entry.CurrentRate.Rate;
                    }

                    var perpValue = perpPosition.Quantity * perpPosition.EntryPrice;
                    var perpSide = perpPosition.Side;

                    // Long: negative rate = receive (positive), positive rate = pay (negative)
                    // Short: negative rate = pay (negative), positive rate = receive (positive)
                    estimatedFunding = rate * perpValue * (perpSide == PositionSide.Long ? -1 : 1);
                }

                // Total funding P&L = historical + estimated
                var totalFundingPnL = historicalNetFunding + estimatedFunding;

                // If negative funding P&L and not recently notified, send alert
                if (totalFundingPnL < 0)
                {
                    // Get or create a singleton lock for this execution ID
                    var lockObj = _executionLocks.GetOrAdd(execution.Id, _ => new SemaphoreSlim(1, 1));

                    await lockObj.WaitAsync();
                    try
                    {
                        // Check cooldown using cache (double-checked locking pattern)
                        var cooldownCacheKey = $"{CACHE_KEY_PREFIX_NOTIFICATION_COOLDOWN}{execution.Id}";
                        if (_cache.TryGetValue<DateTime>(cooldownCacheKey, out var lastNotification))
                        {
                            if (DateTime.UtcNow - lastNotification < _notificationCooldown)
                            {
                                continue; // Skip, too soon since last notification
                            }
                        }

                        // Set cooldown BEFORE sending notification (atomic check-and-set)
                        var now = DateTime.UtcNow;
                        _cache.Set(cooldownCacheKey, now, _notificationCooldown);

                        // Create notification
                        var notification = new NotificationDto
                        {
                            Type = NotificationType.NegativeFunding,
                            Severity = NotificationSeverity.Warning,
                            Title = "Execution Losing Money on Funding",
                            Message = $"{execution.Symbol} on {execution.Exchange}: Fund P&L ${totalFundingPnL:F2} (negative). Historical: ${historicalNetFunding:F2}, Estimated: ${estimatedFunding:F2}.",
                            Data = new
                            {
                                executionId = execution.Id,
                                symbol = execution.Symbol,
                                exchange = execution.Exchange,
                                totalFundingPnL = totalFundingPnL,
                                historicalNetFunding = historicalNetFunding,
                                estimatedFunding = estimatedFunding,
                                positionsCount = positions.Count
                            },
                            AutoClose = false, // Persistent notification for critical issue
                            AutoCloseDelay = null
                        };

                        await _signalRService.SendNotificationAsync(userId, notification);

                        _logger.LogInformation(
                            "Sent negative funding notification: {Symbol} on {Exchange}, Total Fund P&L: {TotalFundingPnL}, Historical: {HistoricalNetFunding}, Estimated: {EstimatedFunding}",
                            execution.Symbol, execution.Exchange, totalFundingPnL, historicalNetFunding, estimatedFunding);
                    }
                    finally
                    {
                        lockObj.Release();
                    }
                }
                else
                {
                    // If funding is now positive, remove cooldown from cache
                    var cooldownCacheKey = $"{CACHE_KEY_PREFIX_NOTIFICATION_COOLDOWN}{execution.Id}";
                    _cache.Remove(cooldownCacheKey);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking negative funding for user {UserId}", userId);
        }
        finally
        {
            _logger.LogDebug("<<< EXIT CheckNegativeFundingForUserAsync for user {UserId} at {Timestamp}",
                userId, DateTime.UtcNow.ToString("HH:mm:ss.fff"));
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
