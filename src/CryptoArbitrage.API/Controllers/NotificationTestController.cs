using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Data.Entities;
using System.Security.Claims;
using CryptoArbitrage.API.Services.Notifications;

namespace CryptoArbitrage.API.Controllers;

[Authorize]
[ApiController]
[Route("api/[controller]")]
public class NotificationTestController : BaseController
{
    private readonly INotificationService _notificationService;

    public NotificationTestController(
        INotificationService notificationService,
        ILogger<NotificationTestController> logger)
        : base(logger)
    {
        _notificationService = notificationService;
    }

    /// <summary>
    /// Test: Send a negative funding notification
    /// </summary>
    [HttpPost("negative-funding")]
    public async Task<IActionResult> TestNegativeFunding()
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            await _notificationService.CheckNegativeFundingForUserAsync(userId);

            Logger.LogInformation("Sent test negative funding notification to user {UserId}", userId);
            return Ok(new { message = "Negative funding check triggered" });
        }, "testing negative funding notification");
    }

    /// <summary>
    /// Test: Send execution state change notification
    /// </summary>
    [HttpPost("execution-state")]
    public async Task<IActionResult> TestExecutionStateChange(
        [FromQuery] string symbol = "BTCUSDT",
        [FromQuery] string exchange = "Binance",
        [FromQuery] ExecutionState oldState = ExecutionState.Stopped,
        [FromQuery] ExecutionState newState = ExecutionState.Running)
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            await _notificationService.NotifyExecutionStateChangeAsync(
                userId,
                executionId: 999,
                symbol,
                exchange,
                oldState,
                newState);

            Logger.LogInformation(
                "Sent test execution state change notification: {Symbol} on {Exchange} from {OldState} to {NewState}",
                symbol, exchange, oldState, newState);

            return Ok(new {
                message = "Execution state change notification sent",
                symbol,
                exchange,
                oldState = oldState.ToString(),
                newState = newState.ToString()
            });
        }, "testing execution state change notification");
    }

    /// <summary>
    /// Test: Send exchange connectivity notification
    /// </summary>
    [HttpPost("exchange-connectivity")]
    public async Task<IActionResult> TestExchangeConnectivity(
        [FromQuery] string exchange = "Binance",
        [FromQuery] bool isConnected = false,
        [FromQuery] string? errorMessage = "Connection timeout after 30 seconds")
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            await _notificationService.NotifyExchangeConnectivityAsync(
                userId,
                exchange,
                isConnected,
                errorMessage);

            Logger.LogInformation(
                "Sent test exchange connectivity notification: {Exchange} {Status}",
                exchange, isConnected ? "connected" : "disconnected");

            return Ok(new {
                message = "Exchange connectivity notification sent",
                exchange,
                isConnected,
                errorMessage
            });
        }, "testing exchange connectivity notification");
    }

    /// <summary>
    /// Test: Send large opportunity notification
    /// </summary>
    [HttpPost("large-opportunity")]
    public async Task<IActionResult> TestLargeOpportunity(
        [FromQuery] string symbol = "ETHUSDT",
        [FromQuery] decimal annualizedSpread = 25.5m,
        [FromQuery] decimal estimatedProfit = 1250.00m)
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            await _notificationService.NotifyLargeOpportunityAsync(
                userId,
                symbol,
                annualizedSpread,
                estimatedProfit);

            Logger.LogInformation(
                "Sent test large opportunity notification: {Symbol} with {Spread}% spread, ${Profit} profit",
                symbol, annualizedSpread, estimatedProfit);

            return Ok(new {
                message = "Large opportunity notification sent",
                symbol,
                annualizedSpread,
                estimatedProfit
            });
        }, "testing large opportunity notification");
    }

    /// <summary>
    /// Test: Send liquidation risk notification
    /// </summary>
    [HttpPost("liquidation-risk")]
    public async Task<IActionResult> TestLiquidationRisk(
        [FromQuery] string symbol = "BTCUSDT",
        [FromQuery] string exchange = "Bybit",
        [FromQuery] decimal marginPercent = 15.5m)
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            await _notificationService.NotifyLiquidationRiskAsync(
                userId,
                symbol,
                exchange,
                marginPercent);

            Logger.LogInformation(
                "Sent test liquidation risk notification: {Symbol} on {Exchange} at {MarginPercent}%",
                symbol, exchange, marginPercent);

            return Ok(new {
                message = "Liquidation risk notification sent",
                symbol,
                exchange,
                marginPercent
            });
        }, "testing liquidation risk notification");
    }

    /// <summary>
    /// Test: Send all notification types at once
    /// </summary>
    [HttpPost("all")]
    public async Task<IActionResult> TestAllNotifications()
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized();

            // 1. Execution state change (Success - auto-close)
            await _notificationService.NotifyExecutionStateChangeAsync(
                userId, 999, "BTCUSDT", "Binance",
                ExecutionState.Stopped, ExecutionState.Running);

            await Task.Delay(500); // Small delay between notifications

            // 2. Large opportunity (Success - auto-close)
            await _notificationService.NotifyLargeOpportunityAsync(
                userId, "ETHUSDT", 25.5m, 1250.00m);

            await Task.Delay(500);

            // 3. Exchange connectivity issue (Error - persistent)
            await _notificationService.NotifyExchangeConnectivityAsync(
                userId, "Binance", false, "Connection timeout after 30 seconds");

            await Task.Delay(500);

            // 4. Liquidation risk (Error - persistent)
            await _notificationService.NotifyLiquidationRiskAsync(
                userId, "BTCUSDT", "Bybit", 15.5m);

            await Task.Delay(500);

            // 5. Exchange reconnected (Success - auto-close)
            await _notificationService.NotifyExchangeConnectivityAsync(
                userId, "Binance", true, null);

            Logger.LogInformation("Sent all test notifications to user {UserId}", userId);

            return Ok(new {
                message = "All notification types sent successfully",
                count = 5,
                types = new[] {
                    "execution-state-change",
                    "large-opportunity",
                    "exchange-disconnected",
                    "liquidation-risk",
                    "exchange-reconnected"
                }
            });
        }, "testing all notifications");
    }
}
