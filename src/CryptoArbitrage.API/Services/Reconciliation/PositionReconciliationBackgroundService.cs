using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services.Reconciliation.Configuration;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.Reconciliation;

/// <summary>
/// Background service that periodically reconciles positions with transaction history
/// Runs every 30 seconds after TransactionHistoryCollector completes
/// </summary>
public class PositionReconciliationBackgroundService : BackgroundService
{
    private readonly ILogger<PositionReconciliationBackgroundService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly PositionReconciliationConfiguration _config;

    private DateTime? _lastSuccessfulRun;
    private int _consecutiveFailures = 0;

    public PositionReconciliationBackgroundService(
        ILogger<PositionReconciliationBackgroundService> logger,
        IServiceProvider serviceProvider,
        PositionReconciliationConfiguration config)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_config.IsEnabled)
        {
            _logger.LogInformation("PositionReconciliationBackgroundService is disabled in configuration");
            return;
        }

        _logger.LogInformation("PositionReconciliationBackgroundService started with interval {Interval}s",
            _config.ReconciliationIntervalSeconds);

        // Initial delay to allow TransactionHistoryCollector to run first
        await Task.Delay(TimeSpan.FromSeconds(35), stoppingToken); // 5 seconds after TransactionHistoryCollector

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var startTime = DateTime.UtcNow;

                _logger.LogDebug("Starting reconciliation cycle");

                await RunReconciliationCycleAsync(stoppingToken);

                var duration = DateTime.UtcNow - startTime;

                _lastSuccessfulRun = DateTime.UtcNow;
                _consecutiveFailures = 0;

                _logger.LogInformation("Reconciliation cycle completed in {Duration}ms",
                    duration.TotalMilliseconds);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("PositionReconciliationBackgroundService stopping");
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in reconciliation cycle");
                _consecutiveFailures++;

                if (_consecutiveFailures >= 3)
                {
                    var backoffDelay = TimeSpan.FromSeconds(_config.ReconciliationIntervalSeconds * 2);
                    _logger.LogWarning(
                        "Reconciliation has {Failures} consecutive failures. Backing off for {Delay}s",
                        _consecutiveFailures, backoffDelay.TotalSeconds);
                    await Task.Delay(backoffDelay, stoppingToken);
                    _consecutiveFailures = 0;
                }
            }

            // Wait for next cycle
            try
            {
                await Task.Delay(
                    TimeSpan.FromSeconds(_config.ReconciliationIntervalSeconds),
                    stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        _logger.LogInformation("PositionReconciliationBackgroundService stopped");
    }

    private async Task RunReconciliationCycleAsync(CancellationToken cancellationToken)
    {
        using var scope = _serviceProvider.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
        var reconciliationService = scope.ServiceProvider.GetRequiredService<PositionReconciliationService>();

        // Find positions that need reconciliation
        var positionsToReconcile = await dbContext.Positions
            .Where(p => p.ReconciliationStatus == ReconciliationStatus.Preliminary
                || (_config.RetryPartiallyReconciled && p.ReconciliationStatus == ReconciliationStatus.PartiallyReconciled))
            .OrderBy(p => p.ClosedAt ?? p.OpenedAt) // Oldest first
            .Take(_config.MaxPositionsPerCycle)
            .ToListAsync(cancellationToken);

        if (positionsToReconcile.Count == 0)
        {
            _logger.LogDebug("No positions require reconciliation");

            // Check for stale positions
            await MarkStalePositionsAsync(dbContext, cancellationToken);
            return;
        }

        _logger.LogInformation("Reconciling {Count} positions", positionsToReconcile.Count);

        var successCount = 0;
        var failureCount = 0;
        var statusChanges = new Dictionary<string, int>();

        foreach (var position in positionsToReconcile)
        {
            try
            {
                var result = await reconciliationService.ReconcilePositionAsync(
                    position, dbContext, cancellationToken);

                if (result.Success)
                {
                    successCount++;

                    // Track status transitions
                    var transition = $"{result.StartStatus}→{result.EndStatus}";
                    statusChanges[transition] = statusChanges.GetValueOrDefault(transition) + 1;
                }
                else
                {
                    failureCount++;
                    _logger.LogWarning("Failed to reconcile Position {PositionId}: {Error}",
                        position.Id, result.ErrorMessage);
                }
            }
            catch (Exception ex)
            {
                failureCount++;
                _logger.LogError(ex, "Exception reconciling Position {PositionId}", position.Id);
            }
        }

        _logger.LogInformation(
            "Reconciliation cycle: {Success} succeeded, {Failure} failed. Status changes: {Changes}",
            successCount, failureCount,
            string.Join(", ", statusChanges.Select(kv => $"{kv.Key}({kv.Value})")));

        // Check for stale positions
        await MarkStalePositionsAsync(dbContext, cancellationToken);
    }

    /// <summary>
    /// Mark positions as StaleUnreconciled if closed >24 hours ago and not fully reconciled
    /// </summary>
    private async Task MarkStalePositionsAsync(
        ArbitrageDbContext dbContext,
        CancellationToken cancellationToken)
    {
        var staleThreshold = DateTime.UtcNow.AddHours(-_config.StaleThresholdHours);

        var stalePositions = await dbContext.Positions
            .Where(p => p.Status == PositionStatus.Closed
                && p.ClosedAt.HasValue
                && p.ClosedAt.Value < staleThreshold
                && p.ReconciliationStatus != ReconciliationStatus.FullyReconciled
                && p.ReconciliationStatus != ReconciliationStatus.StaleUnreconciled)
            .ToListAsync(cancellationToken);

        if (stalePositions.Count > 0)
        {
            foreach (var position in stalePositions)
            {
                position.ReconciliationStatus = ReconciliationStatus.StaleUnreconciled;
                _logger.LogWarning(
                    "Marked Position {PositionId} as StaleUnreconciled (closed {HoursAgo:F1} hours ago)",
                    position.Id, (DateTime.UtcNow - position.ClosedAt.Value).TotalHours);
            }

            await dbContext.SaveChangesAsync(cancellationToken);
            _logger.LogInformation("Marked {Count} positions as StaleUnreconciled", stalePositions.Count);
        }
    }

    public override Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("PositionReconciliationBackgroundService stop requested");
        return base.StopAsync(cancellationToken);
    }
}
