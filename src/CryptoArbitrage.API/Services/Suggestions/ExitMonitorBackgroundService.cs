using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Models.Suggestions;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.Streaming;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Hosting;

namespace CryptoArbitrage.API.Services.Suggestions;

/// <summary>
/// Background service that monitors active positions and generates exit signals
/// </summary>
public class ExitMonitorBackgroundService : BackgroundService
{
    private readonly ILogger<ExitMonitorBackgroundService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly ExitStrategyMonitor _exitMonitor;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;
    private readonly SignalRBroadcaster _broadcaster;

    private int _monitorIntervalSeconds = 300; // Default: 5 minutes
    private bool _isEnabled = true;

    public ExitMonitorBackgroundService(
        ILogger<ExitMonitorBackgroundService> logger,
        IServiceProvider serviceProvider,
        IConfiguration configuration,
        ExitStrategyMonitor exitMonitor,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        SignalRBroadcaster broadcaster)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _configuration = configuration;
        _exitMonitor = exitMonitor;
        _marketDataRepository = marketDataRepository;
        _broadcaster = broadcaster;

        // Load configuration
        _monitorIntervalSeconds = configuration.GetValue<int>("ExitMonitor:IntervalSeconds", 300);
        _isEnabled = configuration.GetValue<bool>("ExitMonitor:Enabled", true);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_isEnabled)
        {
            _logger.LogInformation("ExitMonitorBackgroundService is disabled in configuration");
            return;
        }

        _logger.LogInformation(
            "ExitMonitorBackgroundService started with interval {Interval}s",
            _monitorIntervalSeconds);

        // Initial delay to allow other services to start
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var startTime = DateTime.UtcNow;

                await MonitorActivePositionsAsync(stoppingToken);

                var duration = DateTime.UtcNow - startTime;
                _logger.LogDebug(
                    "ExitMonitorBackgroundService completed monitoring cycle in {Duration}ms",
                    duration.TotalMilliseconds);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("ExitMonitorBackgroundService stopping");
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "ExitMonitorBackgroundService encountered error during monitoring cycle");
            }

            // Wait for next cycle
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(_monitorIntervalSeconds), stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        _logger.LogInformation("ExitMonitorBackgroundService stopped");
    }

    private async Task MonitorActivePositionsAsync(CancellationToken cancellationToken)
    {
        // Create a new scope for database access
        using var scope = _serviceProvider.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        try
        {
            // Get all active positions (Status = Open)
            var activePositions = await dbContext.Positions
                .Include(p => p.Execution)
                .Where(p => p.Status == PositionStatus.Open)
                .ToListAsync(cancellationToken);

            if (!activePositions.Any())
            {
                _logger.LogDebug("No active positions to monitor");
                return;
            }

            _logger.LogDebug($"Monitoring {activePositions.Count} active positions");

            // Get current market data snapshot
            var marketSnapshot = await _marketDataRepository.GetAsync(DataCollectionConstants.CacheKeys.MarketDataSnapshot);

            if (marketSnapshot == null)
            {
                _logger.LogWarning("No market data snapshot available for exit monitoring");
                return;
            }

            // Group positions by user for efficient processing
            var positionsByUser = activePositions.GroupBy(p => p.UserId);

            foreach (var userGroup in positionsByUser)
            {
                foreach (var position in userGroup)
                {
                    try
                    {
                        await EvaluatePositionExitConditions(
                            position,
                            position.Execution,
                            marketSnapshot,
                            cancellationToken);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(
                            ex,
                            "Error evaluating exit conditions for Position {PositionId}",
                            position.Id);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error querying active positions");
        }
    }

    private async Task EvaluatePositionExitConditions(
        Position position,
        Execution? execution,
        MarketDataSnapshot marketSnapshot,
        CancellationToken cancellationToken)
    {
        // Evaluate exit conditions
        var exitSignals = _exitMonitor.EvaluateExitConditions(position, execution, marketSnapshot);

        if (!exitSignals.Any())
        {
            return;
        }

        // Filter for triggered signals only
        var triggeredSignals = exitSignals.Where(s => s.IsTriggered).ToList();

        if (!triggeredSignals.Any())
        {
            return;
        }

        _logger.LogInformation(
            "Exit signals triggered for Position {PositionId} ({Symbol}): {Signals}",
            position.Id,
            position.Symbol,
            string.Join(", ", triggeredSignals.Select(s => s.ConditionType)));

        // Broadcast exit signals to user via SignalR
        foreach (var signal in triggeredSignals)
        {
            try
            {
                await BroadcastExitSignal(position, signal);
            }
            catch (Exception ex)
            {
                _logger.LogError(
                    ex,
                    "Error broadcasting exit signal for Position {PositionId}",
                    position.Id);
            }
        }

        // Log summary of all signals
        LogExitSignalSummary(position, triggeredSignals);
    }

    private async Task BroadcastExitSignal(Position position, ExitSignal signal)
    {
        // Broadcast to specific user
        await _broadcaster.BroadcastExitSignal(position.UserId, position.Id, signal);

        _logger.LogInformation(
            "Broadcasted {ConditionType} exit signal for Position {PositionId} to user {UserId}",
            signal.ConditionType,
            position.Id,
            position.UserId);
    }

    private void LogExitSignalSummary(Position position, List<ExitSignal> signals)
    {
        var criticalSignals = signals.Where(s => s.Urgency == ExitUrgency.Critical).ToList();
        var highUrgencySignals = signals.Where(s => s.Urgency == ExitUrgency.High).ToList();

        if (criticalSignals.Any())
        {
            _logger.LogWarning(
                "CRITICAL exit signals for Position {PositionId}: {Signals}",
                position.Id,
                string.Join("; ", criticalSignals.Select(s => s.Message)));
        }
        else if (highUrgencySignals.Any())
        {
            _logger.LogWarning(
                "HIGH urgency exit signals for Position {PositionId}: {Signals}",
                position.Id,
                string.Join("; ", highUrgencySignals.Select(s => s.Message)));
        }
        else
        {
            _logger.LogInformation(
                "Exit signals for Position {PositionId}: {Count} condition(s) met",
                position.Id,
                signals.Count);
        }
    }

    public override Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("ExitMonitorBackgroundService stop requested");
        return base.StopAsync(cancellationToken);
    }
}
