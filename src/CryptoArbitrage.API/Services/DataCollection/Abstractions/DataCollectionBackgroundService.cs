using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services.DataCollection.Abstractions;

/// <summary>
/// Base background service for all data collectors
/// Handles the periodic execution, error handling, and lifecycle management
/// Automatically publishes collection events to the event bus
/// </summary>
public abstract class DataCollectionBackgroundService<TData, TConfig> : BackgroundService
    where TData : class
    where TConfig : CollectorConfiguration
{
    protected readonly ILogger Logger;
    protected readonly IDataCollector<TData, TConfig> Collector;
    protected readonly TConfig Configuration;
    private readonly IDataCollectionEventBus _eventBus;

    private DateTime? _lastSuccessfulRun;
    private int _consecutiveFailures = 0;

    protected DataCollectionBackgroundService(
        ILogger logger,
        IDataCollector<TData, TConfig> collector,
        TConfig configuration,
        IDataCollectionEventBus eventBus)
    {
        Logger = logger;
        Collector = collector;
        Configuration = configuration;
        _eventBus = eventBus;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!Configuration.IsEnabled)
        {
            Logger.LogInformation("{ServiceName} is disabled in configuration", GetType().Name);
            return;
        }

        Logger.LogInformation("{ServiceName} started with interval {Interval}s",
            GetType().Name, Configuration.CollectionIntervalSeconds);

        // Initial delay to allow other services to start
        await Task.Delay(TimeSpan.FromSeconds(2), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var startTime = DateTime.UtcNow;

                Logger.LogDebug("{ServiceName} starting collection cycle", GetType().Name);

                var result = await Collector.CollectAsync(stoppingToken);

                var duration = DateTime.UtcNow - startTime;

                if (result.Success)
                {
                    _lastSuccessfulRun = DateTime.UtcNow;
                    _consecutiveFailures = 0;

                    Logger.LogInformation(
                        "{ServiceName} collected {Count} items in {Duration}ms",
                        GetType().Name, result.ItemsCollected, duration.TotalMilliseconds);

                    await OnCollectionSuccessAsync(result, stoppingToken);
                }
                else
                {
                    _consecutiveFailures++;

                    Logger.LogWarning(
                        "{ServiceName} collection failed: {Error}. Consecutive failures: {Failures}",
                        GetType().Name, result.ErrorMessage, _consecutiveFailures);

                    await OnCollectionFailureAsync(result, stoppingToken);

                    // If too many consecutive failures, increase delay
                    if (_consecutiveFailures >= Configuration.MaxRetryAttempts)
                    {
                        var backoffDelay = TimeSpan.FromSeconds(Configuration.CollectionIntervalSeconds * 2);
                        Logger.LogWarning(
                            "{ServiceName} has {Failures} consecutive failures. Backing off for {Delay}s",
                            GetType().Name, _consecutiveFailures, backoffDelay.TotalSeconds);
                        await Task.Delay(backoffDelay, stoppingToken);
                        _consecutiveFailures = 0; // Reset after backoff
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when stopping
                Logger.LogInformation("{ServiceName} stopping", GetType().Name);
                break;
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "{ServiceName} encountered unexpected error", GetType().Name);
                _consecutiveFailures++;
            }

            // Wait for next collection cycle
            try
            {
                await Task.Delay(
                    TimeSpan.FromSeconds(Configuration.CollectionIntervalSeconds),
                    stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        Logger.LogInformation("{ServiceName} stopped", GetType().Name);
    }

    /// <summary>
    /// Called after successful collection. Publishes event to event bus.
    /// Override to add custom behavior (but call base.OnCollectionSuccessAsync first).
    /// </summary>
    protected virtual async Task OnCollectionSuccessAsync(
        CollectionResult<TData> result,
        CancellationToken cancellationToken)
    {
        try
        {
            // Count items in the result
            var itemCount = result.Data?.Count ?? 0;

            // Publish event to event bus
            await _eventBus.PublishAsync(new DataCollectionEvent<IDictionary<string, TData>>
            {
                EventType = GetEventType(),
                Data = result.Data,
                Timestamp = DateTime.UtcNow,
                CollectionDuration = result.Duration,
                Success = result.Success,
                ItemCount = itemCount
            });

            Logger.LogDebug("Published {EventType} event with {Count} items", GetEventType(), itemCount);
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error publishing {EventType} event", GetEventType());
        }
    }

    /// <summary>
    /// Called after failed collection. Override to add custom behavior.
    /// </summary>
    protected virtual Task OnCollectionFailureAsync(
        CollectionResult<TData> result,
        CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }

    /// <summary>
    /// Get the event type constant for this collector.
    /// Used when publishing collection events to the event bus.
    /// </summary>
    protected abstract string GetEventType();

    public override Task StopAsync(CancellationToken cancellationToken)
    {
        Logger.LogInformation("{ServiceName} stop requested", GetType().Name);
        return base.StopAsync(cancellationToken);
    }
}
