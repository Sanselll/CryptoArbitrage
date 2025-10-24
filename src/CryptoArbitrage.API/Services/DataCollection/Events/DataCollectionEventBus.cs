using System.Collections.Concurrent;

namespace CryptoArbitrage.API.Services.DataCollection.Events;

/// <summary>
/// Simple in-memory event bus implementation using ConcurrentDictionary for thread-safety.
/// Supports pub/sub pattern for data collection events.
/// </summary>
public class DataCollectionEventBus : IDataCollectionEventBus
{
    private readonly ILogger<DataCollectionEventBus> _logger;

    // Store handlers by event type. Key format: "EventType|DataType"
    private readonly ConcurrentDictionary<string, List<Delegate>> _handlers = new();
    private readonly object _handlersLock = new();

    public DataCollectionEventBus(ILogger<DataCollectionEventBus> logger)
    {
        _logger = logger;
    }

    public void Subscribe<T>(string eventType, Func<DataCollectionEvent<T>, Task> handler)
    {
        var key = GetHandlerKey<T>(eventType);

        lock (_handlersLock)
        {
            if (!_handlers.ContainsKey(key))
            {
                _handlers[key] = new List<Delegate>();
            }

            _handlers[key].Add(handler);
        }

        _logger.LogDebug("Subscribed handler for event {EventType}<{DataType}>", eventType, typeof(T).Name);
    }

    public async Task PublishAsync<T>(DataCollectionEvent<T> @event)
    {
        var key = GetHandlerKey<T>(@event.EventType);

        List<Delegate>? handlersToInvoke = null;
        lock (_handlersLock)
        {
            if (_handlers.TryGetValue(key, out var handlers))
            {
                // Create a copy to avoid holding the lock during invocation
                handlersToInvoke = new List<Delegate>(handlers);
            }
        }

        if (handlersToInvoke == null || !handlersToInvoke.Any())
        {
            _logger.LogDebug("No handlers registered for event {EventType}<{DataType}>",
                @event.EventType, typeof(T).Name);
            return;
        }

        _logger.LogDebug("Publishing event {EventType}<{DataType}> to {Count} handlers",
            @event.EventType, typeof(T).Name, handlersToInvoke.Count);

        // Execute all handlers in parallel
        var tasks = handlersToInvoke
            .Cast<Func<DataCollectionEvent<T>, Task>>()
            .Select(handler => SafeInvokeHandlerAsync(handler, @event));

        await Task.WhenAll(tasks);
    }

    private async Task SafeInvokeHandlerAsync<T>(
        Func<DataCollectionEvent<T>, Task> handler,
        DataCollectionEvent<T> @event)
    {
        try
        {
            await handler(@event);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in event handler for {EventType}<{DataType}>",
                @event.EventType, typeof(T).Name);
            // Continue with other handlers even if one fails
        }
    }

    private string GetHandlerKey<T>(string eventType)
    {
        return $"{eventType}|{typeof(T).FullName}";
    }
}
