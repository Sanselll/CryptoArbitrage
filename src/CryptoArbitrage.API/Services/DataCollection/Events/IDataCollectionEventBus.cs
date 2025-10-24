namespace CryptoArbitrage.API.Services.DataCollection.Events;

/// <summary>
/// Simple in-memory event bus for data collection events.
/// Allows publishers (collectors) to emit events and subscribers (aggregators, broadcasters) to react.
/// </summary>
public interface IDataCollectionEventBus
{
    /// <summary>
    /// Subscribe to events of a specific type.
    /// Handler will be called whenever an event of type T with matching eventType is published.
    /// </summary>
    /// <typeparam name="T">Type of data in the event</typeparam>
    /// <param name="eventType">Event type to subscribe to (e.g., "FundingRatesCollected")</param>
    /// <param name="handler">Async handler function</param>
    void Subscribe<T>(string eventType, Func<DataCollectionEvent<T>, Task> handler);

    /// <summary>
    /// Publish an event to all subscribers.
    /// All matching handlers will be invoked in parallel.
    /// </summary>
    /// <typeparam name="T">Type of data in the event</typeparam>
    /// <param name="event">The event to publish</param>
    Task PublishAsync<T>(DataCollectionEvent<T> @event);
}
