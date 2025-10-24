using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the order history collector.
/// PURE collector: Only collects and publishes event - no broadcasting.
/// </summary>
public class OrderHistoryCollectionBackgroundService :
    DataCollectionBackgroundService<List<OrderDto>, OrderHistoryCollectorConfiguration>
{
    public OrderHistoryCollectionBackgroundService(
        ILogger<OrderHistoryCollectionBackgroundService> logger,
        IDataCollector<List<OrderDto>, OrderHistoryCollectorConfiguration> collector,
        OrderHistoryCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.OrderHistoryCollected;
}
