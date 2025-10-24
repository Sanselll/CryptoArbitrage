using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the open orders collector.
/// PURE collector: Only collects and publishes event - no broadcasting.
/// </summary>
public class OpenOrdersCollectionBackgroundService :
    DataCollectionBackgroundService<List<OrderDto>, OpenOrdersCollectorConfiguration>
{
    public OpenOrdersCollectionBackgroundService(
        ILogger<OpenOrdersCollectionBackgroundService> logger,
        IDataCollector<List<OrderDto>, OpenOrdersCollectorConfiguration> collector,
        OpenOrdersCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.OpenOrdersCollected;
}
