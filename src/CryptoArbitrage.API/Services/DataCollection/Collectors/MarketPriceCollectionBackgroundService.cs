using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the market price collector.
/// PURE collector: Only collects and publishes event - no broadcasting or aggregation.
/// </summary>
public class MarketPriceCollectionBackgroundService :
    DataCollectionBackgroundService<MarketDataSnapshot, MarketPriceCollectorConfiguration>
{
    public MarketPriceCollectionBackgroundService(
        ILogger<MarketPriceCollectionBackgroundService> logger,
        IDataCollector<MarketDataSnapshot, MarketPriceCollectorConfiguration> collector,
        MarketPriceCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.MarketPricesCollected;
}
