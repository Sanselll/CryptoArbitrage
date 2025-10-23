using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the historical price collector.
/// Fetches KLines data to enable 24h and 3D average price spread projections.
/// </summary>
public class HistoricalPriceCollectionBackgroundService :
    DataCollectionBackgroundService<Dictionary<string, List<HistoricalPriceDto>>, HistoricalPriceCollectorConfiguration>
{
    public HistoricalPriceCollectionBackgroundService(
        ILogger<HistoricalPriceCollectionBackgroundService> logger,
        IDataCollector<Dictionary<string, List<HistoricalPriceDto>>, HistoricalPriceCollectorConfiguration> collector,
        HistoricalPriceCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.HistoricalPriceCollected;
}
