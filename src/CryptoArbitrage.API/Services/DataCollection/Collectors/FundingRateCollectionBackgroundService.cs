using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the funding rate collector.
/// PURE collector: Only collects and publishes event - no broadcasting or aggregation.
/// </summary>
public class FundingRateCollectionBackgroundService :
    DataCollectionBackgroundService<FundingRateDto, FundingRateCollectorConfiguration>
{
    public FundingRateCollectionBackgroundService(
        ILogger<FundingRateCollectionBackgroundService> logger,
        IDataCollector<FundingRateDto, FundingRateCollectorConfiguration> collector,
        FundingRateCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.FundingRatesCollected;
}
