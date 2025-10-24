using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the funding rate history collector.
/// Updates existing funding rates with 3-day and 24h historical averages.
/// </summary>
public class FundingRateHistoryCollectionBackgroundService :
    DataCollectionBackgroundService<FundingRateDto, FundingRateHistoryCollectorConfiguration>
{
    public FundingRateHistoryCollectionBackgroundService(
        ILogger<FundingRateHistoryCollectionBackgroundService> logger,
        IDataCollector<FundingRateDto, FundingRateHistoryCollectorConfiguration> collector,
        FundingRateHistoryCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.FundingRateHistoryCollected;
}
