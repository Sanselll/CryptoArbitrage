using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the liquidity metrics collector.
/// PURE collector: Only collects and publishes event - no broadcasting.
/// </summary>
public class LiquidityCollectionBackgroundService :
    DataCollectionBackgroundService<LiquidityMetricsDto, LiquidityCollectorConfiguration>
{
    public LiquidityCollectionBackgroundService(
        ILogger<LiquidityCollectionBackgroundService> logger,
        IDataCollector<LiquidityMetricsDto, LiquidityCollectorConfiguration> collector,
        LiquidityCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.LiquidityMetricsCollected;
}
