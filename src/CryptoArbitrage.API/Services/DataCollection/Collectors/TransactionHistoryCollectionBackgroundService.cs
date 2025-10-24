using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the transaction history collector.
/// PURE collector: Only collects and publishes event - no broadcasting.
/// </summary>
public class TransactionHistoryCollectionBackgroundService :
    DataCollectionBackgroundService<List<TransactionDto>, TransactionHistoryCollectorConfiguration>
{
    public TransactionHistoryCollectionBackgroundService(
        ILogger<TransactionHistoryCollectionBackgroundService> logger,
        IDataCollector<List<TransactionDto>, TransactionHistoryCollectorConfiguration> collector,
        TransactionHistoryCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.TransactionHistoryCollected;
}
