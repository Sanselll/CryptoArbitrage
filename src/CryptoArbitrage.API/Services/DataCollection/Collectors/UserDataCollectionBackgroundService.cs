using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Background service that periodically runs the user data collector.
/// PURE collector: Only collects and publishes event - no broadcasting.
/// </summary>
public class UserDataCollectionBackgroundService :
    DataCollectionBackgroundService<UserDataSnapshot, UserDataCollectorConfiguration>
{
    public UserDataCollectionBackgroundService(
        ILogger<UserDataCollectionBackgroundService> logger,
        IDataCollector<UserDataSnapshot, UserDataCollectorConfiguration> collector,
        UserDataCollectorConfiguration configuration,
        IDataCollectionEventBus eventBus)
        : base(logger, collector, configuration, eventBus)
    {
    }

    protected override string GetEventType() => DataCollectionConstants.EventTypes.UserDataCollected;
}
