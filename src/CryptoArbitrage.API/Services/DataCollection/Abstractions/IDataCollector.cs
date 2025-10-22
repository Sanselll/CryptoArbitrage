using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Configuration;

namespace CryptoArbitrage.API.Services.DataCollection.Abstractions;

/// <summary>
/// Interface for data collectors that fetch data from exchanges and store it
/// </summary>
/// <typeparam name="TData">The type of data being collected</typeparam>
/// <typeparam name="TConfig">The configuration type for this collector</typeparam>
public interface IDataCollector<TData, TConfig>
    where TData : class
    where TConfig : CollectorConfiguration
{
    /// <summary>
    /// Configuration for this collector
    /// </summary>
    TConfig Configuration { get; }

    /// <summary>
    /// Collect data from all exchanges
    /// </summary>
    Task<CollectionResult<TData>> CollectAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Get the last collection result (for monitoring/debugging)
    /// </summary>
    CollectionResult<TData>? LastResult { get; }

    /// <summary>
    /// Get the last successful collection time
    /// </summary>
    DateTime? LastSuccessfulCollection { get; }
}
