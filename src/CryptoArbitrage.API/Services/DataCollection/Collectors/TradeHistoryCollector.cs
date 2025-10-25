using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Collects trade history for all users across exchanges
/// </summary>
public class TradeHistoryCollector : IDataCollector<List<TradeDto>, TradeHistoryCollectorConfiguration>
{
    private readonly ILogger<TradeHistoryCollector> _logger;
    private readonly IDataRepository<List<TradeDto>> _repository;
    private readonly TradeHistoryCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly ConnectorManager _connectorManager;

    public TradeHistoryCollectorConfiguration Configuration => _configuration;
    public CollectionResult<List<TradeDto>>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public TradeHistoryCollector(
        ILogger<TradeHistoryCollector> logger,
        IDataRepository<List<TradeDto>> repository,
        TradeHistoryCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        ConnectorManager connectorManager)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _connectorManager = connectorManager;
    }

    public async Task<CollectionResult<List<TradeDto>>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<List<TradeDto>>();

        try
        {
            // Get all enabled user API keys from database using a scoped DbContext
            List<UserExchangeApiKey> userApiKeys;
            using (var scope = _serviceProvider.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
                userApiKeys = await dbContext.UserExchangeApiKeys
                    .Where(k => k.IsEnabled)
                    .ToListAsync(cancellationToken);
            }

            if (!userApiKeys.Any())
            {
                result.Success = true;
                result.ErrorMessage = "No enabled user API keys configured";
                LastResult = result;
                return result;
            }

            _logger.LogDebug("Collecting trade history for {Count} user/exchange combinations", userApiKeys.Count);

            var allTrades = new Dictionary<string, List<TradeDto>>();

            // Get encryption service to decrypt API keys
            using var encryptionScope = _serviceProvider.CreateScope();
            var encryptionService = encryptionScope.ServiceProvider.GetRequiredService<IEncryptionService>();

            // Calculate time range for historical data
            var startTime = DateTime.UtcNow.AddDays(-_configuration.HistoryDays);
            var endTime = DateTime.UtcNow;

            // Collect from each user/exchange combination IN PARALLEL
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var tasks = userApiKeys.Select(async apiKey =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    // Create a new scoped connector for this user
                    using var connectorScope = _serviceProvider.CreateScope();

                    // Decrypt API credentials
                    var decryptedApiKey = encryptionService.Decrypt(apiKey.EncryptedApiKey);
                    var decryptedSecret = encryptionService.Decrypt(apiKey.EncryptedApiSecret);

                    // Get connector using ConnectorManager
                    var connector = await _connectorManager.GetConnectorByNameAsync(
                        connectorScope,
                        apiKey.ExchangeName,
                        decryptedApiKey,
                        decryptedSecret,
                        cancellationToken);

                    if (connector == null)
                    {
                        _logger.LogWarning("Could not create connector for {Exchange} for user {UserId}",
                            apiKey.ExchangeName, apiKey.UserId);
                        return ((string?)null, (List<TradeDto>?)null);
                    }

                    // Fetch trade history
                    List<TradeDto> trades = await connector.GetUserTradesAsync(startTime, endTime, 100);

                    var key = $"tradehistory:{apiKey.UserId}:{apiKey.ExchangeName}";

                    _logger.LogDebug("Collected {Count} trades for user {UserId} on {Exchange}",
                        trades.Count, apiKey.UserId, apiKey.ExchangeName);

                    return (key, trades);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect trade history for user {UserId} on {Exchange}",
                        apiKey.UserId, apiKey.ExchangeName);
                    return ((string?)null, (List<TradeDto>?)null);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var results = await Task.WhenAll(tasks);

            // Aggregate successful results
            foreach (var item in results.Where(r => r.Item1 != null && r.Item2 != null))
            {
                allTrades[item.Item1!] = item.Item2!;
            }

            // Store in memory repository
            if (allTrades.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(10);

                await _repository.StoreBatchAsync(allTrades, ttl, cancellationToken);

                result.Data = allTrades;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation("Collected trade history for {Count} user/exchange combinations, total {TotalTrades} trades",
                    allTrades.Count, allTrades.Values.Sum(t => t.Count));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No trade history collected successfully";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting trade history");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }
}
