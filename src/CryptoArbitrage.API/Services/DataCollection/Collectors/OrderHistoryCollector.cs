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
/// Collects order history for all users across exchanges
/// </summary>
public class OrderHistoryCollector : IDataCollector<List<OrderDto>, OrderHistoryCollectorConfiguration>
{
    private readonly ILogger<OrderHistoryCollector> _logger;
    private readonly IDataRepository<List<OrderDto>> _repository;
    private readonly OrderHistoryCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly ConnectorManager _connectorManager;

    public OrderHistoryCollectorConfiguration Configuration => _configuration;
    public CollectionResult<List<OrderDto>>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public OrderHistoryCollector(
        ILogger<OrderHistoryCollector> logger,
        IDataRepository<List<OrderDto>> repository,
        OrderHistoryCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        ConnectorManager connectorManager)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _connectorManager = connectorManager;
    }

    public async Task<CollectionResult<List<OrderDto>>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<List<OrderDto>>();

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

            _logger.LogDebug("Collecting order history for {Count} user/exchange combinations", userApiKeys.Count);

            var allOrders = new Dictionary<string, List<OrderDto>>();

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
                        _logger.LogWarning("Unknown exchange {Exchange} for user {UserId}",
                            apiKey.ExchangeName, apiKey.UserId);
                        return ((string?)null, (List<OrderDto>?)null);
                    }

                    // Fetch order history
                    List<OrderDto> orders = await connector.GetOrderHistoryAsync(startTime, endTime, 100);

                    var key = $"orderhistory:{apiKey.UserId}:{apiKey.ExchangeName}";

                    _logger.LogDebug("Collected {Count} historical orders for user {UserId} on {Exchange}",
                        orders.Count, apiKey.UserId, apiKey.ExchangeName);

                    return (key, orders);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect order history for user {UserId} on {Exchange}",
                        apiKey.UserId, apiKey.ExchangeName);
                    return ((string?)null, (List<OrderDto>?)null);
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
                allOrders[item.Item1!] = item.Item2!;
            }

            // Store in memory repository
            if (allOrders.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(10);

                await _repository.StoreBatchAsync(allOrders, ttl, cancellationToken);

                result.Data = allOrders;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation("Collected order history for {Count} user/exchange combinations, total {TotalOrders} orders",
                    allOrders.Count, allOrders.Values.Sum(o => o.Count));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No order history collected successfully";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting order history");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }
}
