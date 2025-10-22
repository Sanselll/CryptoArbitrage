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
/// Collects open orders for all users across exchanges
/// </summary>
public class OpenOrdersCollector : IDataCollector<List<OrderDto>, OpenOrdersCollectorConfiguration>
{
    private readonly ILogger<OpenOrdersCollector> _logger;
    private readonly IDataRepository<List<OrderDto>> _repository;
    private readonly OpenOrdersCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;

    public OpenOrdersCollectorConfiguration Configuration => _configuration;
    public CollectionResult<List<OrderDto>>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public OpenOrdersCollector(
        ILogger<OpenOrdersCollector> logger,
        IDataRepository<List<OrderDto>> repository,
        OpenOrdersCollectorConfiguration configuration,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
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

            _logger.LogDebug("Collecting open orders for {Count} user/exchange combinations", userApiKeys.Count);

            var allOrders = new Dictionary<string, List<OrderDto>>();

            // Get encryption service to decrypt API keys
            using var encryptionScope = _serviceProvider.CreateScope();
            var encryptionService = encryptionScope.ServiceProvider.GetRequiredService<IEncryptionService>();

            // Collect from each user/exchange combination
            foreach (var apiKey in userApiKeys)
            {
                try
                {
                    // Create a new scoped connector for this user
                    using var connectorScope = _serviceProvider.CreateScope();
                    object? connector = null;

                    // Decrypt API credentials
                    var decryptedApiKey = encryptionService.Decrypt(apiKey.EncryptedApiKey);
                    var decryptedSecret = encryptionService.Decrypt(apiKey.EncryptedApiSecret);

                    // Get connector based on exchange and connect it
                    switch (apiKey.ExchangeName.ToLower())
                    {
                        case "binance":
                            var binanceConnector = connectorScope.ServiceProvider.GetRequiredService<BinanceConnector>();
                            await binanceConnector.ConnectAsync(decryptedApiKey, decryptedSecret);
                            connector = binanceConnector;
                            break;
                        case "bybit":
                            var bybitConnector = connectorScope.ServiceProvider.GetRequiredService<BybitConnector>();
                            await bybitConnector.ConnectAsync(decryptedApiKey, decryptedSecret);
                            connector = bybitConnector;
                            break;
                        default:
                            _logger.LogWarning("Unknown exchange {Exchange} for user {UserId}",
                                apiKey.ExchangeName, apiKey.UserId);
                            continue;
                    }

                    if (connector == null)
                    {
                        continue;
                    }

                    // Fetch open orders using dynamic
                    dynamic dynamicConnector = connector;
                    List<OrderDto> orders = await dynamicConnector.GetOpenOrdersAsync();

                    var key = $"openorders:{apiKey.UserId}:{apiKey.ExchangeName}";
                    allOrders[key] = orders;

                    _logger.LogDebug("Collected {Count} open orders for user {UserId} on {Exchange}",
                        orders.Count, apiKey.UserId, apiKey.ExchangeName);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect open orders for user {UserId} on {Exchange}",
                        apiKey.UserId, apiKey.ExchangeName);
                }
            }

            // Store in memory repository
            if (allOrders.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(5);

                await _repository.StoreBatchAsync(allOrders, ttl, cancellationToken);

                result.Data = allOrders;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation("Collected open orders for {Count} user/exchange combinations, total {TotalOrders} orders",
                    allOrders.Count, allOrders.Values.Sum(o => o.Count));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No open orders collected successfully";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting open orders");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }
}
