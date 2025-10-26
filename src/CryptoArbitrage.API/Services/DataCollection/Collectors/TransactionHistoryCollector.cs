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
/// Collects transaction history for all users across exchanges
/// </summary>
public class TransactionHistoryCollector : IDataCollector<List<TransactionDto>, TransactionHistoryCollectorConfiguration>
{
    private readonly ILogger<TransactionHistoryCollector> _logger;
    private readonly IDataRepository<List<TransactionDto>> _repository;
    private readonly TransactionHistoryCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;

    public TransactionHistoryCollectorConfiguration Configuration => _configuration;
    public CollectionResult<List<TransactionDto>>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public TransactionHistoryCollector(
        ILogger<TransactionHistoryCollector> logger,
        IDataRepository<List<TransactionDto>> repository,
        TransactionHistoryCollectorConfiguration configuration,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
    }

    public async Task<CollectionResult<List<TransactionDto>>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<List<TransactionDto>>();

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

            _logger.LogInformation("Collecting transaction history for {Count} user/exchange combinations (last {Days} days)",
                userApiKeys.Count, _configuration.HistoryDays);

            var allTransactions = new Dictionary<string, List<TransactionDto>>();

            // Get encryption service to decrypt API keys
            using var encryptionScope = _serviceProvider.CreateScope();
            var encryptionService = encryptionScope.ServiceProvider.GetRequiredService<IEncryptionService>();

            // Calculate time range for historical data
            var startTime = DateTime.UtcNow.AddDays(-_configuration.HistoryDays);
            var endTime = DateTime.UtcNow;

            _logger.LogDebug("Fetching transactions from {StartTime:yyyy-MM-dd HH:mm:ss} to {EndTime:yyyy-MM-dd HH:mm:ss}",
                startTime, endTime);

            // Collect from each user/exchange combination IN PARALLEL
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var tasks = userApiKeys.Select(async apiKey =>
            {
                await semaphore.WaitAsync(cancellationToken);
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
                            return ((string?)null, (List<TransactionDto>?)null);
                    }

                    if (connector == null)
                    {
                        return ((string?)null, (List<TransactionDto>?)null);
                    }

                    // Fetch transaction history using dynamic
                    dynamic dynamicConnector = connector;
                    List<TransactionDto> transactions = await dynamicConnector.GetTransactionsAsync(startTime, endTime, 100);

                    var key = $"transactionhistory:{apiKey.UserId}:{apiKey.ExchangeName}";

                    if (transactions.Count > 0)
                    {
                        var oldestTx = transactions.MinBy(t => t.CreatedAt);
                        var newestTx = transactions.MaxBy(t => t.CreatedAt);
                        _logger.LogInformation("Collected {Count} transactions for user {UserId} on {Exchange} (oldest: {Oldest:yyyy-MM-dd HH:mm}, newest: {Newest:yyyy-MM-dd HH:mm})",
                            transactions.Count, apiKey.UserId, apiKey.ExchangeName,
                            oldestTx?.CreatedAt, newestTx?.CreatedAt);
                    }
                    else
                    {
                        _logger.LogWarning("No transactions found for user {UserId} on {Exchange} in the last {Days} days",
                            apiKey.UserId, apiKey.ExchangeName, _configuration.HistoryDays);
                    }

                    return (key, transactions);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect transaction history for user {UserId} on {Exchange}",
                        apiKey.UserId, apiKey.ExchangeName);
                    return ((string?)null, (List<TransactionDto>?)null);
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
                allTransactions[item.Item1!] = item.Item2!;
            }

            // Store in memory repository
            if (allTransactions.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(15);

                await _repository.StoreBatchAsync(allTransactions, ttl, cancellationToken);

                result.Data = allTransactions;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation("Collected transaction history for {Count} user/exchange combinations, total {TotalTransactions} transactions",
                    allTransactions.Count, allTransactions.Values.Sum(t => t.Count));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No transaction history collected successfully";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting transaction history");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }
}
