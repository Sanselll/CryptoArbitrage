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
/// Collects user data (balances and positions) for all users across exchanges
/// </summary>
public class UserDataCollector : IDataCollector<UserDataSnapshot, UserDataCollectorConfiguration>
{
    private readonly ILogger<UserDataCollector> _logger;
    private readonly IDataRepository<UserDataSnapshot> _repository;
    private readonly UserDataCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;

    public UserDataCollectorConfiguration Configuration => _configuration;
    public CollectionResult<UserDataSnapshot>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public UserDataCollector(
        ILogger<UserDataCollector> logger,
        IDataRepository<UserDataSnapshot> repository,
        UserDataCollectorConfiguration configuration,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
    }

    public async Task<CollectionResult<UserDataSnapshot>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<UserDataSnapshot>();

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
                result.Success = true; // Not an error, just no users configured
                result.ErrorMessage = "No enabled user API keys configured";
                LastResult = result;
                return result;
            }

            _logger.LogDebug("Collecting user data for {Count} user/exchange combinations", userApiKeys.Count);

            var snapshots = new Dictionary<string, UserDataSnapshot>();

            // Get encryption service to decrypt API keys
            using var encryptionScope = _serviceProvider.CreateScope();
            var encryptionService = encryptionScope.ServiceProvider.GetRequiredService<IEncryptionService>();

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
                            return ((string?)null, (UserDataSnapshot?)null);
                    }

                    if (connector == null)
                    {
                        return ((string?)null, (UserDataSnapshot?)null);
                    }

                    // Fetch balance, positions, and fees using dynamic to call connector methods
                    dynamic dynamicConnector = connector;
                    var balance = await dynamicConnector.GetAccountBalanceAsync();
                    var positions = await dynamicConnector.GetOpenPositionsAsync();

                    // Fetch fee information
                    FeeInfoDto? feeInfo = null;
                    try
                    {
                        feeInfo = await dynamicConnector.GetTradingFeesAsync();
                        feeInfo.UserId = apiKey.UserId; // Set user ID on the fee info
                        _logger.LogDebug("Collected fee info for user {UserId} on {Exchange}: Maker={Maker}%, Taker={Taker}%",
                            apiKey.UserId, apiKey.ExchangeName, feeInfo.MakerFeeRate * 100, feeInfo.TakerFeeRate * 100);
                    }
                    catch (Exception feeEx)
                    {
                        _logger.LogWarning(feeEx, "Failed to fetch fee info for user {UserId} on {Exchange}, continuing without fees",
                            apiKey.UserId, apiKey.ExchangeName);
                    }

                    // Enrich positions with database data (Id and ExecutionId)
                    await EnrichPositionsWithDatabaseDataAsync(positions, apiKey.UserId, apiKey.ExchangeName, cancellationToken);

                    // Store count before using in snapshot to avoid dynamic in log call
                    int positionCount = positions.Count;

                    var snapshot = new UserDataSnapshot
                    {
                        UserId = apiKey.UserId,
                        Exchange = apiKey.ExchangeName,
                        Balance = balance,
                        Positions = positions,
                        FeeInfo = feeInfo,
                        CollectedAt = DateTime.UtcNow
                    };

                    var key = $"userdata:{apiKey.UserId}:{apiKey.ExchangeName}";

                    _logger.LogDebug("Collected data for user {UserId} on {Exchange}: {PositionCount} positions",
                        apiKey.UserId, apiKey.ExchangeName, positionCount);

                    return (key, snapshot);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect data for user {UserId} on {Exchange}",
                        apiKey.UserId, apiKey.ExchangeName);
                    return ((string?)null, (UserDataSnapshot?)null);
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
                snapshots[item.Item1!] = item.Item2!;
            }

            // Store in repository (memory cache)
            if (snapshots.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(10);

                await _repository.StoreBatchAsync(snapshots, ttl, cancellationToken);

                result.Data = snapshots;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;
                
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No user data collected successfully";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting user data");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    /// <summary>
    /// Enriches exchange positions with database data (Id and ExecutionId)
    /// Matches positions by Symbol and Side to find the corresponding database record
    /// </summary>
    private async Task EnrichPositionsWithDatabaseDataAsync(
        List<PositionDto> exchangePositions,
        string userId,
        string exchangeName,
        CancellationToken cancellationToken)
    {
        try
        {
            if (!exchangePositions.Any())
            {
                return;
            }

            // Query database for open positions for this user and exchange using a scoped DbContext
            List<Position> dbPositions;
            using (var scope = _serviceProvider.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
                dbPositions = await dbContext.Positions
                    .Where(p => p.UserId == userId &&
                                p.Exchange == exchangeName &&
                                p.Status == PositionStatus.Open)
                    .ToListAsync(cancellationToken);
            }

            if (!dbPositions.Any())
            {
                _logger.LogDebug("No database positions found for user {UserId} on {Exchange}", userId, exchangeName);
                return;
            }

            // Match and enrich each exchange position
            int enrichedCount = 0;
            foreach (var exchangePosition in exchangePositions)
            {
                // Find matching database position by Symbol and Side
                var dbPosition = dbPositions.FirstOrDefault(p =>
                    p.Symbol == exchangePosition.Symbol &&
                    p.Side == exchangePosition.Side);

                if (dbPosition != null)
                {
                    // Copy Id and ExecutionId from database
                    exchangePosition.Id = dbPosition.Id;
                    exchangePosition.ExecutionId = dbPosition.ExecutionId;
                    enrichedCount++;

                    _logger.LogDebug(
                        "Enriched position {Symbol} {Side} with Id={Id}, ExecutionId={ExecutionId}",
                        exchangePosition.Symbol,
                        exchangePosition.Side,
                        dbPosition.Id,
                        dbPosition.ExecutionId);
                }
                else
                {
                    _logger.LogDebug(
                        "No database match found for exchange position {Symbol} {Side}",
                        exchangePosition.Symbol,
                        exchangePosition.Side);
                }
            }

            _logger.LogInformation(
                "Enriched {EnrichedCount}/{TotalCount} exchange positions with database data for user {UserId} on {Exchange}",
                enrichedCount,
                exchangePositions.Count,
                userId,
                exchangeName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enriching positions with database data for user {UserId} on {Exchange}",
                userId, exchangeName);
        }
    }
}
