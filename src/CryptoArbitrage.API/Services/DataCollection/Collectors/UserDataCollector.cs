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
    private readonly ConnectorManager _connectorManager;

    public UserDataCollectorConfiguration Configuration => _configuration;
    public CollectionResult<UserDataSnapshot>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public UserDataCollector(
        ILogger<UserDataCollector> logger,
        IDataRepository<UserDataSnapshot> repository,
        UserDataCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        ConnectorManager connectorManager)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _connectorManager = connectorManager;
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
                        return ((string?)null, (UserDataSnapshot?)null);
                    }

                    // Fetch balance and fees
                    var balance = await connector.GetAccountBalanceAsync();

                    // Fetch fee information
                    FeeInfoDto? feeInfo = null;
                    try
                    {
                        feeInfo = await connector.GetTradingFeesAsync();
                        feeInfo.UserId = apiKey.UserId; // Set user ID on the fee info
                        _logger.LogDebug("Collected fee info for user {UserId} on {Exchange}: Maker={Maker}%, Taker={Taker}%",
                            apiKey.UserId, apiKey.ExchangeName, feeInfo.MakerFeeRate * 100, feeInfo.TakerFeeRate * 100);
                    }
                    catch (Exception feeEx)
                    {
                        _logger.LogWarning(feeEx, "Failed to fetch fee info for user {UserId} on {Exchange}, continuing without fees",
                            apiKey.UserId, apiKey.ExchangeName);
                    }

                    // Fetch positions from DATABASE (source of truth), not exchange API
                    var positions = await GetPositionsFromDatabaseAsync(apiKey.UserId, apiKey.ExchangeName, connector, cancellationToken);

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
    /// Fetches positions from database (source of truth) and optionally updates unrealizedPnL from exchange
    /// </summary>
    private async Task<List<PositionDto>> GetPositionsFromDatabaseAsync(
        string userId,
        string exchangeName,
        IExchangeConnector connector,
        CancellationToken cancellationToken)
    {
        var positionDtos = new List<PositionDto>();

        try
        {
            using (var scope = _serviceProvider.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                // Fetch positions from DATABASE (source of truth)
                var dbPositions = await dbContext.Positions
                    .Where(p => p.UserId == userId &&
                                p.Exchange == exchangeName &&
                                p.Status == PositionStatus.Open)
                    .ToListAsync(cancellationToken);

                if (!dbPositions.Any())
                {
                    _logger.LogDebug("No open positions in database for user {UserId} on {Exchange}", userId, exchangeName);
                    return positionDtos;
                }

                // Get all position IDs to fetch transactions in one query
                var positionIds = dbPositions.Select(p => p.Id).ToList();
                var allTransactions = await dbContext.PositionTransactions
                    .Where(pt => positionIds.Contains(pt.PositionId))
                    .ToListAsync(cancellationToken);

                // Optionally fetch real-time unrealizedPnL from exchange
                Dictionary<string, decimal> exchangePnLMap = new();
                try
                {
                    var exchangePositions = await connector.GetOpenPositionsAsync();
                    foreach (var exPos in exchangePositions)
                    {
                        var key = $"{exPos.Symbol}_{exPos.Side}";
                        exchangePnLMap[key] = exPos.UnrealizedPnL;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to fetch real-time P&L from exchange for {Exchange}, using database values", exchangeName);
                }

                // Convert database positions to DTOs
                foreach (var dbPos in dbPositions)
                {
                    var positionTransactions = allTransactions.Where(pt => pt.PositionId == dbPos.Id).ToList();

                    // Calculate fees from PositionTransaction (single source of truth)
                    var tradingFeePaid = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.Commission || pt.TransactionType == TransactionType.Trade)
                        .Sum(pt => pt.Fee);

                    var totalFundingFeePaid = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee < 0)
                        .Sum(pt => Math.Abs(pt.SignedFee ?? 0));

                    var totalFundingFeeReceived = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee > 0)
                        .Sum(pt => pt.SignedFee ?? 0);

                    // Try to get real-time unrealizedPnL from exchange, fallback to database
                    var pnlKey = $"{dbPos.Symbol}_{dbPos.Side}";
                    var unrealizedPnL = exchangePnLMap.ContainsKey(pnlKey) ? exchangePnLMap[pnlKey] : dbPos.UnrealizedPnL;

                    var dto = new PositionDto
                    {
                        Id = dbPos.Id,
                        ExecutionId = dbPos.ExecutionId,
                        Exchange = dbPos.Exchange,
                        Symbol = dbPos.Symbol,
                        Type = dbPos.Type,
                        Side = dbPos.Side,
                        Status = dbPos.Status,
                        EntryPrice = dbPos.EntryPrice,
                        ExitPrice = dbPos.ExitPrice,
                        Quantity = dbPos.Quantity,
                        Leverage = dbPos.Leverage,
                        InitialMargin = dbPos.InitialMargin,
                        FundingEarnedUsd = dbPos.FundingEarnedUsd,
                        TradingFeesUsd = dbPos.TradingFeesUsd,
                        PricePnLUsd = dbPos.PricePnLUsd,
                        RealizedPnLUsd = dbPos.RealizedPnLUsd,
                        RealizedPnLPct = dbPos.RealizedPnLPct,
                        UnrealizedPnL = unrealizedPnL, // Real-time from exchange or database fallback
                        TradingFeePaid = tradingFeePaid,
                        TotalFundingFeePaid = totalFundingFeePaid,
                        TotalFundingFeeReceived = totalFundingFeeReceived,
                        ReconciliationStatus = dbPos.ReconciliationStatus,
                        ReconciliationCompletedAt = dbPos.ReconciliationCompletedAt,
                        OpenedAt = dbPos.OpenedAt, // Correct timestamp from database
                        ClosedAt = dbPos.ClosedAt,
                        ActiveOpportunityId = dbPos.ExecutionId
                    };

                    positionDtos.Add(dto);
                }

                _logger.LogInformation(
                    "Loaded {Count} open positions from database for user {UserId} on {Exchange}",
                    positionDtos.Count,
                    userId,
                    exchangeName);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading positions from database for user {UserId} on {Exchange}",
                userId, exchangeName);
        }

        return positionDtos;
    }
}
