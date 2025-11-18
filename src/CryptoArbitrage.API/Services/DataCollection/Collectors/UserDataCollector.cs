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
using System.Text.Json;

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

                // Fetch positions from DATABASE (source of truth) for THIS exchange
                var currentExchangePositions = await dbContext.Positions
                    .Where(p => p.UserId == userId &&
                                p.Exchange == exchangeName &&
                                p.Status == PositionStatus.Open)
                    .ToListAsync(cancellationToken);

                if (!currentExchangePositions.Any())
                {
                    _logger.LogDebug("No open positions in database for user {UserId} on {Exchange}", userId, exchangeName);
                    return positionDtos;
                }

                // CRITICAL FIX: For cross-exchange arbitrage, load ALL positions from ALL exchanges
                // that share the same ExecutionIds as positions on current exchange
                var executionIds = currentExchangePositions.Select(p => p.ExecutionId).Distinct().ToList();
                var dbPositions = await dbContext.Positions
                    .Where(p => p.UserId == userId &&
                                executionIds.Contains(p.ExecutionId) &&
                                p.Status == PositionStatus.Open)
                    .ToListAsync(cancellationToken);

                // Log loaded positions for debugging
                var positionSummary = string.Join(", ", dbPositions.Select(p => $"{p.Symbol}_{p.Side}(ID:{p.Id},Exec:{p.ExecutionId},Exch:{p.Exchange})"));
                _logger.LogInformation("Loaded {Count} total positions (across all exchanges) for {ExecCount} executions on {UserId}: {Positions}",
                    dbPositions.Count, executionIds.Count, userId, positionSummary);

                // Check for duplicate positions (same symbol+side) which indicates cleanup issue
                var duplicates = dbPositions.GroupBy(p => $"{p.Symbol}_{p.Side}").Where(g => g.Count() > 1).ToList();
                if (duplicates.Any())
                {
                    foreach (var dup in duplicates)
                    {
                        var ids = string.Join(", ", dup.Select(p => $"ID:{p.Id}(Exec:{p.ExecutionId})"));
                        _logger.LogWarning("DUPLICATE OPEN POSITIONS for {Key}: {Ids} - This causes P&L carryover bug!",
                            dup.Key, ids);
                    }
                }

                // Fetch real-time unrealizedPnL from exchange FIRST (before P&L snapshot calculations)
                // This ensures P&L snapshots use fresh data, not stale database values
                Dictionary<string, decimal> exchangePnLMap = new();
                try
                {
                    var exchangePositions = await connector.GetOpenPositionsAsync();
                    foreach (var exPos in exchangePositions)
                    {
                        var key = $"{exPos.Symbol}_{exPos.Side}";
                        exchangePnLMap[key] = exPos.UnrealizedPnL;
                    }
                    _logger.LogDebug("Fetched real-time P&L for {Count} positions from {Exchange}", exchangePnLMap.Count, exchangeName);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to fetch real-time P&L from exchange for {Exchange}, using database values", exchangeName);
                }

                // PHASE 1: P&L tracking for Phase 1 exit timing features
                // Snapshots taken every collection cycle, keeping last 6 values (matches training environment)
                var currentTime = DateTime.UtcNow;
                var snapshotUpdated = false;

                // Group positions by ExecutionId to calculate NET execution P&L (not individual position P&L)
                // This ensures pnl_history matches the unrealized_pnl_pct calculation in RLPredictionService
                var positionsByExecution = dbPositions.GroupBy(p => p.ExecutionId).ToList();

                foreach (var executionGroup in positionsByExecution)
                {
                    var positions = executionGroup.ToList();

                    // Update UnrealizedPnL from exchange for all positions in this execution
                    decimal totalUnrealizedPnL = 0m;
                    decimal totalPositionSize = 0m;  // FIXED: Use position SIZE, not margin (to match environment & RLPredictionService)

                    foreach (var pos in positions)
                    {
                        var pnlKey = $"{pos.Symbol}_{pos.Side}";
                        var unrealizedPnL = exchangePnLMap.ContainsKey(pnlKey) ? exchangePnLMap[pnlKey] : pos.UnrealizedPnL;

                        // Update database position with fresh exchange UnrealizedPnL
                        if (exchangePnLMap.ContainsKey(pnlKey))
                        {
                            pos.UnrealizedPnL = unrealizedPnL;
                        }

                        totalUnrealizedPnL += unrealizedPnL;
                        // CRITICAL: Use position SIZE (quantity * entryPrice), NOT initial margin
                        // This matches RLPredictionService and ML environment calculation
                        // With 2x leverage: position_size = $1000, margin = $500
                        // We want P&L relative to $1000, not $500
                        totalPositionSize += pos.Quantity * pos.EntryPrice;
                    }

                    // Calculate NET execution P&L percentage (matches RLPredictionService & environment)
                    // Environment: unrealized_pnl_pct = (pnl_usd / (position_size_usd * 2)) * 100
                    // RLPredictionService: totalPnl / totalCapitalUsed * 100 where totalCapitalUsed = longSize + shortSize
                    var currentNetPnlPct = totalPositionSize > 0
                        ? (totalUnrealizedPnL / totalPositionSize) * 100m
                        : 0m;

                    _logger.LogInformation(
                        "Execution {ExecutionId} NET P&L: {NetPnl:F2}% (total unrealized: ${TotalPnl}, total position size: ${TotalSize})",
                        executionGroup.Key, currentNetPnlPct, totalUnrealizedPnL, totalPositionSize);

                    // Update P&L history and peak for ALL positions in this execution with the SAME net P&L
                    foreach (var pos in positions)
                    {
                        var positionAge = DateTime.UtcNow - pos.OpenedAt;

                        // DEFENSIVE FIX: Reset P&L history for very young positions to prevent carryover
                        // If position is younger than collection interval (5 min) but already has P&L history,
                        // it means data is carrying over from a previous position → reset it
                        var isVeryYoung = positionAge.TotalMinutes < 6; // Younger than one collection cycle

                        // Parse existing PnL history from JSON
                        List<decimal> pnlHistory;
                        if (!string.IsNullOrEmpty(pos.PnlHistoryJson))
                        {
                            try
                            {
                                pnlHistory = JsonSerializer.Deserialize<List<decimal>>(pos.PnlHistoryJson) ?? new List<decimal>();

                                // DEFENSIVE: If position is very young but already has data, check if it's TRUE carryover
                                if (isVeryYoung && pnlHistory.Any())
                                {
                                    // Check if this is TRUE carryover (old snapshot time) vs fresh data from recent collector runs
                                    var timeSinceLastCarryoverCheck = pos.LastPnlSnapshotTime.HasValue
                                        ? DateTime.UtcNow - pos.LastPnlSnapshotTime.Value
                                        : TimeSpan.MaxValue;

                                    // If last snapshot is >10 minutes old, it's carryover from previous position → reset
                                    // Otherwise, it's fresh data from recent collector runs (every 5 min) → keep it!
                                    if (timeSinceLastCarryoverCheck.TotalMinutes > 10)
                                    {
                                        _logger.LogWarning(
                                            "Position {PositionId} ({Symbol}) is only {Age:F1} minutes old but has P&L snapshots from {SnapshotAge:F1} minutes ago - resetting carryover data",
                                            pos.Id, pos.Symbol, positionAge.TotalMinutes, timeSinceLastCarryoverCheck.TotalMinutes);
                                        pnlHistory.Clear();
                                        pos.PeakPnlPct = 0m; // Also reset peak
                                    }
                                }
                            }
                            catch (JsonException)
                            {
                                _logger.LogWarning("Failed to parse PnlHistoryJson for Position {PositionId}, resetting", pos.Id);
                                pnlHistory = new List<decimal>();
                            }
                        }
                        else
                        {
                            pnlHistory = new List<decimal>();
                        }

                        // CRITICAL FIX: Only update P&L history every 5 minutes to match ML model expectations
                        // The collector runs every 1 second, but we should only snapshot P&L periodically
                        var timeSinceLastSnapshot = pos.LastPnlSnapshotTime.HasValue
                            ? DateTime.UtcNow - pos.LastPnlSnapshotTime.Value
                            : TimeSpan.MaxValue;

                        bool shouldUpdateHistory = timeSinceLastSnapshot.TotalMinutes >= 5.0;

                        if (shouldUpdateHistory)
                        {
                            // Append current NET execution P&L % (not individual position P&L)
                            // This ensures pnl_history matches unrealized_pnl_pct calculation in RLPredictionService
                            pnlHistory.Add(currentNetPnlPct);

                            // Keep only last 6 snapshots (matches training environment where portfolio.update_pnl() keeps 6)
                            // With 5-minute intervals, this gives us 30 minutes of history
                            if (pnlHistory.Count > 6)
                            {
                                pnlHistory = pnlHistory.TakeLast(6).ToList();
                            }

                            // Update Position fields
                            pos.PnlHistoryJson = JsonSerializer.Serialize(pnlHistory);
                            pos.LastPnlSnapshotTime = currentTime;

                            _logger.LogDebug(
                                "Updated P&L history for Position {PositionId}: added snapshot #{Count} with value {Value:F2}%",
                                pos.Id, pnlHistory.Count, currentNetPnlPct);
                        }
                        else
                        {
                            // Even if not updating history, always update current P&L for real-time monitoring
                            _logger.LogTrace(
                                "Skipping P&L history update for Position {PositionId}: only {MinutesSince:F1} minutes since last snapshot",
                                pos.Id, timeSinceLastSnapshot.TotalMinutes);
                        }

                        // Update PeakPnlPct if current NET P&L is higher
                        // Store same peak for all positions in execution (consistency with pnl_history)
                        if (currentNetPnlPct > pos.PeakPnlPct)
                        {
                            pos.PeakPnlPct = currentNetPnlPct;
                        }

                        snapshotUpdated = true;

                        _logger.LogInformation(
                            "Updated P&L snapshot for Position {PositionId} ({Symbol}, Exec:{ExecutionId}, Age:{Age:F1}min): NET P&L={NetPnl:F2}%, Peak={PeakPct:F2}%, History={Count} snapshots",
                            pos.Id, pos.Symbol, pos.ExecutionId, positionAge.TotalMinutes, currentNetPnlPct, pos.PeakPnlPct, pnlHistory.Count);
                    }
                }

                // Save changes if any position was updated
                if (snapshotUpdated)
                {
                    await dbContext.SaveChangesAsync(cancellationToken);
                    _logger.LogDebug(
                        "Saved P&L snapshots for {Count} positions (user {UserId}, exchange {Exchange})",
                        dbPositions.Count,
                        userId,
                        exchangeName);
                }

                // Get all position IDs to fetch transactions in one query
                var positionIds = dbPositions.Select(p => p.Id).ToList();
                var allTransactions = await dbContext.PositionTransactions
                    .Where(pt => positionIds.Contains(pt.PositionId))
                    .ToListAsync(cancellationToken);

                // Convert database positions to DTOs
                // Note: exchangePnLMap is already populated at the top of this method
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
                        ActiveOpportunityId = dbPos.ExecutionId,

                        // ML features (Phase 1 exit timing) - copy from database
                        EntryApr = dbPos.EntryApr,
                        LastKnownApr = dbPos.LastKnownApr,
                        PeakPnlPct = dbPos.PeakPnlPct,
                        PnlHistoryJson = dbPos.PnlHistoryJson,

                        // Funding interval metadata - copy from database
                        LongFundingIntervalHours = dbPos.LongFundingIntervalHours,
                        ShortFundingIntervalHours = dbPos.ShortFundingIntervalHours
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
