using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Arbitrage.Detection;
using CryptoArbitrage.API.Services.Arbitrage.Execution;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.Streaming;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.Agent;

/// <summary>
/// Background service that manages autonomous trading agents.
/// One agent per user, running continuously with prediction loops.
/// </summary>
public class AgentBackgroundService : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<AgentBackgroundService> _logger;
    private readonly HttpClient _httpClient;
    private readonly string _mlApiBaseUrl;
    private readonly JsonSerializerOptions _jsonOptions;

    // Track running agents: user_id -> CancellationTokenSource
    private readonly Dictionary<string, CancellationTokenSource> _runningAgents = new();
    private readonly object _lock = new();

    // Execution locks to prevent race conditions: user_id -> SemaphoreSlim
    // Prevents multiple predictions from executing simultaneously for the same user
    private readonly Dictionary<string, SemaphoreSlim> _executionLocks = new();

    public AgentBackgroundService(
        IServiceProvider serviceProvider,
        ILogger<AgentBackgroundService> logger,
        IConfiguration configuration)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;

        // Configure ML API client
        var host = configuration["MLApi:Host"] ?? "localhost";
        var port = configuration["MLApi:Port"] ?? "5250";
        _mlApiBaseUrl = $"http://{host}:{port}";

        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_mlApiBaseUrl),
            Timeout = TimeSpan.FromSeconds(30)
        };

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        _logger.LogInformation("AgentBackgroundService initialized. ML API: {BaseUrl}", _mlApiBaseUrl);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("AgentBackgroundService started");

        // Main monitoring loop - check for agent start/stop requests
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckAndStartAgentsAsync(stoppingToken);
                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in AgentBackgroundService monitoring loop");
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
        }

        // Stop all agents on shutdown
        await StopAllAgentsAsync();
        _logger.LogInformation("AgentBackgroundService stopped");
    }

    /// <summary>
    /// Check database for agents that should be running and start them
    /// </summary>
    private async Task CheckAndStartAgentsAsync(CancellationToken stoppingToken)
    {
        using var scope = _serviceProvider.CreateScope();
        var context = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        // Find all sessions with status = Running that aren't already tracked
        var runningSessions = await context.AgentSessions
            .Where(s => s.Status == AgentStatus.Running)
            .Include(s => s.AgentConfiguration)
            .ToListAsync(stoppingToken);

        foreach (var session in runningSessions)
        {
            lock (_lock)
            {
                if (!_runningAgents.ContainsKey(session.UserId))
                {
                    // Start agent for this user
                    var cts = new CancellationTokenSource();
                    _runningAgents[session.UserId] = cts;

                    _logger.LogInformation("Starting agent for user {UserId}", session.UserId);

                    // Start agent loop in background
                    _ = Task.Run(() => RunAgentLoopAsync(session.UserId, session.AgentConfiguration, cts.Token), stoppingToken);
                }
            }
        }

        // Stop agents that are no longer Running in database
        lock (_lock)
        {
            var usersToStop = _runningAgents.Keys
                .Where(userId => !runningSessions.Any(s => s.UserId == userId))
                .ToList();

            foreach (var userId in usersToStop)
            {
                _logger.LogInformation("Stopping agent for user {UserId} (session no longer running)", userId);
                _runningAgents[userId].Cancel();
                _runningAgents.Remove(userId);
            }
        }
    }

    /// <summary>
    /// Main agent loop for a single user
    /// </summary>
    private async Task RunAgentLoopAsync(string userId, AgentConfiguration config, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Agent loop started for user {UserId}", userId);

        var predictionIntervalSeconds = config.PredictionIntervalSeconds;
        var predictionCount = 0;

        _logger.LogInformation("Using prediction interval: {Interval} seconds ({Minutes} minutes)",
            predictionIntervalSeconds, predictionIntervalSeconds / 60.0);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Run one prediction cycle
                    await RunPredictionCycleAsync(userId, config, cancellationToken);
                    predictionCount++;

                    // Wait for next cycle
                    await Task.Delay(TimeSpan.FromSeconds(predictionIntervalSeconds), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in prediction cycle for user {UserId}", userId);
                    await SetAgentErrorAsync(userId, ex.Message);

                    // Back off on errors
                    await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
                }
            }
        }
        finally
        {
            _logger.LogInformation("Agent loop ended for user {UserId}. Total predictions: {Count}", userId, predictionCount);
            lock (_lock)
            {
                _runningAgents.Remove(userId);
            }
        }
    }

    /// <summary>
    /// Run one prediction cycle: get state, call ML API, execute action
    /// </summary>
    private async Task RunPredictionCycleAsync(string userId, AgentConfiguration config, CancellationToken cancellationToken)
    {
        using var scope = _serviceProvider.CreateScope();
        var context = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
        var opportunityRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<ArbitrageOpportunityDto>>();
        var userDataRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<UserDataSnapshot>>();
        var executionService = scope.ServiceProvider.GetRequiredService<ArbitrageExecutionService>();
        var signalR = scope.ServiceProvider.GetRequiredService<ISignalRStreamingService>();

        _logger.LogDebug("Running prediction cycle for user {UserId}", userId);

        // 1. Get current opportunities from cache (already detected by OpportunityAggregator)
        var opportunitiesDict = await opportunityRepository.GetByPatternAsync("opportunity:*");

        // Smart opportunity selection for ML API (up to 10 total)
        // Strategy:
        // - Filter by SubType: Only CrossExchangeFuturesFutures (CFFF)
        // - Always include opportunities with existing positions (for EXIT decisions)
        // - From new opportunities (Good or Medium liquidity):
        //   * Top 7 by highest FundApr
        //   * Top 3 by lowest spread
        //   * Remove duplicates

        var allOpportunities = opportunitiesDict.Values.ToList();

        // FILTER: Only send CrossExchangeFuturesFutures (CFFF) opportunities to ML
        // This is the primary funding arbitrage strategy (long perp on one exchange + short perp on another)
        var cfffOpportunities = allOpportunities
            .Where(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures)
            .ToList();

        _logger.LogInformation("Filtered opportunities by strategy: {Total} total → {CFFF} CrossExchangeFuturesFutures",
            allOpportunities.Count, cfffOpportunities.Count);

        // Separate existing positions from new opportunities
        var existingPositionOpps = cfffOpportunities
            .Where(o => o.IsExistingPosition)
            .ToList();

        var newOpportunities = cfffOpportunities
            .Where(o => !o.IsExistingPosition &&
                       (o.LiquidityStatus == LiquidityStatus.Good || o.LiquidityStatus == LiquidityStatus.Medium))
            .ToList();

        var liquidityCounts = newOpportunities
            .Where(o => o.LiquidityStatus.HasValue)
            .GroupBy(o => o.LiquidityStatus!.Value)
            .ToDictionary(g => g.Key, g => g.Count());

        _logger.LogDebug("Opportunity pool (CFFF only): {Total} total, {Existing} with positions, {New} new (Good: {Good}, Medium: {Medium})",
            cfffOpportunities.Count,
            existingPositionOpps.Count,
            newOpportunities.Count,
            liquidityCounts.GetValueOrDefault(LiquidityStatus.Good, 0),
            liquidityCounts.GetValueOrDefault(LiquidityStatus.Medium, 0));

        // Select top 7 by highest APR
        var topByApr = newOpportunities
            .OrderByDescending(o => o.FundApr)
            .Take(7)
            .ToList();

        // Select top 3 by lowest 30-sample spread average (better stability)
        var topBySpread = newOpportunities
            .Where(o => o.Spread30SampleAvg.HasValue)
            .OrderBy(o => o.Spread30SampleAvg!.Value)
            .Take(3)
            .ToList();

        // Combine APR and spread selections, removing duplicates
        var selectedNew = topByApr
            .Union(topBySpread, new OpportunityComparer())
            .ToList();

        _logger.LogDebug("Selected {Apr} by APR, {Spread} by spread, {Unique} unique new opportunities",
            topByApr.Count, topBySpread.Count, selectedNew.Count);

        // Combine existing positions with selected new opportunities (up to 10 total)
        var userOpportunities = existingPositionOpps
            .Concat(selectedNew)
            .Take(10)
            .ToList();

        _logger.LogInformation("Sending {Total} CFFF opportunities to ML: {Existing} existing positions + {New} new opportunities",
            userOpportunities.Count, existingPositionOpps.Count, Math.Min(selectedNew.Count, 10 - existingPositionOpps.Count));

        if (userOpportunities.Count == 0)
        {
            _logger.LogDebug("No CFFF opportunities available for user {UserId}", userId);
            return;
        }

        // 2. Get current positions from repository (real-time positions from UserDataCollector)
        var positions = await GetPositionsFromRepositoryAsync(userId, userDataRepository, cancellationToken);

        var numExecutionsForLogging = positions.Select(p => p.ExecutionId).Distinct().Count();

        _logger.LogInformation("========== AGENT SENDING TO ML API ==========");
        _logger.LogInformation("User: {UserId}, Executions: {ExecutionCount} ({PositionCount} individual positions)",
            userId, numExecutionsForLogging, positions.Count);
        foreach (var pos in positions)
        {
            _logger.LogInformation("  - Position: {Symbol} {Side} on {Exchange}, ExecutionId: {ExecutionId}",
                pos.Symbol, pos.Side, pos.Exchange, pos.ExecutionId);
        }
        _logger.LogInformation("=============================================");

        // 3. Get market data and funding rate repositories for enriching portfolio state
        var marketDataRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<MarketDataSnapshot>>();
        var fundingRateRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<FundingRateDto>>();

        // 4. Get portfolio state with enriched position data
        var portfolio = await BuildPortfolioStateAsync(userId, positions, context, marketDataRepository, fundingRateRepository, userDataRepository, cancellationToken);

        // 5. Call ML API for prediction
        var prediction = await GetMLPredictionAsync(userId, config, userOpportunities, portfolio);

        _logger.LogInformation("========== ML API RETURNED ==========");
        _logger.LogInformation("Action: {Action}, Symbol: {Symbol}", prediction?.Action, prediction?.OpportunitySymbol);
        _logger.LogInformation("====================================");

        if (prediction == null)
        {
            _logger.LogWarning("ML API returned null prediction for user {UserId}", userId);
            return;
        }

        // 6. Execute action based on prediction
        var executionResult = await ExecuteAgentActionAsync(userId, prediction, userOpportunities, positions, executionService, context, userDataRepository, cancellationToken);

        // 7. Update stats and broadcast decision
        await UpdateAgentStatsAsync(userId, prediction.Action, context);
        await BroadcastAgentDecisionAsync(userId, prediction, userOpportunities, positions, executionResult, signalR);

        // 8. Broadcast update via SignalR
        await BroadcastAgentUpdateAsync(userId, signalR, context);

        _logger.LogDebug("Prediction cycle completed for user {UserId}. Action: {Action}", userId, prediction.Action);
    }

    /// <summary>
    /// Get positions from repository (real-time positions from UserDataCollector)
    /// </summary>
    private async Task<List<PositionDto>> GetPositionsFromRepositoryAsync(
        string userId,
        IDataRepository<UserDataSnapshot> userDataRepository,
        CancellationToken cancellationToken)
    {
        var allPositions = new List<PositionDto>();

        try
        {
            // Get all user data snapshots for this user across all exchanges
            var userDataDict = await userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);

            foreach (var snapshot in userDataDict.Values)
            {
                if (snapshot?.Positions != null)
                {
                    allPositions.AddRange(snapshot.Positions);
                }
            }

            _logger.LogDebug("Retrieved {Count} positions from repository for user {UserId}", allPositions.Count, userId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting positions from repository for user {UserId}", userId);
        }

        return allPositions;
    }

    /// <summary>
    /// Build portfolio state for ML API
    /// </summary>
    private async Task<Dictionary<string, object>> BuildPortfolioStateAsync(
        string userId,
        List<PositionDto> positions,
        ArbitrageDbContext context,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        IDataRepository<FundingRateDto> fundingRateRepository,
        IDataRepository<UserDataSnapshot> userDataRepository,
        CancellationToken cancellationToken)
    {
        // Query real balance data from all exchanges
        var userDataDict = await userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);

        // Calculate total capital across all exchanges
        decimal totalCapital = 0m;
        decimal totalMarginUsed = 0m;

        foreach (var snapshot in userDataDict.Values)
        {
            if (snapshot?.Balance != null)
            {
                // Sum up futures balance (total equity) and margin used across all exchanges
                totalCapital += snapshot.Balance.FuturesBalanceUsd;
                totalMarginUsed += snapshot.Balance.MarginUsed;

                _logger.LogDebug(
                    "Exchange balance for user {UserId}: FuturesBalance=${Balance:F2}, MarginUsed=${Margin:F2}",
                    userId, snapshot.Balance.FuturesBalanceUsd, snapshot.Balance.MarginUsed);
            }
        }

        // Validate that we have balance data
        if (totalCapital == 0m)
        {
            throw new InvalidOperationException(
                $"Unable to retrieve balance data for user {userId}. " +
                "User data collector may not be running or data is stale.");
        }

        // Get total P&L from performance metrics if available
        var latestMetric = await context.PerformanceMetrics
            .Where(pm => pm.UserId == userId)
            .OrderByDescending(pm => pm.Date)
            .FirstOrDefaultAsync();
        var totalPnL = latestMetric?.TotalPnL ?? 0m;

        // Calculate utilization based on real margin used
        var utilization = totalCapital > 0 ? (double)(totalMarginUsed / totalCapital) : 0.0;

        _logger.LogInformation(
            "Portfolio state for user {UserId}: TotalCapital=${Total:F2}, MarginUsed=${Margin:F2}, Utilization={Util:F2}%",
            userId, totalCapital, totalMarginUsed, utilization * 100);

        // Fetch market data and funding rates once for all positions
        var marketSnapshot = await marketDataRepository.GetAsync("market_data_snapshot", cancellationToken);

        // Convert positions to enriched dictionaries with execution data
        var enrichedPositions = new List<Dictionary<string, object>>();
        foreach (var position in positions)
        {
            var enrichedPos = await ConvertPositionDtoToDictAsync(position, positions, marketSnapshot, totalCapital, cancellationToken);
            enrichedPositions.Add(enrichedPos);
        }

        // IMPORTANT: Count EXECUTIONS, not individual positions
        // Each execution has 2 positions (long + short hedge), but the ML model expects execution count
        // The training environment's "position" semantically means "execution"
        var numExecutions = positions.Select(p => p.ExecutionId).Distinct().Count();

        return new Dictionary<string, object>
        {
            ["total_capital"] = (double)totalCapital,
            ["initial_capital"] = (double)totalCapital, // Use current total capital (user's requirement: don't change ML model input structure)
            ["utilization"] = utilization,
            ["total_pnl_pct"] = (double)(totalPnL / totalCapital * 100),
            ["num_positions"] = numExecutions,  // Count of executions, not individual positions (matches training semantics)
            ["positions"] = enrichedPositions
        };
    }

    /// <summary>
    /// Convert PositionDto to enriched dictionary for ML API with execution-specific data
    /// </summary>
    private Task<Dictionary<string, object>> ConvertPositionDtoToDictAsync(
        PositionDto position,
        List<PositionDto> allPositions,
        MarketDataSnapshot? marketSnapshot,
        decimal totalCapital,
        CancellationToken cancellationToken)
    {
        var hoursHeld = (DateTime.UtcNow - position.OpenedAt).TotalHours;
        var positionSizeUsd = (double)position.Quantity * (double)position.EntryPrice;
        var unrealizedPnlPct = positionSizeUsd > 0 ? (double)position.UnrealizedPnL / positionSizeUsd * 100 : 0;

        // Basic position data
        var result = new Dictionary<string, object>
        {
            ["symbol"] = position.Symbol ?? "",
            ["position_size_usd"] = positionSizeUsd,
            ["unrealized_pnl_pct"] = unrealizedPnlPct,
            ["hours_held"] = hoursHeld / 72.0,  // Normalized by 72 hours as expected by ML model
            ["leverage"] = (double)position.Leverage,
            ["position_is_active"] = 1.0  // Always 1.0 for open positions
        };

        // Enrich with execution-specific data (ExecutionId is always present)
        // Find the paired position (for cross-exchange arbitrage, there's a long and short)
        var pairedPosition = allPositions.FirstOrDefault(p =>
            p.ExecutionId == position.ExecutionId &&
            p.Id != position.Id &&
            p.Status == PositionStatus.Open);

        if (pairedPosition != null)
        {
            // Determine which is long and which is short
            var longPos = position.Side == PositionSide.Long ? position : pairedPosition;
            var shortPos = position.Side == PositionSide.Short ? position : pairedPosition;

            // Calculate execution-specific metrics
                var longSizeUsd = (double)longPos.Quantity * (double)longPos.EntryPrice;
                var shortSizeUsd = (double)shortPos.Quantity * (double)shortPos.EntryPrice;

                // Individual leg P&Ls
                result["long_pnl_pct"] = longSizeUsd > 0 ? (double)longPos.UnrealizedPnL / longSizeUsd * 100 : 0;
                result["short_pnl_pct"] = shortSizeUsd > 0 ? (double)shortPos.UnrealizedPnL / shortSizeUsd * 100 : 0;

                // Entry prices for both legs
                var entryLongPrice = (double)longPos.EntryPrice;
                var entryShortPrice = (double)shortPos.EntryPrice;
                result["entry_long_price"] = entryLongPrice;
                result["entry_short_price"] = entryShortPrice;

                // Fetch current prices from market snapshot
                var currentLongPrice = entryLongPrice; // Default to entry price
                var currentShortPrice = entryShortPrice;

                if (marketSnapshot?.PerpPrices != null)
                {
                    // Try to get current price from snapshot
                    if (marketSnapshot.PerpPrices.TryGetValue(longPos.Exchange, out var longExchangePrices) &&
                        longExchangePrices.TryGetValue(longPos.Symbol, out var longPriceDto))
                    {
                        currentLongPrice = (double)longPriceDto.Price;
                    }

                    if (marketSnapshot.PerpPrices.TryGetValue(shortPos.Exchange, out var shortExchangePrices) &&
                        shortExchangePrices.TryGetValue(shortPos.Symbol, out var shortPriceDto))
                    {
                        currentShortPrice = (double)shortPriceDto.Price;
                    }
                }

                result["current_long_price"] = currentLongPrice;
                result["current_short_price"] = currentShortPrice;

                // Funding fees (net funding is already calculated per position)
                var longNetFunding = (double)(longPos.TotalFundingFeeReceived - longPos.TotalFundingFeePaid);
                var shortNetFunding = (double)(shortPos.TotalFundingFeeReceived - shortPos.TotalFundingFeePaid);
                var netFundingUsd = longNetFunding + shortNetFunding;

                result["long_net_funding_usd"] = longNetFunding;
                result["short_net_funding_usd"] = shortNetFunding;

                // Fetch funding rates from market snapshot
                var longFundingRate = 0.0;
                var shortFundingRate = 0.0;

                if (marketSnapshot?.FundingRates != null)
                {
                    // Try to get funding rate from snapshot (keyed by exchange, then filter by symbol)
                    if (marketSnapshot.FundingRates.TryGetValue(longPos.Exchange, out var longExchangeFundingRates))
                    {
                        var longFr = longExchangeFundingRates.FirstOrDefault(fr => fr.Symbol == longPos.Symbol);
                        if (longFr != null)
                        {
                            longFundingRate = (double)longFr.Rate;
                        }
                    }

                    if (marketSnapshot.FundingRates.TryGetValue(shortPos.Exchange, out var shortExchangeFundingRates))
                    {
                        var shortFr = shortExchangeFundingRates.FirstOrDefault(fr => fr.Symbol == shortPos.Symbol);
                        if (shortFr != null)
                        {
                            shortFundingRate = (double)shortFr.Rate;
                        }
                    }
                }

                result["long_funding_rate"] = longFundingRate;
                result["short_funding_rate"] = shortFundingRate;

                // Entry fees (use trading fee paid as approximation, include both legs)
                var entryFeesUsd = (double)(longPos.TradingFeePaid + shortPos.TradingFeePaid);
                result["entry_fees_paid_usd"] = entryFeesUsd;

                // ========== CALCULATE MISSING FIELDS ==========

                // 1. net_funding_ratio - Net funding / total capital used
                var totalCapitalUsed = longSizeUsd + shortSizeUsd;
                var netFundingRatio = totalCapitalUsed > 0 ? netFundingUsd / totalCapitalUsed : 0.0;
                result["net_funding_ratio"] = netFundingRatio;

                // 2. net_funding_rate - Current funding rate differential
                result["net_funding_rate"] = shortFundingRate - longFundingRate;

                // 3. current_spread_pct - Current price spread percentage
                var avgCurrentPrice = (currentLongPrice + currentShortPrice) / 2.0;
                var currentSpreadPct = avgCurrentPrice > 0 ? Math.Abs(currentLongPrice - currentShortPrice) / avgCurrentPrice : 0.0;
                result["current_spread_pct"] = currentSpreadPct;

                // 4. entry_spread_pct - Entry price spread percentage
                var avgEntryPrice = (entryLongPrice + entryShortPrice) / 2.0;
                var entrySpreadPct = avgEntryPrice > 0 ? Math.Abs(entryLongPrice - entryShortPrice) / avgEntryPrice : 0.0;
                result["entry_spread_pct"] = entrySpreadPct;

                // 5. value_to_capital_ratio - Position value / total capital
                var valueToCapitalRatio = (double)totalCapital > 0 ? totalCapitalUsed / (double)totalCapital : 0.0;
                result["value_to_capital_ratio"] = valueToCapitalRatio;

                // 6. funding_efficiency - Net funding / entry fees
                var fundingEfficiency = entryFeesUsd > 0 ? netFundingUsd / entryFeesUsd : 0.0;
                result["funding_efficiency"] = fundingEfficiency;

                // 7. liquidation_distance - Distance to liquidation threshold
                // Calculate based on leverage and current P&L
                // Liquidation occurs when loss reaches (1 / leverage) * 100%
                // Distance = (liquidation_threshold - current_loss_pct) / liquidation_threshold
                var liquidationThreshold = 100.0 / (double)position.Leverage; // e.g., 10% for 10x leverage
                var currentLossPct = Math.Abs(Math.Min(0, unrealizedPnlPct)); // Only consider losses
                var liquidationDistance = liquidationThreshold > 0
                    ? Math.Max(0, (liquidationThreshold - currentLossPct) / liquidationThreshold)
                    : 1.0;
                result["liquidation_distance"] = liquidationDistance;
            }
            else
            {
                // ========== FALLBACK: ExecutionId exists but no paired position found ==========
                _logger.LogWarning(
                    "Position {PositionId} has ExecutionId {ExecutionId} but no paired position found. Providing fallback values.",
                    position.Id, position.ExecutionId);

                // Provide single-leg position data with fallback values
                result["long_pnl_pct"] = position.Side == PositionSide.Long ? unrealizedPnlPct : 0.0;
                result["short_pnl_pct"] = position.Side == PositionSide.Short ? unrealizedPnlPct : 0.0;

                result["entry_long_price"] = position.Side == PositionSide.Long ? (double)position.EntryPrice : 0.0;
                result["entry_short_price"] = position.Side == PositionSide.Short ? (double)position.EntryPrice : 0.0;

                // Try to get current price from market snapshot
                var currentPrice = (double)position.EntryPrice;
                if (marketSnapshot?.PerpPrices != null &&
                    marketSnapshot.PerpPrices.TryGetValue(position.Exchange, out var exchangePrices) &&
                    exchangePrices.TryGetValue(position.Symbol, out var priceDto))
                {
                    currentPrice = (double)priceDto.Price;
                }

                result["current_long_price"] = position.Side == PositionSide.Long ? currentPrice : 0.0;
                result["current_short_price"] = position.Side == PositionSide.Short ? currentPrice : 0.0;

                var netFunding = (double)(position.TotalFundingFeeReceived - position.TotalFundingFeePaid);
                result["long_net_funding_usd"] = position.Side == PositionSide.Long ? netFunding : 0.0;
                result["short_net_funding_usd"] = position.Side == PositionSide.Short ? netFunding : 0.0;

                // Get funding rate from market snapshot
                var fundingRate = 0.0;
                if (marketSnapshot?.FundingRates != null &&
                    marketSnapshot.FundingRates.TryGetValue(position.Exchange, out var exchangeFundingRates))
                {
                    var fr = exchangeFundingRates.FirstOrDefault(f => f.Symbol == position.Symbol);
                    if (fr != null)
                    {
                        fundingRate = (double)fr.Rate;
                    }
                }

                result["long_funding_rate"] = position.Side == PositionSide.Long ? fundingRate : 0.0;
                result["short_funding_rate"] = position.Side == PositionSide.Short ? fundingRate : 0.0;

                var entryFeesUsd = (double)position.TradingFeePaid;
                result["entry_fees_paid_usd"] = entryFeesUsd;

                // Calculate fallback values for missing fields
                result["net_funding_ratio"] = positionSizeUsd > 0 ? netFunding / positionSizeUsd : 0.0;
                result["net_funding_rate"] = 0.0; // No differential for single position
                result["current_spread_pct"] = 0.0; // No spread for single position
                result["entry_spread_pct"] = 0.0;
                result["value_to_capital_ratio"] = (double)totalCapital > 0 ? positionSizeUsd / (double)totalCapital : 0.0;
                result["funding_efficiency"] = entryFeesUsd > 0 ? netFunding / entryFeesUsd : 0.0;

                var liquidationThreshold = 100.0 / (double)position.Leverage;
                var currentLossPct = Math.Abs(Math.Min(0, unrealizedPnlPct));
                var liquidationDistance = liquidationThreshold > 0
                    ? Math.Max(0, (liquidationThreshold - currentLossPct) / liquidationThreshold)
                    : 1.0;
                result["liquidation_distance"] = liquidationDistance;
            }
        // Note: The fallback else block above handles positions without paired positions

        return Task.FromResult(result);
    }

    /// <summary>
    /// Call ML API for prediction (Modular Mode)
    /// </summary>
    private async Task<AgentPrediction?> GetMLPredictionAsync(
        string userId,
        AgentConfiguration config,
        List<ArbitrageOpportunityDto> opportunities,
        Dictionary<string, object> portfolio)
    {
        try
        {
            // Build trading config for ML API
            var tradingConfig = new Dictionary<string, object>
            {
                ["max_leverage"] = config.MaxLeverage,
                ["target_utilization"] = config.TargetUtilization,
                ["max_positions"] = config.MaxPositions,
                ["stop_loss_threshold"] = -0.02,  // Default -2%
                ["liquidation_buffer"] = 0.15     // Default 15%
            };

            // Call /rl/predict/opportunities with up to 10 opportunities
            var payload = new
            {
                opportunities = opportunities.Take(10).Select(ConvertOpportunityToDict).ToList(),
                portfolio = portfolio,
                trading_config = tradingConfig
            };

            var json = JsonSerializer.Serialize(payload, _jsonOptions);

            // DEBUG: Log the actual JSON payload being sent
            _logger.LogInformation("===== JSON PAYLOAD TO ML API =====");
            _logger.LogInformation("Portfolio positions in payload: {Json}",
                JsonSerializer.Serialize(portfolio.ContainsKey("positions") ? portfolio["positions"] : "NO POSITIONS", _jsonOptions));
            _logger.LogInformation("==================================");

            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync("/rl/predict/opportunities", content);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogError("ML API call failed for user {UserId}: {Status}", userId, response.StatusCode);
                return null;
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var mlResponse = JsonSerializer.Deserialize<MLModularResponse>(responseJson, _jsonOptions);

            if (mlResponse == null)
            {
                return new AgentPrediction { Action = "HOLD", Confidence = "LOW" };
            }

            _logger.LogDebug(
                "Agent prediction for user {UserId}: {Action} (Confidence: {Confidence:F1}%, ActionID: {ActionId})",
                userId, mlResponse.Action, mlResponse.Confidence * 100, mlResponse.ActionId);

            // Map modular response to AgentPrediction
            return new AgentPrediction
            {
                Action = mlResponse.Action ?? "HOLD",
                OpportunityIndex = mlResponse.OpportunityIndex,
                OpportunitySymbol = mlResponse.OpportunitySymbol,
                PositionSize = mlResponse.PositionSize,
                SizeMultiplier = mlResponse.SizeMultiplier,
                PositionIndex = mlResponse.PositionIndex,
                Confidence = mlResponse.Confidence >= 0.7 ? "HIGH" : mlResponse.Confidence >= 0.4 ? "MEDIUM" : "LOW",
                EnterProbability = mlResponse.Confidence, // Confidence is the action probability
                StateValue = mlResponse.StateValue
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling ML API for user {UserId}", userId);
            return null;
        }
    }

    /// <summary>
    /// Convert opportunity to dictionary for ML API
    /// ALL 20 fields must match what the ML model expects (rl_predictor.py:_build_opportunity_features)
    /// </summary>
    private Dictionary<string, object> ConvertOpportunityToDict(ArbitrageOpportunityDto opp)
    {
        return new Dictionary<string, object>
        {
            ["symbol"] = opp.Symbol ?? "",

            // Funding rates (fields 1-4)
            ["long_funding_rate"] = (double)opp.LongFundingRate,
            ["short_funding_rate"] = (double)opp.ShortFundingRate,
            ["long_funding_interval_hours"] = opp.LongFundingIntervalHours ?? 8,
            ["short_funding_interval_hours"] = opp.ShortFundingIntervalHours ?? 8,

            // Profit metrics - current (fields 5-6)
            ["fund_profit_8h"] = (double)opp.FundProfit8h,
            ["fund_apr"] = (double)opp.FundApr,

            // Projected profits - 24h and 3d (fields 7-10)
            ["fundProfit8h24hProj"] = (double)(opp.FundProfit8h24hProj ?? 0m),
            ["fundProfit8h3dProj"] = (double)(opp.FundProfit8h3dProj ?? 0m),
            ["fundApr24hProj"] = (double)(opp.FundApr24hProj ?? 0m),
            ["fundApr3dProj"] = (double)(opp.FundApr3dProj ?? 0m),

            // Spread metrics (fields 11-14)
            ["spread30SampleAvg"] = (double)(opp.Spread30SampleAvg ?? 0m),
            ["priceSpread24hAvg"] = (double)(opp.PriceSpread24hAvg ?? 0m),
            ["priceSpread3dAvg"] = (double)(opp.PriceSpread3dAvg ?? 0m),
            ["spread_volatility_stddev"] = (double)(opp.SpreadVolatilityStdDev ?? 0m),

            // Volume and liquidity (fields 15-17)
            ["volume_24h"] = (double)opp.Volume24h,
            ["bidAskSpreadPercent"] = (double)(opp.BidAskSpreadPercent ?? 0m),
            ["orderbookDepthUsd"] = (double)(opp.OrderbookDepthUsd ?? 10000m),  // Default to 10k if missing

            // Profit and cost estimates (fields 18-19)
            ["estimatedProfitPercentage"] = (double)opp.EstimatedProfitPercentage,
            ["positionCostPercent"] = (double)opp.PositionCostPercent,

            // Field 20 is computed: short_funding_rate - long_funding_rate (handled by ML)

            // CRITICAL: Flag for action masking to prevent duplicate entries
            ["has_existing_position"] = opp.IsExistingPosition
        };
    }

    /// <summary>
    /// Execute agent action (ENTER or EXIT)
    /// </summary>
    private async Task<AgentExecutionResult> ExecuteAgentActionAsync(
        string userId,
        AgentPrediction prediction,
        List<ArbitrageOpportunityDto> opportunities,
        List<PositionDto> positions,
        ArbitrageExecutionService executionService,
        ArbitrageDbContext context,
        IDataRepository<UserDataSnapshot> userDataRepository,
        CancellationToken cancellationToken)
    {
        // Get or create execution lock for this user to prevent race conditions
        SemaphoreSlim executionLock;
        lock (_lock)
        {
            if (!_executionLocks.TryGetValue(userId, out executionLock!))
            {
                executionLock = new SemaphoreSlim(1, 1);
                _executionLocks[userId] = executionLock;
            }
        }

        // Prevent concurrent executions for this user - predictions will wait
        _logger.LogDebug("Agent waiting for execution lock for user {UserId}", userId);
        await executionLock.WaitAsync();

        // Initialize execution result
        var result = new AgentExecutionResult { Success = true };

        try
        {
            _logger.LogDebug("Agent acquired execution lock for user {UserId}", userId);

            if (prediction.Action == "ENTER" && prediction.OpportunityIndex.HasValue)
            {
                var oppIndex = prediction.OpportunityIndex.Value;
                if (oppIndex >= 0 && oppIndex < opportunities.Count)
                {
                    var opportunity = opportunities[oppIndex];

                    _logger.LogInformation(
                        "Agent executing ENTER for user {UserId}: {Symbol} (Strategy: {Strategy}, Exchange: {Exchange}, Confidence: {Confidence})",
                        userId, opportunity.Symbol, opportunity.Strategy, opportunity.Exchange, prediction.Confidence);

                    try
                    {
                        // Get user's agent config for position sizing (include AgentConfiguration)
                        var session = await context.AgentSessions
                            .Include(s => s.AgentConfiguration)
                            .FirstOrDefaultAsync(s => s.UserId == userId && s.Status == AgentStatus.Running);

                        if (session == null)
                        {
                            result.Success = false;
                            result.ErrorMessage = "Agent session not found";
                            _logger.LogWarning("Agent session not found for user {UserId}, cannot execute trade", userId);
                            return result;
                        }

                    // Calculate position size based on ML prediction and config
                    var sizeMultiplier = (decimal)(prediction.SizeMultiplier ?? 0.20); // Default to MEDIUM (20%) if not provided

                    // Get portfolio state to calculate available capital
                    // Note: We need to pass userDataRepository from the parent scope
                    var availableCapital = await GetAvailableCapitalAsync(userId, context, userDataRepository, opportunity, cancellationToken);
                    var positionSizeUsd = availableCapital * sizeMultiplier;

                    // Apply minimum and maximum limits
                    var minPositionSize = 10m; // $10 minimum
                    var maxPositionSize = availableCapital * session.AgentConfiguration.TargetUtilization; // Don't exceed target utilization
                    positionSizeUsd = Math.Max(minPositionSize, Math.Min(positionSizeUsd, maxPositionSize));

                    // Build execution request
                    var request = new ExecuteOpportunityRequest
                    {
                        Symbol = opportunity.Symbol,
                        Strategy = opportunity.Strategy,
                        SubType = opportunity.SubType,
                        PositionSizeUsd = positionSizeUsd,
                        Leverage = session.AgentConfiguration.MaxLeverage,

                        // For spot-perpetual strategy
                        Exchange = opportunity.Exchange,
                        FundingRate = opportunity.LongFundingRate, // Use long funding rate as primary

                        // For cross-exchange strategy
                        LongExchange = opportunity.LongExchange,
                        ShortExchange = opportunity.ShortExchange,
                        LongFundingRate = opportunity.LongFundingRate,
                        ShortFundingRate = opportunity.ShortFundingRate,

                        // Spread information
                        SpreadRate = opportunity.SpreadRate,
                        AnnualizedSpread = opportunity.AnnualizedSpread,
                        EstimatedProfitPercentage = opportunity.EstimatedProfitPercentage
                    };

                        _logger.LogInformation(
                            "Agent calling ExecuteOpportunityAsync: {Symbol}, Size: ${Size:F2}, Leverage: {Leverage}x",
                            request.Symbol, positionSizeUsd, request.Leverage);

                        // Set background user context so ExecutionService can find user's API keys
                        using var scope = _serviceProvider.CreateScope();
                        var currentUserService = scope.ServiceProvider.GetRequiredService<ICurrentUserService>();
                        using var userContext = currentUserService.SetBackgroundUserContext(userId);

                        // Execute the trade with proper user context
                        var response = await executionService.ExecuteOpportunityAsync(request);

                        // Update result with execution info
                        result.Success = response.Success;
                        result.AmountUsd = positionSizeUsd;

                        if (response.Success)
                        {
                            _logger.LogInformation(
                                "Agent successfully executed ENTER for {Symbol}: {Message}",
                                opportunity.Symbol, response.Message);
                        }
                        else
                        {
                            result.ErrorMessage = response.ErrorMessage;
                            _logger.LogWarning(
                                "Agent failed to execute ENTER for {Symbol}: {Error}",
                                opportunity.Symbol, response.ErrorMessage);
                        }
                    }
                    catch (Exception ex)
                    {
                        result.Success = false;
                        result.ErrorMessage = ex.Message;
                        _logger.LogError(ex, "Error executing ENTER action for user {UserId}, symbol {Symbol}",
                            userId, opportunity.Symbol);
                    }
                }
            }
            else if (prediction.Action == "EXIT" && positions.Count > 0)
        {
            // Find the position to exit (use prediction.PositionIndex if available)
            PositionDto? positionToExit = null;

            if (prediction.PositionIndex.HasValue && prediction.PositionIndex.Value < positions.Count)
            {
                positionToExit = positions[prediction.PositionIndex.Value];
            }
            else
            {
                // Default to first position if no index specified
                positionToExit = positions.First();
            }

            _logger.LogInformation(
                "Agent executing EXIT for user {UserId}: Position {PositionId}, Symbol: {Symbol} (Confidence: {Confidence})",
                userId, positionToExit.Id, positionToExit.Symbol, prediction.Confidence);

                try
                {
                    // Get the execution ID from the position (ExecutionId is always present now)
                    var executionId = positionToExit.ExecutionId;

                    if (executionId == 0)
                    {
                        result.Success = false;
                        result.ErrorMessage = "Position has invalid execution ID";
                        _logger.LogWarning("Position {PositionId} has invalid execution ID (0), cannot close", positionToExit.Id);
                        return result;
                    }

                    _logger.LogInformation(
                        "Agent calling StopExecutionAsync for Execution {ExecutionId}, Symbol: {Symbol}",
                        executionId, positionToExit.Symbol);

                    // Set background user context so ExecutionService can find user's API keys
                    using var scope = _serviceProvider.CreateScope();
                    var currentUserService = scope.ServiceProvider.GetRequiredService<ICurrentUserService>();
                    using var userContext = currentUserService.SetBackgroundUserContext(userId);

                    // Calculate duration before closing
                    var durationHours = (DateTime.UtcNow - positionToExit.OpenedAt).TotalHours;

                    // Close the execution (which closes all positions in that execution)
                    var response = await executionService.StopExecutionAsync(executionId);

                    // Update result with EXIT info
                    result.Success = response.Success;
                    result.ExecutionId = executionId;
                    result.DurationHours = durationHours;

                    if (response.Success)
                    {
                        // Calculate profit from position
                        result.ProfitUsd = positionToExit.UnrealizedPnL;
                        var entryValue = positionToExit.Quantity * positionToExit.EntryPrice;
                        result.ProfitPct = entryValue > 0 ? (positionToExit.UnrealizedPnL / entryValue) * 100 : 0;

                        _logger.LogInformation(
                            "Agent successfully executed EXIT for Execution {ExecutionId}: {Message}",
                            executionId, response.Message);
                    }
                    else
                    {
                        result.ErrorMessage = response.ErrorMessage;
                        _logger.LogWarning(
                            "Agent failed to execute EXIT for Execution {ExecutionId}: {Error}",
                            executionId, response.ErrorMessage);
                    }
                }
                catch (Exception ex)
                {
                    result.Success = false;
                    result.ErrorMessage = ex.Message;
                    _logger.LogError(ex, "Error executing EXIT action for user {UserId}, position {PositionId}",
                        userId, positionToExit.Id);
                }
            }
        }
        finally
        {
            // Always release the execution lock
            executionLock.Release();
            _logger.LogDebug("Agent released execution lock for user {UserId}", userId);
        }

        return result;
    }

    /// <summary>
    /// Calculate available capital for user based on real exchange balances.
    /// For cross-exchange opportunities, returns the minimum available balance across both exchanges.
    /// </summary>
    private async Task<decimal> GetAvailableCapitalAsync(
        string userId,
        ArbitrageDbContext context,
        IDataRepository<UserDataSnapshot> userDataRepository,
        ArbitrageOpportunityDto opportunity,
        CancellationToken cancellationToken = default)
    {
        List<string> exchangesToQuery = new();

        // Determine which exchanges to query based on strategy
        if (opportunity.Strategy == ArbitrageStrategy.CrossExchange)
        {
            exchangesToQuery.Add(opportunity.LongExchange);
            exchangesToQuery.Add(opportunity.ShortExchange);
        }
        else // SpotPerpetual or other single-exchange strategies
        {
            exchangesToQuery.Add(opportunity.Exchange);
        }

        // Query balance data from repository for each exchange
        var availableBalances = new List<decimal>();

        foreach (var exchange in exchangesToQuery.Where(e => !string.IsNullOrEmpty(e)))
        {
            var key = $"userdata:{userId}:{exchange}";
            var snapshot = await userDataRepository.GetAsync(key, cancellationToken);

            if (snapshot?.Balance == null)
            {
                throw new InvalidOperationException(
                    $"Unable to retrieve balance data for user {userId} on exchange {exchange}. " +
                    "User data collector may not be running or data is stale.");
            }

            var futuresAvailable = snapshot.Balance.FuturesAvailableUsd;
            availableBalances.Add(futuresAvailable);

            _logger.LogDebug(
                "Balance for user {UserId} on {Exchange}: FuturesAvailable=${Available:F2}, MarginUsed=${MarginUsed:F2}",
                userId, exchange, futuresAvailable, snapshot.Balance.MarginUsed);
        }

        if (availableBalances.Count == 0)
        {
            throw new InvalidOperationException(
                $"No exchanges specified for opportunity {opportunity.Symbol}. Cannot determine available capital.");
        }

        // For cross-exchange hedge positions, use the MINIMUM available balance
        // This ensures we can open equal-sized positions on both exchanges
        var availableCapital = availableBalances.Min();

        _logger.LogInformation(
            "Available capital for user {UserId} (Strategy: {Strategy}): ${Available:F2} " +
            "(Exchanges: {Exchanges}, Balances: {Balances})",
            userId,
            opportunity.Strategy,
            availableCapital,
            string.Join(", ", exchangesToQuery),
            string.Join(", ", availableBalances.Select(b => $"${b:F2}")));

        return Math.Max(0, availableCapital);
    }

    /// <summary>
    /// Update agent statistics (session-based)
    /// </summary>
    private async Task UpdateAgentStatsAsync(string userId, string action, ArbitrageDbContext context)
    {
        // Update session decision counters
        var session = await context.AgentSessions
            .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
            .FirstOrDefaultAsync();

        if (session != null)
        {
            // Increment appropriate decision counter based on action
            if (action == "HOLD")
                session.HoldDecisions++;
            else if (action == "ENTER")
                session.EnterDecisions++;
            else if (action == "EXIT")
                session.ExitDecisions++;

            session.UpdatedAt = DateTime.UtcNow;
            await context.SaveChangesAsync();
        }
    }

    /// <summary>
    /// Broadcast agent decision via SignalR
    /// </summary>
    private async Task BroadcastAgentDecisionAsync(
        string userId,
        AgentPrediction prediction,
        List<ArbitrageOpportunityDto> opportunities,
        List<PositionDto> positions,
        AgentExecutionResult executionResult,
        ISignalRStreamingService signalR)
    {
        try
        {
            // Filter out HOLD decisions as per user requirement
            if (prediction.Action == "HOLD")
            {
                _logger.LogDebug("Skipping broadcast for HOLD decision");
                return;
            }

            var decision = new
            {
                action = prediction.Action,
                timestamp = DateTime.UtcNow,
                symbol = prediction.OpportunitySymbol,
                confidence = prediction.Confidence,
                enterProbability = prediction.EnterProbability,
                reasoning = $"State value: {prediction.StateValue:F4}",
                numOpportunities = opportunities.Count,
                numPositions = positions.Count,

                // Execution results
                executionStatus = executionResult.Success ? "success" : "failed",
                errorMessage = executionResult.ErrorMessage,

                // ENTER specific fields
                amountUsd = executionResult.AmountUsd,
                executionId = executionResult.ExecutionId,

                // EXIT specific fields
                profitUsd = executionResult.ProfitUsd,
                profitPct = executionResult.ProfitPct,
                durationHours = executionResult.DurationHours
            };

            await signalR.BroadcastAgentDecisionAsync(userId, decision);
            _logger.LogInformation("Broadcast agent decision: {Action} {Symbol} (Status: {Status})",
                prediction.Action, prediction.OpportunitySymbol, decision.executionStatus);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting agent decision for user {UserId}", userId);
        }
    }

    /// <summary>
    /// Broadcast agent update via SignalR
    /// </summary>
    private async Task BroadcastAgentUpdateAsync(string userId, ISignalRStreamingService signalR, ArbitrageDbContext context)
    {
        try
        {
            // Get latest session for duration and stats
            var session = await context.AgentSessions
                .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
                .OrderByDescending(s => s.StartedAt)
                .FirstOrDefaultAsync();

            if (session != null)
            {
                var durationSeconds = session.StartedAt.HasValue
                    ? (int)(DateTime.UtcNow - session.StartedAt.Value).TotalSeconds
                    : 0;

                // Broadcast status update
                await signalR.BroadcastAgentStatusAsync(userId, "running", durationSeconds, true, null);

                // Calculate stats from session data
                var totalDecisions = session.HoldDecisions + session.EnterDecisions + session.ExitDecisions;
                var totalTrades = session.WinningTrades + session.LosingTrades;
                var winRate = totalTrades > 0 ? (decimal)session.WinningTrades / totalTrades : 0m;

                var statsData = new
                {
                    totalDecisions = totalDecisions,
                    holdDecisions = session.HoldDecisions,
                    enterDecisions = session.EnterDecisions,
                    exitDecisions = session.ExitDecisions,
                    totalTrades = totalTrades,
                    winningTrades = session.WinningTrades,
                    losingTrades = session.LosingTrades,
                    winRate = winRate,
                    totalPnLUsd = session.SessionPnLUsd,
                    totalPnLPct = session.SessionPnLPct,
                    todayPnLUsd = session.SessionPnLUsd,
                    todayPnLPct = session.SessionPnLPct,
                    activePositions = session.ActivePositions
                };

                await signalR.BroadcastAgentStatsAsync(userId, statsData);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error broadcasting agent update for user {UserId}", userId);
        }
    }

    /// <summary>
    /// Set agent to error state
    /// </summary>
    private async Task SetAgentErrorAsync(string userId, string errorMessage)
    {
        using var scope = _serviceProvider.CreateScope();
        var context = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        var session = await context.AgentSessions
            .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
            .FirstOrDefaultAsync();

        if (session != null)
        {
            session.Status = AgentStatus.Error;
            session.ErrorMessage = errorMessage;
            session.UpdatedAt = DateTime.UtcNow;
            await context.SaveChangesAsync();

            _logger.LogError("Agent for user {UserId} set to ERROR state: {Error}", userId, errorMessage);
        }

        // Stop the agent
        lock (_lock)
        {
            if (_runningAgents.TryGetValue(userId, out var cts))
            {
                cts.Cancel();
                _runningAgents.Remove(userId);
            }
        }
    }

    /// <summary>
    /// Stop all running agents
    /// </summary>
    private async Task StopAllAgentsAsync()
    {
        lock (_lock)
        {
            foreach (var cts in _runningAgents.Values)
            {
                cts.Cancel();
            }
            _runningAgents.Clear();
        }

        await Task.CompletedTask;
    }

    public override void Dispose()
    {
        _httpClient?.Dispose();
        base.Dispose();
    }
}

/// <summary>
/// Equality comparer for ArbitrageOpportunityDto to remove duplicates
/// </summary>
public class OpportunityComparer : IEqualityComparer<ArbitrageOpportunityDto>
{
    public bool Equals(ArbitrageOpportunityDto? x, ArbitrageOpportunityDto? y)
    {
        if (ReferenceEquals(x, y)) return true;
        if (x is null || y is null) return false;
        return x.UniqueKey == y.UniqueKey;
    }

    public int GetHashCode(ArbitrageOpportunityDto obj)
    {
        return obj.UniqueKey.GetHashCode();
    }
}

/// <summary>
/// Agent prediction result from ML API
/// </summary>
public class AgentPrediction
{
    public string Action { get; set; } = "HOLD";
    public int? OpportunityIndex { get; set; }
    public string? OpportunitySymbol { get; set; }
    public string? PositionSize { get; set; }  // "SMALL", "MEDIUM", "LARGE"
    public double? SizeMultiplier { get; set; }  // 0.10, 0.20, 0.30
    public int? PositionIndex { get; set; }  // For EXIT actions
    public string Confidence { get; set; } = "LOW";
    public double? EnterProbability { get; set; }
    public double? StateValue { get; set; }
}

/// <summary>
/// ML API modular response (single action)
/// </summary>
public class MLModularResponse
{
    [JsonPropertyName("action")]
    public string? Action { get; set; }

    [JsonPropertyName("action_id")]
    public int ActionId { get; set; }

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("state_value")]
    public double StateValue { get; set; }

    [JsonPropertyName("opportunity_index")]
    public int? OpportunityIndex { get; set; }

    [JsonPropertyName("opportunity_symbol")]
    public string? OpportunitySymbol { get; set; }

    [JsonPropertyName("position_size")]
    public string? PositionSize { get; set; }

    [JsonPropertyName("size_multiplier")]
    public double? SizeMultiplier { get; set; }

    [JsonPropertyName("position_index")]
    public int? PositionIndex { get; set; }
}

/// <summary>
/// ML API opportunities response (DEPRECATED - old format)
/// </summary>
public class MLOpportunitiesResponse
{
    [JsonPropertyName("predictions")]
    public List<MLOpportunityPrediction> Predictions { get; set; } = new();
}

/// <summary>
/// ML API single opportunity prediction (DEPRECATED - old format)
/// </summary>
public class MLOpportunityPrediction
{
    [JsonPropertyName("opportunity_index")]
    public int OpportunityIndex { get; set; }

    [JsonPropertyName("symbol")]
    public string Symbol { get; set; } = "";

    [JsonPropertyName("enter_probability")]
    public double EnterProbability { get; set; }

    [JsonPropertyName("confidence")]
    public string Confidence { get; set; } = "";

    [JsonPropertyName("hold_probability")]
    public double HoldProbability { get; set; }

    [JsonPropertyName("state_value")]
    public double StateValue { get; set; }
}

/// <summary>
/// Execution result for agent action
/// </summary>
public class AgentExecutionResult
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public decimal? AmountUsd { get; set; }
    public int? ExecutionId { get; set; }
    public decimal? ProfitUsd { get; set; }
    public decimal? ProfitPct { get; set; }
    public double? DurationHours { get; set; }
}
