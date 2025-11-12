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
using CryptoArbitrage.API.Services.ML;
using Microsoft.EntityFrameworkCore;
using AgentPrediction = CryptoArbitrage.API.Services.ML.RLPredictionService.AgentPrediction;

namespace CryptoArbitrage.API.Services.Agent;

/// <summary>
/// Background service that manages autonomous trading agents.
/// One agent per user, running continuously with prediction loops.
/// </summary>
public class AgentBackgroundService : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<AgentBackgroundService> _logger;
    private readonly RLPredictionService _rlPredictionService;
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
        RLPredictionService rlPredictionService,
        IConfiguration configuration)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
        _rlPredictionService = rlPredictionService;

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


        // 3. Call RLPredictionService with centralized feature preparation (275 features)
        // Pass positions from repository - same data source as frontend (UserDataSnapshot)
        // This includes: 5 config features, 10 portfolio features, 60 position features, 200 opportunity features
        var prediction = await _rlPredictionService.GetModularActionAsync(userId, userOpportunities, positions, cancellationToken);

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

            // BUG FIX #5: Deduplicate positions by ID (multiple snapshots may contain same positions)
            var countBeforeDedup = allPositions.Count;
            allPositions = allPositions
                .GroupBy(p => p.Id)
                .Select(g => g.First())
                .ToList();

            if (countBeforeDedup != allPositions.Count)
            {
                _logger.LogWarning("Deduplicated {Before} positions to {After} unique positions for user {UserId}",
                    countBeforeDedup, allPositions.Count, userId);
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
                        FundingIntervalHours = opportunity.LongFundingIntervalHours,

                        // For cross-exchange strategy
                        LongExchange = opportunity.LongExchange,
                        ShortExchange = opportunity.ShortExchange,
                        LongFundingRate = opportunity.LongFundingRate,
                        ShortFundingRate = opportunity.ShortFundingRate,
                        LongFundingIntervalHours = opportunity.LongFundingIntervalHours,
                        ShortFundingIntervalHours = opportunity.ShortFundingIntervalHours,

                        // Spread information
                        SpreadRate = opportunity.SpreadRate,
                        AnnualizedSpread = opportunity.AnnualizedSpread,
                        EstimatedProfitPercentage = opportunity.EstimatedProfitPercentage,

                        // Pre-calculated FundApr (ensures consistency with opportunity)
                        FundApr = opportunity.FundApr
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
