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
        var userOpportunities = opportunitiesDict.Values.Take(10).ToList(); // Top 10

        if (userOpportunities.Count == 0)
        {
            _logger.LogDebug("No opportunities available for user {UserId}", userId);
            return;
        }

        // 2. Get current positions from repository (real-time positions from UserDataCollector)
        var positions = await GetPositionsFromRepositoryAsync(userId, userDataRepository, cancellationToken);

        _logger.LogInformation("========== AGENT SENDING TO ML API ==========");
        _logger.LogInformation("User: {UserId}, Positions from repository: {PositionCount}", userId, positions.Count);
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
        var portfolio = await BuildPortfolioStateAsync(userId, positions, context, marketDataRepository, fundingRateRepository, cancellationToken);

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
        var executionResult = await ExecuteAgentActionAsync(userId, prediction, userOpportunities, positions, executionService, context);

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
        CancellationToken cancellationToken)
    {
        // Get user's total capital from performance metrics or default
        var latestMetric = await context.PerformanceMetrics
            .Where(pm => pm.UserId == userId)
            .OrderByDescending(pm => pm.Date)
            .FirstOrDefaultAsync();

        var totalCapital = latestMetric?.AccountBalance ?? 10000m;
        var totalPnL = latestMetric?.TotalPnL ?? 0m;

        // Calculate utilization from PositionDto
        var marginUsed = positions.Sum(p => p.InitialMargin);
        var utilization = totalCapital > 0 ? (double)(marginUsed / totalCapital) : 0.0;

        // Fetch market data and funding rates once for all positions
        var marketSnapshot = await marketDataRepository.GetAsync("market_data_snapshot", cancellationToken);

        // Convert positions to enriched dictionaries with execution data
        var enrichedPositions = new List<Dictionary<string, object>>();
        foreach (var position in positions)
        {
            var enrichedPos = await ConvertPositionDtoToDictAsync(position, positions, marketSnapshot, cancellationToken);
            enrichedPositions.Add(enrichedPos);
        }

        return new Dictionary<string, object>
        {
            ["total_capital"] = (double)totalCapital,
            ["initial_capital"] = 10000.0, // TODO: Track initial capital
            ["utilization"] = utilization,
            ["total_pnl_pct"] = (double)(totalPnL / totalCapital * 100),
            ["num_positions"] = positions.Count,
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
            ["position_age_hours"] = hoursHeld,
            ["leverage"] = (double)position.Leverage
        };

        // If position is part of an execution, enrich with execution-specific data
        if (position.ExecutionId.HasValue)
        {
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
                result["entry_long_price"] = (double)longPos.EntryPrice;
                result["entry_short_price"] = (double)shortPos.EntryPrice;

                // Fetch current prices from market snapshot
                var currentLongPrice = (double)longPos.EntryPrice; // Default to entry price
                var currentShortPrice = (double)shortPos.EntryPrice;

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

                // Entry fees (use trading fee paid as approximation)
                result["entry_fees_paid_usd"] = (double)position.TradingFeePaid;
            }
        }

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
    /// </summary>
    private Dictionary<string, object> ConvertOpportunityToDict(ArbitrageOpportunityDto opp)
    {
        return new Dictionary<string, object>
        {
            ["symbol"] = opp.Symbol ?? "",
            ["long_funding_rate"] = opp.LongFundingRate,
            ["short_funding_rate"] = opp.ShortFundingRate,
            ["long_funding_interval_hours"] = opp.LongFundingIntervalHours,
            ["short_funding_interval_hours"] = opp.ShortFundingIntervalHours,
            ["fund_profit_8h"] = opp.FundProfit8h,
            ["fund_apr"] = opp.FundApr,
            ["volume_24h"] = (double)opp.Volume24h
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
        ArbitrageDbContext context)
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
                    var availableCapital = await GetAvailableCapitalAsync(userId, context);
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
                    // Get the execution ID from the position
                    var executionId = positionToExit.ExecutionId;

                    if (executionId == null || executionId == 0)
                    {
                        result.Success = false;
                        result.ErrorMessage = "Position has no execution ID";
                        _logger.LogWarning("Position {PositionId} has no execution ID, cannot close", positionToExit.Id);
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
                    var response = await executionService.StopExecutionAsync(executionId.Value);

                    // Update result with EXIT info
                    result.Success = response.Success;
                    result.ExecutionId = executionId.Value;
                    result.DurationHours = durationHours;

                    if (response.Success)
                    {
                        // Calculate profit from position
                        result.ProfitUsd = positionToExit.UnrealizedPnL;
                        var entryValue = positionToExit.Quantity * positionToExit.EntryPrice;
                        result.ProfitPct = entryValue > 0 ? (positionToExit.UnrealizedPnL / entryValue) * 100 : 0;

                        _logger.LogInformation(
                            "Agent successfully executed EXIT for Execution {ExecutionId}: {Message}",
                            executionId.Value, response.Message);
                    }
                    else
                    {
                        result.ErrorMessage = response.ErrorMessage;
                        _logger.LogWarning(
                            "Agent failed to execute EXIT for Execution {ExecutionId}: {Error}",
                            executionId.Value, response.ErrorMessage);
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
    /// Calculate available capital for user (total equity - used margin)
    /// </summary>
    private async Task<decimal> GetAvailableCapitalAsync(string userId, ArbitrageDbContext context)
    {
        // Get all open positions for user
        var openPositions = await context.Positions
            .Where(p => p.UserId == userId && p.Status == PositionStatus.Open)
            .ToListAsync();

        // Calculate used margin
        var usedMargin = openPositions.Sum(p => p.InitialMargin);

        // For now, assume user has $10000 total capital (this should come from actual balance)
        // TODO: Get actual account balance from exchange connectors
        var totalCapital = 10000m;

        var availableCapital = totalCapital - usedMargin;

        _logger.LogDebug(
            "Available capital for user {UserId}: Total=${Total:F2}, Used=${Used:F2}, Available=${Available:F2}",
            userId, totalCapital, usedMargin, availableCapital);

        return Math.Max(0, availableCapital);
    }

    /// <summary>
    /// Update agent statistics
    /// </summary>
    private async Task UpdateAgentStatsAsync(string userId, string action, ArbitrageDbContext context)
    {
        // Update session prediction count
        var session = await context.AgentSessions
            .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
            .FirstOrDefaultAsync();

        if (session != null)
        {
            session.TotalPredictions++;
            session.UpdatedAt = DateTime.UtcNow;
        }

        // Update or create AgentStats
        var stats = await context.AgentStats
            .Where(s => s.UserId == userId)
            .OrderByDescending(s => s.UpdatedAt)
            .FirstOrDefaultAsync();

        if (stats == null)
        {
            stats = new AgentStats
            {
                UserId = userId,
                TotalDecisions = 0,
                HoldDecisions = 0,
                EnterDecisions = 0,
                ExitDecisions = 0,
                TotalTrades = 0,
                WinningTrades = 0,
                LosingTrades = 0,
                WinRate = 0,
                TotalPnLUsd = 0,
                TotalPnLPct = 0,
                TodayPnLUsd = 0,
                TodayPnLPct = 0,
                ActivePositions = 0,
                StatsPeriodStart = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            };
            context.AgentStats.Add(stats);
        }

        // Increment decision counters
        stats.TotalDecisions++;
        if (action == "HOLD")
            stats.HoldDecisions++;
        else if (action == "ENTER")
            stats.EnterDecisions++;
        else if (action == "EXIT")
            stats.ExitDecisions++;

        stats.UpdatedAt = DateTime.UtcNow;

        await context.SaveChangesAsync();
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
            // Get latest session for duration
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
            }

            // Get and broadcast latest stats
            var stats = await context.AgentStats
                .Where(s => s.UserId == userId)
                .OrderByDescending(s => s.UpdatedAt)
                .FirstOrDefaultAsync();

            if (stats != null)
            {
                var statsData = new
                {
                    totalDecisions = stats.TotalDecisions,
                    holdDecisions = stats.HoldDecisions,
                    enterDecisions = stats.EnterDecisions,
                    exitDecisions = stats.ExitDecisions,
                    totalTrades = stats.TotalTrades,
                    winningTrades = stats.WinningTrades,
                    losingTrades = stats.LosingTrades,
                    winRate = stats.WinRate,
                    totalPnLUsd = stats.TotalPnLUsd,
                    totalPnLPct = stats.TotalPnLPct,
                    todayPnLUsd = stats.TodayPnLUsd,
                    todayPnLPct = stats.TodayPnLPct,
                    activePositions = stats.ActivePositions
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
