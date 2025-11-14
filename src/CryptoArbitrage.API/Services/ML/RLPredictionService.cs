using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.Agent;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// Service for getting RL (Reinforcement Learning) predictions from the Python ML API (V3)
/// Centralizes ALL feature preparation (203 features total - V3 refactoring) matching the training environment
/// V3 Changes: 301→203 dims (Portfolio 10→3, Execution 100→85, Opportunity 190→110)
/// Evaluates ENTER probabilities for opportunities and EXIT probabilities for positions
/// </summary>
public class RLPredictionService : IDisposable
{
    private readonly ILogger<RLPredictionService> _logger;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly JsonSerializerOptions _deserializeOptions;

    // Data access dependencies for feature preparation
    private readonly ArbitrageDbContext _context;
    private readonly IDataRepository<UserDataSnapshot> _userDataRepository;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;
    private readonly IDataRepository<FundingRateDto> _fundingRateRepository;
    private readonly IAgentConfigurationService _agentConfigService;

    public RLPredictionService(
        ILogger<RLPredictionService> logger,
        IConfiguration configuration,
        ArbitrageDbContext context,
        IDataRepository<UserDataSnapshot> userDataRepository,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        IDataRepository<FundingRateDto> fundingRateRepository,
        IAgentConfigurationService agentConfigService)
    {
        _logger = logger;
        _context = context;
        _userDataRepository = userDataRepository;
        _marketDataRepository = marketDataRepository;
        _fundingRateRepository = fundingRateRepository;
        _agentConfigService = agentConfigService;

        // Get configuration (same ML API server, different port if needed)
        var host = configuration["MLApi:Host"] ?? "localhost";
        var port = configuration["MLApi:Port"] ?? "5250";  // Updated port for RL API
        _baseUrl = $"http://{host}:{port}";

        // Create HTTP client with timeout
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromSeconds(30)
        };

        // JSON serialization options (snake_case for Python)
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        // Deserialization options (snake_case from Python)
        _deserializeOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            PropertyNameCaseInsensitive = true
        };
    }

    /// <summary>
    /// Evaluate ENTER probabilities for ALL opportunities (batch evaluation)
    /// </summary>
    /// <param name="opportunities">List of opportunities to evaluate (any number)</param>
    /// <param name="portfolio">Current portfolio state</param>
    /// <returns>Dictionary mapping opportunity unique key to RL prediction</returns>
    public async Task<Dictionary<string, RLPredictionDto>> EvaluateOpportunitiesAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        RLPortfolioState portfolio)
    {
        var oppList = opportunities.ToList();  // No limit - ML API handles batching
        var result = new Dictionary<string, RLPredictionDto>();

        if (oppList.Count == 0)
            return result;

        try
        {
            // Build request payload
            var payload = new
            {
                opportunities = oppList.Select(ConvertOpportunityToFeatures).ToList(),
                portfolio = portfolio
            };

            // Serialize and send request
            var json = JsonSerializer.Serialize(payload, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            _logger.LogDebug("Calling RL API /rl/predict/opportunities with {Count} opportunities", oppList.Count);
            var response = await _httpClient.PostAsync("/rl/predict/opportunities", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("RL API prediction failed: {Status} - {Error}", response.StatusCode, error);
                return result;
            }

            // Deserialize response
            var responseJson = await response.Content.ReadAsStringAsync();
            var apiResponse = JsonSerializer.Deserialize<RLOpportunitiesResponse>(responseJson, _deserializeOptions);

            if (apiResponse?.Predictions == null)
            {
                _logger.LogWarning("RL API returned null predictions");
                return result;
            }

            // Map predictions back to opportunities
            foreach (var prediction in apiResponse.Predictions)
            {
                if (prediction.OpportunityIndex >= 0 && prediction.OpportunityIndex < oppList.Count)
                {
                    var opp = oppList[prediction.OpportunityIndex];
                    result[opp.UniqueKey] = new RLPredictionDto
                    {
                        ActionProbability = prediction.EnterProbability,
                        HoldProbability = prediction.HoldProbability,
                        Confidence = prediction.Confidence,
                        StateValue = prediction.StateValue,
                        ModelVersion = apiResponse.ModelVersion
                    };
                }
            }

            _logger.LogInformation("RL predictions received for {Count} opportunities", result.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get RL predictions for opportunities");
            return result;
        }
    }

    /// <summary>
    /// Evaluate opportunities with COMPLETE feature preparation (203 features total - V3 refactoring)
    /// This is the NEW centralized method that prepares ALL features internally
    /// Features breakdown (V3):
    /// - Config: 5 features (from AgentConfiguration)
    /// - Portfolio: 3 features (V3: was 10, removed historical metrics)
    /// - Execution: 85 features (5 slots × 17 features - V3: was 100, added velocities)
    /// - Opportunities: 110 features (10 slots × 11 features - V3: was 190, removed market quality)
    /// </summary>
    /// <param name="userId">User ID to prepare context for</param>
    /// <param name="opportunities">List of opportunities to evaluate</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary mapping opportunity unique key to RL prediction</returns>
    public async Task<Dictionary<string, RLPredictionDto>> EvaluateOpportunitiesWithFullContextAsync(
        string userId,
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        CancellationToken cancellationToken = default)
    {
        var oppList = opportunities.ToList();
        var result = new Dictionary<string, RLPredictionDto>();

        if (oppList.Count == 0)
            return result;

        try
        {
            _logger.LogInformation("=== RLPredictionService: Preparing COMPLETE 203-feature payload (V3) ===");

            // 1. Get or create AgentConfiguration (5 config features)
            var agentConfig = await _agentConfigService.GetOrCreateConfigurationAsync(userId);
            var tradingConfig = BuildTradingConfig(agentConfig);
            _logger.LogDebug("Config features: MaxLeverage={MaxLeverage}, TargetUtil={TargetUtil}, MaxPos={MaxPos}",
                tradingConfig.MaxLeverage, tradingConfig.TargetUtilization, tradingConfig.MaxPositions);

            // 2. Get active positions from database (DEPRECATED - should use UserDataSnapshot repository)
            var positions = await _context.Positions
                .Where(p => p.UserId == userId && p.Status == PositionStatus.Open)
                .Include(p => p.Execution)
                .ToListAsync(cancellationToken);
            var numExecutions = positions.Select(p => p.ExecutionId).Distinct().Count();
            _logger.LogDebug("Found {Count} active positions across {Executions} executions", positions.Count, numExecutions);

            // 3. Calculate best available APR from ALL opportunities (including current positions)
            // CRITICAL FIX: Include ALL opportunities, even those we have positions in
            // The model needs to compare current position APR against all opportunities (including itself)
            // Previously excluding current positions caused false "better opportunity" signals
            var bestAvailableApr = oppList.Any() ? oppList.Max(o => (float)o.FundApr) : 0f;
            _logger.LogDebug("Best available APR: {BestApr:F2}% (from ALL {Count} opportunities)",
                bestAvailableApr, oppList.Count);

            // 4. Build complete portfolio state (3 portfolio features + 85 execution features) - V3 refactoring
            var portfolio = await BuildCompletePortfolioStateAsync(userId, positions, agentConfig, bestAvailableApr, oppList, cancellationToken);
            _logger.LogInformation("Portfolio features: Capital=${Capital:F2}, Util={Util:F2}%, NumPos={NumPos}, AvgPnl={AvgPnl:F2}%",
                portfolio.Capital, portfolio.MarginUtilization, portfolio.NumPositions, portfolio.AvgPositionPnlPct);
            _logger.LogInformation("Position features: {Count} slots prepared ({Active} active, {Empty} empty)",
                portfolio.Positions.Count,
                portfolio.Positions.Count(p => p.IsActive),
                portfolio.Positions.Count(p => !p.IsActive));

            // 5. Build request payload
            var payload = new
            {
                opportunities = oppList.Select(ConvertOpportunityToFeatures).ToList(),
                portfolio = portfolio,
                trading_config = tradingConfig
            };

            // 6. Serialize and send request
            var json = JsonSerializer.Serialize(payload, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            _logger.LogInformation("Calling ML API /rl/predict/opportunities with {Count} opportunities", oppList.Count);
            _logger.LogDebug("Payload size: {Size} bytes", json.Length);

            var response = await _httpClient.PostAsync("/rl/predict/opportunities", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("ML API prediction failed: {Status} - {Error}", response.StatusCode, error);
                return result;
            }

            // 6. Deserialize response
            var responseJson = await response.Content.ReadAsStringAsync();
            var apiResponse = JsonSerializer.Deserialize<RLOpportunitiesResponse>(responseJson, _deserializeOptions);

            if (apiResponse?.Predictions == null)
            {
                _logger.LogWarning("ML API returned null predictions");
                return result;
            }

            // 7. Map predictions back to opportunities
            foreach (var prediction in apiResponse.Predictions)
            {
                if (prediction.OpportunityIndex >= 0 && prediction.OpportunityIndex < oppList.Count)
                {
                    var opp = oppList[prediction.OpportunityIndex];
                    result[opp.UniqueKey] = new RLPredictionDto
                    {
                        ActionProbability = prediction.EnterProbability,
                        HoldProbability = prediction.HoldProbability,
                        Confidence = prediction.Confidence,
                        StateValue = prediction.StateValue,
                        ModelVersion = apiResponse.ModelVersion
                    };
                }
            }

            _logger.LogInformation("✅ Complete 203-feature predictions received for {Count} opportunities (V3)", result.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get RL predictions with full context for user {UserId}", userId);
            return result;
        }
    }

    /// <summary>
    /// Get modular action decision with COMPLETE feature preparation (for AgentBackgroundService) - V3
    /// Returns a SINGLE action decision (ENTER/EXIT/HOLD) based on all 203 features (V3 refactoring)
    /// This is the method AgentBackgroundService should use
    /// </summary>
    public async Task<AgentPrediction?> GetModularActionAsync(
        string userId,
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        List<PositionDto> repositoryPositions,
        CancellationToken cancellationToken = default)
    {
        var oppList = opportunities.ToList();

        if (oppList.Count == 0)
            return new AgentPrediction { Action = "HOLD", Confidence = "LOW" };

        try
        {
            _logger.LogInformation("=== RLPredictionService: Preparing modular action with COMPLETE 203-feature payload (V3) ===");

            // 1. Get or create AgentConfiguration (5 config features)
            var agentConfig = await _agentConfigService.GetOrCreateConfigurationAsync(userId);
            if (agentConfig == null)
            {
                _logger.LogError("AgentConfiguration is null for user {UserId}, cannot proceed", userId);
                return null;
            }
            var tradingConfig = BuildTradingConfig(agentConfig);

            _logger.LogInformation("Trading config: MaxLeverage={MaxLeverage}, TargetUtilization={TargetUtil}, MaxPositions={MaxPos}",
                tradingConfig.MaxLeverage, tradingConfig.TargetUtilization, tradingConfig.MaxPositions);

            // 2. Convert PositionDto from repository to Position entities (for position features)
            // Using same data source as frontend (UserDataSnapshot) instead of direct database query
            var positions = ConvertPositionDtosToEntities(repositoryPositions, userId);
            _logger.LogInformation("Using {Count} positions from UserDataSnapshot repository (same source as frontend)", positions.Count);

            // 3. Calculate best available APR from ALL opportunities (including current positions)
            // CRITICAL FIX: Include ALL opportunities, even those we have positions in
            // The model needs to compare current position APR against all opportunities (including itself)
            // Previously excluding current positions caused false "better opportunity" signals
            var bestAvailableApr = oppList.Any() ? oppList.Max(o => (float)o.FundApr) : 0f;
            _logger.LogDebug("Best available APR: {BestApr:F2}% (from ALL {Count} opportunities)",
                bestAvailableApr, oppList.Count);

            // 4. Build complete portfolio state (3 portfolio features + 85 execution features) - V3 refactoring
            var portfolio = await BuildCompletePortfolioStateAsync(userId, positions, agentConfig, bestAvailableApr, oppList, cancellationToken);

            // 5. Build request payload for modular mode
            var payload = new
            {
                opportunities = oppList.Select(ConvertOpportunityToFeatures).ToList(),
                portfolio = portfolio,
                trading_config = tradingConfig
            };

            // 6. Serialize and send request
            var json = JsonSerializer.Serialize(payload, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            _logger.LogInformation("Calling ML API /rl/predict/opportunities (modular mode) with {Count} opportunities", oppList.Count);

            var response = await _httpClient.PostAsync("/rl/predict/opportunities", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("ML API prediction failed: {Status} - {Error}", response.StatusCode, error);
                return null;
            }

            // 6. Deserialize modular response
            var responseJson = await response.Content.ReadAsStringAsync();
            var mlResponse = JsonSerializer.Deserialize<MLModularResponse>(responseJson, _deserializeOptions);

            if (mlResponse == null)
            {
                return new AgentPrediction { Action = "HOLD", Confidence = "LOW" };
            }

            _logger.LogInformation("ML API returned action: {Action} (Confidence: {Confidence:P0}, OpportunityIndex: {Index}, Symbol: {Symbol})",
                mlResponse.Action, mlResponse.Confidence, mlResponse.OpportunityIndex, mlResponse.OpportunitySymbol);

            // 7. Map modular response to AgentPrediction
            return new AgentPrediction
            {
                Action = mlResponse.Action ?? "HOLD",
                OpportunityIndex = mlResponse.OpportunityIndex,
                OpportunitySymbol = mlResponse.OpportunitySymbol,
                PositionSize = mlResponse.PositionSize,
                SizeMultiplier = mlResponse.SizeMultiplier,
                PositionIndex = mlResponse.PositionIndex,
                Confidence = mlResponse.Confidence >= 0.7 ? "HIGH" : mlResponse.Confidence >= 0.4 ? "MEDIUM" : "LOW",
                EnterProbability = mlResponse.Confidence,
                StateValue = mlResponse.StateValue
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get modular action for user {UserId}", userId);
            return null;
        }
    }

    /// <summary>
    /// Evaluate EXIT probabilities for open positions (execution-based)
    /// Groups positions by ExecutionId and evaluates each execution SEPARATELY
    /// Makes one API call per execution to respect Simple Mode constraint (1 execution per call)
    /// </summary>
    /// <param name="positions">List of positions to evaluate</param>
    /// <param name="portfolio">Current portfolio state</param>
    /// <param name="opportunities">Optional: current opportunities for context</param>
    /// <returns>Dictionary mapping position ID to RL prediction</returns>
    public async Task<Dictionary<int, RLPredictionDto>> EvaluatePositionsAsync(
        IEnumerable<PositionDto> positions,
        RLPortfolioState portfolio,
        IEnumerable<ArbitrageOpportunityDto>? opportunities = null)
    {
        var result = new Dictionary<int, RLPredictionDto>();
        var posList = positions.ToList();

        // DEPRECATED: This method uses the legacy /rl/predict/positions endpoint which is no longer supported
        // Position evaluation is now handled through the opportunities endpoint in Modular Mode
        _logger.LogDebug("EvaluatePositionsAsync called but is deprecated - returning empty results. Use opportunities endpoint instead.");
        return await Task.FromResult(result);

        /* LEGACY CODE - DISABLED
        if (posList.Count == 0)
            return result;

        try
        {
            // Group positions by ExecutionId to create execution-level features
            var executionGroups = posList
                .GroupBy(p => p.ExecutionId)
                .Select(g => g.ToList())
                .ToList(); // Process ALL executions

            if (executionGroups.Count == 0)
            {
                _logger.LogWarning("No positions with ExecutionId found for RL evaluation");
                return result;
            }

            // Convert opportunities to List for matching
            var opportunitiesList = opportunities?.ToList();

            _logger.LogInformation("Evaluating {Count} executions using Simple Mode (1 API call per execution)", executionGroups.Count);

            // Process each execution SEPARATELY with its own API call
            // This respects the ML server's Simple Mode constraint (1 execution per call)
            var executionIndex = 0;
            foreach (var executionPositions in executionGroups)
            {
                try
                {
                    // Match this execution to its specific opportunity
                    var matchedOpportunity = FindMatchingOpportunity(executionPositions, opportunitiesList);

                    // Convert execution to features
                    var executionFeatures = ConvertExecutionToFeatures(executionPositions, matchedOpportunity);

                    // Build request payload with SINGLE execution
                    // CRITICAL: ML server reads execution features from portfolio.positions[0], NOT from positions parameter!
                    // The positions parameter is only used for the symbol
                    var payload = new
                    {
                        positions = new[] { executionFeatures },  // Only used for symbol extraction
                        portfolio = new
                        {
                            total_capital = portfolio.Capital,
                            capital = portfolio.Capital,
                            initial_capital = portfolio.InitialCapital,
                            margin_utilization = portfolio.MarginUtilization,
                            capital_utilization = portfolio.CapitalUtilization,
                            total_pnl_pct = portfolio.TotalPnlPct,
                            max_drawdown = portfolio.Drawdown,
                            drawdown = portfolio.Drawdown,
                            // CRITICAL: Put THIS execution's features here - ML server reads from positions[0]
                            positions = new[] { executionFeatures }
                        },
                        opportunity = matchedOpportunity != null
                            ? ConvertOpportunityToFeatures(matchedOpportunity)
                            : null
                    };

                    // Serialize and send request for THIS execution
                    var json = JsonSerializer.Serialize(payload, _jsonOptions);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    var executionId = executionPositions.FirstOrDefault()?.ExecutionId ?? 0;
                    _logger.LogDebug("Calling RL API for execution {ExecutionId} (index {Index}/{Total})",
                        executionId, executionIndex + 1, executionGroups.Count);

                    var response = await _httpClient.PostAsync("/rl/predict/positions", content);

                    if (!response.IsSuccessStatusCode)
                    {
                        var error = await response.Content.ReadAsStringAsync();
                        _logger.LogError("RL API prediction failed for execution {ExecutionId}: {Status} - {Error}",
                            executionId, response.StatusCode, error);
                        continue; // Skip this execution but continue with others
                    }

                    // Deserialize response
                    var responseJson = await response.Content.ReadAsStringAsync();
                    var apiResponse = JsonSerializer.Deserialize<RLPositionsResponse>(responseJson, _deserializeOptions);

                    if (apiResponse?.Predictions == null || apiResponse.Predictions.Count == 0)
                    {
                        _logger.LogWarning("RL API returned no predictions for execution {ExecutionId}", executionId);
                        continue;
                    }

                    // Get prediction for this execution (should be first and only prediction)
                    var prediction = apiResponse.Predictions[0];
                    var rlPrediction = new RLPredictionDto
                    {
                        ActionProbability = prediction.ExitProbability,
                        HoldProbability = prediction.HoldProbability,
                        Confidence = prediction.Confidence,
                        StateValue = prediction.StateValue,
                        ModelVersion = apiResponse.ModelVersion
                    };

                    // Apply the SAME prediction to ALL positions in this execution
                    foreach (var pos in executionPositions)
                    {
                        result[pos.Id] = rlPrediction;
                    }

                    _logger.LogDebug("Execution {ExecutionId}: EXIT={ExitProb:P2}, HOLD={HoldProb:P2}, Confidence={Confidence}",
                        executionId, prediction.ExitProbability, prediction.HoldProbability, prediction.Confidence);
                }
                catch (Exception ex)
                {
                    var executionId = executionPositions.FirstOrDefault()?.ExecutionId ?? 0;
                    _logger.LogError(ex, "Failed to evaluate execution {ExecutionId}", executionId);
                    // Continue with next execution
                }

                executionIndex++;
            }

            _logger.LogInformation("RL predictions completed for {SuccessCount}/{TotalCount} executions ({PositionCount} positions)",
                result.Count / 2, executionGroups.Count, result.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get RL predictions for positions");
            return result;
        }
        */
    }

    /// <summary>
    /// Convert opportunity DTO to feature dictionary for Python API (V3)
    /// Provides raw opportunity data - Python ML API extracts 11 features (V3: was 19)
    /// V3 removed: market quality features (volume, spread, depth, cost)
    /// </summary>
    private object ConvertOpportunityToFeatures(ArbitrageOpportunityDto opp)
    {
        return new
        {
            symbol = opp.Symbol,
            long_funding_rate = (float)opp.LongFundingRate,
            short_funding_rate = (float)opp.ShortFundingRate,
            long_funding_interval_hours = opp.LongFundingIntervalHours ?? 8,
            short_funding_interval_hours = opp.ShortFundingIntervalHours ?? 8,
            fund_profit_8h = (float)opp.FundProfit8h,
            fundProfit8h24hProj = (float)(opp.FundProfit8h24hProj ?? 0),
            fundProfit8h3dProj = (float)(opp.FundProfit8h3dProj ?? 0),
            fund_apr = (float)opp.FundApr,
            fundApr24hProj = (float)(opp.FundApr24hProj ?? 0),
            fundApr3dProj = (float)(opp.FundApr3dProj ?? 0),
            spread30SampleAvg = (float)(opp.Spread30SampleAvg ?? 0),
            priceSpread24hAvg = (float)(opp.PriceSpread24hAvg ?? 0),
            priceSpread3dAvg = (float)(opp.PriceSpread3dAvg ?? 0),
            spread_volatility_stddev = (float)(opp.SpreadVolatilityStdDev ?? 0),
            volume_24h = (float)opp.Volume24h,
            bidAskSpreadPercent = (float)(opp.BidAskSpreadPercent ?? 0),
            orderbookDepthUsd = (float)(opp.OrderbookDepthUsd ?? 0),
            estimatedProfitPercentage = (float)opp.EstimatedProfitPercentage,
            positionCostPercent = (float)opp.PositionCostPercent,
            has_existing_position = opp.IsExistingPosition  // CRITICAL: For action masking
        };
    }

    /// <summary>
    /// Convert execution (2 positions: long + short) to feature dictionary for Python API (V3)
    /// Provides raw position data - Python ML API extracts 17 features per slot (V3: was 20)
    /// V3 added: velocity tracking (pnl, funding, spread)
    /// </summary>
    /// <param name="executionPositions">List of positions in this execution (should be 2: long + short)</param>
    /// <param name="opportunity">Optional current opportunity for blending estimated funding</param>
    private object ConvertExecutionToFeatures(
        List<PositionDto> executionPositions,
        ArbitrageOpportunityDto? opportunity = null)
    {
        // Execution must have exactly 2 positions (long + short)
        var longPos = executionPositions.FirstOrDefault(p => p.Side == PositionSide.Long);
        var shortPos = executionPositions.FirstOrDefault(p => p.Side == PositionSide.Short);

        if (longPos == null || shortPos == null)
        {
            _logger.LogWarning("Execution missing long or short position, using fallback values");
            // Return minimal features if execution is incomplete
            return new
            {
                symbol = executionPositions.FirstOrDefault()?.Symbol ?? "UNKNOWN",
                unrealized_pnl_pct = 0.0f,
                position_age_hours = 0.0f,
                long_net_funding_usd = 0.0f,
                short_net_funding_usd = 0.0f,
                total_net_funding_usd = 0.0f,
                short_funding_rate = 0.0f,
                long_funding_rate = 0.0f,
                current_long_price = 0.0f,
                current_short_price = 0.0f,
                entry_long_price = 0.0f,
                entry_short_price = 0.0f,
                position_size_usd = 0.0f,
                entry_fees_paid_usd = 0.0f,
                long_pnl_pct = 0.0f,
                short_pnl_pct = 0.0f
            };
        }

        // Calculate execution-level metrics
        var hoursHeld = (float)(DateTime.UtcNow - longPos.OpenedAt).TotalHours;

        // Net P&L % (combined long + short)
        var totalInitialMargin = longPos.InitialMargin + shortPos.InitialMargin;
        var totalUnrealizedPnl = longPos.UnrealizedPnL + shortPos.UnrealizedPnL;
        var netPnlPct = totalInitialMargin > 0
            ? (float)((totalUnrealizedPnl / totalInitialMargin) * 100)
            : 0.0f;

        // Individual side P&L %
        var longPnlPct = longPos.InitialMargin > 0
            ? (float)((longPos.UnrealizedPnL / longPos.InitialMargin) * 100)
            : 0.0f;
        var shortPnlPct = shortPos.InitialMargin > 0
            ? (float)((shortPos.UnrealizedPnL / shortPos.InitialMargin) * 100)
            : 0.0f;

        // Funding fees - Extract BOTH paid and received from BOTH positions
        // IMPORTANT: Both LONG and SHORT positions can have BOTH paid AND received funding
        // depending on funding rates. We must calculate NET funding for each side.
        var longFundingReceived = (float)longPos.TotalFundingFeeReceived;
        var longFundingPaid = (float)longPos.TotalFundingFeePaid;
        var shortFundingReceived = (float)shortPos.TotalFundingFeeReceived;
        var shortFundingPaid = (float)shortPos.TotalFundingFeePaid;

        // Calculate net funding (positive = profit, negative = cost)
        var longNetFunding = longFundingReceived - longFundingPaid;
        var shortNetFunding = shortFundingReceived - shortFundingPaid;
        var totalNetFunding = longNetFunding + shortNetFunding;

        // Estimate funding rates from cumulative NET fees
        var positionSizeUsd = (float)longPos.InitialMargin;  // Size per side
        var longFundingRate = hoursHeld > 0 && positionSizeUsd > 0
            ? longNetFunding / positionSizeUsd / hoursHeld * 8  // Normalize to 8h rate
            : 0.0f;
        var shortFundingRate = hoursHeld > 0 && positionSizeUsd > 0
            ? shortNetFunding / positionSizeUsd / hoursHeld * 8
            : 0.0f;

        // BLENDING LOGIC: For NEW executions (< 8 hours), blend estimated (from opportunity) with actual funding
        // For MATURE executions (>= 8 hours), use only actual funding
        const float BLEND_THRESHOLD_HOURS = 8.0f;
        if (hoursHeld < BLEND_THRESHOLD_HOURS && opportunity != null)
        {
            // Get estimated funding rates from active opportunity
            // Determine which exchange is LONG and which is SHORT to match rates correctly
            var longExchange = longPos.Exchange;
            var shortExchange = shortPos.Exchange;

            // Match opportunity's long/short to position's long/short based on exchange
            var estimatedLongRate = longExchange == opportunity.LongExchange
                ? (float)opportunity.LongFundingRate
                : (float)opportunity.ShortFundingRate;
            var estimatedShortRate = shortExchange == opportunity.ShortExchange
                ? (float)opportunity.ShortFundingRate
                : (float)opportunity.LongFundingRate;

            // Blend factor: 0 = all estimated, 1 = all actual
            var blendFactor = hoursHeld / BLEND_THRESHOLD_HOURS;

            // Blend: new execution uses estimated, mature execution uses actual
            var originalLongRate = longFundingRate;
            var originalShortRate = shortFundingRate;

            longFundingRate = longFundingRate * blendFactor + estimatedLongRate * (1 - blendFactor);
            shortFundingRate = shortFundingRate * blendFactor + estimatedShortRate * (1 - blendFactor);

            _logger.LogDebug("Blended funding rates for {Hours:F1}h execution (factor={Factor:F2}): " +
                "long {Actual:F6}→{Blended:F6}, short {ActualShort:F6}→{BlendedShort:F6}",
                hoursHeld, blendFactor,
                originalLongRate, longFundingRate, originalShortRate, shortFundingRate);
        }

        // Current prices (use entry price + P&L to estimate current price)
        var longCurrentPrice = longPos.ExitPrice.HasValue
            ? (float)longPos.ExitPrice.Value
            : (float)(longPos.EntryPrice * (1 + longPos.UnrealizedPnL / longPos.InitialMargin));
        var shortCurrentPrice = shortPos.ExitPrice.HasValue
            ? (float)shortPos.ExitPrice.Value
            : (float)(shortPos.EntryPrice * (1 - shortPos.UnrealizedPnL / shortPos.InitialMargin));

        // Entry/trading fees (estimate ~0.02% maker fee per side)
        var entryFeesUsd = positionSizeUsd * 2 * 0.0002f;

        return new
        {
            symbol = longPos.Symbol,
            unrealized_pnl_pct = netPnlPct,
            position_age_hours = hoursHeld,
            // Send NET funding for each side (positive = profit, negative = cost)
            long_net_funding_usd = longNetFunding,
            short_net_funding_usd = shortNetFunding,
            total_net_funding_usd = totalNetFunding,
            short_funding_rate = shortFundingRate,
            long_funding_rate = longFundingRate,
            current_long_price = longCurrentPrice,
            current_short_price = shortCurrentPrice,
            entry_long_price = (float)longPos.EntryPrice,
            entry_short_price = (float)shortPos.EntryPrice,
            position_size_usd = positionSizeUsd,
            entry_fees_paid_usd = entryFeesUsd,
            long_pnl_pct = longPnlPct,
            short_pnl_pct = shortPnlPct,
            leverage = (float)longPos.Leverage  // For ML to calculate liquidation_distance
        };
    }

    #region Feature Preparation Helper Methods

    /// <summary>
    /// Build trading config from AgentConfiguration (5 features)
    /// </summary>
    private RLTradingConfig BuildTradingConfig(AgentConfiguration config)
    {
        return new RLTradingConfig
        {
            MaxLeverage = config.MaxLeverage,
            TargetUtilization = config.TargetUtilization,
            MaxPositions = config.MaxPositions,
            StopLossThreshold = -0.02m,  // Default: -2% stop loss
            LiquidationBuffer = 0.15m    // Default: 15% buffer from liquidation
        };
    }

    /// <summary>
    /// Convert PositionDto from UserDataSnapshot repository to Position entities
    /// This ensures ML predictor uses the same data source as the frontend
    /// </summary>
    private List<Position> ConvertPositionDtosToEntities(List<PositionDto> dtos, string userId)
    {
        return dtos.Select(dto => new Position
        {
            Id = dto.Id,
            ExecutionId = dto.ExecutionId,
            UserId = userId,
            Exchange = dto.Exchange,
            Symbol = dto.Symbol,
            Type = dto.Type,
            Side = dto.Side,
            Status = dto.Status,
            EntryPrice = dto.EntryPrice,
            ExitPrice = dto.ExitPrice,
            Quantity = dto.Quantity,
            Leverage = dto.Leverage,
            InitialMargin = dto.InitialMargin,
            FundingEarnedUsd = dto.FundingEarnedUsd,
            TradingFeesUsd = dto.TradingFeesUsd,
            PricePnLUsd = dto.PricePnLUsd,
            RealizedPnLUsd = dto.RealizedPnLUsd,
            RealizedPnLPct = dto.RealizedPnLPct,
            UnrealizedPnL = dto.UnrealizedPnL,
            ReconciliationStatus = dto.ReconciliationStatus,
            ReconciliationCompletedAt = dto.ReconciliationCompletedAt,
            OpenedAt = dto.OpenedAt,
            ClosedAt = dto.ClosedAt,

            // ML features (Phase 1 exit timing) - now available from DTO
            EntryApr = dto.EntryApr,
            PeakPnlPct = dto.PeakPnlPct,
            PnlHistoryJson = dto.PnlHistoryJson
            // Note: Position entity doesn't have TotalFundingFeePaid, TotalFundingFeeReceived, TradingFeePaid, ActiveOpportunityId
            // These are DTO-specific properties that aren't needed for ML feature calculation
        }).ToList();
    }

    // REMOVED: GetActivePositionsAsync - No longer needed as we use UserDataSnapshot repository
    // The ML predictor now uses the same data source as the frontend (UserDataSnapshot)
    // instead of querying the database directly

    /// <summary>
    /// Build complete portfolio state (V3: 3 portfolio features + 85 execution features = 5 slots × 17 features)
    /// V3 Changes: Portfolio 10→3 (removed historical), Execution 100→85 (added velocities)
    /// Matches environment.py:_get_observation() portfolio section
    /// </summary>
    private async Task<RLPortfolioState> BuildCompletePortfolioStateAsync(
        string userId,
        List<Position> positions,
        AgentConfiguration config,
        float bestAvailableApr,
        List<ArbitrageOpportunityDto> opportunities,
        CancellationToken cancellationToken)
    {
        // 1. Get total capital from UserDataSnapshot
        var totalCapital = await GetTotalCapitalAsync(userId, cancellationToken);
        var marginUsed = await GetMarginUsedAsync(userId, cancellationToken);

        // 2. Calculate utilization (margin used / total capital)
        var utilization = totalCapital > 0 ? (float)(marginUsed / totalCapital) : 0.0f;

        // 3. Calculate average position P&L
        var avgPnlPct = CalculateAvgPositionPnl(positions, totalCapital);

        // 4. Calculate total P&L % from position P&L (MATCHES TEST ENVIRONMENT EXACTLY)
        // Test environment (portfolio.py:897-898):
        //   total_pnl_pct = (total_capital - initial_capital) / initial_capital * 100
        // Where total_capital starts at $10k and accumulates position P&L as they close
        //
        // In production, we calculate the equivalent by summing unrealized P&L from ALL positions
        // divided by total capital (portfolio.py:222-232 for individual position P&L)
        var totalPnlPct = totalCapital > 0 && positions.Any()
            ? (float)(positions.Sum(p => p.UnrealizedPnL) / totalCapital * 100m)
            : 0.0f;

        // 5. Get max drawdown
        var drawdown = await GetMaxDrawdownAsync(userId, cancellationToken);

        // 6. Episode progress removed (was session-based, not needed without TradingSessionService)
        // In test environment, this tracks progress through 72-hour episodes
        // In production, we don't have fixed-length episodes, so set to 0
        var episodeProgress = 0.0f;

        // 7. Get market data snapshot (needed for liquidation distance and position states)
        var marketSnapshot = await _marketDataRepository.GetAsync("market_data_snapshot", cancellationToken);

        // 8. Calculate minimum liquidation distance across all positions
        var minLiqDistance = CalculateMinLiquidationDistance(positions, marketSnapshot);

        // 9. Calculate capital utilization (total value of positions / capital)
        var capitalUtil = CalculateCapitalUtilization(positions, totalCapital);

        // 10. Build position features (5 slots × 17 features each) - V3: Added velocities, removed historical
        var positionStates = await BuildPositionStatesAsync(positions, totalCapital, marketSnapshot, bestAvailableApr, opportunities, cancellationToken);

        // Enhanced logging for P&L verification
        var sumPositionPnls = positions.Count > 0 ? positions.Sum(p => p.UnrealizedPnL) : 0m;
        var verifyPnlPct = totalCapital > 0 ? (float)(sumPositionPnls / totalCapital * 100) : 0f;

        _logger.LogInformation(
            "Portfolio State: Capital=${Capital}, TotalPnlPct={TotalPnlPct}% (verify={VerifyPnl}%), " +
            "AvgPosPnl={AvgPnl}%, NumPositions={NumPositions}, MarginUtil={MarginUtil}%, " +
            "EpisodeProgress={EpisodeProgress:F2}, SumPosPnl=${SumPnl}",
            totalCapital, totalPnlPct, verifyPnlPct, avgPnlPct,
            positions.Select(p => p.ExecutionId).Distinct().Count(),
            utilization * 100, episodeProgress, sumPositionPnls);

        return new RLPortfolioState
        {
            // Core metrics
            Capital = totalCapital,
            InitialCapital = totalCapital,  // Set to current capital (we calculate total_pnl_pct from position P&L instead)
            NumPositions = positions.Select(p => p.ExecutionId).Distinct().Count(),
            MarginUtilization = utilization * 100,  // Feature #3: margin locked / capital (as percentage 0-100)

            // P&L metrics
            AvgPositionPnlPct = avgPnlPct,
            TotalPnlPct = totalPnlPct,
            Drawdown = drawdown,

            // Progress and risk
            EpisodeProgress = episodeProgress,
            MinLiquidationDistance = minLiqDistance,
            CapitalUtilization = capitalUtil,  // Feature #10: total notional / capital (as percentage 0-100)

            // Position details
            Positions = positionStates
        };
    }

    /// <summary>
    /// Build position states (V3: 17 features per execution, up to 5 slots - was 20)
    /// V3: Added velocities (pnl, funding, spread), removed net funding metrics
    /// Matches portfolio.py:get_execution_state()
    /// </summary>
    private async Task<List<RLPositionState>> BuildPositionStatesAsync(
        List<Position> positions,
        decimal totalCapital,
        MarketDataSnapshot? marketSnapshot,
        float bestAvailableApr,
        List<ArbitrageOpportunityDto> opportunities,
        CancellationToken cancellationToken)
    {
        var result = new List<RLPositionState>();

        // Group by ExecutionId (each execution = 1 position state with long + short)
        var executionGroups = positions
            .GroupBy(p => p.ExecutionId)
            .Take(5);  // Max 5 slots

        foreach (var execution in executionGroups)
        {
            var longPos = execution.FirstOrDefault(p => p.Side == PositionSide.Long);
            var shortPos = execution.FirstOrDefault(p => p.Side == PositionSide.Short);

            if (longPos == null || shortPos == null)
            {
                _logger.LogWarning("Execution {ExecutionId} missing long or short leg, skipping", execution.Key);
                continue;
            }

            // Get current prices
            var longPrice = GetCurrentPrice(longPos.Exchange, longPos.Symbol, marketSnapshot);
            var shortPrice = GetCurrentPrice(shortPos.Exchange, shortPos.Symbol, marketSnapshot);

            // Get current funding rates
            var longFR = await GetCurrentFundingRateAsync(longPos.Exchange, longPos.Symbol, cancellationToken);
            var shortFR = await GetCurrentFundingRateAsync(shortPos.Exchange, shortPos.Symbol, cancellationToken);

            // Calculate net funding from historical transactions
            var longFunding = longPos.FundingEarnedUsd;  // Using the calculated field
            var shortFunding = shortPos.FundingEarnedUsd;
            var totalFunding = longFunding + shortFunding;

            // Calculate position size
            var positionSize = longPos.Quantity * longPos.EntryPrice;

            // Calculate position age for logging
            var positionAge = (DateTime.UtcNow - longPos.OpenedAt).TotalHours;

            // CRITICAL FIX: Use P&L history directly from database (no delay)
            // Test environment (portfolio.py:243-245) appends to pnl_history every step (1 hour)
            // Production: UserDataCollector appends every 5 minutes
            // We send whatever history exists to match test behavior where pnl_history
            // starts accumulating from the first hour
            var pnlHistoryToUse = ParsePnlHistory(longPos.PnlHistoryJson);
            _logger.LogDebug(
                "Using P&L history for {Symbol} (age={Age:F1}h): {Count} samples",
                longPos.Symbol, positionAge, pnlHistoryToUse.Count);

            // Build position state with raw data fields for ML predictor
            var state = new RLPositionState
            {
                // Symbol for logging and debugging
                Symbol = longPos.Symbol,

                // ===== FIELDS USED DIRECTLY BY ML PREDICTOR =====

                // P&L metrics (ML reads these directly)
                UnrealizedPnlPct = CalculateNetPnlPct(longPos, shortPos, longPrice, shortPrice),
                LongPnlPct = CalculateSidePnlPct(longPos, longPrice),
                ShortPnlPct = CalculateSidePnlPct(shortPos, shortPrice),

                // Risk metric (ML reads directly)
                LiquidationDistance = CalculateLiquidationDistance(longPos, shortPos, longPrice, shortPrice),

                // ===== RAW DATA FIELDS (ML Calculates Features From These) =====

                // Time (ML will normalize by 72h)
                PositionAgeHours = (float)positionAge,

                // Funding amounts (USD)
                LongNetFundingUsd = (float)longFunding,
                ShortNetFundingUsd = (float)shortFunding,

                // Individual funding rates
                ShortFundingRate = (float)shortFR,
                LongFundingRate = (float)longFR,

                // Raw prices
                CurrentLongPrice = (float)longPrice,
                CurrentShortPrice = (float)shortPrice,
                EntryLongPrice = (float)longPos.EntryPrice,
                EntryShortPrice = (float)shortPos.EntryPrice,

                // Position sizing
                PositionSizeUsd = (float)positionSize,
                // FALLBACK: Calculate entry fees if database has 0 (legacy positions)
                // Entry fees: 0.01% maker fee × 2 sides = 0.02% = 0.0002
                EntryFeesPaidUsd = (longPos.TradingFeesUsd + shortPos.TradingFeesUsd) > 0
                    ? (float)(longPos.TradingFeesUsd + shortPos.TradingFeesUsd)
                    : (float)((longPos.Quantity * longPos.EntryPrice + shortPos.Quantity * shortPos.EntryPrice) * 0.0002m),

                // ===== PHASE 1 EXIT TIMING FEATURES =====

                // For pnl_velocity: P&L history (ML calculates velocity from this)
                // Test environment: appends every hour starting from hour 1
                // Production: UserDataCollector appends every 5 minutes
                // We send whatever history exists (no delay)
                PnlHistory = pnlHistoryToUse,

                // For peak_drawdown: Peak P&L percentage reached (ML calculates drawdown from this)
                // Test environment: tracks peak from beginning (portfolio.py:247-250)
                // Production: UserDataCollector tracks peak from beginning
                // We send whatever peak exists (no delay)
                PeakPnlPct = (float)longPos.PeakPnlPct,

                // For apr_ratio: APR at entry time (ML calculates ratio vs current APR)
                // Read from Position entity (now populated from database via UserDataCollector → PositionDto)
                EntryApr = (float)longPos.EntryApr,

                // ===== PHASE 2 APR COMPARISON FEATURES =====
                // Calculate APR comparison features (features 18-20)

                // Feature 18: Current Position APR (direct from opportunities, matching test_inference)
                // Use raw opportunity APR without smoothing to match test behavior
                CurrentPositionApr = GetPositionAprFromOpportunities(longPos.Symbol, opportunities),

                // Feature 19: Best Available APR (max APR among current opportunities)
                BestAvailableApr = bestAvailableApr,

                // Feature 20: APR Advantage (current - best, negative = better opportunities exist)
                AprAdvantage = GetPositionAprFromOpportunities(longPos.Symbol, opportunities) - bestAvailableApr,

                // ===== DEPRECATED FIELDS (kept for backward compatibility) =====
                IsActive = true,
                PnlPct = CalculateNetPnlPct(longPos, shortPos, longPrice, shortPrice),
                HoursHeld = (float)(DateTime.UtcNow - longPos.OpenedAt).TotalHours,  // Actual hours, NOT normalized
                NetFundingRatio = positionSize > 0 ? (float)(totalFunding / positionSize) : 0.0f,
                FundingRate = (float)(shortFR - longFR),
                CurrentSpreadPct = CalculateSpreadPct(longPrice, shortPrice),
                EntrySpreadPct = CalculateSpreadPct(longPos.EntryPrice, shortPos.EntryPrice),
                ValueToCapitalRatio = totalCapital > 0 ? (float)(positionSize / totalCapital) : 0.0f,
                FundingEfficiency = CalculateFundingEfficiency(totalFunding, longPos.TradingFeesUsd, shortPos.TradingFeesUsd),
                LiquidationDistancePct = CalculateLiquidationDistance(longPos, shortPos, longPrice, shortPrice),
                TotalNetFundingUsd = (float)totalFunding,
                PositionIsActive = 1.0f
            };

            result.Add(state);
        }

        // Fill remaining slots with inactive/empty states (all zeros)
        while (result.Count < 5)
        {
            result.Add(new RLPositionState
            {
                // Symbol for empty slot
                Symbol = "",

                // ML predictor reads these directly (all zeros for empty slot)
                UnrealizedPnlPct = 0,
                LongPnlPct = 0,
                ShortPnlPct = 0,
                LiquidationDistance = 1.0f,  // 1.0 = safe (far from liquidation)

                // Raw data fields for ML (all zeros)
                PositionAgeHours = 0,
                LongNetFundingUsd = 0,
                ShortNetFundingUsd = 0,
                ShortFundingRate = 0,
                LongFundingRate = 0,
                CurrentLongPrice = 0,
                CurrentShortPrice = 0,
                EntryLongPrice = 0,
                EntryShortPrice = 0,
                PositionSizeUsd = 0,
                EntryFeesPaidUsd = 0,

                // Phase 1 exit timing features (empty slot values)
                PnlHistory = new List<float>(),
                PeakPnlPct = 0f,
                EntryApr = 0f,

                // Phase 2 APR comparison features (empty slot values)
                CurrentPositionApr = 0f,
                BestAvailableApr = 0f,
                AprAdvantage = 0f,

                // Deprecated fields (kept for compatibility)
                IsActive = false,
                PnlPct = 0,
                HoursHeld = 0,
                NetFundingRatio = 0,
                FundingRate = 0,
                CurrentSpreadPct = 0,
                EntrySpreadPct = 0,
                ValueToCapitalRatio = 0,
                FundingEfficiency = 0,
                LiquidationDistancePct = 1.0f,
                TotalNetFundingUsd = 0,
                PositionIsActive = 0.0f
            });
        }

        return result;
    }

    #endregion

    #region Data Query Methods

    private async Task<decimal> GetTotalCapitalAsync(string userId, CancellationToken cancellationToken)
    {
        var userDataDict = await _userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);
        decimal total = 0m;

        foreach (var snapshot in userDataDict.Values)
        {
            if (snapshot?.Balance != null)
            {
                total += snapshot.Balance.FuturesBalanceUsd;
            }
        }

        return total > 0 ? total : 10000m;  // Default to 10k if no data
    }

    private async Task<decimal> GetMarginUsedAsync(string userId, CancellationToken cancellationToken)
    {
        var userDataDict = await _userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);
        decimal total = 0m;

        foreach (var snapshot in userDataDict.Values)
        {
            if (snapshot?.Balance != null)
            {
                total += snapshot.Balance.MarginUsed;
            }
        }

        return total;
    }

    private async Task<float> CalculateTotalPnlPctFromUserDataAsync(string userId, decimal totalCapital, CancellationToken cancellationToken)
    {
        // DEPRECATED: This method is kept for backward compatibility but no longer used
        // It was reading stale data from UserDataSnapshot.Balance.UnrealizedPnL
        // Now we calculate directly from positions in CalculateTotalPnlPctFromPositions

        // Get UserDataSnapshot for all exchanges
        var userDataDict = await _userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);

        if (!userDataDict.Any() || totalCapital == 0)
            return 0.0f;

        // Sum unrealized P&L from all open positions across all exchanges
        decimal totalUnrealizedPnl = 0m;

        foreach (var snapshot in userDataDict.Values)
        {
            if (snapshot?.Balance != null)
            {
                totalUnrealizedPnl += snapshot.Balance.UnrealizedPnL;
            }
        }

        // Calculate percentage: (unrealized P&L / capital) * 100
        var pnlPct = (float)(totalUnrealizedPnl / totalCapital * 100);

        _logger.LogDebug(
            "Total P&L (from UserData): UnrealizedPnL=${UnrealizedPnl}, Capital=${Capital}, PnlPct={PnlPct}%",
            totalUnrealizedPnl, totalCapital, pnlPct);

        return pnlPct;
    }

    /// <summary>
    /// Calculate total P&L percentage directly from positions.
    /// This ensures the value matches what the ML model expects (same as test environment).
    /// </summary>
    private float CalculateTotalPnlPctFromPositions(List<Position> positions, decimal totalCapital)
    {
        if (totalCapital == 0)
            return 0.0f;

        // Sum unrealized P&L from all positions directly
        decimal totalUnrealizedPnl = positions.Sum(p => p.UnrealizedPnL);

        // Calculate percentage: (unrealized P&L / capital) * 100
        var pnlPct = (float)(totalUnrealizedPnl / totalCapital * 100);

        _logger.LogDebug(
            "Total P&L (from positions): UnrealizedPnL=${UnrealizedPnl}, Capital=${Capital}, PnlPct={PnlPct}%, NumPositions={NumPositions}",
            totalUnrealizedPnl, totalCapital, pnlPct, positions.Count);

        return pnlPct;
    }

    private async Task<float> GetTotalPnlPctAsync(string userId, CancellationToken cancellationToken)
    {
        var latestMetric = await _context.PerformanceMetrics
            .Where(pm => pm.UserId == userId)
            .OrderByDescending(pm => pm.Date)
            .FirstOrDefaultAsync(cancellationToken);

        if (latestMetric == null)
            return 0.0f;

        var capital = await GetTotalCapitalAsync(userId, cancellationToken);
        return capital > 0 ? (float)(latestMetric.TotalPnL / capital * 100) : 0.0f;
    }

    private async Task<float> GetMaxDrawdownAsync(string userId, CancellationToken cancellationToken)
    {
        // TODO: Implement actual drawdown tracking
        // For now, return 0 (no drawdown)
        return 0.0f;
    }

    private async Task<decimal> GetCurrentFundingRateAsync(string exchange, string symbol, CancellationToken cancellationToken)
    {
        var key = DataCollectionConstants.CacheKeys.BuildFundingRateKey(exchange, symbol);
        try
        {
            var fundingRate = await _fundingRateRepository.GetAsync(key, cancellationToken);
            return fundingRate?.Rate ?? 0m;
        }
        catch
        {
            return 0m;
        }
    }

    private decimal GetCurrentPrice(string exchange, string symbol, MarketDataSnapshot? marketSnapshot)
    {
        if (marketSnapshot?.PerpPrices == null)
            return 0m;

        // PerpPrices is Dictionary<exchange, Dictionary<symbol, PriceDto>>
        if (marketSnapshot.PerpPrices.TryGetValue(exchange, out var exchangePrices) &&
            exchangePrices.TryGetValue(symbol, out var priceDto))
        {
            return priceDto.Price;
        }
        return 0m;
    }

    #endregion

    #region Calculation Methods

    private float CalculateAvgPositionPnl(List<Position> positions, decimal totalCapital)
    {
        // Match test environment: only calculate from CLOSED positions
        // This prevents the model from seeing negative P&L immediately on open positions
        var closedPositions = positions.Where(p => p.Status == PositionStatus.Closed).ToList();

        if (!closedPositions.Any() || totalCapital == 0)
            return 0.0f; // No closed positions yet, return 0

        var executionGroups = closedPositions.GroupBy(p => p.ExecutionId);
        var pnls = new List<float>();

        foreach (var execution in executionGroups)
        {
            // For closed positions, use RealizedPnLUsd instead of UnrealizedPnL
            var totalPnl = execution.Sum(p => p.RealizedPnLUsd);
            var totalMargin = execution.Sum(p => p.InitialMargin);
            if (totalMargin > 0)
            {
                pnls.Add((float)(totalPnl / totalMargin * 100));
            }
        }

        return pnls.Any() ? pnls.Average() : 0.0f;
    }

    private float CalculateMinLiquidationDistance(List<Position> positions, MarketDataSnapshot? marketSnapshot)
    {
        // Match ground truth: portfolio.py:get_min_liquidation_distance() (lines 523-546)
        if (!positions.Any())
            return 1.0f;  // 1.0 = safe (no positions, no liquidation risk)

        // Calculate minimum liquidation distance across all position pairs
        var minDistance = 1.0f;

        var executionGroups = positions.GroupBy(p => p.ExecutionId);

        foreach (var execution in executionGroups)
        {
            var longPos = execution.FirstOrDefault(p => p.Side == PositionSide.Long);
            var shortPos = execution.FirstOrDefault(p => p.Side == PositionSide.Short);

            if (longPos == null || shortPos == null)
                continue;  // Skip incomplete executions

            // Get current prices for this position
            var longPrice = GetCurrentPrice(longPos.Exchange, longPos.Symbol, marketSnapshot);
            var shortPrice = GetCurrentPrice(shortPos.Exchange, shortPos.Symbol, marketSnapshot);

            if (longPrice == 0 || shortPrice == 0)
                continue;  // Skip if no price data available

            // Calculate liquidation distance for this execution
            var distance = CalculateLiquidationDistance(longPos, shortPos, longPrice, shortPrice);

            // Track minimum (most at risk)
            minDistance = Math.Min(minDistance, distance);
        }

        return minDistance;
    }

    private float CalculateCapitalUtilization(List<Position> positions, decimal totalCapital)
    {
        if (!positions.Any() || totalCapital == 0)
            return 0.0f;

        // Group positions by ExecutionId to calculate total notional value per execution
        // Training environment formula: sum(position_size_usd * 2) for each execution
        // Backend equivalent: sum((long_qty * long_price) + (short_qty * short_price)) for each execution
        var totalNotionalValue = positions
            .GroupBy(p => p.ExecutionId)
            .Sum(executionGroup =>
            {
                // Calculate total notional for this execution (sum of both legs)
                return executionGroup.Sum(pos => pos.Quantity * pos.EntryPrice);
            });

        return (float)(totalNotionalValue / totalCapital * 100);
    }

    private float CalculateNetPnlPct(Position longPos, Position shortPos, decimal longPrice, decimal shortPrice)
    {
        // Calculate position sizes
        var longValue = longPos.Quantity * longPos.EntryPrice;
        var shortValue = shortPos.Quantity * shortPos.EntryPrice;

        // Match ML environment: total_capital_used = position_size_usd * 2 (both long and short positions)
        var totalCapitalUsed = longValue + shortValue;

        if (totalCapitalUsed == 0)
            return 0.0f;

        // CRITICAL FIX: Use UnrealizedPnL from database (which comes from exchange and INCLUDES funding fees)
        // Environment calculation: unrealized_pnl_usd = long_price_pnl + short_price_pnl + net_funding - entry_fees
        // Exchange APIs provide unrealizedPnL that already includes funding fees
        // DO NOT recalculate from scratch using only price changes, as that misses funding
        var totalPnl = longPos.UnrealizedPnL + shortPos.UnrealizedPnL;

        // Use position size, NOT margin, to match ML training
        // With 2x leverage: $10 profit on $1000 position = 1%, not 2%
        return (float)(totalPnl / totalCapitalUsed * 100);
    }

    private float CalculateSpreadPct(decimal price1, decimal price2)
    {
        if (price1 == 0 || price2 == 0)
            return 0.0f;

        // CRITICAL: Return decimal (0.01 = 1%), NOT percentage (1.0 = 1%) to match ML environment
        return (float)(Math.Abs(price1 - price2) / ((price1 + price2) / 2));
    }

    private float CalculateFundingEfficiency(decimal netFunding, decimal longFees, decimal shortFees)
    {
        var totalFees = longFees + shortFees;
        if (totalFees == 0)
            return 0.0f;

        return (float)(netFunding / totalFees);
    }

    private float CalculateSidePnlPct(Position pos, decimal currentPrice)
    {
        // Match ML environment: use position size, not margin
        var positionSize = pos.Quantity * pos.EntryPrice;

        if (positionSize == 0)
            return 0.0f;

        // CRITICAL FIX: Use UnrealizedPnL from database (includes funding fees)
        // This matches the environment which includes funding in long_pnl_pct and short_pnl_pct
        return (float)(pos.UnrealizedPnL / positionSize * 100);
    }

    private float CalculateLiquidationDistance(Position longPos, Position shortPos, decimal longPrice, decimal shortPrice)
    {
        // Match ML environment formula (portfolio.py:318-341)
        // Liquidation occurs when loss reaches ~(1/leverage) of position value

        // Get leverage (use max of both positions' leverage)
        var leverage = Math.Max(longPos.Leverage, shortPos.Leverage);
        if (leverage == 0)
            return 1.0f;  // Safe distance if no leverage

        // Long liquidation price: entry * (1 - 0.9 / leverage)
        var longLiqPrice = longPos.EntryPrice * (1 - 0.9m / leverage);
        var longDistance = longPrice > 0
            ? (float)(Math.Abs(longPrice - longLiqPrice) / longPrice)
            : 1.0f;

        // Short liquidation price: entry * (1 + 0.9 / leverage)
        var shortLiqPrice = shortPos.EntryPrice * (1 + 0.9m / leverage);
        var shortDistance = shortPrice > 0
            ? (float)(Math.Abs(shortLiqPrice - shortPrice) / shortPrice)
            : 1.0f;

        // Return minimum distance (most at risk)
        return Math.Min(longDistance, shortDistance);
    }

    private float CalculateEntryApr(decimal longFundingRate, decimal shortFundingRate)
    {
        // Calculate APR from funding rates
        // APR = (short_funding_rate - long_funding_rate) * 365 * 3
        // Where:
        //   - Net funding rate = short rate - long rate (arbitrage profit)
        //   - 365 = days per year
        //   - 3 = number of 8-hour funding intervals per day (24h / 8h = 3)
        // NOTE: Using current rates as proxy for entry rates since historical rates aren't stored
        var netFundingRate = shortFundingRate - longFundingRate;
        var apr = (float)(netFundingRate * 365 * 3 * 100);  // Convert to percentage
        return apr;
    }

    private float CalculateCurrentPositionApr(decimal shortFundingRate, decimal longFundingRate,
        decimal longIntervalHours, decimal shortIntervalHours)
    {
        // DEPRECATED: This method has a bug - it subtracts funding rates before converting to daily
        // which is incorrect when intervals differ (e.g., 1h vs 8h)
        // Use GetPositionAprFromOpportunities instead for correct APR calculation

        var netFundingRate = shortFundingRate - longFundingRate;
        var avgIntervalHours = (longIntervalHours + shortIntervalHours) / 2m;

        // Avoid division by zero
        if (avgIntervalHours <= 0)
            avgIntervalHours = 8m;  // Fallback to default 8h

        var paymentsPerDay = 24m / avgIntervalHours;
        var apr = (float)(netFundingRate * paymentsPerDay * 365m * 100m);

        return apr;
    }

    private float GetPositionAprFromOpportunities(string symbol, List<ArbitrageOpportunityDto> opportunities)
    {
        // Lookup the matching opportunity for this position's symbol
        // The opportunity APR is calculated correctly (converts each funding rate to daily independently)
        var matchingOpp = opportunities.FirstOrDefault(o => o.Symbol == symbol);
        if (matchingOpp != null)
        {
            return (float)matchingOpp.FundApr;
        }

        // Fallback: If no matching opportunity found, return 0
        // This can happen if the opportunity expired but position still exists
        _logger.LogWarning("No matching opportunity found for position symbol {Symbol}, using APR=0", symbol);
        return 0f;
    }

    private float GetSmoothedPositionApr(string symbol, decimal entryApr, List<ArbitrageOpportunityDto> opportunities)
    {
        // Get current market APR for this symbol
        var currentMarketApr = GetPositionAprFromOpportunities(symbol, opportunities);

        // If no market APR available or very close to entry, use entry APR
        // This prevents sudden drops when opportunity disappears temporarily
        if (currentMarketApr == 0f)
        {
            _logger.LogDebug("No market APR for {Symbol}, using entry APR {EntryApr}", symbol, entryApr);
            return (float)entryApr;
        }

        // Apply exponential smoothing between entry APR and current market APR
        // Alpha = 0.1 means 10% weight to current, 90% to entry
        // This strongly reduces sensitivity to rapid APR changes for maximum position stability
        const float alpha = 0.1f;
        var smoothedApr = (alpha * currentMarketApr) + ((1 - alpha) * (float)entryApr);

        _logger.LogDebug("APR smoothing for {Symbol}: Entry={EntryApr:F0}, Market={MarketApr:F0}, Smoothed={SmoothedApr:F0}",
            symbol, entryApr, currentMarketApr, smoothedApr);

        return smoothedApr;
    }

    private List<float> ParsePnlHistory(string? pnlHistoryJson)
    {
        // Parse JSON array of hourly P&L snapshots from Position.PnlHistoryJson
        if (string.IsNullOrEmpty(pnlHistoryJson))
        {
            return new List<float>();  // Return empty list if no history yet
        }

        try
        {
            var pnlHistory = System.Text.Json.JsonSerializer.Deserialize<List<decimal>>(pnlHistoryJson);
            return pnlHistory?.Select(p => (float)p).ToList() ?? new List<float>();
        }
        catch (System.Text.Json.JsonException ex)
        {
            _logger.LogWarning(ex, "Failed to parse PnlHistoryJson, returning empty list");
            return new List<float>();
        }
    }

    #endregion

    public void Dispose()
    {
        _httpClient.Dispose();
    }

    #region Response DTOs

    /// <summary>
    /// Response from /rl/predict/opportunities endpoint
    /// </summary>
    private class RLOpportunitiesResponse
    {
        [JsonPropertyName("predictions")]
        public List<RLOpportunityPrediction> Predictions { get; set; } = new();

        [JsonPropertyName("model_version")]
        public string ModelVersion { get; set; } = string.Empty;
    }

    private class RLOpportunityPrediction
    {
        [JsonPropertyName("opportunity_index")]
        public int OpportunityIndex { get; set; }

        [JsonPropertyName("symbol")]
        public string Symbol { get; set; } = string.Empty;

        [JsonPropertyName("enter_probability")]
        public float EnterProbability { get; set; }

        [JsonPropertyName("confidence")]
        public string Confidence { get; set; } = "LOW";

        [JsonPropertyName("hold_probability")]
        public float HoldProbability { get; set; }

        [JsonPropertyName("state_value")]
        public float StateValue { get; set; }
    }

    /// <summary>
    /// Response from /rl/predict/positions endpoint
    /// </summary>
    private class RLPositionsResponse
    {
        [JsonPropertyName("predictions")]
        public List<RLPositionPrediction> Predictions { get; set; } = new();

        [JsonPropertyName("model_version")]
        public string ModelVersion { get; set; } = string.Empty;
    }

    /// <summary>
    /// Find the most appropriate opportunity for a given execution based on symbol and exchange matching
    /// </summary>
    private ArbitrageOpportunityDto? FindMatchingOpportunity(
        List<PositionDto> executionPositions,
        List<ArbitrageOpportunityDto>? opportunities)
    {
        if (opportunities == null || opportunities.Count == 0 || executionPositions.Count == 0)
            return null;

        var firstPosition = executionPositions[0];
        var symbol = firstPosition.Symbol;

        // Find opportunities that match the symbol
        var matchingOpportunities = opportunities.Where(opp => opp.Symbol == symbol).ToList();

        if (matchingOpportunities.Count == 0)
            return null;

        // Try to match by exchange (for spot-perp or cross-exchange)
        var longPos = executionPositions.FirstOrDefault(p => p.Side == PositionSide.Long);
        var shortPos = executionPositions.FirstOrDefault(p => p.Side == PositionSide.Short);

        if (longPos != null && shortPos != null)
        {
            // Try to match cross-exchange opportunity (long exchange + short exchange)
            var exactMatch = matchingOpportunities.FirstOrDefault(opp =>
                (opp.LongExchange == longPos.Exchange && opp.ShortExchange == shortPos.Exchange) ||
                (opp.Exchange == longPos.Exchange && longPos.Exchange == shortPos.Exchange) // Same exchange spot-perp
            );

            if (exactMatch != null)
                return exactMatch;
        }

        // Return first matching opportunity by symbol if no exact match
        return matchingOpportunities.First();
    }

    private class RLPositionPrediction
    {
        [JsonPropertyName("position_index")]
        public int PositionIndex { get; set; }

        [JsonPropertyName("symbol")]
        public string Symbol { get; set; } = string.Empty;

        [JsonPropertyName("exit_probability")]
        public float ExitProbability { get; set; }

        [JsonPropertyName("confidence")]
        public string Confidence { get; set; } = "LOW";

        [JsonPropertyName("hold_probability")]
        public float HoldProbability { get; set; }

        [JsonPropertyName("state_value")]
        public float StateValue { get; set; }
    }

    /// <summary>
    /// Agent prediction result (used by AgentBackgroundService)
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

    #endregion
}
