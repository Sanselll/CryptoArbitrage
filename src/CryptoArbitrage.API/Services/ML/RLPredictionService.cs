using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// Service for getting RL (Reinforcement Learning) predictions from the Python ML API
/// Evaluates ENTER probabilities for opportunities and EXIT probabilities for positions
/// </summary>
public class RLPredictionService : IDisposable
{
    private readonly ILogger<RLPredictionService> _logger;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly JsonSerializerOptions _deserializeOptions;

    public RLPredictionService(ILogger<RLPredictionService> logger, IConfiguration configuration)
    {
        _logger = logger;

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

        if (posList.Count == 0)
            return result;

        try
        {
            // Group positions by ExecutionId to create execution-level features
            var executionGroups = posList
                .Where(p => p.ExecutionId.HasValue)
                .GroupBy(p => p.ExecutionId!.Value)
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
                            utilization = portfolio.Utilization,
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
    }

    /// <summary>
    /// Convert opportunity DTO to feature dictionary for Python API
    /// Matches the 22 features expected by the RL model
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
            positionCostPercent = (float)opp.PositionCostPercent
        };
    }

    /// <summary>
    /// Convert execution (2 positions: long + short) to feature dictionary for Python API
    /// Matches the 10 execution-level features expected by the RL model
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
            short_pnl_pct = shortPnlPct
        };
    }

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

    #endregion
}
