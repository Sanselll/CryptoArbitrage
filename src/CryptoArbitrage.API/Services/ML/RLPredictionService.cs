using System.Text;
using System.Text.Json;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Agent;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// V2 RL Prediction Service - Simplified with Unified Feature Builder
///
/// This service sends RAW data to the ML API. All feature engineering is done
/// in Python by the UnifiedFeatureBuilder class.
///
/// Key principles:
/// - Backend collects raw data from repositories
/// - No feature calculations in C#
/// - Python handles ALL feature preparation
/// - Single source of truth for features
/// </summary>
public class RLPredictionService
{
    private readonly HttpClient _httpClient;
    private readonly IAgentConfigurationService _agentConfigService;
    private readonly IDataRepository<UserDataSnapshot> _userDataRepository;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;
    private readonly ILogger<RLPredictionService> _logger;

    // ML API endpoint
    private const string ML_API_BASE_URL = "http://localhost:5250";
    private const string PREDICT_ENDPOINT = "/rl/predict";

    public RLPredictionService(
        HttpClient httpClient,
        IAgentConfigurationService agentConfigService,
        IDataRepository<UserDataSnapshot> userDataRepository,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        ILogger<RLPredictionService> logger)
    {
        _httpClient = httpClient;
        _agentConfigService = agentConfigService;
        _userDataRepository = userDataRepository;
        _marketDataRepository = marketDataRepository;
        _logger = logger;

        _httpClient.BaseAddress = new Uri(ML_API_BASE_URL);
        _httpClient.Timeout = TimeSpan.FromSeconds(10);
    }

    /// <summary>
    /// Get ML prediction for opportunities using V2 API (unified feature builder)
    /// </summary>
    public async Task<RLPredictionResponseV2?> GetPredictionAsync(
        List<OpportunityRawData> opportunities,
        List<PositionRawData> positions,
        TradingConfigRawData tradingConfig,
        decimal totalCapital,
        decimal capitalUtilization)
    {
        try
        {
            // Build raw data request
            var request = new RLRawDataRequest
            {
                TradingConfig = tradingConfig,
                Portfolio = new PortfolioRawData
                {
                    Positions = positions,
                    TotalCapital = totalCapital,
                    CapitalUtilization = capitalUtilization
                },
                Opportunities = opportunities
            };

            // Serialize using JsonPropertyName attributes (respects both snake_case and camelCase)
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = null,  // Respect JsonPropertyName attributes
                WriteIndented = false
            };

            var json = JsonSerializer.Serialize(request, options);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            _logger.LogInformation($"Sending prediction request (UnifiedFeatureBuilder): {opportunities.Count} opportunities, {positions.Count(p => p.IsActive)} positions");

            // Send request to ML API
            var response = await _httpClient.PostAsync(PREDICT_ENDPOINT, content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                _logger.LogError($"ML API error ({response.StatusCode}): {errorContent}");
                return null;
            }

            // Parse response
            var responseJson = await response.Content.ReadAsStringAsync();
            var prediction = JsonSerializer.Deserialize<RLPredictionResponseV2>(responseJson, options);

            if (prediction != null)
            {
                _logger.LogInformation($"Received prediction: Action={prediction.Action}, Confidence={prediction.Confidence:F2}");
            }

            return prediction;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Failed to connect to ML API");
            return null;
        }
        catch (TaskCanceledException ex)
        {
            _logger.LogError(ex, "ML API request timeout");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling ML API");
            return null;
        }
    }

    /// <summary>
    /// Get modular action for autonomous trading agent
    /// This is the primary method used by AgentBackgroundService
    /// </summary>
    public async Task<AgentPrediction?> GetModularActionAsync(
        string userId,
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        List<PositionDto> positions,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var oppList = opportunities.ToList();

            if (oppList.Count == 0)
                return new AgentPrediction { Action = "HOLD", Confidence = "LOW" };

            // 1. Get agent configuration
            var agentConfig = await _agentConfigService.GetOrCreateConfigurationAsync(userId);
            if (agentConfig == null)
            {
                _logger.LogError("AgentConfiguration is null for user {UserId}, cannot proceed", userId);
                return null;
            }

            // 2. Build trading config
            var tradingConfig = BuildTradingConfigRawData(agentConfig);

            // 3. Build opportunity raw data
            var opportunityData = oppList.Select(BuildOpportunityRawData).ToList();

            // 4. Build position raw data from current positions (async - retrieves current prices & funding rates)
            var positionData = await BuildPositionRawDataFromDtos(positions, oppList, cancellationToken);

            // 5. Calculate portfolio metrics from user balance data
            var userDataDict = await _userDataRepository.GetByPatternAsync($"userdata:{userId}:*", cancellationToken);
            decimal totalCapital = userDataDict.Values
                .Where(s => s?.Balance != null)
                .Sum(s => s!.Balance!.OperationalBalanceUsd);

            // Fallback to default if no balance data available
            if (totalCapital == 0m)
            {
                _logger.LogWarning("No balance data found for user {UserId}, using default $10,000", userId);
                totalCapital = 10000m;
            }

            decimal capitalUtilization = positionData.Any() && totalCapital != 0
                ? SanitizeDecimal(positionData.Sum(p => p.PositionSizeUsd) / totalCapital * 100m)
                : 0m;

            // 6. Call base prediction method
            var prediction = await GetPredictionAsync(
                opportunityData,
                positionData,
                tradingConfig,
                totalCapital,
                capitalUtilization);

            if (prediction == null)
                return new AgentPrediction { Action = "HOLD", Confidence = "LOW" };

            // 7. Map to AgentPrediction format
            var result = new AgentPrediction
            {
                Action = prediction.Action,
                Confidence = prediction.Confidence > 0.7m ? "HIGH"
                           : prediction.Confidence > 0.4m ? "MEDIUM"
                           : "LOW",
                StateValue = (double?)prediction.StateValue,
                OpportunityIndex = prediction.OpportunityIndex,
                OpportunitySymbol = prediction.OpportunitySymbol,
                PositionSize = prediction.PositionSize,
                SizeMultiplier = (double?)prediction.SizeMultiplier,
                PositionIndex = prediction.PositionIndex
            };

            _logger.LogInformation("Agent prediction: Action={Action}, Confidence={Confidence}, Symbol={Symbol}",
                result.Action, result.Confidence, result.OpportunitySymbol);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in GetModularActionAsync for user {UserId}", userId);
            return null;
        }
    }

    /// <summary>
    /// Build opportunity raw data from ArbitrageOpportunityDto
    /// </summary>
    private OpportunityRawData BuildOpportunityRawData(ArbitrageOpportunityDto opp)
    {
        return new OpportunityRawData
        {
            Symbol = opp.Symbol,
            LongExchange = opp.LongExchange,
            ShortExchange = opp.ShortExchange,
            FundProfit8h = opp.FundProfit8h,
            FundProfit8h24hProj = opp.FundProfit8h24hProj ?? 0m,
            FundProfit8h3dProj = opp.FundProfit8h3dProj ?? 0m,
            FundApr = opp.FundApr,
            FundApr24hProj = opp.FundApr24hProj ?? 0m,
            FundApr3dProj = opp.FundApr3dProj ?? 0m,
            Spread30SampleAvg = opp.Spread30SampleAvg ?? 0m,
            PriceSpread24hAvg = opp.PriceSpread24hAvg ?? 0m,
            PriceSpread3dAvg = opp.PriceSpread3dAvg ?? 0m,
            SpreadVolatilityStddev = opp.SpreadVolatilityStdDev ?? 0m,
            HasExistingPosition = opp.IsExistingPosition
        };
    }

    /// <summary>
    /// Build trading config raw data from AgentConfiguration
    /// </summary>
    public TradingConfigRawData BuildTradingConfigRawData(AgentConfiguration config)
    {
        return new TradingConfigRawData
        {
            MaxLeverage = config.MaxLeverage,
            TargetUtilization = config.TargetUtilization,
            MaxPositions = config.MaxPositions
            // StopLossThreshold and LiquidationBuffer will use default values from TradingConfigRawData
        };
    }

    /// <summary>
    /// Get current price for a symbol on an exchange from market data snapshot
    /// </summary>
    private async Task<decimal?> GetCurrentPriceAsync(
        string exchange,
        string symbol,
        PositionType positionType,
        CancellationToken cancellationToken = default)
    {
        try
        {
            const string key = "market_data_snapshot";
            var snapshot = await _marketDataRepository.GetAsync(key, cancellationToken);

            if (snapshot == null)
            {
                _logger.LogDebug("Market data snapshot not available");
                return null;
            }

            // Choose spot or perpetual prices based on position type
            var priceDict = positionType == PositionType.Spot
                ? snapshot.SpotPrices
                : snapshot.PerpPrices;

            if (priceDict.TryGetValue(exchange, out var exchangePrices) &&
                exchangePrices.TryGetValue(symbol, out var priceDto))
            {
                return priceDto.Price;
            }

            _logger.LogDebug("Price not found for {Exchange}:{Symbol} ({Type})",
                exchange, symbol, positionType);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving price for {Exchange}:{Symbol}", exchange, symbol);
            return null;
        }
    }

    /// <summary>
    /// Get current funding rate for a symbol on an exchange from market data snapshot
    /// </summary>
    private async Task<decimal?> GetCurrentFundingRateAsync(
        string exchange,
        string symbol,
        CancellationToken cancellationToken = default)
    {
        try
        {
            const string key = "market_data_snapshot";
            var snapshot = await _marketDataRepository.GetAsync(key, cancellationToken);

            if (snapshot == null)
            {
                _logger.LogDebug("Market data snapshot not available for funding rates");
                return null;
            }

            // Find funding rate for this exchange and symbol
            if (snapshot.FundingRates.TryGetValue(exchange, out var exchangeRates))
            {
                var fundingRate = exchangeRates.FirstOrDefault(r => r.Symbol == symbol);
                if (fundingRate != null)
                {
                    return fundingRate.Rate;
                }
            }

            _logger.LogDebug("Funding rate not found for {Exchange}:{Symbol}", exchange, symbol);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving funding rate for {Exchange}:{Symbol}", exchange, symbol);
            return null;
        }
    }

    /// <summary>
    /// Build position raw data from PositionDto list
    /// Pairs long/short positions by ExecutionId for arbitrage positions
    /// </summary>
    private async Task<List<PositionRawData>> BuildPositionRawDataFromDtos(
        List<PositionDto> positions,
        IEnumerable<ArbitrageOpportunityDto> currentOpportunities,
        CancellationToken cancellationToken = default)
    {
        var positionData = new List<PositionRawData>();

        // Group positions by ExecutionId to pair long/short sides
        var groupedByExecution = positions
            .Where(p => p.Status == PositionStatus.Open)
            .GroupBy(p => p.ExecutionId);

        foreach (var executionGroup in groupedByExecution)
        {
            var positionsList = executionGroup.ToList();

            // Find long and short positions
            var longPosition = positionsList.FirstOrDefault(p => p.Side == PositionSide.Long);
            var shortPosition = positionsList.FirstOrDefault(p => p.Side == PositionSide.Short);

            if (longPosition == null || shortPosition == null)
            {
                _logger.LogWarning("Execution {ExecutionId} missing long or short position, skipping", executionGroup.Key);
                continue;
            }

            // Calculate position age in hours
            var positionAgeHours = (decimal)(DateTime.UtcNow - longPosition.OpenedAt).TotalHours;

            // Find current APR from opportunities for this symbol
            var currentOpp = currentOpportunities.FirstOrDefault(o =>
                o.Symbol == longPosition.Symbol &&
                ((o.LongExchange == longPosition.Exchange && o.ShortExchange == shortPosition.Exchange) ||
                 (o.ShortExchange == longPosition.Exchange && o.LongExchange == shortPosition.Exchange)));

            var currentApr = currentOpp?.FundApr ?? (longPosition.LastKnownApr != 0 ? longPosition.LastKnownApr : longPosition.EntryApr);

            // Get funding interval hours from position data (stored when position was opened)
            var longFundingIntervalHours = (int)longPosition.LongFundingIntervalHours;
            var shortFundingIntervalHours = (int)shortPosition.ShortFundingIntervalHours;

            // Get current prices from market data snapshot
            var currentLongPrice = await GetCurrentPriceAsync(
                longPosition.Exchange,
                longPosition.Symbol,
                longPosition.Type,
                cancellationToken) ?? longPosition.EntryPrice;

            var currentShortPrice = await GetCurrentPriceAsync(
                shortPosition.Exchange,
                shortPosition.Symbol,
                shortPosition.Type,
                cancellationToken) ?? shortPosition.EntryPrice;

            // Get current funding rates from market data snapshot
            var longFundingRate = await GetCurrentFundingRateAsync(
                longPosition.Exchange,
                longPosition.Symbol,
                cancellationToken) ?? 0m;

            var shortFundingRate = await GetCurrentFundingRateAsync(
                shortPosition.Exchange,
                shortPosition.Symbol,
                cancellationToken) ?? 0m;

            // === CALCULATE P&L EXACTLY LIKE PYTHON PORTFOLIO ===
            // Matches: ml_pipeline/models/rl/core/portfolio.py lines 200-240

            // Get position size (same for both legs in arbitrage)
            decimal positionSizeUsd = longPosition.InitialMargin;

            // 1. Calculate PRICE P&L for each leg (in USD)
            decimal longPriceChangePct = longPosition.EntryPrice != 0
                ? ((currentLongPrice - longPosition.EntryPrice) / longPosition.EntryPrice)
                : 0m;
            decimal longPricePnlUsd = positionSizeUsd * longPriceChangePct;

            decimal shortPriceChangePct = shortPosition.EntryPrice != 0
                ? ((shortPosition.EntryPrice - currentShortPrice) / shortPosition.EntryPrice)
                : 0m;
            decimal shortPricePnlUsd = positionSizeUsd * shortPriceChangePct;

            // 2. Get NET FUNDING (from PositionTransaction - same source as frontend)
            decimal netFundingUsd = (longPosition.NetFundingFee) + (shortPosition.NetFundingFee);

            // 3. Get TOTAL TRADING FEES (from PositionTransaction)
            decimal totalFeesUsd = (longPosition.TradingFeePaid) + (shortPosition.TradingFeePaid);

            // 4. Calculate TOTAL P&L (net arbitrage P&L)
            decimal unrealizedPnlUsd = longPricePnlUsd + shortPricePnlUsd + netFundingUsd - totalFeesUsd;

            // 5. Calculate PERCENTAGES (relative to total capital = both legs)
            decimal totalCapitalUsd = positionSizeUsd * 2;

            var unrealizedPnlPct = totalCapitalUsd != 0
                ? SanitizeDecimal((unrealizedPnlUsd / totalCapitalUsd) * 100m)
                : 0m;

            // Individual leg P&L percentages (relative to each leg's position size)
            var longPnlPct = positionSizeUsd != 0
                ? SanitizeDecimal((longPricePnlUsd / positionSizeUsd) * 100m)
                : 0m;

            var shortPnlPct = positionSizeUsd != 0
                ? SanitizeDecimal((shortPricePnlUsd / positionSizeUsd) * 100m)
                : 0m;

            // === END P&L CALCULATION ===

            // Calculate liquidation distance (simplified formula)
            var longLiqDistance = CalculateLiquidationDistance(longPosition.Leverage);
            var shortLiqDistance = CalculateLiquidationDistance(shortPosition.Leverage);
            var minLiqDistance = Math.Min(longLiqDistance, shortLiqDistance);

            positionData.Add(new PositionRawData
            {
                IsActive = true,
                Symbol = longPosition.Symbol,
                PositionSizeUsd = longPosition.InitialMargin + shortPosition.InitialMargin,
                PositionAgeHours = positionAgeHours,
                Leverage = longPosition.Leverage,

                // Entry prices
                EntryLongPrice = longPosition.EntryPrice,
                EntryShortPrice = shortPosition.EntryPrice,

                // Current prices from market data
                CurrentLongPrice = currentLongPrice,
                CurrentShortPrice = currentShortPrice,

                // P&L
                UnrealizedPnlPct = unrealizedPnlPct,
                LongPnlPct = longPnlPct,
                ShortPnlPct = shortPnlPct,

                // Funding rates from market data
                LongFundingRate = longFundingRate,
                ShortFundingRate = shortFundingRate,
                LongFundingIntervalHours = longFundingIntervalHours,
                ShortFundingIntervalHours = shortFundingIntervalHours,

                // APR
                EntryApr = longPosition.EntryApr,
                CurrentPositionApr = currentApr,

                // Risk
                LiquidationDistance = minLiqDistance
            });
        }

        return positionData;
    }

    /// <summary>
    /// Calculate liquidation distance based on leverage (simplified formula)
    /// </summary>
    private decimal CalculateLiquidationDistance(decimal leverage)
    {
        if (leverage <= 0) return 1.0m;

        // Simplified: liquidation distance ≈ (1 - margin_ratio) where margin_ratio = 1/leverage
        // With 90% buffer factor for safety
        return 0.9m / leverage;
    }

    /// <summary>
    /// Sanitize decimal value to ensure it's JSON-compliant.
    /// Replaces infinity and NaN with safe fallback values.
    /// </summary>
    private decimal SanitizeDecimal(decimal value, decimal fallback = 0m)
    {
        // C# decimal type doesn't have infinity or NaN by design,
        // but we can check for very large values that might cause issues
        const decimal MAX_SAFE_VALUE = 1_000_000_000m; // 1 billion

        if (value > MAX_SAFE_VALUE)
            return MAX_SAFE_VALUE;
        if (value < -MAX_SAFE_VALUE)
            return -MAX_SAFE_VALUE;

        return value;
    }

}

/// <summary>
/// Agent prediction result - simplified wrapper for ML API response
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
