using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
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
    /// Evaluate EXIT probabilities for open positions
    /// </summary>
    /// <param name="positions">List of positions to evaluate (max 3)</param>
    /// <param name="portfolio">Current portfolio state</param>
    /// <param name="opportunities">Optional: current opportunities for context</param>
    /// <returns>Dictionary mapping position ID to RL prediction</returns>
    public async Task<Dictionary<int, RLPredictionDto>> EvaluatePositionsAsync(
        IEnumerable<PositionDto> positions,
        RLPortfolioState portfolio,
        IEnumerable<ArbitrageOpportunityDto>? opportunities = null)
    {
        var posList = positions.Take(3).ToList();  // Limit to 3 (model constraint)
        var result = new Dictionary<int, RLPredictionDto>();

        if (posList.Count == 0)
            return result;

        try
        {
            // Build request payload
            var payload = new
            {
                positions = posList.Select(ConvertPositionToFeatures).ToList(),
                portfolio = portfolio,
                opportunities = opportunities?.Take(5).Select(ConvertOpportunityToFeatures).ToList() ?? new List<object>()
            };

            // Serialize and send request
            var json = JsonSerializer.Serialize(payload, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            _logger.LogDebug("Calling RL API /rl/predict/positions with {Count} positions", posList.Count);
            var response = await _httpClient.PostAsync("/rl/predict/positions", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("RL API position prediction failed: {Status} - {Error}", response.StatusCode, error);
                return result;
            }

            // Deserialize response
            var responseJson = await response.Content.ReadAsStringAsync();
            var apiResponse = JsonSerializer.Deserialize<RLPositionsResponse>(responseJson, _deserializeOptions);

            if (apiResponse?.Predictions == null)
            {
                _logger.LogWarning("RL API returned null position predictions");
                return result;
            }

            // Map predictions back to positions
            foreach (var prediction in apiResponse.Predictions)
            {
                if (prediction.PositionIndex >= 0 && prediction.PositionIndex < posList.Count)
                {
                    var pos = posList[prediction.PositionIndex];
                    result[pos.Id] = new RLPredictionDto
                    {
                        ActionProbability = prediction.ExitProbability,
                        HoldProbability = prediction.HoldProbability,
                        Confidence = prediction.Confidence,
                        StateValue = prediction.StateValue,
                        ModelVersion = apiResponse.ModelVersion
                    };
                }
            }

            _logger.LogInformation("RL predictions received for {Count} positions", result.Count);
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
    /// Convert position DTO to feature dictionary for Python API
    /// </summary>
    private object ConvertPositionToFeatures(PositionDto pos)
    {
        var hoursHeld = pos.ClosedAt.HasValue
            ? (float)(pos.ClosedAt.Value - pos.OpenedAt).TotalHours
            : (float)(DateTime.UtcNow - pos.OpenedAt).TotalHours;

        var pnlPct = pos.InitialMargin > 0
            ? (float)((pos.UnrealizedPnL / pos.InitialMargin) * 100)
            : 0.0f;

        // Estimate current funding rate from net funding fees
        var fundingRate = hoursHeld > 0
            ? (float)(pos.NetFundingFee / pos.InitialMargin / (decimal)hoursHeld * 8 * 100)  // Normalize to 8h rate
            : 0.0f;

        return new
        {
            symbol = pos.Symbol,
            pnl_pct = pnlPct,
            hours_held = hoursHeld,
            funding_rate = fundingRate
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
