using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// HTTP client for calling the Python ML API server
/// Provides ML predictions by calling Flask API at localhost:5053
/// </summary>
public class PythonMLApiClient : IDisposable
{
    private readonly ILogger<PythonMLApiClient> _logger;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly JsonSerializerOptions _jsonOptions;

    public PythonMLApiClient(ILogger<PythonMLApiClient> logger, IConfiguration configuration)
    {
        _logger = logger;

        // Get configuration
        var host = configuration["MLApi:Host"] ?? "localhost";
        var port = configuration["MLApi:Port"] ?? "5053";
        _baseUrl = $"http://{host}:{port}";

        // Create HTTP client with timeout
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromSeconds(30)
        };

        // JSON serialization options (camelCase for Python)
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Check if the ML API server is healthy and responding
    /// </summary>
    public async Task<bool> HealthCheckAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/health");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                _logger.LogInformation("ML API health check successful: {Content}", content);
                return true;
            }

            _logger.LogWarning("ML API health check failed with status: {Status}", response.StatusCode);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ML API health check failed");
            return false;
        }
    }

    /// <summary>
    /// Score a single opportunity using Python ML models
    /// </summary>
    public async Task<MLPredictionResult> ScoreOpportunityAsync(ArbitrageOpportunityDto opportunity)
    {
        try
        {
            // Serialize opportunity to JSON
            var json = JsonSerializer.Serialize(opportunity, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // Call Python API
            var response = await _httpClient.PostAsync("/predict", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("ML API prediction failed: {Status} - {Error}", response.StatusCode, error);

                return new MLPredictionResult
                {
                    IsSuccess = false,
                    ErrorMessage = $"API error: {response.StatusCode} - {error}"
                };
            }

            // Deserialize response
            var responseJson = await response.Content.ReadAsStringAsync();
            var pythonResult = JsonSerializer.Deserialize<PythonPredictionResult>(responseJson, _jsonOptions);

            if (pythonResult == null)
            {
                throw new InvalidOperationException("Failed to deserialize Python prediction result");
            }

            return new MLPredictionResult
            {
                PredictedProfitPercent = pythonResult.PredictedProfitPercent,
                SuccessProbability = pythonResult.SuccessProbability,
                PredictedDurationHours = pythonResult.PredictedDurationHours,
                CompositeScore = pythonResult.CompositeScore,
                IsSuccess = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to score opportunity {Symbol} on {LongExchange}-{ShortExchange}",
                opportunity.Symbol, opportunity.LongExchange, opportunity.ShortExchange);

            return new MLPredictionResult
            {
                IsSuccess = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Score multiple opportunities in batch
    /// </summary>
    public async Task<List<MLPredictionResult>> ScoreOpportunitiesBatchAsync(IEnumerable<ArbitrageOpportunityDto> opportunities)
    {
        var opportunityList = opportunities.ToList();

        if (opportunityList.Count == 0)
            return new List<MLPredictionResult>();

        try
        {
            // Serialize opportunities to JSON array
            var json = JsonSerializer.Serialize(opportunityList, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // Call Python API
            var response = await _httpClient.PostAsync("/predict/batch", content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                _logger.LogError("ML API batch prediction failed: {Status} - {Error}", response.StatusCode, error);

                // Return error results for all
                return opportunityList.Select(_ => new MLPredictionResult
                {
                    IsSuccess = false,
                    ErrorMessage = $"API error: {response.StatusCode}"
                }).ToList();
            }

            // Deserialize response
            var responseJson = await response.Content.ReadAsStringAsync();
            var pythonResults = JsonSerializer.Deserialize<List<PythonPredictionResult>>(responseJson, _jsonOptions);

            if (pythonResults == null || pythonResults.Count != opportunityList.Count)
            {
                throw new InvalidOperationException("Invalid batch prediction result");
            }

            return pythonResults.Select(r => new MLPredictionResult
            {
                PredictedProfitPercent = r.PredictedProfitPercent,
                SuccessProbability = r.SuccessProbability,
                PredictedDurationHours = r.PredictedDurationHours,
                CompositeScore = r.CompositeScore,
                IsSuccess = true
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to score {Count} opportunities in batch", opportunityList.Count);

            // Return error results for all
            return opportunityList.Select(_ => new MLPredictionResult
            {
                IsSuccess = false,
                ErrorMessage = ex.Message
            }).ToList();
        }
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }

    /// <summary>
    /// Python prediction result structure (matches Flask API response)
    /// </summary>
    private class PythonPredictionResult
    {
        [JsonPropertyName("predicted_profit_percent")]
        public float PredictedProfitPercent { get; set; }

        [JsonPropertyName("success_probability")]
        public float SuccessProbability { get; set; }

        [JsonPropertyName("predicted_duration_hours")]
        public float PredictedDurationHours { get; set; }

        [JsonPropertyName("composite_score")]
        public float CompositeScore { get; set; }
    }
}
