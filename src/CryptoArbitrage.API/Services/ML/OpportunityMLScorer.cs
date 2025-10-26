using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// Scores arbitrage opportunities using trained XGBoost models via Python ML API
/// </summary>
public class OpportunityMLScorer
{
    private readonly ILogger<OpportunityMLScorer> _logger;
    private readonly PythonMLApiClient _mlApiClient;

    public OpportunityMLScorer(
        ILogger<OpportunityMLScorer> logger,
        PythonMLApiClient mlApiClient)
    {
        _logger = logger;
        _mlApiClient = mlApiClient;
    }

    /// <summary>
    /// Score a single opportunity with ML predictions
    /// </summary>
    public async Task<MLPredictionResult> ScoreOpportunityAsync(ArbitrageOpportunityDto opportunity)
    {
        return await _mlApiClient.ScoreOpportunityAsync(opportunity);
    }

    /// <summary>
    /// Score multiple opportunities in batch (more efficient)
    /// </summary>
    public async Task<List<MLPredictionResult>> ScoreOpportunitiesBatchAsync(IEnumerable<ArbitrageOpportunityDto> opportunities)
    {
        return await _mlApiClient.ScoreOpportunitiesBatchAsync(opportunities);
    }

    /// <summary>
    /// Score and enrich opportunities with ML predictions
    /// Updates the opportunities in-place with ML prediction fields
    /// </summary>
    public async Task ScoreAndEnrichOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        if (opportunities.Count == 0) return;

        var predictions = await ScoreOpportunitiesBatchAsync(opportunities);

        for (int i = 0; i < opportunities.Count; i++)
        {
            var prediction = predictions[i];
            if (prediction.IsSuccess)
            {
                opportunities[i].MLPredictedProfitPercent = (decimal)prediction.PredictedProfitPercent;
                opportunities[i].MLSuccessProbability = (decimal)prediction.SuccessProbability;
                opportunities[i].MLPredictedDurationHours = (decimal)prediction.PredictedDurationHours;
                opportunities[i].MLCompositeScore = (decimal)prediction.CompositeScore;
            }
        }
    }

    /// <summary>
    /// Rank opportunities by ML composite score
    /// </summary>
    public async Task<List<ArbitrageOpportunityDto>> RankOpportunitiesAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        int topN = 10)
    {
        var opportunityList = opportunities.ToList();
        await ScoreAndEnrichOpportunitiesAsync(opportunityList);

        return opportunityList
            .Where(o => o.MLCompositeScore.HasValue)
            .OrderByDescending(o => o.MLCompositeScore!.Value)
            .Take(topN)
            .ToList();
    }

    /// <summary>
    /// Filter opportunities by minimum success probability
    /// </summary>
    public async Task<List<ArbitrageOpportunityDto>> FilterBySuccessProbabilityAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities,
        float minSuccessProbability = 0.5f)
    {
        var opportunityList = opportunities.ToList();
        await ScoreAndEnrichOpportunitiesAsync(opportunityList);

        return opportunityList
            .Where(o => o.MLSuccessProbability.HasValue && o.MLSuccessProbability.Value >= (decimal)minSuccessProbability)
            .OrderByDescending(o => o.MLCompositeScore ?? 0m)
            .ToList();
    }
}

/// <summary>
/// ML prediction result for a single opportunity
/// </summary>
public class MLPredictionResult
{
    public float PredictedProfitPercent { get; set; }
    public float SuccessProbability { get; set; }
    public float PredictedDurationHours { get; set; }
    public float CompositeScore { get; set; }
    public bool IsSuccess { get; set; }
    public string? ErrorMessage { get; set; }
}
