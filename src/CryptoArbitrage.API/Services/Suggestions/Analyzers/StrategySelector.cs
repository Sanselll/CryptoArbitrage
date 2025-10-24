using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Selects the optimal trading strategy for each opportunity
/// </summary>
public class StrategySelector
{
    private readonly ILogger<StrategySelector> _logger;
    private readonly FundingStrategyAnalyzer _fundingAnalyzer;
    private readonly SpreadStrategyAnalyzer _spreadAnalyzer;
    private readonly HybridStrategyAnalyzer _hybridAnalyzer;

    public StrategySelector(
        ILogger<StrategySelector> logger,
        FundingStrategyAnalyzer fundingAnalyzer,
        SpreadStrategyAnalyzer spreadAnalyzer,
        HybridStrategyAnalyzer hybridAnalyzer)
    {
        _logger = logger;
        _fundingAnalyzer = fundingAnalyzer;
        _spreadAnalyzer = spreadAnalyzer;
        _hybridAnalyzer = hybridAnalyzer;
    }

    /// <summary>
    /// Evaluates all strategies and selects the optimal one
    /// Returns the recommended strategy type and its score
    /// </summary>
    public (RecommendedStrategyType Strategy, decimal Score, List<string> Factors) SelectOptimalStrategy(ArbitrageOpportunityDto opportunity)
    {
        var fundingFactors = new List<string>();
        var spreadFactors = new List<string>();
        var hybridFactors = new List<string>();

        // Evaluate all three strategies
        decimal fundingScore = _fundingAnalyzer.AnalyzeFundingQuality(opportunity, fundingFactors);
        decimal spreadScore = _spreadAnalyzer.AnalyzeSpreadQuality(opportunity, spreadFactors);
        decimal hybridScore = _hybridAnalyzer.AnalyzeHybridPotential(opportunity, hybridFactors);

        _logger.LogDebug(
            "Strategy Selection for {Symbol}: Funding={Funding}, Spread={Spread}, Hybrid={Hybrid}",
            opportunity.Symbol, fundingScore, spreadScore, hybridScore);

        // Select the strategy with the highest score
        var strategies = new[]
        {
            (Type: RecommendedStrategyType.FundingOnly, Score: fundingScore, Factors: fundingFactors),
            (Type: RecommendedStrategyType.SpreadOnly, Score: spreadScore, Factors: spreadFactors),
            (Type: RecommendedStrategyType.Hybrid, Score: hybridScore, Factors: hybridFactors)
        };

        var bestStrategy = strategies.OrderByDescending(s => s.Score).First();

        // Add strategy selection reasoning
        var selectionFactors = new List<string>
        {
            $"Selected {bestStrategy.Type} strategy with score {bestStrategy.Score:F1}"
        };

        // Add comparison context if scores are close
        var secondBest = strategies.OrderByDescending(s => s.Score).Skip(1).First();
        decimal scoreDifference = bestStrategy.Score - secondBest.Score;

        if (scoreDifference < 5m)
        {
            selectionFactors.Add($"⚠️ Close competition with {secondBest.Type} (score {secondBest.Score:F1}) - consider both");
        }
        else if (scoreDifference > 20m)
        {
            selectionFactors.Add($"✓ Clear winner - {scoreDifference:F1} points ahead of next best strategy");
        }

        // Combine selection reasoning with strategy-specific factors
        selectionFactors.AddRange(bestStrategy.Factors);

        _logger.LogInformation(
            "Selected {Strategy} strategy for {Symbol} with score {Score}",
            bestStrategy.Type, opportunity.Symbol, bestStrategy.Score);

        return (bestStrategy.Type, bestStrategy.Score, selectionFactors);
    }

    /// <summary>
    /// Gets the score for a specific strategy type
    /// </summary>
    public decimal GetStrategyScore(ArbitrageOpportunityDto opportunity, RecommendedStrategyType strategyType)
    {
        var factors = new List<string>();

        return strategyType switch
        {
            RecommendedStrategyType.FundingOnly => _fundingAnalyzer.AnalyzeFundingQuality(opportunity, factors),
            RecommendedStrategyType.SpreadOnly => _spreadAnalyzer.AnalyzeSpreadQuality(opportunity, factors),
            RecommendedStrategyType.Hybrid => _hybridAnalyzer.AnalyzeHybridPotential(opportunity, factors),
            _ => 0m
        };
    }
}
