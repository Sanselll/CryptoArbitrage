using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes hybrid strategy combining both funding and spread opportunities
/// </summary>
public class HybridStrategyAnalyzer
{
    private readonly ILogger<HybridStrategyAnalyzer> _logger;
    private readonly FundingStrategyAnalyzer _fundingAnalyzer;
    private readonly SpreadStrategyAnalyzer _spreadAnalyzer;

    public HybridStrategyAnalyzer(
        ILogger<HybridStrategyAnalyzer> logger,
        FundingStrategyAnalyzer fundingAnalyzer,
        SpreadStrategyAnalyzer spreadAnalyzer)
    {
        _logger = logger;
        _fundingAnalyzer = fundingAnalyzer;
        _spreadAnalyzer = spreadAnalyzer;
    }

    /// <summary>
    /// Analyzes hybrid strategy potential and returns a score (0-100)
    /// </summary>
    public decimal AnalyzeHybridPotential(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Hybrid only makes sense for cross-exchange strategies with both funding and spread components
        if (opportunity.Strategy != ArbitrageStrategy.CrossExchange)
        {
            return 0m;
        }

        // Get individual strategy scores
        var fundingFactors = new List<string>();
        var spreadFactors = new List<string>();

        decimal fundingScore = _fundingAnalyzer.AnalyzeFundingQuality(opportunity, fundingFactors);
        decimal spreadScore = _spreadAnalyzer.AnalyzeSpreadQuality(opportunity, spreadFactors);

        // Analyze synergy between funding and spread
        decimal synergyScore = AnalyzeSynergy(opportunity, fundingScore, spreadScore, scoringFactors);

        // Determine optimal ratio between funding and spread focus
        decimal fundingWeight = DetermineFundingWeight(opportunity, fundingScore, spreadScore);
        decimal spreadWeight = 1m - fundingWeight;

        // Calculate weighted hybrid score with synergy bonus
        decimal baseScore = (fundingScore * fundingWeight) + (spreadScore * spreadWeight);
        decimal hybridScore = baseScore + (synergyScore * 0.1m); // Synergy adds up to 10 points

        scoringFactors.Add($"Hybrid strategy: {fundingWeight:P0} funding + {spreadWeight:P0} spread");
        scoringFactors.Add($"Funding component score: {fundingScore:F1}");
        scoringFactors.Add($"Spread component score: {spreadScore:F1}");
        scoringFactors.Add($"Synergy bonus: +{synergyScore * 0.1m:F1}");

        // Add key factors from individual analyses
        AddTopFactors(scoringFactors, fundingFactors, "Funding");
        AddTopFactors(scoringFactors, spreadFactors, "Spread");

        _logger.LogDebug(
            "Hybrid Analysis for {Symbol}: Funding={Funding}, Spread={Spread}, Synergy={Synergy}, Hybrid={Hybrid}",
            opportunity.Symbol, fundingScore, spreadScore, synergyScore, hybridScore);

        return Math.Round(Math.Min(100m, hybridScore), 2);
    }

    private decimal AnalyzeSynergy(ArbitrageOpportunityDto opportunity, decimal fundingScore, decimal spreadScore, List<string> scoringFactors)
    {
        decimal score = 0m;

        // Check if both strategies are viable (both > 50)
        if (fundingScore > 50m && spreadScore > 50m)
        {
            scoringFactors.Add("✓ Both funding and spread strategies viable - strong synergy");
            score += 50m;

            // Additional bonus if both are strong (both > 70)
            if (fundingScore > 70m && spreadScore > 70m)
            {
                scoringFactors.Add("✓ Both strategies strong - excellent hybrid opportunity");
                score += 30m;
            }
        }
        else if (fundingScore < 40m && spreadScore < 40m)
        {
            scoringFactors.Add("⚠️ Both strategies weak - poor hybrid opportunity");
            return 0m;
        }
        else
        {
            scoringFactors.Add("Moderate hybrid potential - one strategy stronger than other");
            score += 25m;
        }

        // Check for complementary risk profiles
        // Funding is ongoing, spread is one-time capture
        // Good synergy if spread can be captured while setting up funding position
        bool hasImmediateSpread = (opportunity.EstimatedProfitPercentage > 0.3m);
        bool hasPositiveFunding = (opportunity.FundApr > 15m);

        if (hasImmediateSpread && hasPositiveFunding)
        {
            scoringFactors.Add("✓ Immediate spread capture + ongoing funding income");
            score += 20m;
        }

        return Math.Min(100m, score);
    }

    private decimal DetermineFundingWeight(ArbitrageOpportunityDto opportunity, decimal fundingScore, decimal spreadScore)
    {
        // If one strategy significantly outperforms the other, weight it more heavily
        decimal scoreDifference = Math.Abs(fundingScore - spreadScore);

        if (scoreDifference < 10m)
        {
            // Scores are close - balanced hybrid (50/50)
            return 0.5m;
        }
        else if (scoreDifference < 25m)
        {
            // Moderate difference - weight toward better strategy (60/40 or 40/60)
            return fundingScore > spreadScore ? 0.6m : 0.4m;
        }
        else
        {
            // Large difference - heavily weight better strategy (75/25 or 25/75)
            return fundingScore > spreadScore ? 0.75m : 0.25m;
        }
    }

    private void AddTopFactors(List<string> targetList, List<string> sourceList, string prefix)
    {
        // Add top 2 most important factors from each analysis
        int factorsToAdd = Math.Min(2, sourceList.Count);
        for (int i = 0; i < factorsToAdd; i++)
        {
            // Prioritize warning factors (containing ⚠️) and positive factors (containing ✓)
            var factor = sourceList[i];
            if (factor.Contains("⚠️") || factor.Contains("✓") || factor.Contains("Excellent") || factor.Contains("Good"))
            {
                targetList.Add($"[{prefix}] {factor}");
            }
        }
    }
}
