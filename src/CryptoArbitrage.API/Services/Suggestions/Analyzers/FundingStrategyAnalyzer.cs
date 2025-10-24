using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes funding rate strategy quality and consistency
/// </summary>
public class FundingStrategyAnalyzer
{
    private readonly ILogger<FundingStrategyAnalyzer> _logger;

    // Thresholds for funding rate analysis
    private const decimal EXCELLENT_APR_THRESHOLD = 50m; // 50% APR
    private const decimal GOOD_APR_THRESHOLD = 30m; // 30% APR
    private const decimal MINIMUM_APR_THRESHOLD = 15m; // 15% APR

    private const decimal EXCELLENT_BREAK_EVEN_HOURS = 24m; // 1 day or less
    private const decimal GOOD_BREAK_EVEN_HOURS = 48m; // 2 days or less
    private const decimal ACCEPTABLE_BREAK_EVEN_HOURS = 96m; // 4 days or less

    public FundingStrategyAnalyzer(ILogger<FundingStrategyAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Analyzes funding rate quality and returns a score (0-100)
    /// </summary>
    public decimal AnalyzeFundingQuality(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal consistencyScore = AnalyzeFundingConsistency(opportunity, scoringFactors);
        decimal profitabilityScore = AnalyzeFundingProfitability(opportunity, scoringFactors);
        decimal breakEvenScore = AnalyzeBreakEvenTime(opportunity, scoringFactors);
        decimal reversalRiskScore = AnalyzeReversalRisk(opportunity, scoringFactors);

        // Weighted average: Consistency 30%, Profitability 30%, BreakEven 25%, Reversal Risk 15%
        decimal overallScore = (consistencyScore * 0.30m) + (profitabilityScore * 0.30m) +
                              (breakEvenScore * 0.25m) + (reversalRiskScore * 0.15m);

        _logger.LogDebug(
            "Funding Analysis for {Symbol}: Consistency={Consistency}, Profit={Profit}, BreakEven={BreakEven}, Reversal={Reversal}, Overall={Overall}",
            opportunity.Symbol, consistencyScore, profitabilityScore, breakEvenScore, reversalRiskScore, overallScore);

        return Math.Round(overallScore, 2);
    }

    /// <summary>
    /// Gets detailed sub-scores for funding strategy (quality and profit potential separately)
    /// </summary>
    public (decimal FundingQuality, decimal ProfitPotential) GetDetailedScores(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal consistencyScore = AnalyzeFundingConsistency(opportunity, scoringFactors);
        decimal profitabilityScore = AnalyzeFundingProfitability(opportunity, scoringFactors);
        decimal breakEvenScore = AnalyzeBreakEvenTime(opportunity, scoringFactors);
        decimal reversalRiskScore = AnalyzeReversalRisk(opportunity, scoringFactors);

        // Funding quality = consistency + reversal risk
        decimal fundingQuality = (consistencyScore * 0.7m) + (reversalRiskScore * 0.3m);

        // Profit potential = profitability + break-even time
        decimal profitPotential = (profitabilityScore * 0.6m) + (breakEvenScore * 0.4m);

        return (Math.Round(fundingQuality, 2), Math.Round(profitPotential, 2));
    }

    private decimal AnalyzeFundingConsistency(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Check if we have historical data
        bool has24hProj = opportunity.FundApr24hProj.HasValue;
        bool has3dProj = opportunity.FundApr3dProj.HasValue;

        if (!has24hProj && !has3dProj)
        {
            scoringFactors.Add("⚠️ No historical funding data available");
            return 50m; // Neutral score without historical data
        }

        decimal currentApr = opportunity.FundApr;
        decimal score = 100m;

        // Compare current vs 24h average
        if (has24hProj)
        {
            decimal apr24h = opportunity.FundApr24hProj!.Value;
            decimal deviation24h = Math.Abs(currentApr - apr24h) / Math.Max(Math.Abs(apr24h), 0.01m);

            if (deviation24h <= 0.15m) // Within 15%
            {
                scoringFactors.Add($"Strong 24h consistency: Current {currentApr:F1}% vs 24h avg {apr24h:F1}%");
            }
            else if (deviation24h <= 0.30m) // Within 30%
            {
                scoringFactors.Add($"Moderate 24h consistency: Current {currentApr:F1}% vs 24h avg {apr24h:F1}%");
                score -= 15m;
            }
            else
            {
                scoringFactors.Add($"⚠️ Poor 24h consistency: Current {currentApr:F1}% vs 24h avg {apr24h:F1}%");
                score -= 30m;
            }
        }

        // Compare current vs 3D average
        if (has3dProj)
        {
            decimal apr3d = opportunity.FundApr3dProj!.Value;
            decimal deviation3d = Math.Abs(currentApr - apr3d) / Math.Max(Math.Abs(apr3d), 0.01m);

            if (deviation3d <= 0.20m) // Within 20%
            {
                scoringFactors.Add($"Strong 3D consistency: Current {currentApr:F1}% vs 3D avg {apr3d:F1}%");
            }
            else if (deviation3d <= 0.40m) // Within 40%
            {
                scoringFactors.Add($"Moderate 3D consistency: Current {currentApr:F1}% vs 3D avg {apr3d:F1}%");
                score -= 15m;
            }
            else
            {
                scoringFactors.Add($"⚠️ Poor 3D consistency: Current {currentApr:F1}% vs 3D avg {apr3d:F1}%");
                score -= 30m;
            }
        }

        // Check if current is better or worse than historical
        if (has24hProj && has3dProj)
        {
            decimal apr24h = opportunity.FundApr24hProj!.Value;
            decimal apr3d = opportunity.FundApr3dProj!.Value;

            if (currentApr > apr24h && currentApr > apr3d)
            {
                scoringFactors.Add("✓ Current rate improving vs historical averages");
                score += 10m;
            }
            else if (currentApr < apr24h && currentApr < apr3d)
            {
                scoringFactors.Add("⚠️ Current rate declining vs historical averages");
                score -= 10m;
            }
        }

        return Math.Max(0m, Math.Min(100m, score));
    }

    private decimal AnalyzeFundingProfitability(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal currentApr = opportunity.FundApr;

        if (currentApr >= EXCELLENT_APR_THRESHOLD)
        {
            scoringFactors.Add($"Excellent funding APR: {currentApr:F1}%");
            return 100m;
        }
        else if (currentApr >= GOOD_APR_THRESHOLD)
        {
            scoringFactors.Add($"Good funding APR: {currentApr:F1}%");
            // Linear scale from 75-100
            return 75m + (25m * ((currentApr - GOOD_APR_THRESHOLD) / (EXCELLENT_APR_THRESHOLD - GOOD_APR_THRESHOLD)));
        }
        else if (currentApr >= MINIMUM_APR_THRESHOLD)
        {
            scoringFactors.Add($"Moderate funding APR: {currentApr:F1}%");
            // Linear scale from 50-75
            return 50m + (25m * ((currentApr - MINIMUM_APR_THRESHOLD) / (GOOD_APR_THRESHOLD - MINIMUM_APR_THRESHOLD)));
        }
        else if (currentApr > 0)
        {
            scoringFactors.Add($"⚠️ Low funding APR: {currentApr:F1}%");
            // Linear scale from 0-50
            return 50m * (currentApr / MINIMUM_APR_THRESHOLD);
        }
        else
        {
            scoringFactors.Add($"⚠️ Negative funding APR: {currentApr:F1}% - avoid");
            return 0m;
        }
    }

    private decimal AnalyzeBreakEvenTime(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        if (!opportunity.BreakEvenTimeHours.HasValue)
        {
            return 50m; // Neutral if no break-even data
        }

        decimal breakEvenHours = opportunity.BreakEvenTimeHours.Value;

        if (breakEvenHours <= EXCELLENT_BREAK_EVEN_HOURS)
        {
            scoringFactors.Add($"Excellent break-even time: {breakEvenHours:F1} hours");
            return 100m;
        }
        else if (breakEvenHours <= GOOD_BREAK_EVEN_HOURS)
        {
            scoringFactors.Add($"Good break-even time: {breakEvenHours:F1} hours");
            // Linear scale from 80-100
            return 80m + (20m * (1 - ((breakEvenHours - EXCELLENT_BREAK_EVEN_HOURS) / (GOOD_BREAK_EVEN_HOURS - EXCELLENT_BREAK_EVEN_HOURS))));
        }
        else if (breakEvenHours <= ACCEPTABLE_BREAK_EVEN_HOURS)
        {
            scoringFactors.Add($"Acceptable break-even time: {breakEvenHours:F1} hours");
            // Linear scale from 50-80
            return 50m + (30m * (1 - ((breakEvenHours - GOOD_BREAK_EVEN_HOURS) / (ACCEPTABLE_BREAK_EVEN_HOURS - GOOD_BREAK_EVEN_HOURS))));
        }
        else
        {
            scoringFactors.Add($"⚠️ Long break-even time: {breakEvenHours:F1} hours");
            // Declining score for very long break-even times
            return Math.Max(0m, 50m * (1 - Math.Min(1, (breakEvenHours - ACCEPTABLE_BREAK_EVEN_HOURS) / 200m)));
        }
    }

    private decimal AnalyzeReversalRisk(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // If we have both current and historical data, assess reversal risk
        bool has24hProj = opportunity.FundApr24hProj.HasValue;
        bool has3dProj = opportunity.FundApr3dProj.HasValue;

        if (!has24hProj && !has3dProj)
        {
            return 50m; // Neutral without historical data
        }

        decimal currentApr = opportunity.FundApr;
        decimal score = 100m;

        // Check for sign flip risk (positive → negative or vice versa)
        if (has24hProj)
        {
            decimal apr24h = opportunity.FundApr24hProj!.Value;
            if (Math.Sign(currentApr) != Math.Sign(apr24h) && Math.Abs(apr24h) > 5m)
            {
                scoringFactors.Add("⚠️ Funding rate sign recently flipped - high reversal risk");
                score -= 40m;
            }
        }

        if (has3dProj)
        {
            decimal apr3d = opportunity.FundApr3dProj!.Value;
            if (Math.Sign(currentApr) != Math.Sign(apr3d) && Math.Abs(apr3d) > 5m)
            {
                scoringFactors.Add("⚠️ Funding rate direction unstable over 3 days");
                score -= 30m;
            }
        }

        // Check for volatility (large swings between current and historical)
        if (has24hProj && has3dProj)
        {
            decimal apr24h = opportunity.FundApr24hProj!.Value;
            decimal apr3d = opportunity.FundApr3dProj!.Value;
            decimal volatility = Math.Abs(apr24h - apr3d);

            if (volatility > 20m)
            {
                scoringFactors.Add($"⚠️ High funding rate volatility: {volatility:F1}% swing");
                score -= 20m;
            }
            else if (volatility < 5m)
            {
                scoringFactors.Add($"✓ Low funding rate volatility: {volatility:F1}% swing");
                // Stability bonus
            }
        }

        return Math.Max(0m, score);
    }
}
