using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes price spread strategy quality for cross-exchange arbitrage
/// </summary>
public class SpreadStrategyAnalyzer
{
    private readonly ILogger<SpreadStrategyAnalyzer> _logger;

    // Thresholds for spread analysis
    private const decimal EXCELLENT_SPREAD_THRESHOLD = 1.0m; // 1% spread
    private const decimal GOOD_SPREAD_THRESHOLD = 0.5m; // 0.5% spread
    private const decimal MINIMUM_SPREAD_THRESHOLD = 0.3m; // 0.3% spread (minimum profitability)

    private const decimal POSITION_COST = 0.2m; // 0.2% position cost (4 trades × 0.05%)

    public SpreadStrategyAnalyzer(ILogger<SpreadStrategyAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Analyzes price spread quality and returns a score (0-100)
    /// </summary>
    public decimal AnalyzeSpreadQuality(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Only applicable for cross-exchange strategies
        if (opportunity.Strategy != ArbitrageStrategy.CrossExchange)
        {
            return 0m;
        }

        decimal spreadScore = AnalyzeCurrentSpread(opportunity, scoringFactors);
        decimal consistencyScore = AnalyzeSpreadConsistency(opportunity, scoringFactors);
        decimal profitMarginScore = AnalyzeProfitMargin(opportunity, scoringFactors);
        decimal meanReversionScore = AnalyzeMeanReversion(opportunity, scoringFactors);

        // Weighted average: Current Spread 35%, Consistency 25%, Profit Margin 25%, Mean Reversion 15%
        decimal overallScore = (spreadScore * 0.35m) + (consistencyScore * 0.25m) +
                              (profitMarginScore * 0.25m) + (meanReversionScore * 0.15m);

        _logger.LogDebug(
            "Spread Analysis for {Symbol}: Spread={Spread}, Consistency={Consistency}, Margin={Margin}, MeanRev={MeanRev}, Overall={Overall}",
            opportunity.Symbol, spreadScore, consistencyScore, profitMarginScore, meanReversionScore, overallScore);

        return Math.Round(overallScore, 2);
    }

    /// <summary>
    /// Gets the spread efficiency score (current spread + consistency)
    /// </summary>
    public decimal GetSpreadEfficiencyScore(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Only applicable for cross-exchange strategies
        if (opportunity.Strategy != ArbitrageStrategy.CrossExchange)
        {
            return 0m;
        }

        decimal spreadScore = AnalyzeCurrentSpread(opportunity, scoringFactors);
        decimal consistencyScore = AnalyzeSpreadConsistency(opportunity, scoringFactors);

        // Spread efficiency = current spread quality + consistency
        decimal efficiency = (spreadScore * 0.6m) + (consistencyScore * 0.4m);

        return Math.Round(efficiency, 2);
    }

    private decimal AnalyzeCurrentSpread(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal currentSpread = Math.Abs(opportunity.SpreadRate) * 100m; // Convert to percentage

        if (currentSpread >= EXCELLENT_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Excellent price spread: {currentSpread:F2}%");
            return 100m;
        }
        else if (currentSpread >= GOOD_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Good price spread: {currentSpread:F2}%");
            // Linear scale from 75-100
            return 75m + (25m * ((currentSpread - GOOD_SPREAD_THRESHOLD) / (EXCELLENT_SPREAD_THRESHOLD - GOOD_SPREAD_THRESHOLD)));
        }
        else if (currentSpread >= MINIMUM_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Moderate price spread: {currentSpread:F2}%");
            // Linear scale from 50-75
            return 50m + (25m * ((currentSpread - MINIMUM_SPREAD_THRESHOLD) / (GOOD_SPREAD_THRESHOLD - MINIMUM_SPREAD_THRESHOLD)));
        }
        else if (currentSpread > 0)
        {
            scoringFactors.Add($"⚠️ Low price spread: {currentSpread:F2}% - marginal profitability");
            // Linear scale from 0-50
            return 50m * (currentSpread / MINIMUM_SPREAD_THRESHOLD);
        }
        else
        {
            scoringFactors.Add("⚠️ No profitable price spread");
            return 0m;
        }
    }

    private decimal AnalyzeSpreadConsistency(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        bool has24hAvg = opportunity.PriceSpread24hAvg.HasValue;
        bool has3dAvg = opportunity.PriceSpread3dAvg.HasValue;

        if (!has24hAvg && !has3dAvg)
        {
            scoringFactors.Add("⚠️ No historical spread data available");
            return 50m; // Neutral score without historical data
        }

        decimal currentSpread = Math.Abs(opportunity.SpreadRate);
        decimal score = 100m;

        // Compare current vs 24h average
        if (has24hAvg)
        {
            decimal spread24h = Math.Abs(opportunity.PriceSpread24hAvg!.Value);
            decimal deviation24h = Math.Abs(currentSpread - spread24h) / Math.Max(spread24h, 0.001m);

            if (deviation24h <= 0.20m) // Within 20%
            {
                scoringFactors.Add($"Strong 24h spread consistency: Current {currentSpread:P2} vs 24h avg {spread24h:P2}");
            }
            else if (deviation24h <= 0.40m) // Within 40%
            {
                scoringFactors.Add($"Moderate 24h spread consistency: Current {currentSpread:P2} vs 24h avg {spread24h:P2}");
                score -= 15m;
            }
            else
            {
                scoringFactors.Add($"⚠️ Poor 24h spread consistency: Current {currentSpread:P2} vs 24h avg {spread24h:P2}");
                score -= 30m;
            }
        }

        // Compare current vs 3D average
        if (has3dAvg)
        {
            decimal spread3d = Math.Abs(opportunity.PriceSpread3dAvg!.Value);
            decimal deviation3d = Math.Abs(currentSpread - spread3d) / Math.Max(spread3d, 0.001m);

            if (deviation3d <= 0.25m) // Within 25%
            {
                scoringFactors.Add($"Strong 3D spread consistency: Current {currentSpread:P2} vs 3D avg {spread3d:P2}");
            }
            else if (deviation3d <= 0.50m) // Within 50%
            {
                scoringFactors.Add($"Moderate 3D spread consistency: Current {currentSpread:P2} vs 3D avg {spread3d:P2}");
                score -= 15m;
            }
            else
            {
                scoringFactors.Add($"⚠️ Poor 3D spread consistency: Current {currentSpread:P2} vs 3D avg {spread3d:P2}");
                score -= 30m;
            }
        }

        return Math.Max(0m, Math.Min(100m, score));
    }

    private decimal AnalyzeProfitMargin(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal estimatedProfit = opportunity.EstimatedProfitPercentage;

        // Calculate time efficiency (daily profit rate)
        // Spread strategies typically have ~4-24h holding period
        decimal holdingHours = 4m; // Quick capture assumption for spread
        decimal dailyProfitRate = estimatedProfit * (24m / holdingHours);
        decimal annualizedAPR = dailyProfitRate * 365m;

        // Score based on absolute profit AND time efficiency
        decimal absoluteProfitScore;
        if (estimatedProfit >= 0.8m) // 0.8% net profit
        {
            scoringFactors.Add($"Excellent profit margin: {estimatedProfit:F2}% (Daily: {dailyProfitRate:F2}%, APR: {annualizedAPR:F1}%)");
            absoluteProfitScore = 100m;
        }
        else if (estimatedProfit >= 0.5m) // 0.5% net profit
        {
            scoringFactors.Add($"Good profit margin: {estimatedProfit:F2}% (Daily: {dailyProfitRate:F2}%, APR: {annualizedAPR:F1}%)");
            // Linear scale from 75-100
            absoluteProfitScore = 75m + (25m * ((estimatedProfit - 0.5m) / 0.3m));
        }
        else if (estimatedProfit >= 0.3m) // 0.3% net profit (breakeven at MINIMUM_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Moderate profit margin: {estimatedProfit:F2}% (Daily: {dailyProfitRate:F2}%, APR: {annualizedAPR:F1}%)");
            // Linear scale from 50-75
            absoluteProfitScore = 50m + (25m * ((estimatedProfit - 0.3m) / 0.2m));
        }
        else if (estimatedProfit > 0)
        {
            scoringFactors.Add($"⚠️ Low profit margin: {estimatedProfit:F2}% (Daily: {dailyProfitRate:F2}%, APR: {annualizedAPR:F1}%)");
            // Linear scale from 0-50
            absoluteProfitScore = 50m * (estimatedProfit / 0.3m);
        }
        else
        {
            scoringFactors.Add($"⚠️ Negative profit margin: {estimatedProfit:F2}% - avoid");
            absoluteProfitScore = 0m;
        }

        return absoluteProfitScore;
    }

    private decimal AnalyzeMeanReversion(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Analyze if the current spread is likely to persist or revert
        bool has24hAvg = opportunity.PriceSpread24hAvg.HasValue;
        bool has3dAvg = opportunity.PriceSpread3dAvg.HasValue;

        if (!has24hAvg && !has3dAvg)
        {
            return 50m; // Neutral without historical data
        }

        decimal currentSpread = Math.Abs(opportunity.SpreadRate);
        decimal score = 100m;

        // Check if current spread is wider than historical (good for immediate capture)
        if (has24hAvg && has3dAvg)
        {
            decimal spread24h = Math.Abs(opportunity.PriceSpread24hAvg!.Value);
            decimal spread3d = Math.Abs(opportunity.PriceSpread3dAvg!.Value);
            decimal avgHistorical = (spread24h + spread3d) / 2;

            if (currentSpread > avgHistorical * 1.3m)
            {
                // Current spread significantly wider than historical - opportunity might close quickly
                scoringFactors.Add($"✓ Current spread {currentSpread:P2} wider than avg {avgHistorical:P2} - good capture opportunity");
                score += 20m; // Bonus for wider-than-average spread
            }
            else if (currentSpread < avgHistorical * 0.7m)
            {
                // Current spread narrower than historical - might not be sustainable
                scoringFactors.Add($"⚠️ Current spread {currentSpread:P2} narrower than avg {avgHistorical:P2} - may widen further");
                score -= 20m;
            }
            else
            {
                scoringFactors.Add($"Current spread {currentSpread:P2} near historical avg {avgHistorical:P2}");
            }

            // Check for stability (24h and 3d averages close to each other)
            decimal historicalStability = Math.Abs(spread24h - spread3d) / Math.Max(spread3d, 0.001m);
            if (historicalStability < 0.15m)
            {
                scoringFactors.Add("✓ Stable historical spread pattern");
            }
            else if (historicalStability > 0.40m)
            {
                scoringFactors.Add("⚠️ Volatile historical spread pattern");
                score -= 15m;
            }
        }

        return Math.Max(0m, Math.Min(100m, score));
    }
}
