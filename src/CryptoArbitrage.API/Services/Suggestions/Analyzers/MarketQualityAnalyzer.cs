using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes market quality metrics (liquidity, volume, spread)
/// </summary>
public class MarketQualityAnalyzer
{
    private readonly ILogger<MarketQualityAnalyzer> _logger;

    // Thresholds for market quality scoring
    private const decimal EXCELLENT_VOLUME_THRESHOLD = 10_000_000m; // $10M+
    private const decimal GOOD_VOLUME_THRESHOLD = 1_000_000m; // $1M+
    private const decimal MINIMUM_VOLUME_THRESHOLD = 100_000m; // $100K

    private const decimal EXCELLENT_SPREAD_THRESHOLD = 0.05m; // 0.05%
    private const decimal GOOD_SPREAD_THRESHOLD = 0.10m; // 0.10%
    private const decimal ACCEPTABLE_SPREAD_THRESHOLD = 0.20m; // 0.20%

    public MarketQualityAnalyzer(ILogger<MarketQualityAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Analyzes market quality and returns a score (0-100)
    /// </summary>
    public decimal AnalyzeMarketQuality(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal volumeScore = AnalyzeVolume(opportunity, scoringFactors);
        decimal liquidityScore = AnalyzeLiquidity(opportunity, scoringFactors);
        decimal spreadScore = AnalyzeBidAskSpread(opportunity, scoringFactors);

        // Weighted average: Volume 40%, Liquidity 30%, Spread 30%
        decimal overallScore = (volumeScore * 0.4m) + (liquidityScore * 0.3m) + (spreadScore * 0.3m);

        _logger.LogDebug(
            "Market Quality Analysis for {Symbol}: Volume={VolumeScore}, Liquidity={LiquidityScore}, Spread={SpreadScore}, Overall={OverallScore}",
            opportunity.Symbol, volumeScore, liquidityScore, spreadScore, overallScore);

        return Math.Round(overallScore, 2);
    }

    private decimal AnalyzeVolume(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal volume = opportunity.Volume24h;

        if (volume >= EXCELLENT_VOLUME_THRESHOLD)
        {
            scoringFactors.Add($"Excellent 24h volume: ${volume:N0}");
            return 100m;
        }
        else if (volume >= GOOD_VOLUME_THRESHOLD)
        {
            scoringFactors.Add($"Good 24h volume: ${volume:N0}");
            // Linear scale from 70-100 between 1M and 10M
            return 70m + (30m * ((volume - GOOD_VOLUME_THRESHOLD) / (EXCELLENT_VOLUME_THRESHOLD - GOOD_VOLUME_THRESHOLD)));
        }
        else if (volume >= MINIMUM_VOLUME_THRESHOLD)
        {
            scoringFactors.Add($"Moderate 24h volume: ${volume:N0}");
            // Linear scale from 40-70 between 100K and 1M
            return 40m + (30m * ((volume - MINIMUM_VOLUME_THRESHOLD) / (GOOD_VOLUME_THRESHOLD - MINIMUM_VOLUME_THRESHOLD)));
        }
        else
        {
            scoringFactors.Add($"⚠️ Low 24h volume: ${volume:N0}");
            // Linear scale from 0-40 for volumes below 100K
            return Math.Max(0m, 40m * (volume / MINIMUM_VOLUME_THRESHOLD));
        }
    }

    private decimal AnalyzeLiquidity(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        switch (opportunity.LiquidityStatus)
        {
            case LiquidityStatus.Good:
                scoringFactors.Add("Good liquidity status");
                return 100m;

            case LiquidityStatus.Medium:
                scoringFactors.Add("⚠️ Medium liquidity status");
                return 60m;

            case LiquidityStatus.Low:
                scoringFactors.Add("⚠️ Low liquidity status - high slippage risk");
                return 20m;

            default:
                scoringFactors.Add("⚠️ Unknown liquidity status");
                return 50m;
        }
    }

    private decimal AnalyzeBidAskSpread(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        if (!opportunity.BidAskSpreadPercent.HasValue)
        {
            // No spread data available, assume moderate score
            return 50m;
        }

        decimal spread = opportunity.BidAskSpreadPercent.Value;

        if (spread <= EXCELLENT_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Excellent bid-ask spread: {spread:P2}");
            return 100m;
        }
        else if (spread <= GOOD_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Good bid-ask spread: {spread:P2}");
            // Linear scale from 80-100
            return 80m + (20m * (1 - ((spread - EXCELLENT_SPREAD_THRESHOLD) / (GOOD_SPREAD_THRESHOLD - EXCELLENT_SPREAD_THRESHOLD))));
        }
        else if (spread <= ACCEPTABLE_SPREAD_THRESHOLD)
        {
            scoringFactors.Add($"Acceptable bid-ask spread: {spread:P2}");
            // Linear scale from 50-80
            return 50m + (30m * (1 - ((spread - GOOD_SPREAD_THRESHOLD) / (ACCEPTABLE_SPREAD_THRESHOLD - GOOD_SPREAD_THRESHOLD))));
        }
        else
        {
            scoringFactors.Add($"⚠️ Wide bid-ask spread: {spread:P2} - execution risk");
            // Declining score for spreads > 0.2%
            return Math.Max(0m, 50m * (1 - Math.Min(1, (spread - ACCEPTABLE_SPREAD_THRESHOLD) / 0.5m)));
        }
    }
}
