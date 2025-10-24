using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes various risk factors for arbitrage opportunities
/// </summary>
public class RiskAnalyzer
{
    private readonly ILogger<RiskAnalyzer> _logger;

    public RiskAnalyzer(ILogger<RiskAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Calculates overall risk score (0-100, higher = less risky)
    /// </summary>
    public decimal CalculateRiskScore(
        ArbitrageOpportunityDto opportunity,
        RecommendedStrategyType strategy,
        decimal suggestedLeverage,
        List<string> scoringFactors)
    {
        decimal spreadVolatilityRisk = AnalyzeSpreadVolatility(opportunity, scoringFactors);
        decimal fundingReversalRisk = AnalyzeFundingReversalRisk(opportunity, strategy, scoringFactors);
        decimal liquidationRisk = AnalyzeLiquidationRisk(opportunity, suggestedLeverage, scoringFactors);
        decimal marketConditionRisk = AnalyzeMarketConditions(opportunity, scoringFactors);

        // Weighted risk score
        // Lower risk = higher score (inverted for readability)
        decimal overallRisk = (spreadVolatilityRisk * 0.30m) +
                              (fundingReversalRisk * 0.30m) +
                              (liquidationRisk * 0.25m) +
                              (marketConditionRisk * 0.15m);

        scoringFactors.Add($"Overall risk safety: {overallRisk:F1}/100 (higher = safer)");

        return Math.Round(overallRisk, 2);
    }

    /// <summary>
    /// Analyzes spread volatility risk
    /// </summary>
    private decimal AnalyzeSpreadVolatility(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        // Only applicable for cross-exchange spread strategies
        if (opportunity.Strategy != ArbitrageStrategy.CrossExchange)
        {
            return 100m; // No spread volatility risk for spot-perpetual
        }

        bool has24hAvg = opportunity.PriceSpread24hAvg.HasValue;
        bool has3dAvg = opportunity.PriceSpread3dAvg.HasValue;

        if (!has24hAvg || !has3dAvg)
        {
            scoringFactors.Add("⚠️ No historical spread data - unknown volatility risk");
            return 50m; // Unknown = neutral risk
        }

        decimal currentSpread = Math.Abs(opportunity.SpreadRate);
        decimal spread24h = Math.Abs(opportunity.PriceSpread24hAvg!.Value);
        decimal spread3d = Math.Abs(opportunity.PriceSpread3dAvg!.Value);

        // Calculate spread stability (how much 24h and 3d averages differ)
        decimal avgSpread = (spread24h + spread3d) / 2m;
        decimal spreadVariance = Math.Abs(spread24h - spread3d) / Math.Max(avgSpread, 0.0001m);

        // Calculate current deviation from average
        decimal currentDeviation = Math.Abs(currentSpread - avgSpread) / Math.Max(avgSpread, 0.0001m);

        // Score based on stability and deviation
        decimal stabilityScore = 100m;

        if (spreadVariance < 0.15m)
        {
            scoringFactors.Add($"✓ Very stable spread pattern (24h/3d variance: {spreadVariance:P1})");
        }
        else if (spreadVariance < 0.30m)
        {
            scoringFactors.Add($"Moderate spread stability (24h/3d variance: {spreadVariance:P1})");
            stabilityScore -= 20m;
        }
        else
        {
            scoringFactors.Add($"⚠️ High spread volatility (24h/3d variance: {spreadVariance:P1})");
            stabilityScore -= 40m;
        }

        // Penalize if current spread is extreme outlier
        if (currentDeviation > 0.50m)
        {
            scoringFactors.Add($"⚠️ Current spread {currentSpread:P2} is extreme outlier from avg {avgSpread:P2}");
            stabilityScore -= 30m;
        }
        else if (currentDeviation > 0.30m)
        {
            scoringFactors.Add($"Current spread {currentSpread:P2} significantly differs from avg {avgSpread:P2}");
            stabilityScore -= 15m;
        }

        return Math.Max(0m, Math.Min(100m, stabilityScore));
    }

    /// <summary>
    /// Analyzes funding rate reversal risk
    /// </summary>
    private decimal AnalyzeFundingReversalRisk(
        ArbitrageOpportunityDto opportunity,
        RecommendedStrategyType strategy,
        List<string> scoringFactors)
    {
        // Only relevant for funding-based strategies
        if (strategy == RecommendedStrategyType.SpreadOnly)
        {
            return 100m; // No funding reversal risk for spread-only
        }

        // Get funding data
        decimal currentLongRate = opportunity.LongFundingRate;
        decimal currentShortRate = opportunity.ShortFundingRate;

        if (currentLongRate == 0m && currentShortRate == 0m)
        {
            scoringFactors.Add("⚠️ No funding rate data - cannot assess reversal risk");
            return 50m;
        }

        decimal netCurrentRate = currentLongRate - currentShortRate;
        decimal score = 100m;

        // Check if funding is approaching zero or already negative
        if (Math.Abs(netCurrentRate) < 0.0001m)
        {
            scoringFactors.Add("⚠️ CRITICAL: Net funding rate near zero - high reversal risk!");
            score = 20m;
        }
        else if (Math.Abs(netCurrentRate) < 0.0005m)
        {
            scoringFactors.Add("⚠️ Very low net funding rate - reversal risk present");
            score -= 40m;
        }
        else
        {
            // Funding rate is reasonable
            scoringFactors.Add($"Funding rate: {netCurrentRate:P3} (Long: {currentLongRate:P3}, Short: {currentShortRate:P3})");
        }

        return Math.Max(0m, Math.Min(100m, score));
    }

    /// <summary>
    /// Analyzes liquidation risk based on leverage
    /// </summary>
    private decimal AnalyzeLiquidationRisk(
        ArbitrageOpportunityDto opportunity,
        decimal leverage,
        List<string> scoringFactors)
    {
        // Estimate distance to liquidation
        // For leveraged position, liquidation typically occurs around (1 / leverage) price move
        // e.g., 5x leverage = ~20% price move triggers liquidation

        if (leverage <= 1m)
        {
            scoringFactors.Add("✓ No leverage - no liquidation risk");
            return 100m;
        }

        // Calculate approximate liquidation distance
        decimal liquidationDistance = 1m / leverage;

        // Estimate price volatility from spread data (rough proxy)
        decimal estimatedVolatility = 0.05m; // Default 5% volatility
        if (opportunity.PriceSpread24hAvg.HasValue)
        {
            // Use spread variance as volatility proxy
            estimatedVolatility = Math.Max(estimatedVolatility, Math.Abs(opportunity.PriceSpread24hAvg.Value) * 10m);
        }

        // Calculate "safety buffer" = liquidation distance / volatility
        decimal safetyBuffer = liquidationDistance / Math.Max(estimatedVolatility, 0.01m);

        decimal score;
        if (safetyBuffer >= 5.0m)
        {
            scoringFactors.Add($"✓ Low liquidation risk: {leverage:F1}x leverage, {liquidationDistance:P1} to liquidation ({safetyBuffer:F1}x volatility)");
            score = 100m;
        }
        else if (safetyBuffer >= 3.0m)
        {
            scoringFactors.Add($"Moderate liquidation risk: {leverage:F1}x leverage, {liquidationDistance:P1} to liquidation ({safetyBuffer:F1}x volatility)");
            score = 70m + (30m * ((safetyBuffer - 3.0m) / 2.0m));
        }
        else if (safetyBuffer >= 2.0m)
        {
            scoringFactors.Add($"⚠️ Elevated liquidation risk: {leverage:F1}x leverage, {liquidationDistance:P1} to liquidation ({safetyBuffer:F1}x volatility)");
            score = 40m + (30m * ((safetyBuffer - 2.0m) / 1.0m));
        }
        else if (safetyBuffer >= 1.0m)
        {
            scoringFactors.Add($"⚠️ HIGH liquidation risk: {leverage:F1}x leverage, {liquidationDistance:P1} to liquidation ({safetyBuffer:F1}x volatility)");
            score = 20m * safetyBuffer;
        }
        else
        {
            scoringFactors.Add($"⚠️ CRITICAL liquidation risk: {leverage:F1}x leverage, liquidation distance < volatility!");
            score = 0m;
        }

        return score;
    }

    /// <summary>
    /// Analyzes general market condition risks
    /// </summary>
    private decimal AnalyzeMarketConditions(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal score = 100m;

        // Check volume (low volume = higher execution risk)
        if (opportunity.Volume24h < 100_000m)
        {
            scoringFactors.Add("⚠️ Very low volume - high execution risk");
            score -= 40m;
        }
        else if (opportunity.Volume24h < 500_000m)
        {
            scoringFactors.Add("Low volume - moderate execution risk");
            score -= 20m;
        }

        // Check liquidity status
        if (opportunity.LiquidityStatus == LiquidityStatus.Low)
        {
            scoringFactors.Add("⚠️ Low liquidity - price impact risk");
            score -= 30m;
        }
        else if (opportunity.LiquidityStatus == LiquidityStatus.Medium)
        {
            scoringFactors.Add("Medium liquidity - some price impact expected");
            score -= 15m;
        }

        return Math.Max(0m, score);
    }
}
