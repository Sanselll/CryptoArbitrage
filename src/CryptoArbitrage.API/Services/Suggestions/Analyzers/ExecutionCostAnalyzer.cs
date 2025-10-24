using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.Suggestions.Analyzers;

/// <summary>
/// Analyzes execution costs and validates profit targets against realistic trading costs
/// </summary>
public class ExecutionCostAnalyzer
{
    private readonly ILogger<ExecutionCostAnalyzer> _logger;

    // Execution cost constants
    private const decimal DEFAULT_SLIPPAGE_PERCENT = 0.05m; // 0.05% default slippage per trade
    private const decimal HIGH_SLIPPAGE_PERCENT = 0.10m; // 0.10% for low liquidity

    public ExecutionCostAnalyzer(ILogger<ExecutionCostAnalyzer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Calculates total execution cost and validates against profit target
    /// Returns (ExecutionCostPercent, ExecutionSafetyScore, ProfitAfterCosts)
    /// </summary>
    public (decimal ExecutionCost, decimal SafetyScore, decimal ProfitAfterCosts) AnalyzeExecutionCosts(
        ArbitrageOpportunityDto opportunity,
        decimal profitTargetPercent,
        List<string> scoringFactors)
    {
        decimal totalExecutionCost = CalculateTotalExecutionCost(opportunity, scoringFactors);
        decimal profitAfterCosts = profitTargetPercent - totalExecutionCost;
        decimal safetyScore = CalculateExecutionSafetyScore(profitTargetPercent, totalExecutionCost, scoringFactors);

        scoringFactors.Add($"Execution cost: {totalExecutionCost:F3}%, Profit after costs: {profitAfterCosts:F3}%");

        return (totalExecutionCost, safetyScore, profitAfterCosts);
    }

    /// <summary>
    /// Calculates total execution cost including spreads, slippage, and position costs
    /// </summary>
    private decimal CalculateTotalExecutionCost(ArbitrageOpportunityDto opportunity, List<string> scoringFactors)
    {
        decimal entryCost = 0m;
        decimal exitCost = 0m;
        decimal slippageCost = 0m;
        decimal positionCost = opportunity.PositionCostPercent;

        // Determine slippage based on liquidity
        decimal slippageRate = DetermineSlippageRate(opportunity);

        if (opportunity.Strategy == ArbitrageStrategy.SpotPerpetual)
        {
            // Spot-Perpetual: 2 trades per entry (buy spot, sell futures), 2 trades per exit
            // Each trade crosses the spread once
            decimal bidAskSpread = opportunity.BidAskSpreadPercent ?? 0.1m; // Default 0.1% if not available

            entryCost = bidAskSpread * 2m; // Open both legs
            exitCost = bidAskSpread * 2m;   // Close both legs
            slippageCost = slippageRate * 4m; // 4 total trades

            scoringFactors.Add($"Entry cost: {entryCost:F3}% (2 trades × {bidAskSpread:F3}% spread)");
            scoringFactors.Add($"Exit cost: {exitCost:F3}% (2 trades × {bidAskSpread:F3}% spread)");
        }
        else // CrossExchange
        {
            // Cross-Exchange: 2 trades per entry (one on each exchange), 2 trades per exit
            decimal bidAskSpread = opportunity.BidAskSpreadPercent ?? 0.1m; // Default 0.1% if not available

            entryCost = bidAskSpread * 2m; // Open position on both exchanges
            exitCost = bidAskSpread * 2m;   // Close position on both exchanges
            slippageCost = slippageRate * 4m; // 4 total trades

            scoringFactors.Add($"Entry cost: {entryCost:F3}% (2 trades × {bidAskSpread:F3}% spread)");
            scoringFactors.Add($"Exit cost: {exitCost:F3}% (2 trades × {bidAskSpread:F3}% spread)");
        }

        scoringFactors.Add($"Slippage cost: {slippageCost:F3}% (4 trades × {slippageRate:F3}%)");
        scoringFactors.Add($"Position cost: {positionCost:F3}% (funding + fees)");

        decimal totalCost = entryCost + exitCost + slippageCost + positionCost;

        return totalCost;
    }

    /// <summary>
    /// Determines slippage rate based on liquidity metrics
    /// </summary>
    private decimal DetermineSlippageRate(ArbitrageOpportunityDto opportunity)
    {
        // Use liquidity status to determine slippage
        return opportunity.LiquidityStatus switch
        {
            LiquidityStatus.Good => DEFAULT_SLIPPAGE_PERCENT,
            LiquidityStatus.Medium => DEFAULT_SLIPPAGE_PERCENT * 1.5m,
            LiquidityStatus.Low => HIGH_SLIPPAGE_PERCENT,
            _ => DEFAULT_SLIPPAGE_PERCENT
        };
    }

    /// <summary>
    /// Calculates execution safety score based on profit-to-cost ratio
    /// </summary>
    private decimal CalculateExecutionSafetyScore(decimal profitTarget, decimal executionCost, List<string> scoringFactors)
    {
        if (executionCost <= 0)
        {
            return 100m; // No cost = perfect safety
        }

        decimal profitToCostRatio = profitTarget / executionCost;

        // Scoring based on profit-to-cost ratio
        if (profitToCostRatio >= 5.0m)
        {
            // Profit is 5x+ execution cost - excellent safety margin
            scoringFactors.Add($"✓ Excellent safety: Profit {profitTarget:F3}% is {profitToCostRatio:F1}x execution cost");
            return 100m;
        }
        else if (profitToCostRatio >= 3.0m)
        {
            // Profit is 3-5x execution cost - good safety margin
            scoringFactors.Add($"✓ Good safety: Profit {profitTarget:F3}% is {profitToCostRatio:F1}x execution cost");
            return 75m + (25m * ((profitToCostRatio - 3.0m) / 2.0m));
        }
        else if (profitToCostRatio >= 2.0m)
        {
            // Profit is 2-3x execution cost - acceptable but tight
            scoringFactors.Add($"Acceptable safety: Profit {profitTarget:F3}% is {profitToCostRatio:F1}x execution cost");
            return 50m + (25m * ((profitToCostRatio - 2.0m) / 1.0m));
        }
        else if (profitToCostRatio >= 1.0m)
        {
            // Profit is 1-2x execution cost - risky, small margin for error
            scoringFactors.Add($"⚠️ Low safety: Profit {profitTarget:F3}% is only {profitToCostRatio:F1}x execution cost");
            return 25m * (profitToCostRatio - 1.0m);
        }
        else
        {
            // Profit less than execution cost - guaranteed loss!
            scoringFactors.Add($"⚠️ CRITICAL: Profit {profitTarget:F3}% < execution cost {executionCost:F3}% - Guaranteed loss!");
            return 0m;
        }
    }
}
