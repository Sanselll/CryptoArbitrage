using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.Suggestions;
using CryptoArbitrage.API.Services.Suggestions.Analyzers;

namespace CryptoArbitrage.API.Services.Suggestions;

/// <summary>
/// Main service for generating AI-driven trading suggestions for arbitrage opportunities
/// </summary>
public class OpportunitySuggestionService
{
    private readonly ILogger<OpportunitySuggestionService> _logger;
    private readonly StrategySelector _strategySelector;
    private readonly MarketQualityAnalyzer _marketQualityAnalyzer;
    private readonly ExecutionCostAnalyzer _executionCostAnalyzer;
    private readonly RiskAnalyzer _riskAnalyzer;
    private readonly IConfiguration _configuration;

    // Configuration constants with defaults
    private const decimal DEFAULT_BASE_POSITION_SIZE = 1000m; // $1000
    private const decimal DEFAULT_MIN_POSITION_SIZE = 100m; // $100
    private const decimal DEFAULT_MAX_POSITION_SIZE = 10000m; // $10,000
    private const decimal DEFAULT_BASE_LEVERAGE = 3m; // 3x
    private const decimal DEFAULT_MAX_LEVERAGE = 10m; // 10x
    private const decimal DEFAULT_PROFIT_TARGET_MULTIPLIER = 2.0m; // 2x position cost

    public OpportunitySuggestionService(
        ILogger<OpportunitySuggestionService> logger,
        StrategySelector strategySelector,
        MarketQualityAnalyzer marketQualityAnalyzer,
        ExecutionCostAnalyzer executionCostAnalyzer,
        RiskAnalyzer riskAnalyzer,
        IConfiguration configuration)
    {
        _logger = logger;
        _strategySelector = strategySelector;
        _marketQualityAnalyzer = marketQualityAnalyzer;
        _executionCostAnalyzer = executionCostAnalyzer;
        _riskAnalyzer = riskAnalyzer;
        _configuration = configuration;
    }

    /// <summary>
    /// Generates a complete suggestion for an arbitrage opportunity
    /// </summary>
    /// <param name="opportunity">The arbitrage opportunity to analyze</param>
    /// <param name="availableBalances">Dictionary of available balances by exchange (optional for balance-aware suggestions)</param>
    public OpportunitySuggestion GenerateSuggestion(ArbitrageOpportunityDto opportunity, IDictionary<string, decimal>? availableBalances = null)
    {
        try
        {
            var scoringFactors = new List<string>();

            // Step 1: Select optimal strategy and get its score
            var (recommendedStrategy, strategyScore, strategyFactors) = _strategySelector.SelectOptimalStrategy(opportunity);
            scoringFactors.AddRange(strategyFactors);

            // Step 2: Analyze market quality
            decimal marketScore = _marketQualityAnalyzer.AnalyzeMarketQuality(opportunity, scoringFactors);

            // Step 3: Get detailed score breakdown
            var (fundingQuality, profitPotential, spreadEfficiency) = GetDetailedScoreBreakdown(opportunity, recommendedStrategy, scoringFactors);

            // Step 4: Calculate total available balance
            decimal totalAvailableBalance = CalculateTotalAvailableBalance(opportunity, availableBalances, scoringFactors);

            // Step 5: Calculate position sizing (balance-aware)
            decimal suggestedSize = CalculatePositionSize(strategyScore, marketScore, opportunity, totalAvailableBalance, scoringFactors);

            // Step 6: Determine leverage (with margin validation)
            decimal suggestedLeverage = DetermineLeverage(strategyScore, recommendedStrategy, suggestedSize, totalAvailableBalance, scoringFactors);

            // Step 7: Calculate holding period and exit parameters (with funding cycle optimization)
            var (holdingPeriod, profitTarget, maxHolding) = CalculateHoldingParameters(
                opportunity, recommendedStrategy, strategyScore, scoringFactors);

            // Step 8: Calculate time efficiency score
            decimal timeEfficiency = CalculateTimeEfficiency(profitTarget, holdingPeriod, scoringFactors);

            // Step 9: Analyze execution costs and safety
            var (executionCost, executionSafety, profitAfterCosts) = _executionCostAnalyzer.AnalyzeExecutionCosts(
                opportunity, profitTarget, scoringFactors);

            // Step 10: Calculate risk score
            decimal riskScore = _riskAnalyzer.CalculateRiskScore(opportunity, recommendedStrategy, suggestedLeverage, scoringFactors);

            // Step 11: Calculate overall confidence score (NEW FORMULA)
            decimal confidenceScore = CalculateConfidenceScore(strategyScore, marketScore, executionSafety, riskScore, scoringFactors);

            // Step 12: Determine entry recommendation
            EntryRecommendation entryRec = DetermineEntryRecommendation(confidenceScore, scoringFactors);

            // Step 13: Generate reasoning and warnings
            string reasoning = GenerateReasoning(opportunity, recommendedStrategy, confidenceScore, entryRec);
            var warnings = GenerateWarnings(opportunity, confidenceScore, marketScore, scoringFactors);

            // Build the suggestion
            var suggestion = new OpportunitySuggestion
            {
                ConfidenceScore = confidenceScore,
                RecommendedStrategy = recommendedStrategy,
                EntryRecommendation = entryRec,
                SuggestedPositionSizeUsd = suggestedSize,
                SuggestedLeverage = suggestedLeverage,
                SuggestedHoldingPeriodHours = holdingPeriod,
                ProfitTargetPercent = profitTarget,
                MaxHoldingHours = maxHolding,
                ScoreBreakdown = new ScoreBreakdown
                {
                    FundingQualityScore = fundingQuality,
                    ProfitPotentialScore = profitPotential,
                    SpreadEfficiencyScore = spreadEfficiency,
                    MarketQualityScore = marketScore,
                    TimeEfficiencyScore = timeEfficiency,
                    RiskScore = riskScore,
                    ExecutionSafetyScore = executionSafety,
                    ExecutionCostPercent = executionCost,
                    ProfitAfterCosts = profitAfterCosts,
                    OverallConfidence = confidenceScore,
                    ScoringFactors = scoringFactors
                },
                Reasoning = reasoning,
                Warnings = warnings,
                GeneratedAt = DateTime.UtcNow
            };

            _logger.LogInformation(
                "Generated suggestion for {Symbol}: Strategy={Strategy}, Confidence={Confidence}, Recommendation={Recommendation}, Size=${Size}",
                opportunity.Symbol, recommendedStrategy, confidenceScore, entryRec, suggestedSize);

            return suggestion;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating suggestion for opportunity {Symbol}", opportunity.Symbol);

            // Return a conservative fallback suggestion
            return new OpportunitySuggestion
            {
                ConfidenceScore = 0m,
                EntryRecommendation = EntryRecommendation.Skip,
                Warnings = new List<string> { "Error generating suggestion - please review manually" },
                GeneratedAt = DateTime.UtcNow
            };
        }
    }

    private (decimal FundingQuality, decimal ProfitPotential, decimal SpreadEfficiency) GetDetailedScoreBreakdown(
        ArbitrageOpportunityDto opportunity, RecommendedStrategyType strategy, List<string> scoringFactors)
    {
        decimal fundingQuality = 0m;
        decimal profitPotential = 0m;
        decimal spreadEfficiency = 0m;

        // Get detailed scores based on strategy type
        if (strategy == RecommendedStrategyType.FundingOnly || strategy == RecommendedStrategyType.Hybrid)
        {
            var fundingAnalyzer = _strategySelector.GetType().GetField("_fundingAnalyzer",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(_strategySelector);

            if (fundingAnalyzer != null)
            {
                var method = fundingAnalyzer.GetType().GetMethod("GetDetailedScores");
                if (method != null)
                {
                    var tempFactors = new List<string>();
                    var result = method.Invoke(fundingAnalyzer, new object[] { opportunity, tempFactors });
                    if (result is ValueTuple<decimal, decimal> scores)
                    {
                        fundingQuality = scores.Item1;
                        profitPotential = scores.Item2;
                    }
                }
            }
        }

        if (strategy == RecommendedStrategyType.SpreadOnly || strategy == RecommendedStrategyType.Hybrid)
        {
            var spreadAnalyzer = _strategySelector.GetType().GetField("_spreadAnalyzer",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(_strategySelector);

            if (spreadAnalyzer != null)
            {
                var method = spreadAnalyzer.GetType().GetMethod("GetSpreadEfficiencyScore");
                if (method != null)
                {
                    var tempFactors = new List<string>();
                    var result = method.Invoke(spreadAnalyzer, new object[] { opportunity, tempFactors });
                    if (result is decimal score)
                    {
                        spreadEfficiency = score;
                    }
                }
            }
        }

        // For hybrid, average the scores; for single-strategy, use what we have
        if (strategy == RecommendedStrategyType.Hybrid)
        {
            // Already have both, keep as is
        }
        else if (strategy == RecommendedStrategyType.FundingOnly)
        {
            // Spread efficiency should be low for funding-only
            spreadEfficiency = 30m;
        }
        else if (strategy == RecommendedStrategyType.SpreadOnly)
        {
            // Funding scores should be lower for spread-only
            fundingQuality = 40m;
            profitPotential = 50m;
        }

        return (fundingQuality, profitPotential, spreadEfficiency);
    }

    private decimal CalculateConfidenceScore(decimal strategyScore, decimal marketScore, decimal executionSafety, decimal riskScore, List<string> scoringFactors)
    {
        // NEW FORMULA: Balanced scoring across all critical dimensions
        // Strategy Quality: 40% (funding/spread opportunity quality)
        // Market Quality: 20% (liquidity, volume)
        // Execution Safety: 20% (profit vs costs ratio)
        // Risk Score: 20% (volatility, funding reversal, liquidation risks)

        decimal confidence = (strategyScore * 0.40m) +
                            (marketScore * 0.20m) +
                            (executionSafety * 0.20m) +
                            (riskScore * 0.20m);

        scoringFactors.Add($"Overall confidence: {confidence:F1} = Strategy {strategyScore:F1}×40% + Market {marketScore:F1}×20% + Execution {executionSafety:F1}×20% + Risk {riskScore:F1}×20%");

        return Math.Round(Math.Min(100m, confidence), 2);
    }

    private decimal CalculateTimeEfficiency(decimal profitPercent, decimal holdingHours, List<string> scoringFactors)
    {
        // Calculate time-adjusted profit efficiency
        // Convert to daily rate for comparison
        decimal dailyRate = profitPercent * (24m / Math.Max(holdingHours, 1m));
        decimal annualizedAPR = dailyRate * 365m;

        // Score based on daily profit rate
        decimal score;
        if (dailyRate >= 1.0m) // 1%+ per day = 365%+ APR
        {
            score = 100m;
            scoringFactors.Add($"✓ Excellent time efficiency: {profitPercent:F2}% in {holdingHours:F0}h = {dailyRate:F2}%/day ({annualizedAPR:F0}% APR)");
        }
        else if (dailyRate >= 0.5m) // 0.5-1% per day = 182-365% APR
        {
            score = 75m + (25m * ((dailyRate - 0.5m) / 0.5m));
            scoringFactors.Add($"✓ Good time efficiency: {profitPercent:F2}% in {holdingHours:F0}h = {dailyRate:F2}%/day ({annualizedAPR:F0}% APR)");
        }
        else if (dailyRate >= 0.2m) // 0.2-0.5% per day = 73-182% APR
        {
            score = 50m + (25m * ((dailyRate - 0.2m) / 0.3m));
            scoringFactors.Add($"Moderate time efficiency: {profitPercent:F2}% in {holdingHours:F0}h = {dailyRate:F2}%/day ({annualizedAPR:F0}% APR)");
        }
        else if (dailyRate >= 0.1m) // 0.1-0.2% per day = 36-73% APR
        {
            score = 25m + (25m * ((dailyRate - 0.1m) / 0.1m));
            scoringFactors.Add($"⚠️ Low time efficiency: {profitPercent:F2}% in {holdingHours:F0}h = {dailyRate:F2}%/day ({annualizedAPR:F0}% APR)");
        }
        else
        {
            score = Math.Max(0m, 25m * (dailyRate / 0.1m));
            scoringFactors.Add($"⚠️ Poor time efficiency: {profitPercent:F2}% in {holdingHours:F0}h = {dailyRate:F2}%/day ({annualizedAPR:F0}% APR)");
        }

        return Math.Round(score, 2);
    }

    private EntryRecommendation DetermineEntryRecommendation(decimal confidenceScore, List<string> scoringFactors)
    {
        if (confidenceScore >= 80m)
        {
            scoringFactors.Add("✓ Strong confidence - recommended entry");
            return EntryRecommendation.StrongBuy;
        }
        else if (confidenceScore >= 60m)
        {
            scoringFactors.Add("Moderate confidence - consider entry with caution");
            return EntryRecommendation.Buy;
        }
        else if (confidenceScore >= 40m)
        {
            scoringFactors.Add("⚠️ Low confidence - hold and monitor");
            return EntryRecommendation.Hold;
        }
        else
        {
            scoringFactors.Add("⚠️ Very low confidence - skip this opportunity");
            return EntryRecommendation.Skip;
        }
    }

    private decimal CalculateTotalAvailableBalance(ArbitrageOpportunityDto opportunity, IDictionary<string, decimal>? availableBalances, List<string> scoringFactors)
    {
        if (availableBalances == null || !availableBalances.Any())
        {
            scoringFactors.Add("⚠️ No balance data available - using default position sizing");
            return decimal.MaxValue; // No limit if balances not provided
        }

        // Determine which exchanges are needed
        var requiredExchanges = opportunity.Strategy == ArbitrageStrategy.SpotPerpetual
            ? new[] { opportunity.Exchange }
            : new[] { opportunity.LongExchange, opportunity.ShortExchange };

        var exchangeBalances = new List<decimal>();
        foreach (var exchange in requiredExchanges)
        {
            if (availableBalances.TryGetValue(exchange, out decimal balance))
            {
                exchangeBalances.Add(balance);
                scoringFactors.Add($"Available on {exchange}: ${balance:N2}");
            }
            else
            {
                scoringFactors.Add($"⚠️ No balance data for {exchange}");
                exchangeBalances.Add(0m);
            }
        }

        // CRITICAL FIX: For cross-exchange, use MINIMUM balance (bottleneck principle)
        // Each exchange needs to hold its own margin simultaneously
        decimal effectiveBalance;
        if (opportunity.Strategy == ArbitrageStrategy.CrossExchange)
        {
            effectiveBalance = exchangeBalances.Min();
            scoringFactors.Add($"Cross-exchange bottleneck: Using minimum balance ${effectiveBalance:N2} (each exchange needs separate margin)");
        }
        else
        {
            effectiveBalance = exchangeBalances.Sum();
        }

        // Apply safety margin (use only 80% of available balance to prevent liquidation)
        decimal safetyMarginPercent = _configuration.GetValue<decimal>("Suggestions:SafetyMarginPercent", 20m);
        decimal safeBalance = effectiveBalance * (1m - (safetyMarginPercent / 100m));

        scoringFactors.Add($"Effective balance: ${effectiveBalance:N2}, Safe limit (after {safetyMarginPercent}% margin): ${safeBalance:N2}");

        return safeBalance;
    }

    private decimal CalculatePositionSize(decimal confidenceScore, decimal marketScore, ArbitrageOpportunityDto opportunity, decimal availableBalance, List<string> scoringFactors)
    {
        decimal baseSize = _configuration.GetValue<decimal>("Suggestions:BasePositionSize", DEFAULT_BASE_POSITION_SIZE);
        decimal minSize = _configuration.GetValue<decimal>("Suggestions:MinPositionSize", DEFAULT_MIN_POSITION_SIZE);
        decimal maxSize = _configuration.GetValue<decimal>("Suggestions:MaxPositionSize", DEFAULT_MAX_POSITION_SIZE);

        // Scale by confidence (0.5x to 1.5x)
        decimal confidenceFactor = 0.5m + (confidenceScore / 100m);

        // Scale by liquidity (volume factor)
        decimal liquidityFactor = CalculateLiquidityFactor(opportunity.Volume24h);

        // Calculate suggested size
        decimal suggestedSize = baseSize * confidenceFactor * liquidityFactor;

        // Clamp to configured min/max
        suggestedSize = Math.Max(minSize, Math.Min(maxSize, suggestedSize));

        // Apply balance limit (most important constraint)
        if (availableBalance < decimal.MaxValue)
        {
            if (suggestedSize > availableBalance)
            {
                scoringFactors.Add($"⚠️ Position size limited by available balance: ${availableBalance:N2} (was ${suggestedSize:N2})");
                suggestedSize = availableBalance;
            }
        }

        scoringFactors.Add($"Final position size: ${suggestedSize:N0} (Base ${baseSize:N0} × Confidence {confidenceFactor:F2} × Liquidity {liquidityFactor:F2})");

        return Math.Round(suggestedSize, 0);
    }

    private decimal CalculateLiquidityFactor(decimal volume24h)
    {
        // Scale position size based on liquidity
        // $100K volume → 0.3x, $1M → 1.0x, $10M+ → 1.5x
        if (volume24h >= 10_000_000m) return 1.5m;
        if (volume24h >= 1_000_000m) return 1.0m;
        if (volume24h >= 100_000m) return 0.5m + (0.5m * ((volume24h - 100_000m) / 900_000m));
        return 0.3m;
    }

    private decimal DetermineLeverage(decimal confidenceScore, RecommendedStrategyType strategy, decimal positionSize, decimal availableBalance, List<string> scoringFactors)
    {
        decimal baseLeverage = _configuration.GetValue<decimal>("Suggestions:BaseLeverage", DEFAULT_BASE_LEVERAGE);
        decimal maxLeverage = _configuration.GetValue<decimal>("Suggestions:MaxLeverage", DEFAULT_MAX_LEVERAGE);

        // Higher confidence → higher leverage (but conservative)
        decimal leverageFactor = 0.5m + (confidenceScore / 200m); // 0.5x to 1.0x
        decimal suggestedLeverage = baseLeverage * leverageFactor;

        // Funding strategies can use slightly higher leverage than spread strategies
        if (strategy == RecommendedStrategyType.FundingOnly)
        {
            suggestedLeverage *= 1.2m;
        }
        else if (strategy == RecommendedStrategyType.SpreadOnly)
        {
            suggestedLeverage *= 0.8m; // More conservative for spread capture
        }

        // Clamp to reasonable range (1x to maxLeverage)
        suggestedLeverage = Math.Max(1m, Math.Min(maxLeverage, suggestedLeverage));

        // Validate margin requirements
        if (availableBalance < decimal.MaxValue && positionSize > 0)
        {
            // Calculate required margin with buffer
            decimal maintenanceMarginBuffer = _configuration.GetValue<decimal>("Suggestions:MaintenanceMarginBuffer", 20m);
            decimal requiredMargin = positionSize / suggestedLeverage;
            decimal marginWithBuffer = requiredMargin * (1m + (maintenanceMarginBuffer / 100m));

            scoringFactors.Add($"Margin check: Position ${positionSize:N0} at {suggestedLeverage:F1}x requires ${requiredMargin:N0} + {maintenanceMarginBuffer}% buffer = ${marginWithBuffer:N0}");

            // Auto-reduce leverage if insufficient margin
            if (marginWithBuffer > availableBalance)
            {
                // Calculate max safe leverage
                decimal maxSafeLeverage = positionSize / (availableBalance * (100m / (100m + maintenanceMarginBuffer)));
                maxSafeLeverage = Math.Max(1m, Math.Floor(maxSafeLeverage * 10m) / 10m); // Round down to 0.1x

                if (maxSafeLeverage < suggestedLeverage)
                {
                    scoringFactors.Add($"⚠️ Leverage reduced from {suggestedLeverage:F1}x to {maxSafeLeverage:F1}x due to margin constraints (available: ${availableBalance:N0})");
                    suggestedLeverage = maxSafeLeverage;
                }
            }
            else
            {
                scoringFactors.Add($"✓ Sufficient margin available: ${availableBalance:N0} covers ${marginWithBuffer:N0}");
            }
        }

        scoringFactors.Add($"Final leverage: {suggestedLeverage:F1}x");

        return Math.Round(suggestedLeverage, 1);
    }

    private (decimal HoldingPeriod, decimal ProfitTarget, decimal MaxHolding) CalculateHoldingParameters(
        ArbitrageOpportunityDto opportunity, RecommendedStrategyType strategy, decimal confidenceScore, List<string> scoringFactors)
    {
        // Extract funding interval (varies by exchange: 1h, 4h, 8h, etc.)
        decimal fundingInterval = opportunity.LongFundingIntervalHours ?? opportunity.ShortFundingIntervalHours ?? 8m;
        decimal breakEven = opportunity.BreakEvenTimeHours ?? 48m;

        decimal holdingPeriod;
        decimal profitTarget;
        decimal maxHolding;

        if (strategy == RecommendedStrategyType.FundingOnly)
        {
            // Calculate cycles to break-even
            decimal cyclesToBreakEven = Math.Ceiling(breakEven / fundingInterval);

            // Hold for 1.5x break-even cycles (rounded to whole cycles)
            decimal targetCycles = Math.Ceiling(cyclesToBreakEven * 1.5m);
            holdingPeriod = targetCycles * fundingInterval;
            maxHolding = targetCycles * 2m * fundingInterval;

            // Calculate actual funding profit over holding period
            // Funding rate per cycle = APR / (hours per year / interval)
            decimal hoursPerYear = 365m * 24m;
            decimal cyclesPerYear = hoursPerYear / fundingInterval;
            decimal fundingRatePerCycle = opportunity.FundApr / cyclesPerYear;
            decimal expectedFundingProfit = fundingRatePerCycle * targetCycles;

            // Target = funding income - position cost
            profitTarget = expectedFundingProfit - opportunity.PositionCostPercent;

            // Add funding time info
            scoringFactors.Add($"Funding interval: {fundingInterval}h ({cyclesPerYear:F0} cycles/year)");
            scoringFactors.Add($"Break-even: {breakEven:F1}h ({cyclesToBreakEven:F0} cycles)");
            scoringFactors.Add($"Funding strategy: Hold {holdingPeriod:F0}h ({targetCycles:F0} cycles × {fundingRatePerCycle:F3}% = {expectedFundingProfit:F2}% - {opportunity.PositionCostPercent:F2}% cost = {profitTarget:F2}% target)");
        }
        else if (strategy == RecommendedStrategyType.SpreadOnly)
        {
            // Spread strategies: quick capture
            holdingPeriod = 4m; // 4 hours for spread to close
            maxHolding = 24m; // Max 1 day

            // Capture 75% of estimated spread (conservative)
            decimal spreadCapture = opportunity.EstimatedProfitPercentage * 0.75m;
            profitTarget = spreadCapture;

            scoringFactors.Add($"Spread strategy: Quick capture in {holdingPeriod:F0}h");
            scoringFactors.Add($"Target {profitTarget:F2}% ({opportunity.EstimatedProfitPercentage:F2}% spread × 75% capture)");
        }
        else // Hybrid
        {
            // Calculate funding cycles for holding period
            decimal targetCycles = Math.Max(3m, Math.Ceiling(breakEven / fundingInterval));
            holdingPeriod = targetCycles * fundingInterval;
            maxHolding = targetCycles * 2m * fundingInterval;

            // Calculate funding component
            decimal hoursPerYear = 365m * 24m;
            decimal cyclesPerYear = hoursPerYear / fundingInterval;
            decimal fundingRatePerCycle = opportunity.FundApr / cyclesPerYear;
            decimal fundingProfit = fundingRatePerCycle * targetCycles;

            // Calculate spread component (70% capture)
            decimal spreadProfit = opportunity.EstimatedProfitPercentage * 0.7m;

            // Combined target
            profitTarget = fundingProfit + spreadProfit - opportunity.PositionCostPercent;

            scoringFactors.Add($"Hybrid strategy: Hold {holdingPeriod:F0}h ({targetCycles:F0} funding cycles)");
            scoringFactors.Add($"Target: {fundingProfit:F2}% funding + {spreadProfit:F2}% spread - {opportunity.PositionCostPercent:F2}% cost = {profitTarget:F2}%");
        }

        return (Math.Round(holdingPeriod, 0), Math.Round(profitTarget, 3), Math.Round(maxHolding, 0));
    }

    private string GenerateReasoning(ArbitrageOpportunityDto opportunity, RecommendedStrategyType strategy, decimal confidence, EntryRecommendation recommendation)
    {
        var parts = new List<string>
        {
            $"Confidence: {confidence:F1}/100",
            $"Recommended strategy: {strategy}",
            $"Recommendation: {recommendation}"
        };

        if (strategy == RecommendedStrategyType.FundingOnly)
        {
            parts.Add($"Focus on earning {opportunity.FundApr:F1}% APR from funding rates");
        }
        else if (strategy == RecommendedStrategyType.SpreadOnly)
        {
            parts.Add($"Focus on capturing {opportunity.EstimatedProfitPercentage:F2}% price spread");
        }
        else
        {
            parts.Add($"Combine spread capture with funding income for optimal returns");
        }

        return string.Join(". ", parts) + ".";
    }

    private List<string> GenerateWarnings(ArbitrageOpportunityDto opportunity, decimal confidence, decimal marketScore, List<string> scoringFactors)
    {
        var warnings = new List<string>();

        // Extract warnings from scoring factors
        warnings.AddRange(scoringFactors.Where(f => f.Contains("⚠️")));

        // Add specific warnings based on conditions
        if (confidence < 60m)
        {
            warnings.Add("Low confidence score - higher risk opportunity");
        }

        if (marketScore < 50m)
        {
            warnings.Add("Poor market quality - consider execution risks");
        }

        if (opportunity.LiquidityStatus == LiquidityStatus.Low)
        {
            warnings.Add("Low liquidity - expect slippage on execution");
        }

        if (!string.IsNullOrEmpty(opportunity.LiquidityWarning))
        {
            warnings.Add(opportunity.LiquidityWarning);
        }

        return warnings;
    }
}
