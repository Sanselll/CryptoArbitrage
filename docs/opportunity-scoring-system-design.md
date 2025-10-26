# Opportunity Scoring System Design

## Overview

Comprehensive scoring system to predict profitability of arbitrage opportunity execution, determine optimal position sizing, estimate hold duration, and assess risks.

## Requirements Summary

- **Primary Goal**: Maximum flexibility with adjustable weights
- **Position Sizing**: Best-in-class approach (Kelly Criterion + Risk-adjusted)
- **Hold Duration**: Both time-based exit signals AND condition-based triggers
- **Risk Focus**: Execution risk, rate volatility, concentration risk, market correlation

---

## Architecture Overview

```
1. Multi-Factor Scoring Engine
   ├── Profitability Score (0-100)
   ├── Risk Score (0-100)
   ├── Execution Quality Score (0-100)
   ├── Timing Score (0-100)
   └── Composite Score (weighted combination)

2. Position Sizing Engine
   ├── Kelly Criterion Calculator
   ├── Risk-Adjusted Modifier
   ├── Concentration Limiter
   └── Portfolio Correlation Adjuster

3. Hold Duration Optimizer
   ├── Time-Based Exit Predictor
   ├── Condition-Based Monitor
   └── Dynamic Rebalancing Suggester
```

---

## 1. Multi-Factor Scoring System

### A. Profitability Score (0-100)

**Formula:**
```
ProfitabilityScore = (
  CurrentAPR_Score * 0.30 +
  Proj24h_Score * 0.30 +
  Proj3D_Score * 0.25 +
  BreakEven_Score * 0.15
)
```

**Components:**

1. **CurrentAPR_Score**: Normalize `FundApr` or `EstimatedProfitPercentage`
   - 0-20% APR → 0-50 points
   - 20-50% APR → 50-80 points
   - 50%+ APR → 80-100 points

2. **Proj24h_Score**: Score `FundApr24hProj`
   - Penalize if 24h projection < current rate (rate declining)
   - Bonus if 24h projection ≥ current rate (rate stable/rising)

3. **Proj3D_Score**: Score `FundApr3dProj`
   - Higher weight if 3D avg > 24h avg (sustained rate)
   - Lower weight if 3D avg < 24h avg (declining trend)

4. **BreakEven_Score**: Based on `BreakEvenTimeHours`
   - < 8 hours → 100 points
   - 8-24 hours → 75 points
   - 24-72 hours → 50 points
   - > 72 hours → 25 points

---

### B. Risk Score (0-100, lower is riskier)

**Formula:**
```
RiskScore = (
  VolatilityScore * 0.35 +
  StabilityScore * 0.35 +
  CorrelationScore * 0.30
)
```

**Components:**

1. **VolatilityScore** (for cross-exchange opportunities):
   ```csharp
   if (SpreadVolatilityCv.HasValue)
   {
       // Coefficient of Variation: lower is better
       // CV < 0.2 (low volatility) → 100 points
       // CV 0.2-0.5 (medium) → 70 points
       // CV > 0.5 (high) → 30 points
       score = Math.Max(0, 100 - (SpreadVolatilityCv.Value * 200));
   }
   ```

2. **StabilityScore**: Compare current vs historical rates
   ```csharp
   // Rate consistency check
   var current = FundProfit8h;
   var avg24h = FundProfit8h24hProj ?? current;
   var avg3d = FundProfit8h3dProj ?? current;

   var deviation24h = Math.Abs(current - avg24h) / avg24h;
   var deviation3d = Math.Abs(current - avg3d) / avg3d;

   // < 10% deviation → 100 points
   // 10-25% deviation → 70 points
   // > 25% deviation → 30 points
   ```

3. **CorrelationScore**: Track correlation with other active positions
   - Same symbol, different strategy → 70% correlation penalty
   - Same exchange → 50% correlation penalty
   - Different symbol/exchange → Low correlation bonus

---

### C. Execution Quality Score (0-100)

**Formula:**
```
ExecutionScore = (
  LiquidityScore * 0.40 +
  VolumeScore * 0.35 +
  SpreadScore * 0.25
)
```

**Components:**

1. **LiquidityScore**: Based on `LiquidityStatus`
   ```csharp
   switch (LiquidityStatus)
   {
       case Good: score = 100;
       case Medium: score = 60;
       case Low: score = 20;
   }

   // Adjust by orderbook depth
   if (OrderbookDepthUsd >= 100_000) score += 10;
   if (OrderbookDepthUsd < 25_000) score -= 20;
   ```

2. **VolumeScore**: 24h volume relative to position size
   ```csharp
   // Volume should be at least 100x position size for safe execution
   var volumeRatio = Volume24h / estimatedPositionSize;
   if (volumeRatio >= 100) score = 100;
   else if (volumeRatio >= 50) score = 80;
   else if (volumeRatio >= 25) score = 60;
   else score = 30;
   ```

3. **SpreadScore**: Bid-ask spread impact
   ```csharp
   var spreadPercent = BidAskSpreadPercent ?? 0;
   if (spreadPercent <= 0.1m) score = 100; // 0.1% or less
   else if (spreadPercent <= 0.5m) score = 75;
   else score = 50;
   ```

---

### D. Timing Score (0-100)

Evaluates if NOW is the right time to enter:

```csharp
TimingScore = (
  RateTrendScore * 0.40 +
  MarketConditionScore * 0.35 +
  PositionWindowScore * 0.25
)
```

**Components:**

1. **RateTrendScore**: Is the rate improving or declining?
   ```csharp
   // Compare current rate to moving averages
   if (FundProfit8h > FundProfit8h24hProj && FundProfit8h24hProj > FundProfit8h3dProj)
       score = 100; // Uptrend: rate accelerating
   else if (FundProfit8h >= FundProfit8h3dProj)
       score = 70; // Stable
   else
       score = 40; // Downtrend: wait for reversal
   ```

2. **MarketConditionScore**: Check if funding interval is approaching
   ```csharp
   // Ideally enter just after funding payment to maximize hold time
   var hoursUntilNextFunding = CalculateHoursUntilNextFunding();
   if (hoursUntilNextFunding > 6) score = 100; // Full cycle ahead
   else if (hoursUntilNextFunding > 3) score = 70;
   else score = 40; // Too close to next payment
   ```

3. **PositionWindowScore**: How long can we realistically hold?
   ```csharp
   var estimatedHoldTime = CalculateOptimalHoldTime();
   if (estimatedHoldTime >= BreakEvenTimeHours * 2) score = 100;
   else if (estimatedHoldTime >= BreakEvenTimeHours * 1.5) score = 80;
   else if (estimatedHoldTime >= BreakEvenTimeHours) score = 60;
   else score = 20; // Not enough time to break even
   ```

---

### E. Composite Score Calculation

```csharp
public decimal CalculateCompositeScore(
    decimal profitScore,
    decimal riskScore,
    decimal executionScore,
    decimal timingScore,
    ScoreWeights weights)
{
    var compositeScore =
        profitScore * weights.ProfitWeight +
        riskScore * weights.RiskWeight +
        executionScore * weights.ExecutionWeight +
        timingScore * weights.TimingWeight;

    return Math.Clamp(compositeScore, 0, 100);
}

// Configurable weights (sum to 1.0)
public class ScoreWeights
{
    public decimal ProfitWeight { get; set; } = 0.35m;
    public decimal RiskWeight { get; set; } = 0.30m;
    public decimal ExecutionWeight { get; set; } = 0.25m;
    public decimal TimingWeight { get; set; } = 0.10m;
}
```

---

## 2. Position Sizing Engine

### Hybrid Kelly Criterion + Risk-Adjusted Sizing

```csharp
public class PositionSizingEngine
{
    public decimal CalculateOptimalPositionSize(
        ArbitrageOpportunityDto opp,
        decimal availableCapital,
        List<ArbitrageOpportunityDto> activePositions)
    {
        // Step 1: Kelly Criterion base size
        var kellySuggestion = CalculateKellySize(opp);

        // Step 2: Risk-adjusted modifier
        var riskAdjustedSize = ApplyRiskAdjustment(
            kellySuggestion,
            opp.RiskScore,
            opp.ExecutionScore);

        // Step 3: Concentration limits
        var concentrationLimitedSize = ApplyConcentrationLimits(
            riskAdjustedSize,
            opp,
            activePositions,
            availableCapital);

        // Step 4: Correlation adjustment
        var finalSize = ApplyCorrelationAdjustment(
            concentrationLimitedSize,
            opp,
            activePositions);

        return Math.Clamp(
            finalSize,
            _config.MinPositionSizeUsd,
            _config.MaxPositionSizeUsd);
    }
}
```

### Kelly Criterion Implementation

```csharp
private decimal CalculateKellySize(ArbitrageOpportunityDto opp)
{
    // Kelly Formula: f = (bp - q) / b
    // f = fraction of capital to bet
    // b = odds (profit ratio)
    // p = probability of win
    // q = probability of loss (1-p)

    // For arbitrage: estimate win probability from historical data
    decimal winProbability = EstimateWinProbability(opp);

    // Profit ratio: expected return / risk
    decimal profitRatio = CalculateProfitRatio(opp);

    // Kelly fraction
    decimal kellyFraction =
        (profitRatio * winProbability - (1 - winProbability)) / profitRatio;

    // Use fractional Kelly (0.25 to 0.50 Kelly) for safety
    var conservativeKelly = kellyFraction * 0.35m;

    return availableCapital * Math.Max(0, conservativeKelly);
}

private decimal EstimateWinProbability(ArbitrageOpportunityDto opp)
{
    // Base probability from opportunity type
    decimal baseProbability = opp.SubType switch
    {
        StrategySubType.CrossExchangeFuturesFutures => 0.85m, // High probability
        StrategySubType.CrossExchangeFuturesPriceSpread => 0.70m, // Medium (execution risk)
        StrategySubType.SpotPerpetualSameExchange => 0.80m,
        _ => 0.75m
    };

    // Adjust based on metrics
    if (opp.LiquidityStatus == LiquidityStatus.Good) baseProbability += 0.05m;
    if (opp.LiquidityStatus == LiquidityStatus.Low) baseProbability -= 0.15m;

    // Adjust based on rate stability
    if (opp.FundProfit8h3dProj.HasValue && opp.FundProfit8h > 0)
    {
        var stability = opp.FundProfit8h / opp.FundProfit8h3dProj.Value;
        if (stability >= 0.8m && stability <= 1.2m)
            baseProbability += 0.05m; // Stable rate
        else if (stability < 0.6m || stability > 1.5m)
            baseProbability -= 0.10m; // Volatile rate
    }

    return Math.Clamp(baseProbability, 0.50m, 0.95m);
}

private decimal CalculateProfitRatio(ArbitrageOpportunityDto opp)
{
    // Expected return (use average of current and projected)
    var expectedReturn = opp.FundProfit8h;
    if (opp.FundProfit8h24hProj.HasValue)
        expectedReturn = (expectedReturn + opp.FundProfit8h24hProj.Value) / 2;

    // Risk: cost + potential slippage + spread volatility
    var costRisk = opp.PositionCostPercent;
    var slippageRisk = (opp.BidAskSpreadPercent ?? 0.1m) * 100m;
    var volatilityRisk = opp.SpreadVolatilityCv.HasValue
        ? opp.SpreadVolatilityCv.Value * 10m
        : 0.5m;

    var totalRisk = costRisk + slippageRisk + volatilityRisk;

    return expectedReturn / Math.Max(0.01m, totalRisk);
}
```

### Concentration Limits

```csharp
private decimal ApplyConcentrationLimits(
    decimal suggestedSize,
    ArbitrageOpportunityDto opp,
    List<ArbitrageOpportunityDto> activePositions,
    decimal availableCapital)
{
    // Rule 1: Max 20% of capital in single symbol
    var symbolExposure = activePositions
        .Where(p => p.Symbol == opp.Symbol)
        .Sum(p => p.PositionSize);

    var maxSymbolSize = (availableCapital * 0.20m) - symbolExposure;
    suggestedSize = Math.Min(suggestedSize, maxSymbolSize);

    // Rule 2: Max 30% of capital on single exchange
    var exchangeExposure = activePositions
        .Where(p => p.LongExchange == opp.LongExchange ||
                    p.ShortExchange == opp.LongExchange)
        .Sum(p => p.PositionSize);

    var maxExchangeSize = (availableCapital * 0.30m) - exchangeExposure;
    suggestedSize = Math.Min(suggestedSize, maxExchangeSize);

    // Rule 3: Max 0.4% of 24h volume
    var maxVolumeBasedSize = opp.Volume24h * 0.004m;
    suggestedSize = Math.Min(suggestedSize, maxVolumeBasedSize);

    return Math.Max(0, suggestedSize);
}
```

---

## 3. Hold Duration Optimizer

### Time-Based Exit Predictor

```csharp
public class HoldDurationCalculator
{
    public HoldDurationRecommendation CalculateOptimalHoldTime(
        ArbitrageOpportunityDto opp)
    {
        // Strategy 1: Break-even + safety margin
        var minHoldTime = opp.BreakEvenTimeHours * 1.5m; // 50% safety margin

        // Strategy 2: Rate trend analysis
        var trendBasedHoldTime = CalculateTrendBasedHoldTime(opp);

        // Strategy 3: Funding cycle optimization
        var fundingOptimalHoldTime = CalculateFundingCycleOptimal(opp);

        // Take conservative (longer) estimate
        var recommendedHoldTime = Math.Max(
            minHoldTime,
            Math.Max(trendBasedHoldTime, fundingOptimalHoldTime)
        );

        return new HoldDurationRecommendation
        {
            MinimumHoldHours = minHoldTime,
            RecommendedHoldHours = recommendedHoldTime,
            MaximumHoldHours = CalculateMaxHoldTime(opp),
            ConfidenceLevel = CalculateConfidence(opp)
        };
    }

    private decimal CalculateTrendBasedHoldTime(ArbitrageOpportunityDto opp)
    {
        // If rate is declining, suggest shorter hold
        if (opp.FundProfit8h > opp.FundProfit8h24hProj &&
            opp.FundProfit8h24hProj > opp.FundProfit8h3dProj)
        {
            // Uptrend: hold longer (3-5 days)
            return 72m; // 3 days
        }
        else if (opp.FundProfit8h < opp.FundProfit8h24hProj)
        {
            // Downtrend: exit after break-even + 1 cycle
            var interval = opp.LongFundingIntervalHours ?? 8;
            return opp.BreakEvenTimeHours + interval;
        }
        else
        {
            // Stable: hold for 1-2 days
            return 36m;
        }
    }

    private decimal CalculateFundingCycleOptimal(ArbitrageOpportunityDto opp)
    {
        // Optimize around funding intervals
        var longInterval = opp.LongFundingIntervalHours ?? 8;
        var shortInterval = opp.ShortFundingIntervalHours ?? 8;
        var avgInterval = (longInterval + shortInterval) / 2m;

        // Target 6-10 funding cycles for statistical significance
        return avgInterval * 8;
    }

    private decimal CalculateMaxHoldTime(ArbitrageOpportunityDto opp)
    {
        // Never hold longer than 7 days (rate environment changes)
        // Unless it's exceptionally stable
        if (opp.SpreadVolatilityCv.HasValue && opp.SpreadVolatilityCv.Value < 0.15m)
            return 168m; // 7 days for stable opportunities
        else
            return 120m; // 5 days maximum
    }
}
```

### Condition-Based Exit Monitor

```csharp
public class ExitConditionMonitor
{
    public ExitSignal EvaluateExitConditions(
        ArbitrageOpportunityDto opp,
        ActivePositionDto activePosition,
        MarketDataSnapshot currentMarket)
    {
        var signals = new List<ExitTrigger>();

        // Trigger 1: Rate deterioration
        var currentRate = GetCurrentFundingRate(opp, currentMarket);
        if (currentRate < opp.FundProfit8h * 0.5m)
        {
            signals.Add(new ExitTrigger
            {
                Type = "RateDeterioration",
                Severity = "High",
                Message = "Funding rate dropped below 50% of original"
            });
        }

        // Trigger 2: Liquidity degradation
        var currentLiquidity = GetCurrentLiquidity(opp, currentMarket);
        if (currentLiquidity.Status == LiquidityStatus.Low &&
            opp.LiquidityStatus == LiquidityStatus.Good)
        {
            signals.Add(new ExitTrigger
            {
                Type = "LiquidityDegradation",
                Severity = "Medium",
                Message = "Liquidity declined significantly"
            });
        }

        // Trigger 3: Better opportunity available
        var betterOpportunities = FindBetterOpportunities(
            activePosition,
            currentMarket);
        if (betterOpportunities.Any())
        {
            signals.Add(new ExitTrigger
            {
                Type = "BetterOpportunityAvailable",
                Severity = "Low",
                Message = $"Found {betterOpportunities.Count} better opportunities"
            });
        }

        // Trigger 4: Target profit reached
        var profitPercent = activePosition.RealizedProfitPercent;
        var targetProfit = activePosition.ExpectedProfitPercent * 0.80m; // 80% of target
        if (profitPercent >= targetProfit)
        {
            signals.Add(new ExitTrigger
            {
                Type = "TargetReached",
                Severity = "Info",
                Message = "Target profit achieved, consider taking profit"
            });
        }

        // Trigger 5: Maximum hold time exceeded
        var holdDuration = DateTime.UtcNow - activePosition.ExecutedAt;
        if (holdDuration.TotalHours > activePosition.MaximumHoldHours)
        {
            signals.Add(new ExitTrigger
            {
                Type = "MaxHoldTimeExceeded",
                Severity = "High",
                Message = "Maximum hold time exceeded"
            });
        }

        return new ExitSignal
        {
            ShouldExit = signals.Any(s => s.Severity == "High"),
            ShouldConsiderExit = signals.Any(s => s.Severity == "Medium"),
            Triggers = signals,
            RecommendedAction = DetermineRecommendedAction(signals)
        };
    }
}
```

---

## 4. New DTO Fields Required

Add to `ArbitrageOpportunityDto`:

```csharp
// Scoring fields
public decimal ProfitabilityScore { get; set; }
public decimal RiskScore { get; set; }
public decimal ExecutionQualityScore { get; set; }
public decimal TimingScore { get; set; }
public decimal CompositeScore { get; set; }

// Position sizing
public decimal RecommendedPositionSize { get; set; }
public decimal MinimumPositionSize { get; set; }
public decimal MaximumPositionSize { get; set; }
public decimal WinProbability { get; set; }

// Hold duration
public decimal? MinimumHoldHours { get; set; }
public decimal? RecommendedHoldHours { get; set; }
public decimal? MaximumHoldHours { get; set; }
public decimal? HoldConfidenceLevel { get; set; }

// Risk metrics
public decimal ConcentrationRisk { get; set; } // 0-1 scale
public decimal CorrelationRisk { get; set; } // 0-1 scale
public string[] RiskWarnings { get; set; } = Array.Empty<string>();

// For position tracking (if not in separate Position DTO)
public decimal PositionSize { get; set; }
public decimal RealizedProfitPercent { get; set; }
public decimal ExpectedProfitPercent { get; set; }
```

---

## 5. Configuration

Add to `ArbitrageConfig`:

```csharp
// Scoring weights (must sum to 1.0)
public decimal ProfitWeight { get; set; } = 0.35m;
public decimal RiskWeight { get; set; } = 0.30m;
public decimal ExecutionWeight { get; set; } = 0.25m;
public decimal TimingWeight { get; set; } = 0.10m;

// Kelly Criterion parameters
public decimal KellyFraction { get; set; } = 0.35m; // Conservative (35% of full Kelly)
public decimal MaxKellyPositionPercent { get; set; } = 15m; // Max 15% of capital per position

// Concentration limits
public decimal MaxSymbolExposurePercent { get; set; } = 20m; // Max 20% per symbol
public decimal MaxExchangeExposurePercent { get; set; } = 30m; // Max 30% per exchange
public decimal MaxVolumeImpactPercent { get; set; } = 0.4m; // Max 0.4% of 24h volume

// Hold duration parameters
public decimal MinSafetyMarginMultiplier { get; set; } = 1.5m; // 50% above break-even
public int MaxHoldDays { get; set; } = 7; // Never hold longer than 7 days
public int MinFundingCycles { get; set; } = 6; // Target minimum cycles

// Risk thresholds
public decimal MaxVolatilityCv { get; set; } = 0.5m; // Reject if CV > 0.5
public decimal MinStabilityScore { get; set; } = 50m; // Minimum rate stability
```

---

## 6. Implementation Services

### Service Structure

```
Services/
├── Scoring/
│   ├── OpportunityScoringService.cs
│   ├── ProfitabilityScorer.cs
│   ├── RiskScorer.cs
│   ├── ExecutionQualityScorer.cs
│   └── TimingScorer.cs
├── PositionSizing/
│   ├── PositionSizingService.cs
│   ├── KellyCriterionCalculator.cs
│   ├── RiskAdjustmentEngine.cs
│   └── ConcentrationLimiter.cs
├── HoldDuration/
│   ├── HoldDurationService.cs
│   ├── TimeBasedExitPredictor.cs
│   └── ExitConditionMonitor.cs (Background Service)
└── RiskAnalysis/
    ├── RiskAnalysisService.cs
    ├── VolatilityAnalyzer.cs
    ├── CorrelationAnalyzer.cs
    └── ConcentrationAnalyzer.cs
```

---

## 7. Integration Flow

```
Market Data Collection
    ↓
Opportunity Detection
    ↓
Opportunity Enrichment (Volume, Liquidity)
    ↓
[NEW] Opportunity Scoring ← Risk Analysis
    ↓
[NEW] Position Sizing ← Active Positions Context
    ↓
[NEW] Hold Duration Calculation
    ↓
Opportunity Ranking (by Composite Score)
    ↓
SignalR Broadcast to UI
    ↓
[Background] Exit Condition Monitoring
```

---

## 8. UI/Dashboard Enhancements

### Opportunity Card Additions:
- **Score Breakdown**: Visual breakdown of 4 score components
- **Position Recommendation**: Size + win probability + confidence
- **Hold Duration**: Min/Recommended/Max with timeline visual
- **Risk Indicators**: Warning badges for high-risk factors

### New Dashboard Widgets:
- **Portfolio Risk Dashboard**: Concentration, correlation, exposure metrics
- **Active Position Monitor**: Real-time exit signals and recommendations
- **Performance Analytics**: Track actual vs predicted profitability

---

## 9. Testing & Validation

### Unit Tests:
- Test each scoring component independently
- Validate Kelly Criterion calculations
- Test concentration limit enforcement
- Verify hold duration calculations

### Integration Tests:
- End-to-end scoring pipeline
- Position sizing with various portfolio states
- Exit condition triggering scenarios

### Backtesting:
- Score historical opportunities
- Compare predicted vs actual profitability
- Optimize weights based on historical performance
- Validate position sizing recommendations

---

## 10. Future Enhancements

1. **Adaptive Weights**: ML model to auto-adjust scoring weights based on market regime
2. **Portfolio Optimization**: Modern Portfolio Theory (MPT) for multi-opportunity selection
3. **Predictive Models**: Time-series forecasting for funding rate predictions
4. **Risk-Parity**: Size positions based on equal risk contribution
5. **Sentiment Analysis**: Incorporate market sentiment into timing scores
