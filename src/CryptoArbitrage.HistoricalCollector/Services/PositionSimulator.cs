using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using CryptoArbitrage.HistoricalCollector.Config;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Simulates position executions from historical snapshots
/// Generates training data for ML models using realistic exit strategies
/// </summary>
public class PositionSimulator
{
    private readonly ILogger<PositionSimulator> _logger;

    // Exit strategies to simulate (instead of fixed durations)
    private readonly List<ExitStrategyConfig> _exitStrategies;

    // Default position size for simulation
    private const decimal DEFAULT_POSITION_SIZE = 1000m;

    public PositionSimulator(ILogger<PositionSimulator> logger, List<ExitStrategyConfig>? customStrategies = null)
    {
        _logger = logger;
        _exitStrategies = customStrategies ?? ExitStrategyConfig.Presets.Recommended;
    }

    /// <summary>
    /// Simulate positions for all opportunities using realistic exit strategies
    /// </summary>
    public async Task<List<SimulatedExecution>> SimulateAllPositions(
        List<HistoricalMarketSnapshot> snapshots,
        List<ExitStrategyConfig>? customStrategies = null)
    {
        var strategies = customStrategies ?? _exitStrategies;

        _logger.LogInformation(
            "Starting position simulation for {SnapshotCount} snapshots with {StrategyCount} exit strategies",
            snapshots.Count, strategies.Count);

        var simulations = new List<SimulatedExecution>();

        if (!snapshots.Any())
        {
            _logger.LogWarning("No snapshots provided for simulation");
            return simulations;
        }

        var firstSnapshotTime = snapshots.First().Timestamp;
        var lastSnapshotTime = snapshots.Last().Timestamp;
        var totalTimeCoverage = (lastSnapshotTime - firstSnapshotTime).TotalHours;

        _logger.LogInformation(
            "Time coverage: {TotalHours:F1}h ({Start} to {End})",
            totalTimeCoverage, firstSnapshotTime.ToString("yyyy-MM-dd HH:mm"),
            lastSnapshotTime.ToString("yyyy-MM-dd HH:mm"));

        _logger.LogInformation("Exit strategies: {Strategies}",
            string.Join(", ", strategies.Select(s => s.Name)));

        // Track per-strategy statistics
        var strategyStats = strategies.ToDictionary(s => s.Name, s => new StrategyStats());

        // Process ALL snapshots
        for (int i = 0; i < snapshots.Count; i++)
        {
            var entrySnapshot = snapshots[i];

            // For each opportunity detected at this time
            foreach (var opportunity in entrySnapshot.Opportunities)
            {
                // Simulate each exit strategy
                foreach (var strategy in strategies)
                {
                    // Check if we have enough future data for this strategy's max hold time
                    var requiredExitTime = entrySnapshot.Timestamp.AddHours((double)strategy.MaxHoldHours);
                    if (requiredExitTime > lastSnapshotTime)
                    {
                        // Not enough future data for this strategy
                        strategyStats[strategy.Name].Skipped++;
                        continue;
                    }

                    try
                    {
                        var simulation = SimulatePositionWithStrategy(
                            opportunity,
                            snapshots,
                            i,
                            strategy);

                        if (simulation != null)
                        {
                            simulations.Add(simulation);
                            var stats = strategyStats[strategy.Name];
                            stats.Total++;

                            if (simulation.WasProfitable)
                                stats.Profitable++;

                            if (simulation.HitProfitTarget)
                                stats.HitProfitTarget++;

                            if (simulation.HitStopLoss)
                                stats.HitStopLoss++;

                            // Track exit reasons
                            if (!stats.ExitReasons.ContainsKey(simulation.ExitReason))
                                stats.ExitReasons[simulation.ExitReason] = 0;
                            stats.ExitReasons[simulation.ExitReason]++;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex,
                            "Failed to simulate position for {Symbol} at {Time} with {Strategy} strategy",
                            opportunity.Symbol, entrySnapshot.Timestamp, strategy.Name);
                    }
                }
            }

            if ((i + 1) % 100 == 0 || i == snapshots.Count - 1)
            {
                var percentComplete = ((i + 1) * 100.0) / snapshots.Count;
                _logger.LogInformation(
                    "Progress: {Percent:F1}% - Processed {Current}/{Total} snapshots ({Simulations} simulations generated)",
                    percentComplete, i + 1, snapshots.Count, simulations.Count);
            }
        }

        var profitableCount = simulations.Count(s => s.WasProfitable);
        var profitablePercent = simulations.Any() ? (profitableCount * 100.0 / simulations.Count) : 0;

        _logger.LogInformation(
            "Simulation complete: {Total} simulations generated, {Profitable} profitable ({Percent:F1}%)",
            simulations.Count, profitableCount, profitablePercent);

        // Log per-strategy statistics
        _logger.LogInformation("\nResults by strategy:");
        foreach (var strategy in strategies)
        {
            var stats = strategyStats[strategy.Name];
            var winRate = stats.Total > 0 ? (stats.Profitable * 100.0 / stats.Total) : 0;
            var targetHitRate = stats.Total > 0 ? (stats.HitProfitTarget * 100.0 / stats.Total) : 0;
            var stopLossRate = stats.Total > 0 ? (stats.HitStopLoss * 100.0 / stats.Total) : 0;

            _logger.LogInformation(
                "  {Strategy}: {Count} simulations ({Skipped} skipped)",
                strategy.Name, stats.Total, stats.Skipped);
            _logger.LogInformation(
                "    Win rate: {WinRate:F1}%, Profit target: {TargetRate:F1}%, Stop loss: {StopRate:F1}%",
                winRate, targetHitRate, stopLossRate);

            // Log exit reason distribution
            if (stats.ExitReasons.Any())
            {
                var reasonStr = string.Join(", ", stats.ExitReasons
                    .OrderByDescending(kvp => kvp.Value)
                    .Select(kvp => $"{kvp.Key}: {kvp.Value}"));
                _logger.LogInformation("    Exit reasons: {Reasons}", reasonStr);
            }
        }

        if (simulations.Any())
        {
            var avgProfit = simulations.Average(s => s.ActualProfitPercent);
            _logger.LogInformation(
                "\nOverall: Average profit: {AvgProfit:F3}%, Min: {Min:F3}%, Max: {Max:F3}%",
                avgProfit,
                simulations.Min(s => s.ActualProfitPercent),
                simulations.Max(s => s.ActualProfitPercent));
        }

        return simulations;
    }

    private class StrategyStats
    {
        public int Total { get; set; }
        public int Skipped { get; set; }
        public int Profitable { get; set; }
        public int HitProfitTarget { get; set; }
        public int HitStopLoss { get; set; }
        public Dictionary<string, int> ExitReasons { get; set; } = new();
    }

    /// <summary>
    /// Simulate a single position using a specific exit strategy
    /// </summary>
    private SimulatedExecution? SimulatePositionWithStrategy(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        ExitStrategyConfig strategy)
    {
        var entrySnapshot = snapshots[entryIndex];

        // Calculate entry prices with slippage
        var entryPrices = CalculateEntryPrices(opportunity, entrySnapshot);

        // Store entry funding differential for reversal detection
        var entryFundingDifferential = Math.Abs(opportunity.ShortFundingRate - opportunity.LongFundingRate);
        var entryVolatility = opportunity.SpreadVolatilityCv ?? 0.1m;

        // Determine optimal exit based on strategy
        var exitResult = DetermineOptimalExit(
            opportunity,
            snapshots,
            entryIndex,
            entryPrices,
            strategy,
            entryFundingDifferential,
            entryVolatility);

        if (exitResult == null)
            return null;

        var exitSnapshot = snapshots[exitResult.ExitSnapshotIndex];

        // Calculate exit prices with slippage
        var exitPrices = CalculateExitPrices(opportunity, exitSnapshot);

        // Simulate funding payments during hold period
        var fundingPayments = SimulateFundingPayments(
            opportunity,
            snapshots,
            entryIndex,
            exitResult.ExitSnapshotIndex);

        // Calculate total PnL
        var pnl = CalculatePnL(
            opportunity,
            entryPrices,
            exitPrices,
            fundingPayments,
            DEFAULT_POSITION_SIZE);

        // Calculate total fees
        var totalFeesUsd = DEFAULT_POSITION_SIZE * (opportunity.PositionCostPercent / 100);

        // Build simulation result
        return new SimulatedExecution
        {
            // Snapshot
            OpportunitySnapshotJson = JsonSerializer.Serialize(opportunity),
            EntryTime = entrySnapshot.Timestamp,
            ExitTime = exitSnapshot.Timestamp,
            HourOfDay = entrySnapshot.Timestamp.Hour,
            DayOfWeek = (int)entrySnapshot.Timestamp.DayOfWeek,

            // Market context
            BtcPriceAtEntry = entrySnapshot.BtcPrice,
            MarketRegimeAtEntry = entrySnapshot.MarketRegime,

            // Opportunity features
            Symbol = opportunity.Symbol,
            Strategy = opportunity.SubType.ToString(),
            LongExchange = opportunity.LongExchange,
            ShortExchange = opportunity.ShortExchange,

            // Funding rate details
            LongFundingRate = opportunity.LongFundingRate,
            ShortFundingRate = opportunity.ShortFundingRate,
            LongFundingIntervalHours = opportunity.LongFundingIntervalHours,
            ShortFundingIntervalHours = opportunity.ShortFundingIntervalHours,
            LongNextFundingTimeMinutes = opportunity.LongNextFundingTime.HasValue
                ? (decimal?)(opportunity.LongNextFundingTime.Value - entrySnapshot.Timestamp).TotalMinutes
                : null,
            ShortNextFundingTimeMinutes = opportunity.ShortNextFundingTime.HasValue
                ? (decimal?)(opportunity.ShortNextFundingTime.Value - entrySnapshot.Timestamp).TotalMinutes
                : null,

            // Price spread
            CurrentPriceSpreadPercent = opportunity.CurrentPriceSpreadPercent,

            // Profitability metrics
            FundProfit8h = opportunity.FundProfit8h,
            FundApr = opportunity.FundApr,
            FundProfit8h24hProj = opportunity.FundProfit8h24hProj,
            FundApr24hProj = opportunity.FundApr24hProj,
            FundBreakEvenTime24hProj = opportunity.FundBreakEvenTime24hProj,
            FundProfit8h3dProj = opportunity.FundProfit8h3dProj,
            FundApr3dProj = opportunity.FundApr3dProj,
            FundBreakEvenTime3dProj = opportunity.FundBreakEvenTime3dProj,
            BreakEvenTimeHours = opportunity.BreakEvenTimeHours,

            // Price spread statistics
            PriceSpread24hAvg = opportunity.PriceSpread24hAvg,
            PriceSpread3dAvg = opportunity.PriceSpread3dAvg,

            // Risk metrics
            SpreadVolatilityCv = opportunity.SpreadVolatilityCv,
            SpreadVolatilityStdDev = opportunity.SpreadVolatilityStdDev,
            Spread30SampleAvg = opportunity.Spread30SampleAvg,

            Volume24h = opportunity.Volume24h,
            LongVolume24h = opportunity.LongVolume24h,
            ShortVolume24h = opportunity.ShortVolume24h,
            BidAskSpreadPercent = opportunity.BidAskSpreadPercent,
            OrderbookDepthUsd = opportunity.OrderbookDepthUsd,
            LiquidityStatus = opportunity.LiquidityStatus?.ToString(),

            PositionCostPercent = opportunity.PositionCostPercent,
            PositionSizeUsd = DEFAULT_POSITION_SIZE,

            // Target variables (from strategy)
            StrategyName = strategy.Name,
            ExitReason = exitResult.Reason.ToString(),
            ActualHoldHours = exitResult.HoursHeld,
            ActualProfitPercent = pnl.TotalProfitPercent,
            ActualProfitUsd = pnl.TotalProfitUsd,
            WasProfitable = pnl.TotalProfitPercent > 0,
            HitProfitTarget = exitResult.HitProfitTarget,
            HitStopLoss = exitResult.HitStopLoss,

            // Performance metrics
            PeakUnrealizedProfitPercent = exitResult.PeakProfitPercent,
            MaxDrawdownPercent = exitResult.MaxDrawdownPercent,
            FundingPaymentsCount = fundingPayments.Count,
            TotalFundingEarnedUsd = fundingPayments.Sum(f => f.Amount),

            // Execution quality (slippage is already included in entry/exit prices)
            TotalFeesUsd = totalFeesUsd,

            // Prices (already include slippage - long entry has +slippage, exit has -slippage, etc.)
            EntryLongPrice = entryPrices.LongPrice,
            EntryShortPrice = entryPrices.ShortPrice,
            ExitLongPrice = exitPrices.LongPrice,
            ExitShortPrice = exitPrices.ShortPrice
        };
    }

    /// <summary>
    /// Determine optimal exit point based on strategy rules
    /// Monitors position continuously and exits when conditions are met
    /// </summary>
    private ExitResult? DetermineOptimalExit(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) entryPrices,
        ExitStrategyConfig strategy,
        decimal entryFundingDifferential,
        decimal entryVolatility)
    {
        var entryTime = snapshots[entryIndex].Timestamp;
        var maxExitTime = entryTime.AddHours((double)strategy.MaxHoldHours);
        var minExitTime = entryTime.AddHours((double)strategy.MinHoldHours);

        decimal peakProfit = 0m;
        decimal maxDrawdown = 0m;
        decimal trailingStopTriggered = decimal.MinValue;
        bool trailingStopActive = false;

        // Sample at intervals (e.g., every 30 minutes)
        var sampleIntervalHours = strategy.SampleIntervalHours;
        var currentTime = entryTime.AddHours((double)sampleIntervalHours);

        while (currentTime <= maxExitTime)
        {
            // Find snapshot closest to current time
            var snapshotIndex = FindSnapshotIndexClosestTo(snapshots, currentTime, entryIndex);
            if (snapshotIndex == -1 || snapshotIndex >= snapshots.Count)
            {
                // Insufficient data
                return new ExitResult
                {
                    ExitSnapshotIndex = snapshots.Count - 1,
                    ExitTime = snapshots.Last().Timestamp,
                    Reason = ExitReason.INSUFFICIENT_DATA,
                    ProfitPercent = 0,
                    PeakProfitPercent = peakProfit,
                    MaxDrawdownPercent = maxDrawdown,
                    HoursHeld = (decimal)(snapshots.Last().Timestamp - entryTime).TotalHours,
                    HitProfitTarget = false,
                    HitStopLoss = false
                };
            }

            var snapshot = snapshots[snapshotIndex];
            var hoursHeld = (decimal)(snapshot.Timestamp - entryTime).TotalHours;

            // Calculate current unrealized P&L
            var currentLongPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol);
            var currentShortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol);

            if (currentLongPrice == 0 || currentShortPrice == 0)
            {
                currentTime = currentTime.AddHours((double)sampleIntervalHours);
                continue;
            }

            // Unrealized PnL (percentage)
            var longPnl = (currentLongPrice - entryPrices.LongPrice) / entryPrices.LongPrice;
            var shortPnl = (entryPrices.ShortPrice - currentShortPrice) / entryPrices.ShortPrice;
            var unrealizedPnl = ((longPnl + shortPnl) / 2) * 100;

            // Update peak and drawdown
            peakProfit = Math.Max(peakProfit, unrealizedPnl);
            maxDrawdown = Math.Min(maxDrawdown, unrealizedPnl);

            // Only check exit conditions after minimum hold time
            if (snapshot.Timestamp >= minExitTime)
            {
                // 1. STOP LOSS (highest priority)
                if (strategy.StopLossPercent.HasValue && unrealizedPnl <= strategy.StopLossPercent.Value)
                {
                    return new ExitResult
                    {
                        ExitSnapshotIndex = snapshotIndex,
                        ExitTime = snapshot.Timestamp,
                        Reason = ExitReason.STOP_LOSS,
                        ProfitPercent = unrealizedPnl,
                        PeakProfitPercent = peakProfit,
                        MaxDrawdownPercent = maxDrawdown,
                        HoursHeld = hoursHeld,
                        HitProfitTarget = false,
                        HitStopLoss = true
                    };
                }

                // 2. VOLATILITY SPIKE EXIT
                if (strategy.VolatilitySpikeMultiplier.HasValue)
                {
                    var currentVolatility = opportunity.SpreadVolatilityCv ?? entryVolatility;
                    if (currentVolatility > entryVolatility * strategy.VolatilitySpikeMultiplier.Value)
                    {
                        return new ExitResult
                        {
                            ExitSnapshotIndex = snapshotIndex,
                            ExitTime = snapshot.Timestamp,
                            Reason = ExitReason.VOLATILITY_SPIKE,
                            ProfitPercent = unrealizedPnl,
                            PeakProfitPercent = peakProfit,
                            MaxDrawdownPercent = maxDrawdown,
                            HoursHeld = hoursHeld,
                            HitProfitTarget = false,
                            HitStopLoss = false
                        };
                    }
                }

                // 3. PROFIT TARGET
                if (strategy.ProfitTargetPercent.HasValue && unrealizedPnl >= strategy.ProfitTargetPercent.Value)
                {
                    return new ExitResult
                    {
                        ExitSnapshotIndex = snapshotIndex,
                        ExitTime = snapshot.Timestamp,
                        Reason = ExitReason.PROFIT_TARGET,
                        ProfitPercent = unrealizedPnl,
                        PeakProfitPercent = peakProfit,
                        MaxDrawdownPercent = maxDrawdown,
                        HoursHeld = hoursHeld,
                        HitProfitTarget = true,
                        HitStopLoss = false
                    };
                }

                // 4. TRAILING STOP
                if (strategy.TrailingStopPercent.HasValue && strategy.TrailingStopActivationPercent.HasValue)
                {
                    // Activate trailing stop once profit threshold is hit
                    if (!trailingStopActive && unrealizedPnl >= strategy.TrailingStopActivationPercent.Value)
                    {
                        trailingStopActive = true;
                        trailingStopTriggered = unrealizedPnl - strategy.TrailingStopPercent.Value;
                    }

                    // Update trailing stop level as profit increases
                    if (trailingStopActive)
                    {
                        var newTrailingLevel = unrealizedPnl - strategy.TrailingStopPercent.Value;
                        trailingStopTriggered = Math.Max(trailingStopTriggered, newTrailingLevel);

                        // Exit if profit drops below trailing stop
                        if (unrealizedPnl <= trailingStopTriggered)
                        {
                            return new ExitResult
                            {
                                ExitSnapshotIndex = snapshotIndex,
                                ExitTime = snapshot.Timestamp,
                                Reason = ExitReason.TRAILING_STOP,
                                ProfitPercent = unrealizedPnl,
                                PeakProfitPercent = peakProfit,
                                MaxDrawdownPercent = maxDrawdown,
                                HoursHeld = hoursHeld,
                                HitProfitTarget = peakProfit >= (strategy.ProfitTargetPercent ?? 0),
                                HitStopLoss = false
                            };
                        }
                    }
                }

                // 5. FUNDING REVERSAL
                if (strategy.FundingReversalThreshold.HasValue)
                {
                    // Get current funding rates
                    var (currentLongRate, currentShortRate) = GetFundingRatesAt(snapshot, opportunity);
                    var currentDifferential = Math.Abs(currentShortRate - currentLongRate);

                    // Exit if differential dropped significantly
                    if (entryFundingDifferential > 0 &&
                        currentDifferential < entryFundingDifferential * strategy.FundingReversalThreshold.Value)
                    {
                        return new ExitResult
                        {
                            ExitSnapshotIndex = snapshotIndex,
                            ExitTime = snapshot.Timestamp,
                            Reason = ExitReason.FUNDING_REVERSAL,
                            ProfitPercent = unrealizedPnl,
                            PeakProfitPercent = peakProfit,
                            MaxDrawdownPercent = maxDrawdown,
                            HoursHeld = hoursHeld,
                            HitProfitTarget = unrealizedPnl >= (strategy.ProfitTargetPercent ?? 0),
                            HitStopLoss = false
                        };
                    }
                }
            }

            // Move to next sample time
            currentTime = currentTime.AddHours((double)sampleIntervalHours);
        }

        // 6. MAX HOLD TIME REACHED
        var finalIndex = FindSnapshotIndexClosestTo(snapshots, maxExitTime, entryIndex);
        if (finalIndex == -1)
            return null;

        var finalSnapshot = snapshots[finalIndex];
        var finalLongPrice = GetPrice(finalSnapshot, opportunity.LongExchange, opportunity.Symbol);
        var finalShortPrice = GetPrice(finalSnapshot, opportunity.ShortExchange, opportunity.Symbol);

        var finalLongPnl = (finalLongPrice - entryPrices.LongPrice) / entryPrices.LongPrice;
        var finalShortPnl = (entryPrices.ShortPrice - finalShortPrice) / entryPrices.ShortPrice;
        var finalPnl = ((finalLongPnl + finalShortPnl) / 2) * 100;

        return new ExitResult
        {
            ExitSnapshotIndex = finalIndex,
            ExitTime = finalSnapshot.Timestamp,
            Reason = ExitReason.MAX_HOLD_TIME,
            ProfitPercent = finalPnl,
            PeakProfitPercent = Math.Max(peakProfit, finalPnl),
            MaxDrawdownPercent = Math.Min(maxDrawdown, finalPnl),
            HoursHeld = (decimal)(finalSnapshot.Timestamp - entryTime).TotalHours,
            HitProfitTarget = peakProfit >= (strategy.ProfitTargetPercent ?? 0),
            HitStopLoss = false
        };
    }

    private int FindSnapshotIndexClosestTo(
        List<HistoricalMarketSnapshot> snapshots,
        DateTime targetTime,
        int startIndex)
    {
        for (int i = startIndex; i < snapshots.Count; i++)
        {
            if (snapshots[i].Timestamp >= targetTime)
                return i;
        }

        return -1; // Not enough data
    }

    private (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) CalculateEntryPrices(
        ArbitrageOpportunityDto opportunity,
        HistoricalMarketSnapshot snapshot)
    {
        // Get current market prices
        var longPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol);
        var shortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol);

        // Estimate slippage based on volume and liquidity
        var longSlippage = EstimateSlippage(
            DEFAULT_POSITION_SIZE,
            opportunity.LongVolume24h ?? opportunity.Volume24h,
            opportunity.BidAskSpreadPercent ?? 0.01m);

        var shortSlippage = EstimateSlippage(
            DEFAULT_POSITION_SIZE,
            opportunity.ShortVolume24h ?? opportunity.Volume24h,
            opportunity.BidAskSpreadPercent ?? 0.01m);

        // Long position: BUY (pay ask price + slippage)
        var longEntryPrice = longPrice * (1 + longSlippage);

        // Short position: SELL (receive bid price - slippage)
        var shortEntryPrice = shortPrice * (1 - shortSlippage);

        return (longEntryPrice, shortEntryPrice, longSlippage * 100, shortSlippage * 100);
    }

    private (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) CalculateExitPrices(
        ArbitrageOpportunityDto opportunity,
        HistoricalMarketSnapshot snapshot)
    {
        var longPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol);
        var shortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol);

        var longSlippage = EstimateSlippage(
            DEFAULT_POSITION_SIZE,
            opportunity.LongVolume24h ?? opportunity.Volume24h,
            opportunity.BidAskSpreadPercent ?? 0.01m);

        var shortSlippage = EstimateSlippage(
            DEFAULT_POSITION_SIZE,
            opportunity.ShortVolume24h ?? opportunity.Volume24h,
            opportunity.BidAskSpreadPercent ?? 0.01m);

        // Long position: SELL (receive bid price - slippage)
        var longExitPrice = longPrice * (1 - longSlippage);

        // Short position: BUY (pay ask price + slippage)
        var shortExitPrice = shortPrice * (1 + shortSlippage);

        return (longExitPrice, shortExitPrice, longSlippage * 100, shortSlippage * 100);
    }

    private decimal EstimateSlippage(
        decimal positionSize,
        decimal volume24h,
        decimal bidAskSpreadPercent)
    {
        // Base slippage = half of bid-ask spread
        var baseSlippage = bidAskSpreadPercent / 2;

        // Estimate orderbook depth as 0.1% of 24h volume
        var estimatedDepth = volume24h * 0.001m;

        if (estimatedDepth == 0)
            return baseSlippage * 5; // High slippage if no volume data

        // Additional slippage based on position size vs orderbook depth
        var depthRatio = positionSize / estimatedDepth;

        if (depthRatio < 0.01m) // <1% of depth
            return baseSlippage;
        else if (depthRatio < 0.05m) // 1-5%
            return baseSlippage * 1.5m;
        else if (depthRatio < 0.10m) // 5-10%
            return baseSlippage * 2.5m;
        else // >10% - significant market impact
            return baseSlippage * 5m;
    }

    private List<FundingPayment> SimulateFundingPayments(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        int exitIndex)
    {
        var payments = new List<FundingPayment>();

        var longInterval = opportunity.LongFundingIntervalHours ?? 8;
        var shortInterval = opportunity.ShortFundingIntervalHours ?? 8;

        var entryTime = snapshots[entryIndex].Timestamp;
        var exitTime = snapshots[exitIndex].Timestamp;

        // Calculate funding times (assuming 00:00, 08:00, 16:00 UTC for 8h intervals)
        var fundingTimes = CalculateFundingTimes(entryTime, exitTime, Math.Min(longInterval, shortInterval));

        foreach (var fundingTime in fundingTimes)
        {
            // Find snapshot closest to funding time
            var snapshotIndex = FindClosestSnapshotIndex(snapshots, fundingTime, entryIndex, exitIndex);
            if (snapshotIndex == -1)
                continue;

            var snapshot = snapshots[snapshotIndex];

            // Get funding rates at this time
            var (longRate, shortRate) = GetFundingRatesAt(snapshot, opportunity);

            // Calculate funding payment
            // Long position: we PAY if rate is positive, RECEIVE if negative
            // Short position: we RECEIVE if rate is positive, PAY if negative
            var netRate = shortRate - longRate;
            var netFunding = netRate * DEFAULT_POSITION_SIZE;

            payments.Add(new FundingPayment
            {
                Timestamp = fundingTime,
                LongRate = longRate,
                ShortRate = shortRate,
                NetRate = netRate,
                Amount = netFunding
            });
        }

        return payments;
    }

    private List<DateTime> CalculateFundingTimes(DateTime start, DateTime end, int intervalHours)
    {
        var times = new List<DateTime>();
        var current = start;

        // Round to next funding time
        var hoursToAdd = intervalHours - (current.Hour % intervalHours);
        current = current.Date.AddHours(current.Hour + hoursToAdd);

        while (current <= end)
        {
            times.Add(current);
            current = current.AddHours(intervalHours);
        }

        return times;
    }

    private (decimal longRate, decimal shortRate) GetFundingRatesAt(
        HistoricalMarketSnapshot snapshot,
        ArbitrageOpportunityDto opportunity)
    {
        var longRate = 0m;
        var shortRate = 0m;

        if (snapshot.FundingRates.ContainsKey(opportunity.LongExchange))
        {
            var rate = snapshot.FundingRates[opportunity.LongExchange]
                .FirstOrDefault(r => r.Symbol == opportunity.Symbol);
            longRate = rate?.Rate ?? 0;
        }

        if (snapshot.FundingRates.ContainsKey(opportunity.ShortExchange))
        {
            var rate = snapshot.FundingRates[opportunity.ShortExchange]
                .FirstOrDefault(r => r.Symbol == opportunity.Symbol);
            shortRate = rate?.Rate ?? 0;
        }

        return (longRate, shortRate);
    }


    private (decimal TotalProfitPercent, decimal TotalProfitUsd) CalculatePnL(
        ArbitrageOpportunityDto opportunity,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) entryPrices,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) exitPrices,
        List<FundingPayment> fundingPayments,
        decimal positionSize)
    {
        // Long position PnL (BUY low, SELL high)
        // Entry: paid ask + slippage, Exit: received bid - slippage
        var longPnl = (exitPrices.LongPrice - entryPrices.LongPrice) / entryPrices.LongPrice;

        // Short position PnL (SELL high, BUY low)
        // Entry: received bid - slippage, Exit: paid ask + slippage
        var shortPnl = (entryPrices.ShortPrice - exitPrices.ShortPrice) / entryPrices.ShortPrice;

        // Price PnL (average of long and short)
        // For neutral arbitrage, this should be near zero as price movements cancel out
        // Any non-zero value is due to price spread changes + slippage impact
        var pricePnl = (longPnl + shortPnl) / 2;

        // Funding PnL (net funding received from short minus funding paid on long)
        var fundingPnl = fundingPayments.Sum(f => f.Amount) / positionSize;

        // Total PnL before fees
        var totalPnlBeforeFees = pricePnl + fundingPnl;

        // Total PnL after fees (position cost includes entry + exit trading fees)
        // totalPnlBeforeFees is in decimal (0.03 = 3%), multiply by 100 to get percent
        // Then subtract position cost percent (e.g., 0.2% for 0.1% entry + 0.1% exit)
        var totalPnlPercent = (totalPnlBeforeFees * 100) - opportunity.PositionCostPercent;
        var totalPnlUsd = (totalPnlPercent / 100) * positionSize;

        return (totalPnlPercent, totalPnlUsd);
    }

    private decimal GetPrice(HistoricalMarketSnapshot snapshot, string exchange, string symbol)
    {
        if (snapshot.PerpPrices.ContainsKey(exchange) &&
            snapshot.PerpPrices[exchange].ContainsKey(symbol))
        {
            return snapshot.PerpPrices[exchange][symbol].Price;
        }

        return 0;
    }

    private int FindClosestSnapshotIndex(
        List<HistoricalMarketSnapshot> snapshots,
        DateTime targetTime,
        int startIndex,
        int endIndex)
    {
        for (int i = startIndex; i <= endIndex; i++)
        {
            if (snapshots[i].Timestamp >= targetTime)
                return i;
        }

        return -1;
    }

    private class FundingPayment
    {
        public DateTime Timestamp { get; set; }
        public decimal LongRate { get; set; }
        public decimal ShortRate { get; set; }
        public decimal NetRate { get; set; }
        public decimal Amount { get; set; }
    }
}
