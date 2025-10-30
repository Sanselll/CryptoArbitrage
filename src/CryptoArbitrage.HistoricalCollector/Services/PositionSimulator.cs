using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using CryptoArbitrage.HistoricalCollector.Config;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Concurrent;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Simulates position executions from historical snapshots
/// Generates training data for ML models using realistic exit strategies
/// </summary>
public class PositionSimulator
{
    private readonly ILogger<PositionSimulator> _logger;
    private readonly SnapshotGenerator _snapshotGenerator;

    // Exit strategies to simulate (instead of fixed durations)
    private readonly List<ExitStrategyConfig> _exitStrategies;

    // Default position size for simulation
    private const decimal DEFAULT_POSITION_SIZE = 1000m;

    // Counter for unique execution IDs
    private int _executionIdCounter = 0;

    // PERFORMANCE OPTIMIZATION: Thread-safe cache for indexed snapshot lookups (for parallel processing)
    private readonly ConcurrentDictionary<int, SnapshotCache> _snapshotCaches = new();

    /// <summary>
    /// Performance optimization: Pre-indexed cache for fast snapshot data lookups
    /// Replaces O(n) linear searches with O(1) dictionary lookups
    /// </summary>
    private class SnapshotCache
    {
        private readonly Dictionary<string, Dictionary<string, decimal>> _fundingRateCache;
        private readonly Dictionary<(string exchange, string symbol), decimal> _priceCache;

        public SnapshotCache(HistoricalMarketSnapshot snapshot)
        {
            // Pre-index funding rates by exchange+symbol for O(1) lookup
            _fundingRateCache = new Dictionary<string, Dictionary<string, decimal>>();
            foreach (var kvp in snapshot.FundingRates)
            {
                var dict = new Dictionary<string, decimal>();
                foreach (var rate in kvp.Value)
                {
                    dict[rate.Symbol] = rate.Rate;
                }
                _fundingRateCache[kvp.Key] = dict;
            }

            // Pre-index prices by (exchange, symbol) for O(1) lookup
            _priceCache = new Dictionary<(string, string), decimal>();
            foreach (var exchangeKvp in snapshot.PerpPrices)
            {
                foreach (var symbolKvp in exchangeKvp.Value)
                {
                    _priceCache[(exchangeKvp.Key, symbolKvp.Key)] = symbolKvp.Value.Price;
                }
            }
        }

        public decimal GetFundingRate(string exchange, string symbol)
        {
            if (_fundingRateCache.TryGetValue(exchange, out var dict) &&
                dict.TryGetValue(symbol, out var rate))
            {
                return rate;
            }
            return 0;
        }

        public decimal GetPrice(string exchange, string symbol)
        {
            if (_priceCache.TryGetValue((exchange, symbol), out var price))
            {
                return price;
            }
            return 0;
        }
    }

    public PositionSimulator(ILogger<PositionSimulator> logger, List<ExitStrategyConfig>? customStrategies = null)
    {
        _logger = logger;
        _exitStrategies = customStrategies ?? ExitStrategyConfig.Presets.Recommended;

        // Create snapshot generator
        var snapshotLoggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        var snapshotLogger = snapshotLoggerFactory.CreateLogger<SnapshotGenerator>();
        _snapshotGenerator = new SnapshotGenerator(snapshotLogger);
    }

    /// <summary>
    /// PERFORMANCE OPTIMIZATION: Calculate adaptive sampling interval based on volatility
    /// High volatility needs more frequent samples, low volatility can use fewer samples
    /// </summary>
    private decimal GetAdaptiveSampleInterval(ArbitrageOpportunityDto opportunity)
    {
        var spreadVolatility = opportunity.SpreadVolatilityCv ?? 0.03m;

        // Adaptive intervals based on volatility:
        // - High volatility (>5%): Sample every 15 minutes to catch rapid changes
        // - Medium volatility (2-5%): Sample every 30 minutes
        // - Low volatility (<2%): Sample every 60 minutes to save computation

        if (spreadVolatility > 0.05m)
        {
            return 0.25m; // 15 minutes for high volatility
        }
        else if (spreadVolatility > 0.02m)
        {
            return 0.5m; // 30 minutes for medium volatility
        }
        else
        {
            return 1.0m; // 60 minutes for low volatility
        }
    }

    /// <summary>
    /// Simulate positions for all opportunities using optimal hindsight
    /// </summary>
    public async Task<List<SimulatedExecution>> SimulateAllPositions(
        List<HistoricalMarketSnapshot> snapshots,
        List<ExitStrategyConfig>? customStrategies = null)
    {
        _logger.LogInformation(
            "Starting optimal hindsight position simulation for {SnapshotCount} snapshots",
            snapshots.Count);

        // PERFORMANCE OPTIMIZATION: Use thread-safe collection for parallel processing
        var simulations = new ConcurrentBag<SimulatedExecution>();

        if (!snapshots.Any())
        {
            _logger.LogWarning("No snapshots provided for simulation");
            return simulations.ToList();
        }

        var firstSnapshotTime = snapshots.First().Timestamp;
        var lastSnapshotTime = snapshots.Last().Timestamp;
        var totalTimeCoverage = (lastSnapshotTime - firstSnapshotTime).TotalHours;

        _logger.LogInformation(
            "Time coverage: {TotalHours:F1}h ({Start} to {End})",
            totalTimeCoverage, firstSnapshotTime.ToString("yyyy-MM-dd HH:mm"),
            lastSnapshotTime.ToString("yyyy-MM-dd HH:mm"));

        _logger.LogInformation("Using optimal hindsight simulation (adaptive checkpoints, 72h max hold)");
        _logger.LogInformation("PARALLEL PROCESSING: Using {ProcessorCount} cores", Environment.ProcessorCount);

        // PERFORMANCE OPTIMIZATION: Pre-filter snapshots to only include those with enough future data
        // This avoids processing snapshots that will definitely be skipped (need at least 5 minutes of future data)
        var validSnapshotIndices = new List<int>();
        for (int i = 0; i < snapshots.Count; i++)
        {
            var minRequiredExitTime = snapshots[i].Timestamp.AddMinutes(5);
            if (minRequiredExitTime <= lastSnapshotTime)
            {
                validSnapshotIndices.Add(i);
            }
        }

        _logger.LogInformation(
            "Pre-filtered snapshots: {ValidCount}/{TotalCount} snapshots have sufficient future data",
            validSnapshotIndices.Count, snapshots.Count);

        // Track statistics (thread-safe counters)
        var totalOpportunities = 0;
        var processed = 0;

        // PERFORMANCE OPTIMIZATION: Process only valid snapshots in parallel
        Parallel.For(0, validSnapshotIndices.Count, new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        }, i =>
        {
            var snapshotIndex = validSnapshotIndices[i];
            var entrySnapshot = snapshots[snapshotIndex];

            // For each opportunity detected at this time
            foreach (var opportunity in entrySnapshot.Opportunities)
            {
                Interlocked.Increment(ref totalOpportunities);

                try
                {
                    // Generate 9 training rows per opportunity (one for each fixed duration)
                    var fixedDurationSimulations = SimulateWithFixedDurations(
                        opportunity,
                        snapshots,
                        snapshotIndex);

                    if (fixedDurationSimulations != null && fixedDurationSimulations.Any())
                    {
                        // ConcurrentBag doesn't have AddRange, add items individually
                        foreach (var sim in fixedDurationSimulations)
                        {
                            simulations.Add(sim);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "Failed to simulate position for {Symbol} at {Time}",
                        opportunity.Symbol, entrySnapshot.Timestamp);
                }
            }

            // Thread-safe progress logging
            var currentProcessed = Interlocked.Increment(ref processed);
            if (currentProcessed % 100 == 0 || currentProcessed == validSnapshotIndices.Count)
            {
                var percentComplete = (currentProcessed * 100.0) / validSnapshotIndices.Count;
                _logger.LogInformation(
                    "Progress: {Percent:F1}% - Processed {Current}/{Total} valid snapshots ({Simulations} simulations generated)",
                    percentComplete, currentProcessed, validSnapshotIndices.Count, simulations.Count);
            }
        });

        var profitableCount = simulations.Count(s => s.WasProfitable);
        var profitablePercent = simulations.Any() ? (profitableCount * 100.0 / simulations.Count) : 0;
        var skippedSnapshots = snapshots.Count - validSnapshotIndices.Count;

        _logger.LogInformation(
            "Simulation complete: {Total} simulations generated from {Opportunities} opportunities ({SkippedSnapshots} snapshots pre-filtered)",
            simulations.Count, totalOpportunities, skippedSnapshots);

        _logger.LogInformation(
            "Profitable: {Profitable} ({Percent:F1}%)",
            profitableCount, profitablePercent);

        if (simulations.Any())
        {
            var avgProfit = simulations.Average(s => s.ActualProfitPercent);
            var avgHoldHours = simulations.Average(s => s.ActualHoldHours);

            _logger.LogInformation(
                "Profit stats: Avg: {AvgProfit:F3}%, Min: {Min:F3}%, Max: {Max:F3}%",
                avgProfit,
                simulations.Min(s => s.ActualProfitPercent),
                simulations.Max(s => s.ActualProfitPercent));

            _logger.LogInformation(
                "Hold time stats: Avg: {AvgHours:F1}h, Min: {MinHours:F1}h, Max: {MaxHours:F1}h",
                avgHoldHours,
                simulations.Min(s => s.ActualHoldHours),
                simulations.Max(s => s.ActualHoldHours));
        }

        return simulations.ToList();
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
    /// Generate training data with fixed-duration predictions
    /// Creates 9 training rows per opportunity (one for each target duration)
    /// This makes profit/duration predictions feasible compared to unpredictable optimal hindsight
    /// </summary>
    private List<SimulatedExecution> SimulateWithFixedDurations(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex)
    {
        // Fixed durations to evaluate (hours)
        decimal[] targetDurations = { 0.5m, 1m, 2m, 4m, 8m, 12m, 24m, 48m, 72m };

        var results = new List<SimulatedExecution>();

        foreach (var targetHours in targetDurations)
        {
            var simulation = SimulateWithFixedDuration(
                opportunity,
                snapshots,
                entryIndex,
                targetHours);

            if (simulation != null)
            {
                results.Add(simulation);
            }
        }

        return results;
    }

    /// <summary>
    /// Simulate a single position with a fixed exit duration
    /// Evaluates profit at a specific target hold time
    /// </summary>
    private SimulatedExecution? SimulateWithFixedDuration(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        decimal targetHoldHours)
    {
        var entrySnapshot = snapshots[entryIndex];
        var entryTime = entrySnapshot.Timestamp;

        // Calculate entry prices with slippage
        var entryPrices = CalculateEntryPrices(opportunity, entrySnapshot, entryIndex);

        // Calculate target exit time
        var targetExitTime = entryTime.AddHours((double)targetHoldHours);

        // Find snapshot closest to target exit time
        var exitIndex = FindSnapshotIndexClosestTo(snapshots, targetExitTime, entryIndex);
        if (exitIndex == -1 || exitIndex >= snapshots.Count)
        {
            // No data available at target duration
            return null;
        }

        var exitSnapshot = snapshots[exitIndex];
        var actualHoldHours = (decimal)(exitSnapshot.Timestamp - entryTime).TotalHours;

        // Calculate exit prices with slippage
        var exitPrices = CalculateExitPrices(opportunity, exitSnapshot, exitIndex);

        // Simulate funding payments during hold period
        var fundingPayments = SimulateFundingPayments(
            opportunity,
            snapshots,
            entryIndex,
            exitIndex);

        // Calculate total PnL
        var pnl = CalculatePnL(
            opportunity,
            entryPrices,
            exitPrices,
            fundingPayments,
            DEFAULT_POSITION_SIZE);

        // Calculate total fees
        var totalFeesUsd = DEFAULT_POSITION_SIZE * (opportunity.PositionCostPercent / 100);

        // Track peak profit and drawdown during hold period
        var (peakProfit, maxDrawdown) = CalculatePeakAndDrawdown(
            opportunity,
            snapshots,
            entryIndex,
            exitIndex,
            (entryPrices.LongPrice, entryPrices.ShortPrice));

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

            // Fixed-duration prediction feature (NEW)
            TargetHoldHours = targetHoldHours,

            // Target variables (actual outcomes at this duration)
            StrategyName = "FixedDuration",
            ExitReason = $"FIXED_DURATION_{targetHoldHours}h",
            ActualHoldHours = actualHoldHours,
            ActualProfitPercent = pnl.TotalProfitPercent,
            ActualProfitUsd = pnl.TotalProfitUsd,
            WasProfitable = pnl.TotalProfitPercent > 0,
            HitProfitTarget = pnl.TotalProfitPercent > 0.5m,
            HitStopLoss = pnl.TotalProfitPercent < -2.0m,

            // Performance metrics
            PeakUnrealizedProfitPercent = Math.Max(peakProfit, pnl.TotalProfitPercent),
            MaxDrawdownPercent = maxDrawdown,
            FundingPaymentsCount = fundingPayments.Count,
            TotalFundingEarnedUsd = fundingPayments.Sum(f => f.Amount),

            // Execution quality
            TotalFeesUsd = totalFeesUsd,

            // Prices (already include slippage)
            EntryLongPrice = entryPrices.LongPrice,
            EntryShortPrice = entryPrices.ShortPrice,
            ExitLongPrice = exitPrices.LongPrice,
            ExitShortPrice = exitPrices.ShortPrice,

            // No snapshots for fixed-duration training (we don't need exit prediction data)
            Snapshots = new List<PositionSnapshot>()
        };
    }

    /// <summary>
    /// Calculate peak profit and max drawdown during a hold period
    /// </summary>
    private (decimal peakProfit, decimal maxDrawdown) CalculatePeakAndDrawdown(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        int exitIndex,
        (decimal LongPrice, decimal ShortPrice) entryPrices)
    {
        double peakProfit = double.MinValue;
        double maxDrawdown = 0.0;

        // Sample every 5 minutes between entry and exit
        var entryTime = snapshots[entryIndex].Timestamp;
        var exitTime = snapshots[exitIndex].Timestamp;
        var sampleInterval = GetAdaptiveSampleInterval(opportunity);

        var currentTime = entryTime.AddHours((double)sampleInterval);

        while (currentTime <= exitTime)
        {
            var snapshotIndex = FindSnapshotIndexClosestTo(snapshots, currentTime, entryIndex);
            if (snapshotIndex == -1 || snapshotIndex >= snapshots.Count)
                break;

            var snapshot = snapshots[snapshotIndex];

            // Get current prices
            var currentLongPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol, snapshotIndex);
            var currentShortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol, snapshotIndex);

            if (currentLongPrice == 0 || currentShortPrice == 0)
            {
                currentTime = currentTime.AddHours((double)sampleInterval);
                continue;
            }

            // Calculate unrealized P&L (simplified, without funding)
            var longPnl = (double)((currentLongPrice - entryPrices.LongPrice) / entryPrices.LongPrice);
            var shortPnl = (double)((entryPrices.ShortPrice - currentShortPrice) / entryPrices.ShortPrice);
            var pricePnl = ((longPnl + shortPnl) / 2.0) * 100.0;

            // Subtract exit fees
            var exitFeePercent = (double)(opportunity.PositionCostPercent / 2m);
            var netPnl = pricePnl - exitFeePercent;

            peakProfit = Math.Max(peakProfit, netPnl);
            maxDrawdown = Math.Min(maxDrawdown, netPnl);

            currentTime = currentTime.AddHours((double)sampleInterval);
        }

        // Clamp to reasonable ranges before converting to decimal (±1000% max)
        // Extreme values from volatile tokens can overflow decimal conversion
        peakProfit = Math.Clamp(peakProfit, -1000.0, 1000.0);
        maxDrawdown = Math.Clamp(maxDrawdown, -1000.0, 1000.0);

        return ((decimal)peakProfit, (decimal)maxDrawdown);
    }

    /// <summary>
    /// Simulate a single position using optimal hindsight
    /// Evaluates profit every 5 minutes for up to 72 hours and selects the best exit point
    /// </summary>
    private SimulatedExecution? SimulateWithOptimalHindsight(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex)
    {
        var entrySnapshot = snapshots[entryIndex];
        var entryTime = entrySnapshot.Timestamp;

        // Calculate entry prices with slippage
        var entryPrices = CalculateEntryPrices(opportunity, entrySnapshot, entryIndex);

        // Constants for optimal hindsight simulation
        const decimal MAX_HOLD_HOURS = 72m;

        // PERFORMANCE OPTIMIZATION: Use adaptive sampling interval based on volatility
        // High volatility = more frequent samples, Low volatility = fewer samples
        var sampleIntervalHours = GetAdaptiveSampleInterval(opportunity);

        var maxExitTime = entryTime.AddHours((double)MAX_HOLD_HOURS);

        // PERFORMANCE OPTIMIZATION: Use double for intermediate calculations (1.5-2x faster than decimal)
        // Convert to decimal only for final storage
        double bestProfit = double.MinValue;
        int bestExitIndex = -1;
        double peakProfit = double.MinValue;
        double maxDrawdown = 0.0;
        int checkpointsEvaluated = 0;

        // Collect checkpoint data for snapshot generation
        var checkpoints = new List<SnapshotGenerator.CheckpointData>();

        // Track cumulative funding for accurate P&L (use double for speed)
        double cumulativeFunding = 0.0;
        var fundingIntervalHours = (double)Math.Min(
            opportunity.LongFundingIntervalHours ?? 8m,
            opportunity.ShortFundingIntervalHours ?? 8m);

        // Sample adaptively from entry to 72h (or until data runs out)
        var currentTime = entryTime.AddHours((double)sampleIntervalHours);

        while (currentTime <= maxExitTime)
        {
            // Find snapshot closest to current time
            var snapshotIndex = FindSnapshotIndexClosestTo(snapshots, currentTime, entryIndex);
            if (snapshotIndex == -1 || snapshotIndex >= snapshots.Count)
            {
                // No more data available
                break;
            }

            var snapshot = snapshots[snapshotIndex];
            checkpointsEvaluated++;

            // Calculate current unrealized P&L
            var currentLongPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol, snapshotIndex);
            var currentShortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol, snapshotIndex);

            if (currentLongPrice == 0 || currentShortPrice == 0)
            {
                currentTime = currentTime.AddHours((double)sampleIntervalHours);
                continue;
            }

            // PERFORMANCE: Use double arithmetic for calculations (1.5-2x faster)
            // Calculate price-based PnL (percentage)
            var longPnl = (double)((currentLongPrice - entryPrices.LongPrice) / entryPrices.LongPrice);
            var shortPnl = (double)((entryPrices.ShortPrice - currentShortPrice) / entryPrices.ShortPrice);
            var pricePnl = ((longPnl + shortPnl) / 2.0) * 100.0;

            // Calculate cumulative funding up to this checkpoint
            var (currentLongRate, currentShortRate) = GetFundingRatesAt(snapshot, opportunity, snapshotIndex);
            var hoursElapsed = (snapshot.Timestamp - entryTime).TotalHours;

            // Estimate funding payments received (simple approximation based on hours elapsed)
            // More accurate than nothing, accounts for time-weighted funding accumulation
            var fundingPaymentsCount = Math.Floor(hoursElapsed / fundingIntervalHours);
            if (fundingPaymentsCount > 0)
            {
                // Average funding rate differential over the period
                var avgFundingDiff = (double)((opportunity.ShortFundingRate - opportunity.LongFundingRate +
                                     (currentShortRate - currentLongRate)) / 2m);
                cumulativeFunding = avgFundingDiff * fundingPaymentsCount;
            }

            // Calculate NET unrealized P&L including all costs
            // Price P&L + Cumulative Funding - Exit Fees (entry fees are sunk cost)
            var exitFeePercent = (double)(opportunity.PositionCostPercent / 2m); // Half of total cost is exit
            var netUnrealizedPnl = pricePnl + (cumulativeFunding * 100.0) - exitFeePercent;

            // Clamp PnL to realistic range (±50%) to prevent extreme values from volatile tokens
            // Arbitrage profit >50% in hours is unrealistic and indicates data quality issues
            netUnrealizedPnl = Math.Clamp(netUnrealizedPnl, -50.0, 50.0);

            // Update peak and drawdown tracking (use NET P&L)
            peakProfit = Math.Max(peakProfit, netUnrealizedPnl);
            maxDrawdown = Math.Min(maxDrawdown, netUnrealizedPnl);

            // Track best exit point (highest NET profit, even if negative)
            if (netUnrealizedPnl > bestProfit)
            {
                bestProfit = netUnrealizedPnl;
                bestExitIndex = snapshotIndex;
            }

            // Collect checkpoint data for snapshot generation
            // PERFORMANCE: Use double for spread calculation
            var currentSpread = Math.Abs(((double)((currentShortPrice - currentLongPrice) / currentLongPrice)) * 100.0);
            var currentVolume = snapshot.BtcVolume24h; // Use BTC volume as proxy for now

            checkpoints.Add(new SnapshotGenerator.CheckpointData
            {
                Index = checkpointsEvaluated - 1,
                Time = snapshot.Timestamp,
                UnrealizedPnLPercent = (decimal)netUnrealizedPnl, // Convert double back to decimal for storage
                LongPrice = currentLongPrice,
                ShortPrice = currentShortPrice,
                LongFundingRate = currentLongRate,
                ShortFundingRate = currentShortRate,
                SpreadPercent = (decimal)currentSpread, // Convert double back to decimal for storage
                Volume24h = currentVolume,
                CumulativeFundingPercent = (decimal)(cumulativeFunding * 100.0) // Convert double back to decimal for storage
            });

            // Move to next checkpoint (using adaptive interval)
            currentTime = currentTime.AddHours((double)sampleIntervalHours);
        }

        // If no valid exit point was found, return null
        if (bestExitIndex == -1)
            return null;

        var exitSnapshot = snapshots[bestExitIndex];
        var hoursHeld = (decimal)(exitSnapshot.Timestamp - entryTime).TotalHours;

        // Find optimal exit index in checkpoint data
        var optimalExitIndex = checkpoints.FindIndex(c => c.Time == exitSnapshot.Timestamp);

        // Determine intelligent exit reason based on market conditions and P&L
        // Convert double values to decimal for method call
        var exitReason = DetermineExitReason(
            (decimal)bestProfit,
            (decimal)peakProfit,
            hoursHeld,
            opportunity,
            checkpoints,
            optimalExitIndex
        );

        // DISABLED: Exit position snapshot generation (not needed for entry-only predictions)
        // This significantly speeds up training data generation
        // var positionSnapshots = _snapshotGenerator.GenerateSnapshots(
        //     executionId: Interlocked.Increment(ref _executionIdCounter),
        //     opportunity: opportunity,
        //     entrySnapshot: entrySnapshot,
        //     checkpoints: checkpoints,
        //     optimalExitIndex: optimalExitIndex,
        //     exitReason: exitReason,
        //     optimalExitPnL: (decimal)bestProfit
        // );
        var positionSnapshots = new List<PositionSnapshot>();

        // Calculate exit prices with slippage
        var exitPrices = CalculateExitPrices(opportunity, exitSnapshot, bestExitIndex);

        // Simulate funding payments during hold period
        var fundingPayments = SimulateFundingPayments(
            opportunity,
            snapshots,
            entryIndex,
            bestExitIndex);

        // Calculate total PnL
        var pnl = CalculatePnL(
            opportunity,
            entryPrices,
            exitPrices,
            fundingPayments,
            DEFAULT_POSITION_SIZE);

        // Calculate total fees
        var totalFeesUsd = DEFAULT_POSITION_SIZE * (opportunity.PositionCostPercent / 100);

        // Determine if this was a "good" exit based on profitability
        var hitProfitTarget = pnl.TotalProfitPercent > 0.5m; // Consider >0.5% as hitting profit target

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

            // Target variables (optimal hindsight)
            StrategyName = "OptimalHindsight",
            ExitReason = ExitReason.OPTIMAL_HINDSIGHT.ToString(),
            ActualHoldHours = hoursHeld,
            ActualProfitPercent = pnl.TotalProfitPercent,
            ActualProfitUsd = pnl.TotalProfitUsd,
            WasProfitable = pnl.TotalProfitPercent > 0,
            HitProfitTarget = hitProfitTarget,
            HitStopLoss = false, // Not applicable for optimal hindsight

            // Performance metrics (convert double back to decimal for storage)
            // BUG FIX: Ensure peak profit is always >= actual profit (mathematical requirement)
            // peakProfit uses exit-only fees, but actual profit uses full position cost
            // Take the max to prevent peak < actual inconsistency
            PeakUnrealizedProfitPercent = Math.Max((decimal)peakProfit, pnl.TotalProfitPercent),
            MaxDrawdownPercent = (decimal)maxDrawdown,
            FundingPaymentsCount = fundingPayments.Count,
            TotalFundingEarnedUsd = fundingPayments.Sum(f => f.Amount),

            // Execution quality
            TotalFeesUsd = totalFeesUsd,

            // Prices (already include slippage)
            EntryLongPrice = entryPrices.LongPrice,
            EntryShortPrice = entryPrices.ShortPrice,
            ExitLongPrice = exitPrices.LongPrice,
            ExitShortPrice = exitPrices.ShortPrice,

            // Position snapshots (for exit prediction ML training)
            Snapshots = positionSnapshots
        };
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
        var entryPrices = CalculateEntryPrices(opportunity, entrySnapshot, entryIndex);

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
        var exitPrices = CalculateExitPrices(opportunity, exitSnapshot, exitResult.ExitSnapshotIndex);

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
            var currentLongPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol, snapshotIndex);
            var currentShortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol, snapshotIndex);

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
                    var (currentLongRate, currentShortRate) = GetFundingRatesAt(snapshot, opportunity, snapshotIndex);
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
        var finalLongPrice = GetPrice(finalSnapshot, opportunity.LongExchange, opportunity.Symbol, finalIndex);
        var finalShortPrice = GetPrice(finalSnapshot, opportunity.ShortExchange, opportunity.Symbol, finalIndex);

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
        HistoricalMarketSnapshot snapshot,
        int snapshotIndex)
    {
        // Get current market prices
        var longPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol, snapshotIndex);
        var shortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol, snapshotIndex);

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
        HistoricalMarketSnapshot snapshot,
        int snapshotIndex)
    {
        var longPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol, snapshotIndex);
        var shortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol, snapshotIndex);

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
            var (longRate, shortRate) = GetFundingRatesAt(snapshot, opportunity, snapshotIndex);

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
        ArbitrageOpportunityDto opportunity,
        int snapshotIndex)
    {
        // PERFORMANCE OPTIMIZATION: Thread-safe cached lookup O(1) instead of FirstOrDefault() O(n)
        var cache = _snapshotCaches.GetOrAdd(snapshotIndex, _ => new SnapshotCache(snapshot));
        var longRate = cache.GetFundingRate(opportunity.LongExchange, opportunity.Symbol);
        var shortRate = cache.GetFundingRate(opportunity.ShortExchange, opportunity.Symbol);

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

        // Clamp PnL to realistic range (±50%) to prevent extreme values from volatile tokens
        totalPnlPercent = Math.Clamp(totalPnlPercent, -50m, 50m);

        var totalPnlUsd = (totalPnlPercent / 100) * positionSize;

        return (totalPnlPercent, totalPnlUsd);
    }

    private decimal GetPrice(HistoricalMarketSnapshot snapshot, string exchange, string symbol, int snapshotIndex)
    {
        // PERFORMANCE OPTIMIZATION: Thread-safe cached lookup O(1) instead of double dictionary lookup
        var cache = _snapshotCaches.GetOrAdd(snapshotIndex, _ => new SnapshotCache(snapshot));
        return cache.GetPrice(exchange, symbol);
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

    /// <summary>
    /// Determine the actual exit reason based on market conditions and P&L outcomes
    /// </summary>
    private string DetermineExitReason(
        decimal bestProfit,
        decimal peakProfit,
        decimal hoursHeld,
        ArbitrageOpportunityDto opportunity,
        List<SnapshotGenerator.CheckpointData> checkpoints,
        int optimalExitIndex)
    {
        // Priority 1: STOP_LOSS - negative P&L
        if (bestProfit < 0)
        {
            _logger.LogDebug("Exit reason: STOP_LOSS (P&L: {PnL}%)", bestProfit);
            return ExitReason.STOP_LOSS.ToString();
        }

        // Priority 2: MAX_HOLD_TIME - held for max duration (72 hours)
        if (hoursHeld >= 72m)
        {
            _logger.LogDebug("Exit reason: MAX_HOLD_TIME (held: {Hours}h)", hoursHeld);
            return ExitReason.MAX_HOLD_TIME.ToString();
        }

        // Priority 3: TRAILING_STOP - significant drawback from peak
        // If peak was significantly higher than exit (e.g., >0.3% drawback)
        var drawbackFromPeak = peakProfit - bestProfit;
        if (drawbackFromPeak > 0.3m && peakProfit > 0.5m)
        {
            _logger.LogDebug("Exit reason: TRAILING_STOP (peak: {Peak}%, exit: {Exit}%, drawback: {Drawback}%)",
                peakProfit, bestProfit, drawbackFromPeak);
            return ExitReason.TRAILING_STOP.ToString();
        }

        // Priority 4: FUNDING_REVERSAL - funding differential dropped significantly
        if (optimalExitIndex >= 0 && optimalExitIndex < checkpoints.Count)
        {
            var exitCheckpoint = checkpoints[optimalExitIndex];
            var entryFundingDiff = opportunity.ShortFundingRate - opportunity.LongFundingRate;
            var exitFundingDiff = exitCheckpoint.ShortFundingRate - exitCheckpoint.LongFundingRate;

            // Check if funding reversed by >50%
            if (entryFundingDiff != 0)
            {
                var fundingChangeRatio = exitFundingDiff / entryFundingDiff;
                if (fundingChangeRatio < 0.5m)
                {
                    _logger.LogDebug("Exit reason: FUNDING_REVERSAL (entry: {Entry}%, exit: {Exit}%, ratio: {Ratio})",
                        entryFundingDiff, exitFundingDiff, fundingChangeRatio);
                    return ExitReason.FUNDING_REVERSAL.ToString();
                }
            }

            // Priority 5: VOLATILITY_SPIKE - spread volatility increased significantly
            var entryVolatility = opportunity.SpreadVolatilityStdDev ?? 0m;
            var exitSpread = exitCheckpoint.SpreadPercent;
            var entrySpread = opportunity.CurrentPriceSpreadPercent ?? 0m;
            var spreadChange = Math.Abs(exitSpread - entrySpread);

            // If spread changed dramatically (>2x the entry volatility), consider it a volatility spike
            if (entryVolatility > 0 && spreadChange > entryVolatility * 2)
            {
                _logger.LogDebug("Exit reason: VOLATILITY_SPIKE (spread change: {Change}%, entry vol: {Vol}%)",
                    spreadChange, entryVolatility);
                return ExitReason.VOLATILITY_SPIKE.ToString();
            }
        }

        // Priority 6: PROFIT_TARGET - exiting with good profit
        if (bestProfit >= 0.5m)
        {
            _logger.LogDebug("Exit reason: PROFIT_TARGET (P&L: {PnL}%)", bestProfit);
            return ExitReason.PROFIT_TARGET.ToString();
        }

        // Fallback: OPTIMAL_HINDSIGHT - other scenarios
        _logger.LogDebug("Exit reason: OPTIMAL_HINDSIGHT (P&L: {PnL}%, peak: {Peak}%, hours: {Hours})",
            bestProfit, peakProfit, hoursHeld);
        return ExitReason.OPTIMAL_HINDSIGHT.ToString();
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
