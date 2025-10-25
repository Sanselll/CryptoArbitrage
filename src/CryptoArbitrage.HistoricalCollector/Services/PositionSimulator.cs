using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Simulates position executions from historical snapshots
/// Generates training data for ML models
/// </summary>
public class PositionSimulator
{
    private readonly ILogger<PositionSimulator> _logger;

    // Hold durations to simulate (in hours): 30min, 1h, 2h, 4h, 6h, 8h, 12h, 24h, 32h, 48h, 72h
    private readonly decimal[] _holdDurations = new[] { 0.5m, 1m, 2m, 4m, 6m, 8m, 12m, 24m, 32m, 48m, 72m };

    // Default position size for simulation
    private const decimal DEFAULT_POSITION_SIZE = 1000m;

    public PositionSimulator(ILogger<PositionSimulator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Simulate positions for all opportunities across all hold durations
    /// </summary>
    public async Task<List<SimulatedExecution>> SimulateAllPositions(
        List<HistoricalMarketSnapshot> snapshots,
        decimal[]? customHoldDurations = null)
    {
        var holdDurations = customHoldDurations ?? _holdDurations;

        _logger.LogInformation(
            "Starting position simulation for {SnapshotCount} snapshots with {DurationCount} hold durations",
            snapshots.Count, holdDurations.Length);

        var simulations = new List<SimulatedExecution>();
        var maxHoldHours = holdDurations.Max();

        // Calculate how many snapshots we can use (need to leave room for max hold duration)
        var usableSnapshotCount = snapshots.Count - (int)(maxHoldHours * 60); // Convert hours to minutes

        for (int i = 0; i < usableSnapshotCount; i++)
        {
            var entrySnapshot = snapshots[i];

            // For each opportunity detected at this time
            foreach (var opportunity in entrySnapshot.Opportunities)
            {
                // Simulate each hold duration
                foreach (var holdHours in holdDurations)
                {
                    try
                    {
                        var simulation = SimulateSinglePosition(
                            opportunity,
                            snapshots,
                            i,
                            holdHours);

                        if (simulation != null)
                            simulations.Add(simulation);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex,
                            "Failed to simulate position for {Symbol} at {Time} with {Duration}h hold",
                            opportunity.Symbol, entrySnapshot.Timestamp, holdHours);
                    }
                }
            }

            if ((i + 1) % 1000 == 0)
            {
                _logger.LogInformation(
                    "Simulated {Count} snapshots ({Simulations} total simulations)...",
                    i + 1, simulations.Count);
            }
        }

        _logger.LogInformation("Simulation complete: {Count} simulated positions generated", simulations.Count);
        return simulations;
    }

    /// <summary>
    /// Simulate a single position execution
    /// </summary>
    private SimulatedExecution? SimulateSinglePosition(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        decimal holdHours)
    {
        // Find exit snapshot
        var exitIndex = FindExitSnapshotIndex(snapshots, entryIndex, holdHours);
        if (exitIndex == -1)
            return null;

        var entrySnapshot = snapshots[entryIndex];
        var exitSnapshot = snapshots[exitIndex];

        // Calculate entry prices with slippage
        var entryPrices = CalculateEntryPrices(opportunity, entrySnapshot);

        // Calculate exit prices with slippage
        var exitPrices = CalculateExitPrices(opportunity, exitSnapshot);

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

        // Track peak profit and max drawdown
        var (peakProfit, maxDrawdown) = CalculatePeakAndDrawdown(
            opportunity,
            snapshots,
            entryIndex,
            exitIndex,
            entryPrices,
            DEFAULT_POSITION_SIZE);

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

            FundProfit8h = opportunity.FundProfit8h,
            FundApr = opportunity.FundApr,
            FundProfit8h24hProj = opportunity.FundProfit8h24hProj,
            FundProfit8h3dProj = opportunity.FundProfit8h3dProj,
            BreakEvenTimeHours = opportunity.BreakEvenTimeHours,

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

            // Target variables
            ActualHoldHours = (decimal)(exitSnapshot.Timestamp - entrySnapshot.Timestamp).TotalHours,
            ActualProfitPercent = pnl.TotalProfitPercent,
            ActualProfitUsd = pnl.TotalProfitUsd,
            WasProfitable = pnl.TotalProfitPercent > 0,

            // Performance metrics
            PeakUnrealizedProfitPercent = peakProfit,
            MaxDrawdownPercent = maxDrawdown,
            FundingPaymentsCount = fundingPayments.Count,
            TotalFundingEarnedUsd = fundingPayments.Sum(f => f.Amount),

            // Execution quality
            TotalSlippagePercent = pnl.EntrySlippage + pnl.ExitSlippage,
            EntrySlippagePercent = pnl.EntrySlippage,
            ExitSlippagePercent = pnl.ExitSlippage,
            TotalFeesUsd = pnl.TotalFees,

            // Prices
            EntryLongPrice = entryPrices.LongPrice,
            EntryShortPrice = entryPrices.ShortPrice,
            ExitLongPrice = exitPrices.LongPrice,
            ExitShortPrice = exitPrices.ShortPrice
        };
    }

    private int FindExitSnapshotIndex(
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        decimal holdHours)
    {
        var entryTime = snapshots[entryIndex].Timestamp;
        var targetExitTime = entryTime.AddHours((double)holdHours);

        // Find closest snapshot to target exit time
        for (int i = entryIndex + 1; i < snapshots.Count; i++)
        {
            if (snapshots[i].Timestamp >= targetExitTime)
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

    private (decimal peakProfit, decimal maxDrawdown) CalculatePeakAndDrawdown(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        int exitIndex,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) entryPrices,
        decimal positionSize)
    {
        decimal peakProfit = 0;
        decimal maxDrawdown = 0;

        // Sample every 10 snapshots to reduce computation
        for (int i = entryIndex + 1; i <= exitIndex; i += 10)
        {
            var snapshot = snapshots[i];

            var currentLongPrice = GetPrice(snapshot, opportunity.LongExchange, opportunity.Symbol);
            var currentShortPrice = GetPrice(snapshot, opportunity.ShortExchange, opportunity.Symbol);

            // Unrealized PnL
            var longPnl = (currentLongPrice - entryPrices.LongPrice) / entryPrices.LongPrice;
            var shortPnl = (entryPrices.ShortPrice - currentShortPrice) / entryPrices.ShortPrice;
            var unrealizedPnl = ((longPnl + shortPnl) / 2) * 100;

            peakProfit = Math.Max(peakProfit, unrealizedPnl);
            maxDrawdown = Math.Min(maxDrawdown, unrealizedPnl);
        }

        return (peakProfit, maxDrawdown);
    }

    private (decimal TotalProfitPercent, decimal TotalProfitUsd, decimal EntrySlippage, decimal ExitSlippage, decimal TotalFees) CalculatePnL(
        ArbitrageOpportunityDto opportunity,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) entryPrices,
        (decimal LongPrice, decimal ShortPrice, decimal LongSlippage, decimal ShortSlippage) exitPrices,
        List<FundingPayment> fundingPayments,
        decimal positionSize)
    {
        // Long position PnL
        var longPnl = (exitPrices.LongPrice - entryPrices.LongPrice) / entryPrices.LongPrice;

        // Short position PnL
        var shortPnl = (entryPrices.ShortPrice - exitPrices.ShortPrice) / entryPrices.ShortPrice;

        // Price PnL (average of long and short)
        var pricePnl = (longPnl + shortPnl) / 2;

        // Funding PnL
        var fundingPnl = fundingPayments.Sum(f => f.Amount) / positionSize;

        // Total PnL before fees
        var totalPnlBeforeFees = pricePnl + fundingPnl;

        // Calculate fees (use position cost percent from opportunity)
        var feesPercent = opportunity.PositionCostPercent / 100;
        var totalFees = positionSize * feesPercent;

        // Total PnL after fees
        var totalPnlPercent = (totalPnlBeforeFees * 100) - opportunity.PositionCostPercent;
        var totalPnlUsd = (totalPnlPercent / 100) * positionSize;

        return (
            totalPnlPercent,
            totalPnlUsd,
            entryPrices.LongSlippage,
            exitPrices.LongSlippage,
            totalFees
        );
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
