using CryptoArbitrage.HistoricalCollector.Models;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.Extensions.Logging;
using System.Globalization;

namespace CryptoArbitrage.HistoricalCollector.Exporters;

/// <summary>
/// Exports simulated executions to CSV format for ML training
/// </summary>
public class CsvExporter
{
    private readonly ILogger<CsvExporter> _logger;

    public CsvExporter(ILogger<CsvExporter> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Export simulated executions to CSV file
    /// </summary>
    public async Task ExportToCsv(List<SimulatedExecution> simulations, string outputPath)
    {
        _logger.LogInformation("Exporting {Count} simulations to {Path}...", simulations.Count, outputPath);

        // Ensure directory exists
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
        };

        await using var writer = new StreamWriter(outputPath);
        await using var csv = new CsvWriter(writer, config);

        // Register the class map
        csv.Context.RegisterClassMap<SimulatedExecutionMap>();

        // Write records
        await csv.WriteRecordsAsync(simulations);

        _logger.LogInformation("Export complete: {Path}", outputPath);
    }

    /// <summary>
    /// Export position snapshots to CSV file for exit prediction ML training
    /// </summary>
    public async Task ExportSnapshotsToCsv(List<SimulatedExecution> simulations, string outputPath)
    {
        // Flatten all snapshots from all executions
        var allSnapshots = simulations
            .SelectMany(sim => sim.Snapshots)
            .ToList();

        _logger.LogInformation("Exporting {Count} position snapshots to {Path}...", allSnapshots.Count, outputPath);

        // Ensure directory exists
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
        };

        await using var writer = new StreamWriter(outputPath);
        await using var csv = new CsvWriter(writer, config);

        // Register the class map
        csv.Context.RegisterClassMap<PositionSnapshotMap>();

        // Write records
        await csv.WriteRecordsAsync(allSnapshots);

        _logger.LogInformation("Snapshot export complete: {Path} ({Count} snapshots)", outputPath, allSnapshots.Count);
    }

    /// <summary>
    /// CSV mapping for SimulatedExecution
    /// Defines column names and order for ML training
    /// </summary>
    private class SimulatedExecutionMap : ClassMap<SimulatedExecution>
    {
        public SimulatedExecutionMap()
        {
            // ========================================
            // === INPUT FEATURES (X) - For ML Training ===
            // ========================================

            // Timing features
            Map(m => m.EntryTime).Name("entry_time");
            Map(m => m.HourOfDay).Name("hour_of_day");
            Map(m => m.DayOfWeek).Name("day_of_week");

            // Market context
            Map(m => m.Symbol).Name("symbol");
            Map(m => m.Strategy).Name("strategy");
            Map(m => m.LongExchange).Name("long_exchange");
            Map(m => m.ShortExchange).Name("short_exchange");

            // Funding rate details
            Map(m => m.LongFundingRate).Name("long_funding_rate");
            Map(m => m.ShortFundingRate).Name("short_funding_rate");
            Map(m => m.LongFundingIntervalHours).Name("long_funding_interval_hours");
            Map(m => m.ShortFundingIntervalHours).Name("short_funding_interval_hours");
            Map(m => m.LongNextFundingTimeMinutes).Name("long_next_funding_minutes");
            Map(m => m.ShortNextFundingTimeMinutes).Name("short_next_funding_minutes");

            // Profitability projections
            Map(m => m.FundProfit8h).Name("fund_profit_8h");
            Map(m => m.FundApr).Name("fund_apr");
            Map(m => m.FundProfit8h24hProj).Name("fund_profit_8h_24h_proj");
            Map(m => m.FundApr24hProj).Name("fund_apr_24h_proj");
            Map(m => m.FundProfit8h3dProj).Name("fund_profit_8h_3d_proj");
            Map(m => m.FundApr3dProj).Name("fund_apr_3d_proj");

            // Price spread statistics
            Map(m => m.PriceSpread24hAvg).Name("price_spread_24h_avg");
            Map(m => m.PriceSpread3dAvg).Name("price_spread_3d_avg");

            // Risk/volatility metrics
            Map(m => m.Spread30SampleAvg).Name("spread_30sample_avg");
            Map(m => m.SpreadVolatilityStdDev).Name("spread_volatility_stddev");
            Map(m => m.SpreadVolatilityCv).Name("spread_volatility_cv");

            // Volume/liquidity
            Map(m => m.Volume24h).Name("volume_24h");

            // ========================================
            // === TARGET VARIABLES (y) - What We Predict ===
            // ========================================

            // Strategy tracking (NEW)
            Map(m => m.StrategyName).Name("strategy_name");
            Map(m => m.ExitReason).Name("exit_reason");

            // Outcome metrics
            Map(m => m.ActualHoldHours).Name("actual_hold_hours");
            Map(m => m.ActualProfitPercent).Name("actual_profit_pct");
            Map(m => m.WasProfitable).Name("was_profitable");
            Map(m => m.HitProfitTarget).Name("hit_profit_target");
            Map(m => m.HitStopLoss).Name("hit_stop_loss");

            // === OPTIONAL: Additional metrics for analysis ===
            Map(m => m.ExitTime).Name("exit_time");
            Map(m => m.PeakUnrealizedProfitPercent).Name("peak_profit_pct");
            Map(m => m.MaxDrawdownPercent).Name("max_drawdown_pct");

            // Note: Removed unnecessary columns:
            // - MarketRegimeAtEntry, BtcPriceAtEntry (not in user's requested list)
            // - Long/ShortVolume24h (user only requested combined Volume24h)
            // - BidAskSpreadPercent, OrderbookDepthUsd, LiquidityStatus (not requested)
            // - PositionCostPercent, PositionSizeUsd (constant values, not useful for ML)
            // - Entry/ExitLongPrice, Entry/ExitShortPrice (internal calculation details)
            // - FundingPaymentsCount, TotalFundingEarnedUsd, TotalFeesUsd (derivatives)
            // - ActualProfitUsd (can be calculated from ActualProfitPercent if needed)
        }
    }

    /// <summary>
    /// CSV mapping for PositionSnapshot
    /// Defines column names and order for exit prediction ML training
    /// </summary>
    private class PositionSnapshotMap : ClassMap<PositionSnapshot>
    {
        public PositionSnapshotMap()
        {
            // Identification
            Map(m => m.ExecutionId).Name("execution_id");
            Map(m => m.SnapshotIndex).Name("snapshot_index");
            Map(m => m.SnapshotTime).Name("snapshot_time");
            Map(m => m.EntryTime).Name("entry_time");

            // Basic info (static)
            Map(m => m.Symbol).Name("symbol");
            Map(m => m.Strategy).Name("strategy");
            Map(m => m.LongExchange).Name("long_exchange");
            Map(m => m.ShortExchange).Name("short_exchange");

            // Entry timing
            Map(m => m.EntryHourOfDay).Name("entry_hour_of_day");
            Map(m => m.EntryDayOfWeek).Name("entry_day_of_week");

            // Entry funding rates (static)
            Map(m => m.EntryLongFundingRate).Name("entry_long_funding_rate");
            Map(m => m.EntryShortFundingRate).Name("entry_short_funding_rate");
            Map(m => m.EntryFundingRateDifferential).Name("entry_funding_rate_differential");
            Map(m => m.EntryFundProfit8h).Name("entry_fund_profit_8h");
            Map(m => m.EntryFundApr).Name("entry_fund_apr");
            Map(m => m.EntryFundProfit8h24hProj).Name("entry_fund_profit_8h_24h_proj");
            Map(m => m.EntryFundApr24hProj).Name("entry_fund_apr_24h_proj");
            Map(m => m.EntryFundProfit8h3dProj).Name("entry_fund_profit_8h_3d_proj");
            Map(m => m.EntryFundApr3dProj).Name("entry_fund_apr_3d_proj");

            // Entry spread (static)
            Map(m => m.EntryPriceSpreadPercent).Name("entry_price_spread_percent");
            Map(m => m.EntryPriceSpread24hAvg).Name("entry_price_spread_24h_avg");
            Map(m => m.EntryPriceSpread3dAvg).Name("entry_price_spread_3d_avg");
            Map(m => m.EntrySpread30SampleAvg).Name("entry_spread_30sample_avg");
            Map(m => m.EntrySpreadVolatilityStdDev).Name("entry_spread_volatility_stddev");
            Map(m => m.EntrySpreadVolatilityCv).Name("entry_spread_volatility_cv");

            // Entry liquidity (static)
            Map(m => m.EntryVolume24h).Name("entry_volume_24h");
            Map(m => m.EntryBidAskSpreadPercent).Name("entry_bid_ask_spread_percent");
            Map(m => m.EntryOrderbookDepthUsd).Name("entry_orderbook_depth_usd");

            // ML predictions at entry (static)
            Map(m => m.MLPredictedProfitPercent).Name("ml_predicted_profit_percent");
            Map(m => m.MLPredictedSuccessProbability).Name("ml_predicted_success_probability");
            Map(m => m.MLPredictedDurationHours).Name("ml_predicted_duration_hours");

            // Dynamic features (current position state)
            Map(m => m.TimeInPositionHours).Name("time_in_position_hours");
            Map(m => m.CurrentPnLPercent).Name("current_pnl_percent");
            Map(m => m.PeakPnLPercent).Name("peak_pnl_percent");
            Map(m => m.DrawdownFromPeakPercent).Name("drawdown_from_peak_percent");
            Map(m => m.MaxDrawdownPercent).Name("max_drawdown_percent");
            Map(m => m.PnLVelocityPerHour).Name("pnl_velocity_per_hour");

            // Funding (dynamic)
            Map(m => m.FundingPaymentsReceived).Name("funding_payments_received");
            Map(m => m.FundingEarnedUsd).Name("funding_earned_usd");
            Map(m => m.FundingEarnedPercent).Name("funding_earned_percent");

            // Current market state (dynamic)
            Map(m => m.CurrentFundingRateDifferential).Name("current_funding_rate_differential");
            Map(m => m.FundingRateDifferentialChange).Name("funding_rate_differential_change");
            Map(m => m.FundingReversalMagnitude).Name("funding_reversal_magnitude");
            Map(m => m.CurrentPriceSpreadPercent).Name("current_price_spread_percent");
            Map(m => m.SpreadChangeSinceEntryPercent).Name("spread_change_since_entry_percent");
            Map(m => m.CurrentSpreadVolatilityStdDev).Name("current_spread_volatility_stddev");
            Map(m => m.VolatilityChangeRatio).Name("volatility_change_ratio");
            Map(m => m.CurrentVolume24h).Name("current_volume_24h");
            Map(m => m.VolumeChangeRatio).Name("volume_change_ratio");
            Map(m => m.MinutesToNextFunding).Name("minutes_to_next_funding");

            // Engineered features
            Map(m => m.PositionMaturity).Name("position_maturity");
            Map(m => m.HoldEfficiency).Name("hold_efficiency");
            Map(m => m.OptimalTimeReached).Name("optimal_time_reached");
            Map(m => m.ConsecutiveNegativeSamples).Name("consecutive_negative_samples");
            Map(m => m.ConsecutivePositiveSamples).Name("consecutive_positive_samples");

            // Labels (for ML training)
            Map(m => m.ShouldExitNow).Name("should_exit_now");
            Map(m => m.ExitReason).Name("exit_reason");
            Map(m => m.HoursUntilOptimalExit).Name("hours_until_optimal_exit");
            Map(m => m.OptimalExitPnLPercent).Name("optimal_exit_pnl_percent");
            Map(m => m.PotentialPnLLoss).Name("potential_pnl_loss");
        }
    }
}
