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

            // Price spread
            Map(m => m.CurrentPriceSpreadPercent).Name("current_price_spread_pct");

            // Profitability projections
            Map(m => m.FundProfit8h).Name("fund_profit_8h");
            Map(m => m.FundApr).Name("fund_apr");
            Map(m => m.FundProfit8h24hProj).Name("fund_profit_8h_24h_proj");
            Map(m => m.FundApr24hProj).Name("fund_apr_24h_proj");
            Map(m => m.FundBreakEvenTime24hProj).Name("fund_break_even_24h_proj");
            Map(m => m.FundProfit8h3dProj).Name("fund_profit_8h_3d_proj");
            Map(m => m.FundApr3dProj).Name("fund_apr_3d_proj");
            Map(m => m.FundBreakEvenTime3dProj).Name("fund_break_even_3d_proj");
            Map(m => m.BreakEvenTimeHours).Name("break_even_hours");

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
}
