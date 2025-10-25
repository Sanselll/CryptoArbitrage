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
            // === TIMING FEATURES ===
            Map(m => m.EntryTime).Name("entry_time");
            Map(m => m.ExitTime).Name("exit_time");
            Map(m => m.HourOfDay).Name("hour_of_day");
            Map(m => m.DayOfWeek).Name("day_of_week");

            // === MARKET CONTEXT ===
            Map(m => m.BtcPriceAtEntry).Name("btc_price_at_entry");
            Map(m => m.MarketRegimeAtEntry).Name("market_regime");

            // === OPPORTUNITY IDENTIFIERS ===
            Map(m => m.Symbol).Name("symbol");
            Map(m => m.Strategy).Name("strategy");
            Map(m => m.LongExchange).Name("long_exchange");
            Map(m => m.ShortExchange).Name("short_exchange");

            // === PROFITABILITY FEATURES ===
            Map(m => m.FundProfit8h).Name("fund_profit_8h");
            Map(m => m.FundApr).Name("fund_apr");
            Map(m => m.FundProfit8h24hProj).Name("fund_profit_8h_24h_proj");
            Map(m => m.FundProfit8h3dProj).Name("fund_profit_8h_3d_proj");
            Map(m => m.BreakEvenTimeHours).Name("break_even_hours");

            // === RISK FEATURES ===
            Map(m => m.SpreadVolatilityCv).Name("spread_volatility_cv");
            Map(m => m.SpreadVolatilityStdDev).Name("spread_volatility_stddev");
            Map(m => m.Spread30SampleAvg).Name("spread_30sample_avg");

            // === LIQUIDITY FEATURES ===
            Map(m => m.Volume24h).Name("volume_24h");
            Map(m => m.LongVolume24h).Name("long_volume_24h");
            Map(m => m.ShortVolume24h).Name("short_volume_24h");
            Map(m => m.BidAskSpreadPercent).Name("bid_ask_spread_pct");
            Map(m => m.OrderbookDepthUsd).Name("orderbook_depth_usd");
            Map(m => m.LiquidityStatus).Name("liquidity_status");

            // === POSITION DETAILS ===
            Map(m => m.PositionCostPercent).Name("position_cost_pct");
            Map(m => m.PositionSizeUsd).Name("position_size_usd");

            // === PRICE DATA ===
            Map(m => m.EntryLongPrice).Name("entry_long_price");
            Map(m => m.EntryShortPrice).Name("entry_short_price");
            Map(m => m.ExitLongPrice).Name("exit_long_price");
            Map(m => m.ExitShortPrice).Name("exit_short_price");

            // ========================================
            // === TARGET VARIABLES (y) ===
            // ========================================

            Map(m => m.ActualHoldHours).Name("target_hold_hours");
            Map(m => m.ActualProfitPercent).Name("target_profit_pct");
            Map(m => m.ActualProfitUsd).Name("target_profit_usd");
            Map(m => m.WasProfitable).Name("target_was_profitable");

            // === PERFORMANCE METRICS ===
            Map(m => m.PeakUnrealizedProfitPercent).Name("peak_profit_pct");
            Map(m => m.MaxDrawdownPercent).Name("max_drawdown_pct");
            Map(m => m.FundingPaymentsCount).Name("funding_payments_count");
            Map(m => m.TotalFundingEarnedUsd).Name("total_funding_usd");

            // === EXECUTION QUALITY ===
            Map(m => m.TotalSlippagePercent).Name("total_slippage_pct");
            Map(m => m.EntrySlippagePercent).Name("entry_slippage_pct");
            Map(m => m.ExitSlippagePercent).Name("exit_slippage_pct");
            Map(m => m.TotalFeesUsd).Name("total_fees_usd");

            // Ignore OpportunitySnapshotJson (too large for CSV, store separately if needed)
            // Can be used for detailed analysis later
        }
    }
}
