using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services.Persistence;

/// <summary>
/// Persists raw market data to disk for later replay
/// Saves to data/raw/ folder with organized structure
/// </summary>
public class RawDataPersister
{
    private readonly ILogger<RawDataPersister> _logger;
    private const string BaseDataPath = "data/raw";

    public RawDataPersister(ILogger<RawDataPersister> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Save all raw market data to disk in day-specific directories
    /// </summary>
    public async Task<DataCollectionManifest> SaveRawDataAsync(
        DateTime startDate,
        DateTime endDate,
        List<string> exchanges,
        List<string> symbols,
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceKlines,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? liquidityMetrics = null)
    {
        // Validate data before saving
        ValidateCollectedData(startDate, endDate, exchanges, fundingRates, priceKlines);

        // Generate day-specific directory path (data/raw/YYYY-MM-DD)
        string dayDirectory = Path.Combine(BaseDataPath, startDate.ToString("yyyy-MM-dd"));
        _logger.LogInformation("Saving raw market data to {Path}", dayDirectory);

        // Create directory structure for this day
        Directory.CreateDirectory(dayDirectory);
        Directory.CreateDirectory(Path.Combine(dayDirectory, "funding_rates"));
        Directory.CreateDirectory(Path.Combine(dayDirectory, "klines"));
        Directory.CreateDirectory(Path.Combine(dayDirectory, "liquidity"));

        // Save funding rates per exchange
        foreach (var exchange in exchanges)
        {
            if (fundingRates.ContainsKey(exchange))
            {
                var fundingPath = Path.Combine(dayDirectory, "funding_rates", $"{exchange.ToLower()}.json");
                await SaveJsonAsync(fundingPath, fundingRates[exchange]);
                _logger.LogInformation("Saved {Count} funding rates for {Exchange}",
                    fundingRates[exchange].Count, exchange);
            }
        }

        // Save price klines per exchange
        foreach (var exchange in exchanges)
        {
            if (priceKlines.ContainsKey(exchange))
            {
                var klinesPath = Path.Combine(dayDirectory, "klines", $"{exchange.ToLower()}.json");
                await SaveJsonAsync(klinesPath, priceKlines[exchange]);

                var totalKlines = priceKlines[exchange].Values.Sum(list => list.Count);
                _logger.LogInformation("Saved {Count} klines for {Exchange}", totalKlines, exchange);
            }
        }

        // Save liquidity metrics if available
        if (liquidityMetrics != null)
        {
            var liquidityPath = Path.Combine(dayDirectory, "liquidity", "liquidity_metrics.json");
            await SaveJsonAsync(liquidityPath, liquidityMetrics);
            _logger.LogInformation("Saved liquidity metrics");
        }

        // Create manifest
        var manifest = new DataCollectionManifest
        {
            StartDate = startDate,
            EndDate = endDate,
            Exchanges = exchanges,
            Symbols = symbols,
            TotalFundingRates = fundingRates.Values.Sum(list => list.Count),
            TotalPriceKlines = priceKlines.Values.Sum(dict => dict.Values.Sum(list => list.Count)),
            HasLiquidityData = liquidityMetrics != null,
            CollectedAt = DateTime.UtcNow,
            DataPath = dayDirectory
        };

        // Save manifest
        var manifestPath = Path.Combine(dayDirectory, "manifest.json");
        await SaveJsonAsync(manifestPath, manifest);
        _logger.LogInformation("Saved collection manifest to {Path}", manifestPath);

        return manifest;
    }

    private async Task SaveJsonAsync<T>(string filePath, T data)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var json = JsonSerializer.Serialize(data, options);
        await File.WriteAllTextAsync(filePath, json);
    }

    /// <summary>
    /// Validate collected data for quality and correctness
    /// </summary>
    private void ValidateCollectedData(
        DateTime startDate,
        DateTime endDate,
        List<string> exchanges,
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceKlines)
    {
        _logger.LogInformation("Validating collected data quality...");

        var warnings = new List<string>();
        var errors = new List<string>();

        // Validate funding rates
        foreach (var exchange in exchanges)
        {
            if (!fundingRates.ContainsKey(exchange) || !fundingRates[exchange].Any())
            {
                warnings.Add($"No funding rates collected for {exchange}");
                continue;
            }

            var rates = fundingRates[exchange];

            // Check date range
            var outOfRangeRates = rates.Where(r => r.FundingTime < startDate || r.FundingTime > endDate.AddDays(1)).ToList();
            if (outOfRangeRates.Any())
            {
                warnings.Add($"{exchange}: {outOfRangeRates.Count} funding rates outside date range [{startDate:yyyy-MM-dd}, {endDate:yyyy-MM-dd}]");
                _logger.LogWarning("Sample out-of-range funding rates for {Exchange}: {Samples}",
                    exchange, string.Join(", ", outOfRangeRates.Take(3).Select(r => $"{r.Symbol}@{r.FundingTime:yyyy-MM-dd HH:mm}")));
            }

            _logger.LogDebug("Funding rate validation for {Exchange}: {Total} rates, {OutOfRange} out of range",
                exchange, rates.Count, outOfRangeRates.Count);
        }

        // Validate klines
        foreach (var exchange in exchanges)
        {
            if (!priceKlines.ContainsKey(exchange) || !priceKlines[exchange].Any())
            {
                warnings.Add($"No klines collected for {exchange}");
                continue;
            }

            var exchangeKlines = priceKlines[exchange];
            var totalKlines = exchangeKlines.Values.Sum(list => list.Count);

            // Check each symbol's klines
            foreach (var (symbol, klines) in exchangeKlines)
            {
                if (!klines.Any()) continue;

                // Check if klines are sorted
                var sortedKlines = klines.OrderBy(k => k.Timestamp).ToList();
                if (!klines.SequenceEqual(sortedKlines))
                {
                    warnings.Add($"{exchange} {symbol}: Klines are not sorted by timestamp");
                }

                // Detect actual interval by checking consecutive timestamps
                if (klines.Count > 1)
                {
                    var intervals = new List<double>();
                    for (int i = 1; i < Math.Min(klines.Count, 10); i++)
                    {
                        intervals.Add((klines[i].Timestamp - klines[i - 1].Timestamp).TotalMinutes);
                    }

                    var avgInterval = intervals.Average();
                    var expectedInterval = 1.0; // We expect 1-minute klines

                    // Allow 0.1 minute tolerance
                    if (Math.Abs(avgInterval - expectedInterval) > 0.1)
                    {
                        warnings.Add($"{exchange} {symbol}: Kline interval is {avgInterval:F1} minutes, expected {expectedInterval} minutes");
                    }
                }

                // Check date range
                var outOfRangeKlines = klines.Where(k => k.Timestamp < startDate || k.Timestamp > endDate.AddDays(1)).ToList();
                if (outOfRangeKlines.Any())
                {
                    warnings.Add($"{exchange} {symbol}: {outOfRangeKlines.Count}/{klines.Count} klines outside date range");
                }
            }

            _logger.LogDebug("Kline validation for {Exchange}: {Symbols} symbols, {Total} total klines",
                exchange, exchangeKlines.Count, totalKlines);
        }

        // Log validation results
        if (errors.Any())
        {
            _logger.LogError("Data validation found {Count} errors:", errors.Count);
            foreach (var error in errors)
            {
                _logger.LogError("  ❌ {Error}", error);
            }
            throw new InvalidOperationException($"Data validation failed with {errors.Count} errors. See logs for details.");
        }

        if (warnings.Any())
        {
            _logger.LogWarning("Data validation found {Count} warnings:", warnings.Count);
            foreach (var warning in warnings.Take(10)) // Show first 10 warnings
            {
                _logger.LogWarning("  ⚠️  {Warning}", warning);
            }
            if (warnings.Count > 10)
            {
                _logger.LogWarning("  ... and {More} more warnings", warnings.Count - 10);
            }
        }
        else
        {
            _logger.LogInformation("✓ Data validation passed - no issues found");
        }
    }
}
