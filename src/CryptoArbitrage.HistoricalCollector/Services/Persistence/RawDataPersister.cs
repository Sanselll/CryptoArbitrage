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
}
