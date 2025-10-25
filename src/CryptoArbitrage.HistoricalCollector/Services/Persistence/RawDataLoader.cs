using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services.Persistence;

/// <summary>
/// Loads raw market data from disk for replay
/// Reads from data/raw/ folder
/// </summary>
public class RawDataLoader
{
    private readonly ILogger<RawDataLoader> _logger;
    private const string BaseDataPath = "data/raw";

    public RawDataLoader(ILogger<RawDataLoader> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Load collection manifest for a specific date/time range from first available day
    /// </summary>
    public async Task<DataCollectionManifest> LoadManifestAsync(DateTime startDate, DateTime endDate)
    {
        // Try to load manifest from the first day folder
        var currentDate = startDate.Date;
        while (currentDate <= endDate.Date)
        {
            var dayFolder = Path.Combine(BaseDataPath, currentDate.ToString("yyyy-MM-dd"));
            var manifestPath = Path.Combine(dayFolder, "manifest.json");

            if (File.Exists(manifestPath))
            {
                var manifest = await LoadJsonAsync<DataCollectionManifest>(manifestPath);
                _logger.LogInformation("Loaded manifest from {Date}: {Count} symbols, {Exchanges} exchanges",
                    currentDate.ToString("yyyy-MM-dd"), manifest.Symbols.Count, string.Join(", ", manifest.Exchanges));
                return manifest;
            }

            currentDate = currentDate.AddDays(1);
        }

        throw new FileNotFoundException($"No manifest found for date range {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}. Run 'collect' command first.");
    }

    /// <summary>
    /// Load all raw market data for a specific date range from day-by-day folders
    /// </summary>
    public async Task<(
        Dictionary<string, List<FundingRateDto>> FundingRates,
        Dictionary<string, Dictionary<string, List<PriceDto>>> PriceKlines,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? LiquidityMetrics
    )> LoadRawDataAsync(DateTime startDate, DateTime endDate, List<string> exchanges)
    {
        _logger.LogInformation("Loading raw data for {StartDate} to {EndDate} from {Path}",
            startDate.ToString("yyyy-MM-dd"), endDate.ToString("yyyy-MM-dd"), BaseDataPath);

        // Initialize collections for merged data
        var allFundingRates = new Dictionary<string, List<FundingRateDto>>();
        var allPriceKlines = new Dictionary<string, Dictionary<string, List<PriceDto>>>();

        foreach (var exchange in exchanges)
        {
            allFundingRates[exchange] = new List<FundingRateDto>();
            allPriceKlines[exchange] = new Dictionary<string, List<PriceDto>>();
        }

        // Loop through each day in the range
        var currentDate = startDate.Date;
        while (currentDate <= endDate.Date)
        {
            var dayFolder = Path.Combine(BaseDataPath, currentDate.ToString("yyyy-MM-dd"));

            if (!Directory.Exists(dayFolder))
            {
                _logger.LogWarning("Day folder not found: {DayFolder} - skipping", dayFolder);
                currentDate = currentDate.AddDays(1);
                continue;
            }

            _logger.LogDebug("Loading data from {DayFolder}", dayFolder);

            // Load funding rates and klines in parallel for all exchanges
            var loadTasks = exchanges.Select(async exchange =>
            {
                var exchangeLower = exchange.ToLower();

                // Load funding rates
                var fundingPath = Path.Combine(dayFolder, "funding_rates", $"{exchangeLower}.json");
                List<FundingRateDto>? dayRates = null;
                if (File.Exists(fundingPath))
                {
                    dayRates = await LoadJsonAsync<List<FundingRateDto>>(fundingPath);
                    _logger.LogDebug("Loaded {Count} funding rates for {Exchange} on {Date}",
                        dayRates.Count, exchange, currentDate.ToString("yyyy-MM-dd"));
                }

                // Load price klines
                var klinesPath = Path.Combine(dayFolder, "klines", $"{exchangeLower}.json");
                Dictionary<string, List<PriceDto>>? dayKlines = null;
                if (File.Exists(klinesPath))
                {
                    dayKlines = await LoadJsonAsync<Dictionary<string, List<PriceDto>>>(klinesPath);
                    var totalKlines = dayKlines.Values.Sum(list => list.Count);
                    _logger.LogDebug("Loaded {Count} klines for {Exchange} on {Date}",
                        totalKlines, exchange, currentDate.ToString("yyyy-MM-dd"));
                }

                return (exchange, dayRates, dayKlines);
            });

            var results = await Task.WhenAll(loadTasks);

            // Merge results into collections
            foreach (var (exchange, dayRates, dayKlines) in results)
            {
                // Add funding rates
                if (dayRates != null)
                {
                    allFundingRates[exchange].AddRange(dayRates);
                }

                // Add price klines
                if (dayKlines != null)
                {
                    foreach (var (symbol, klines) in dayKlines)
                    {
                        if (!allPriceKlines[exchange].ContainsKey(symbol))
                        {
                            allPriceKlines[exchange][symbol] = new List<PriceDto>();
                        }
                        allPriceKlines[exchange][symbol].AddRange(klines);
                    }
                }
            }

            currentDate = currentDate.AddDays(1);
        }

        // Log totals
        foreach (var exchange in exchanges)
        {
            _logger.LogInformation("Total loaded for {Exchange}: {FundingCount} funding rates, {KlineCount} klines",
                exchange,
                allFundingRates[exchange].Count,
                allPriceKlines[exchange].Values.Sum(list => list.Count));
        }

        // Load shared liquidity metrics (used for all days)
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? liquidityMetrics = null;
        var sharedLiquidityPath = Path.Combine(BaseDataPath, "liquidity_metrics.json");
        if (File.Exists(sharedLiquidityPath))
        {
            liquidityMetrics = await LoadJsonAsync<Dictionary<string, Dictionary<string, LiquidityMetricsDto>>>(sharedLiquidityPath);
            _logger.LogInformation("Loaded shared liquidity metrics from {Path}", sharedLiquidityPath);
        }
        else
        {
            _logger.LogWarning("Shared liquidity metrics not found at {Path} (optional)", sharedLiquidityPath);
        }

        return (allFundingRates, allPriceKlines, liquidityMetrics);
    }

    private async Task<T> LoadJsonAsync<T>(string filePath)
    {
        var json = await File.ReadAllTextAsync(filePath);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        return JsonSerializer.Deserialize<T>(json, options)
            ?? throw new InvalidOperationException($"Failed to deserialize {filePath}");
    }
}
