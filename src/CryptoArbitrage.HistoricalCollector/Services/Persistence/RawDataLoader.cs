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
    /// Load raw market data with sufficient history for calculating averages
    /// Loads target date range + 3 days of history for 3-day averages
    /// Much faster than LoadAllAvailableDataAsync when you only need recent data
    /// </summary>
    public async Task<(
        Dictionary<string, List<FundingRateDto>> FundingRates,
        Dictionary<string, Dictionary<string, List<PriceDto>>> PriceKlines,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? LiquidityMetrics
    )> LoadDataWithHistoryAsync(DateTime startDate, DateTime endDate, List<string> exchanges, int historyDays = 3)
    {
        // Calculate the earliest date we need (target start - history days)
        var dataStartDate = startDate.AddDays(-historyDays);

        _logger.LogInformation(
            "Loading data from {DataStart} to {DataEnd} (target: {TargetStart} to {TargetEnd}, with {HistoryDays}d history)",
            dataStartDate.ToString("yyyy-MM-dd"),
            endDate.ToString("yyyy-MM-dd"),
            startDate.ToString("yyyy-MM-dd"),
            endDate.ToString("yyyy-MM-dd"),
            historyDays);

        // Use existing LoadRawDataAsync with expanded date range
        return await LoadRawDataAsync(dataStartDate, endDate, exchanges);
    }

    /// <summary>
    /// Load ALL available raw market data from all day folders in data/raw/
    /// Scans the directory and loads everything, regardless of dates
    /// WARNING: This can be slow if you have many days of data. Consider using LoadDataWithHistoryAsync instead.
    /// </summary>
    public async Task<(
        Dictionary<string, List<FundingRateDto>> FundingRates,
        Dictionary<string, Dictionary<string, List<PriceDto>>> PriceKlines,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? LiquidityMetrics
    )> LoadAllAvailableDataAsync(List<string> exchanges)
    {
        _logger.LogInformation("Loading ALL available raw data from {Path}", BaseDataPath);

        if (!Directory.Exists(BaseDataPath))
        {
            throw new DirectoryNotFoundException($"Raw data directory not found: {BaseDataPath}. Run 'collect' command first.");
        }

        // Find all day folders (format: yyyy-MM-dd)
        var dayFolders = Directory.GetDirectories(BaseDataPath)
            .Where(dir => DateTime.TryParseExact(
                Path.GetFileName(dir),
                "yyyy-MM-dd",
                null,
                System.Globalization.DateTimeStyles.None,
                out _))
            .OrderBy(dir => dir)
            .ToList();

        if (!dayFolders.Any())
        {
            throw new FileNotFoundException($"No day folders found in {BaseDataPath}. Run 'collect' command first.");
        }

        _logger.LogInformation("Found {Count} day folders to load", dayFolders.Count);

        // Parse dates from folder names to get min/max range
        var dates = dayFolders
            .Select(dir => DateTime.ParseExact(Path.GetFileName(dir), "yyyy-MM-dd", null))
            .ToList();

        var minDate = dates.Min();
        var maxDate = dates.Max();

        _logger.LogInformation("Loading data from {MinDate} to {MaxDate}",
            minDate.ToString("yyyy-MM-dd"), maxDate.ToString("yyyy-MM-dd"));

        // Use existing LoadRawDataAsync to load the entire range
        return await LoadRawDataAsync(minDate, maxDate, exchanges);
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
