using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Collects historical funding rates and calculates 3-day and 24h averages.
/// Updates existing funding rate records in repository with historical averages.
/// Runs independently on its own schedule (typically every 20 minutes).
/// </summary>
public class FundingRateHistoryCollector : IDataCollector<FundingRateDto, FundingRateHistoryCollectorConfiguration>
{
    private readonly ILogger<FundingRateHistoryCollector> _logger;
    private readonly IDataRepository<FundingRateDto> _repository;
    private readonly FundingRateHistoryCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly SymbolDiscoveryService _symbolDiscoveryService;

    public FundingRateHistoryCollectorConfiguration Configuration => _configuration;
    public CollectionResult<FundingRateDto>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public FundingRateHistoryCollector(
        ILogger<FundingRateHistoryCollector> logger,
        IDataRepository<FundingRateDto> repository,
        FundingRateHistoryCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        SymbolDiscoveryService symbolDiscoveryService)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _symbolDiscoveryService = symbolDiscoveryService;
    }

    public async Task<CollectionResult<FundingRateDto>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<FundingRateDto>();

        try
        {
            // Get active symbols from discovery service
            var symbols = await _symbolDiscoveryService.GetActiveSymbolsAsync(cancellationToken);
            if (!symbols.Any())
            {
                result.Success = false;
                result.ErrorMessage = "No symbols available for historical collection";
                LastResult = result;
                return result;
            }

            _logger.LogInformation("Collecting historical funding rates for {Count} symbols", symbols.Count);

            // Retry logic for first run: if no funding rates exist yet, wait and retry
            const int maxRetries = 3;
            const int retryDelayMs = 2000;
            int retryCount = 0;
            int updatedCount = 0;

            while (retryCount <= maxRetries)
            {
                // Get exchange connectors - use scoped services
                using var scope = _serviceProvider.CreateScope();
                var binanceConnector = scope.ServiceProvider.GetService<BinanceConnector>();
                var bybitConnector = scope.ServiceProvider.GetService<BybitConnector>();

                var connectors = new List<(string Name, IExchangeConnector? Connector)>
                {
                    ("Binance", binanceConnector),
                    ("Bybit", bybitConnector)
                };

                // Collect from all exchanges in parallel
                var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
                var tasks = connectors
                    .Where(c => c.Connector != null)
                    .Select(async connector =>
                    {
                        await semaphore.WaitAsync(cancellationToken);
                        try
                        {
                            return await CollectHistoricalAveragesForExchangeAsync(
                                connector.Name,
                                connector.Connector!,
                                symbols,
                                cancellationToken);
                        }
                        finally
                        {
                            semaphore.Release();
                        }
                    });

                var exchangeResults = await Task.WhenAll(tasks);

                // Aggregate results from all exchanges
                var allData = new Dictionary<string, FundingRateDto>();
                updatedCount = 0;
                foreach (var exchangeData in exchangeResults)
                {
                    foreach (var (key, value) in exchangeData)
                    {
                        allData[key] = value;
                        updatedCount++;
                    }
                }

                // If we got results, break out of retry loop
                if (updatedCount > 0 || LastSuccessfulCollection != null)
                {
                    result.Data = allData;
                    result.Success = true;
                    LastSuccessfulCollection = DateTime.UtcNow;

                    _logger.LogInformation(
                        "Updated historical averages for {Count} funding rates from {Exchanges} exchanges",
                        updatedCount,
                        connectors.Count(c => c.Connector != null));
                    break;
                }

                // No results on first run - retry after delay
                if (retryCount < maxRetries)
                {
                    _logger.LogWarning(
                        "No funding rates found in repository (attempt {Attempt}/{MaxAttempts}). Waiting {Delay}ms for FundingRateCollector to populate data...",
                        retryCount + 1, maxRetries + 1, retryDelayMs);
                    await Task.Delay(retryDelayMs, cancellationToken);
                    retryCount++;
                }
                else
                {
                    _logger.LogWarning(
                        "No funding rates available after {Attempts} attempts. Will retry on next collection cycle.",
                        maxRetries + 1);
                    result.Data = allData;
                    result.Success = true; // Don't treat as failure - just no data yet
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting historical funding rates");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    private async Task<Dictionary<string, FundingRateDto>> CollectHistoricalAveragesForExchangeAsync(
        string exchangeName,
        IExchangeConnector connector,
        List<string> symbols,
        CancellationToken cancellationToken)
    {
        var result = new Dictionary<string, FundingRateDto>();

        try
        {
            _logger.LogInformation("Fetching historical funding rates from {Exchange}", exchangeName);

            // Initialize connector for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Calculate time range
            var endTime = DateTime.UtcNow;
            var startTime = endTime.AddDays(-Configuration.HistoryDays);

            // Fetch historical data for all symbols IN PARALLEL
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var historicalTasks = symbols.Select(async symbol =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    // Get existing funding rate from repository
                    var key = DataCollectionConstants.CacheKeys.BuildFundingRateKey(exchangeName, symbol);
                    var existingRate = await _repository.GetAsync(key, cancellationToken);

                    if (existingRate == null)
                    {
                        _logger.LogDebug("No existing funding rate for {Symbol} on {Exchange}, skipping historical update",
                            symbol, exchangeName);
                        return ((string?)null, (FundingRateDto?)null);
                    }

                    // Fetch and calculate historical averages
                    await FetchAndCalculateHistoricalAveragesAsync(
                        connector,
                        exchangeName,
                        existingRate,
                        startTime,
                        endTime,
                        cancellationToken);

                    return (key, existingRate);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var historicalResults = await Task.WhenAll(historicalTasks);

            // Store updated rates back to repository
            foreach (var item in historicalResults.Where(r => r.Item1 != null && r.Item2 != null))
            {
                var key = item.Item1!;
                var rate = item.Item2!;
                result[key] = rate;

                // Update in repository with extended TTL
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromHours(2);

                await _repository.StoreAsync(key, rate, ttl, cancellationToken);
            }

            _logger.LogInformation("Updated {Count} historical averages for {Exchange}",
                result.Count, exchangeName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect historical funding rates from {Exchange}", exchangeName);
        }

        return result;
    }

    /// <summary>
    /// Fetch historical funding rates and calculate 3-day and 24h averages for a single symbol
    /// </summary>
    private async Task FetchAndCalculateHistoricalAveragesAsync(
        IExchangeConnector connector,
        string exchangeName,
        FundingRateDto rate,
        DateTime startTime,
        DateTime endTime,
        CancellationToken cancellationToken)
    {
        try
        {
            var historicalRates = await connector.GetFundingRateHistoryAsync(rate.Symbol, startTime, endTime);

            if (historicalRates.Any())
            {
                // Calculate 3-day time-weighted average
                decimal totalWeightedRate3d = 0;
                int totalWeight3d = 0;

                foreach (var historicalRate in historicalRates)
                {
                    // Weight = number of times this rate occurs per day
                    int periodsPerDay = 24 / historicalRate.FundingIntervalHours;
                    totalWeightedRate3d += historicalRate.Rate * periodsPerDay;
                    totalWeight3d += periodsPerDay;
                }

                // Calculate the 3-day average rate per funding period
                if (totalWeight3d > 0)
                {
                    rate.Average3DayRate = totalWeightedRate3d / totalWeight3d;
                }

                // Calculate 24-hour time-weighted average (using same data, no extra API call)
                var oneDayAgo = DateTime.UtcNow.AddHours(-24);
                var last24hRates = historicalRates.Where(r => r.FundingTime >= oneDayAgo).ToList();

                if (last24hRates.Any())
                {
                    decimal totalWeightedRate24h = 0;
                    int totalWeight24h = 0;

                    foreach (var historicalRate in last24hRates)
                    {
                        int periodsPerDay = 24 / historicalRate.FundingIntervalHours;
                        totalWeightedRate24h += historicalRate.Rate * periodsPerDay;
                        totalWeight24h += periodsPerDay;
                    }

                    if (totalWeight24h > 0)
                    {
                        rate.Average24hRate = totalWeightedRate24h / totalWeight24h;
                    }
                }

                _logger.LogDebug("✓ {Symbol} on {Exchange}: 3D avg = {Avg3D:F6}, 24h avg = {Avg24h:F6}",
                    rate.Symbol, exchangeName, rate.Average3DayRate, rate.Average24hRate);
            }
            else
            {
                _logger.LogWarning("No historical data returned for {Symbol} on {Exchange}",
                    rate.Symbol, exchangeName);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fetch historical data for {Symbol} on {Exchange}",
                rate.Symbol, exchangeName);
        }
    }
}
