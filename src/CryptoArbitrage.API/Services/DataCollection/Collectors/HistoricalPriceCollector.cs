using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Collects historical price data (KLines) for calculating average price spreads.
/// Fetches hourly perpetual futures prices for 3 days to enable 24h and 3D spread projections.
/// Runs independently on its own schedule (typically every hour).
/// </summary>
public class HistoricalPriceCollector : IDataCollector<Dictionary<string, List<HistoricalPriceDto>>, HistoricalPriceCollectorConfiguration>
{
    private readonly ILogger<HistoricalPriceCollector> _logger;
    private readonly IDataRepository<Dictionary<string, List<HistoricalPriceDto>>> _repository;
    private readonly HistoricalPriceCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly SymbolDiscoveryService _symbolDiscoveryService;

    public HistoricalPriceCollectorConfiguration Configuration => _configuration;
    public CollectionResult<Dictionary<string, List<HistoricalPriceDto>>>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public HistoricalPriceCollector(
        ILogger<HistoricalPriceCollector> logger,
        IDataRepository<Dictionary<string, List<HistoricalPriceDto>>> repository,
        HistoricalPriceCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        SymbolDiscoveryService symbolDiscoveryService)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _symbolDiscoveryService = symbolDiscoveryService;
    }

    public async Task<CollectionResult<Dictionary<string, List<HistoricalPriceDto>>>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<Dictionary<string, List<HistoricalPriceDto>>>();

        try
        {
            // Get active symbols from discovery service
            var symbols = await _symbolDiscoveryService.GetActiveSymbolsAsync(cancellationToken);
            if (!symbols.Any())
            {
                result.Success = false;
                result.ErrorMessage = "No symbols available for historical price collection";
                LastResult = result;
                return result;
            }

            _logger.LogInformation("Collecting historical prices for {Count} symbols", symbols.Count);

            // Get exchange connectors - use scoped services
            using var scope = _serviceProvider.CreateScope();
            var binanceConnector = scope.ServiceProvider.GetService<BinanceConnector>();
            var bybitConnector = scope.ServiceProvider.GetService<BybitConnector>();

            var connectors = new List<(string Name, IExchangeConnector? Connector)>
            {
                ("Binance", binanceConnector),
                ("Bybit", bybitConnector)
            };

            // Calculate time range
            var endTime = DateTime.UtcNow;
            var startTime = endTime.AddDays(-Configuration.HistoryDays);

            // Collect from all exchanges in parallel
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var tasks = connectors
                .Where(c => c.Connector != null)
                .Select(async connector =>
                {
                    await semaphore.WaitAsync(cancellationToken);
                    try
                    {
                        return await CollectHistoricalPricesForExchangeAsync(
                            connector.Name,
                            connector.Connector!,
                            symbols,
                            startTime,
                            endTime,
                            cancellationToken);
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                });

            var exchangeResults = await Task.WhenAll(tasks);

            // Aggregate results from all exchanges - key format: "exchange:symbol"
            var allData = new Dictionary<string, List<HistoricalPriceDto>>();
            int totalPrices = 0;
            foreach (var exchangeData in exchangeResults)
            {
                foreach (var (key, value) in exchangeData)
                {
                    allData[key] = value;
                    totalPrices += value.Count;
                }
            }

            if (allData.Any())
            {
                result.Data = new Dictionary<string, Dictionary<string, List<HistoricalPriceDto>>>
                {
                    { "prices", allData }
                };
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation(
                    "Collected {TotalPrices} historical price points across {SymbolCount} symbols from {ExchangeCount} exchanges",
                    totalPrices,
                    allData.Count,
                    connectors.Count(c => c.Connector != null));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No historical prices collected";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting historical prices");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    private async Task<Dictionary<string, List<HistoricalPriceDto>>> CollectHistoricalPricesForExchangeAsync(
        string exchangeName,
        IExchangeConnector connector,
        List<string> symbols,
        DateTime startTime,
        DateTime endTime,
        CancellationToken cancellationToken)
    {
        var result = new Dictionary<string, List<HistoricalPriceDto>>();

        try
        {
            _logger.LogInformation("Fetching historical prices from {Exchange}", exchangeName);

            // Initialize connector for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Fetch historical KLines for all symbols IN PARALLEL
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var historicalTasks = symbols.Select(async symbol =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    // Fetch klines (candlestick data)
                    var klines = await connector.GetKlinesAsync(
                        symbol,
                        startTime,
                        endTime,
                        Configuration.KlineInterval);

                    if (!klines.Any())
                    {
                        _logger.LogDebug("No klines returned for {Symbol} on {Exchange}",
                            symbol, exchangeName);
                        return ((string?)null, (List<HistoricalPriceDto>?)null);
                    }

                    // Convert klines to historical prices (using close price)
                    var historicalPrices = klines.Select(k => new HistoricalPriceDto
                    {
                        Exchange = exchangeName,
                        Symbol = symbol,
                        Price = k.Close,  // Use closing price of each candle
                        Timestamp = k.CloseTime
                    }).ToList();

                    var key = DataCollectionConstants.CacheKeys.BuildKey(exchangeName, symbol);

                    _logger.LogDebug("Collected {Count} historical prices for {Symbol} on {Exchange}",
                        historicalPrices.Count, symbol, exchangeName);

                    return (key, historicalPrices);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to fetch klines for {Symbol} on {Exchange}",
                        symbol, exchangeName);
                    return ((string?)null, (List<HistoricalPriceDto>?)null);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var historicalResults = await Task.WhenAll(historicalTasks);

            // Store historical prices in repository
            foreach (var item in historicalResults.Where(r => r.Item1 != null && r.Item2 != null))
            {
                var key = item.Item1!;
                var prices = item.Item2!;
                result[key] = prices;

                // Update in repository with TTL
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromHours(2);

                await _repository.StoreAsync(key, new Dictionary<string, List<HistoricalPriceDto>> { { key, prices } }, ttl, cancellationToken);
            }

            _logger.LogInformation("Collected historical prices for {Count} symbols on {Exchange}",
                result.Count, exchangeName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect historical prices from {Exchange}", exchangeName);
        }

        return result;
    }
}
