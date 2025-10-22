using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Collects market prices (spot and perpetual) from all configured exchanges
/// </summary>
public class MarketPriceCollector : IDataCollector<MarketDataSnapshot, MarketPriceCollectorConfiguration>
{
    private readonly ILogger<MarketPriceCollector> _logger;
    private readonly IDataRepository<MarketDataSnapshot> _repository;
    private readonly MarketPriceCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly SymbolDiscoveryService _symbolDiscoveryService;

    public MarketPriceCollectorConfiguration Configuration => _configuration;
    public CollectionResult<MarketDataSnapshot>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public MarketPriceCollector(
        ILogger<MarketPriceCollector> logger,
        IDataRepository<MarketDataSnapshot> repository,
        MarketPriceCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        SymbolDiscoveryService symbolDiscoveryService)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _symbolDiscoveryService = symbolDiscoveryService;
    }

    public async Task<CollectionResult<MarketDataSnapshot>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<MarketDataSnapshot>();

        try
        {
            // Get active symbols from discovery service
            var symbols = await _symbolDiscoveryService.GetActiveSymbolsAsync(cancellationToken);
            if (!symbols.Any())
            {
                result.Success = false;
                result.ErrorMessage = "No symbols available for collection";
                LastResult = result;
                return result;
            }

            _logger.LogDebug("Collecting market prices for {Count} symbols from exchanges", symbols.Count);

            // Get exchange connectors - use scoped services
            using var scope = _serviceProvider.CreateScope();
            var binanceConnector = scope.ServiceProvider.GetService<BinanceConnector>();
            var bybitConnector = scope.ServiceProvider.GetService<BybitConnector>();

            var connectors = new List<(string Name, IExchangeConnector? Connector)>
            {
                ("Binance", binanceConnector),
                ("Bybit", bybitConnector)
            };

            // Collect from all exchanges in parallel (with limit)
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var tasks = connectors
                .Where(c => c.Connector != null)
                .Select(async connector =>
                {
                    await semaphore.WaitAsync(cancellationToken);
                    try
                    {
                        return await CollectFromExchangeAsync(
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

            // Aggregate results from all exchanges into a single snapshot
            var snapshot = new MarketDataSnapshot
            {
                SpotPrices = new Dictionary<string, Dictionary<string, PriceDto>>(),
                PerpPrices = new Dictionary<string, Dictionary<string, PriceDto>>(),
                FetchedAt = DateTime.UtcNow
            };

            foreach (var (exchange, spotPrices, perpPrices) in exchangeResults)
            {
                if (spotPrices.Any())
                {
                    snapshot.SpotPrices[exchange] = spotPrices;
                }
                if (perpPrices.Any())
                {
                    snapshot.PerpPrices[exchange] = perpPrices;
                }
            }

            // Store snapshot in repository (memory cache)
            if (snapshot.SpotPrices.Any() || snapshot.PerpPrices.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromMinutes(5);

                var key = DataCollectionConstants.CacheKeys.MarketDataSnapshot;
                await _repository.StoreAsync(key, snapshot, ttl, cancellationToken);

                result.Data = new Dictionary<string, MarketDataSnapshot> { { key, snapshot } };
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation(
                    "Collected market prices from {Exchanges} exchanges: {SpotCount} spot, {PerpCount} perp",
                    snapshot.SpotPrices.Count,
                    snapshot.SpotPrices.Values.Sum(d => d.Count),
                    snapshot.PerpPrices.Values.Sum(d => d.Count));
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No market prices collected from any exchange";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting market prices");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    private async Task<(string Exchange, Dictionary<string, PriceDto> SpotPrices, Dictionary<string, PriceDto> PerpPrices)>
        CollectFromExchangeAsync(
            string exchangeName,
            IExchangeConnector connector,
            List<string> symbols,
            CancellationToken cancellationToken)
    {
        var spotPrices = new Dictionary<string, PriceDto>();
        var perpPrices = new Dictionary<string, PriceDto>();

        try
        {
            _logger.LogDebug("Fetching market prices from {Exchange}", exchangeName);

            // Initialize connector for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Fetch spot prices
            var spotPricesResult = await connector.GetSpotPricesAsync(symbols);
            foreach (var (symbol, price) in spotPricesResult)
            {
                spotPrices[symbol] = price;
            }

            // Fetch perpetual prices
            var perpPricesResult = await connector.GetPerpetualPricesAsync(symbols);
            foreach (var (symbol, price) in perpPricesResult)
            {
                perpPrices[symbol] = price;
            }

            _logger.LogDebug("Fetched {SpotCount} spot prices and {PerpCount} perp prices from {Exchange}",
                spotPrices.Count, perpPrices.Count, exchangeName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect market prices from {Exchange}", exchangeName);
        }

        return (exchangeName, spotPrices, perpPrices);
    }
}
