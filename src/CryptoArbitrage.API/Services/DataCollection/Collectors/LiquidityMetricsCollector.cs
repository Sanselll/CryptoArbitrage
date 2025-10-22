using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection.Collectors;

/// <summary>
/// Collects liquidity metrics (bid-ask spread, orderbook depth) from all configured exchanges
/// </summary>
public class LiquidityMetricsCollector : IDataCollector<LiquidityMetricsDto, LiquidityCollectorConfiguration>
{
    private readonly ILogger<LiquidityMetricsCollector> _logger;
    private readonly IDataRepository<LiquidityMetricsDto> _repository;
    private readonly LiquidityCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly SymbolDiscoveryService _symbolDiscoveryService;

    public LiquidityCollectorConfiguration Configuration => _configuration;
    public CollectionResult<LiquidityMetricsDto>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public LiquidityMetricsCollector(
        ILogger<LiquidityMetricsCollector> logger,
        IDataRepository<LiquidityMetricsDto> repository,
        LiquidityCollectorConfiguration configuration,
        IServiceProvider serviceProvider,
        SymbolDiscoveryService symbolDiscoveryService)
    {
        _logger = logger;
        _repository = repository;
        _configuration = configuration;
        _serviceProvider = serviceProvider;
        _symbolDiscoveryService = symbolDiscoveryService;
    }

    public async Task<CollectionResult<LiquidityMetricsDto>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new CollectionResult<LiquidityMetricsDto>();

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
            var allMetrics = new Dictionary<string, LiquidityMetricsDto>();
            var tasks = connectors
                .Where(c => c.Connector != null)
                .Select(async connector =>
                {
                    return await CollectFromExchangeAsync(
                        connector.Name,
                        connector.Connector!,
                        symbols,
                        cancellationToken);
                });

            var exchangeResults = await Task.WhenAll(tasks);

            // Store each metric in repository
            var ttl = Configuration.CacheTtlMinutes.HasValue
                ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                : TimeSpan.FromMinutes(10);

            int successCount = 0;
            foreach (var (exchange, metricsDict) in exchangeResults)
            {
                foreach (var (symbol, metrics) in metricsDict)
                {
                    if (metrics != null)
                    {
                        var key = DataCollectionConstants.CacheKeys.BuildLiquidityMetricKey(exchange, symbol);
                        await _repository.StoreAsync(key, metrics, ttl, cancellationToken);
                        allMetrics[key] = metrics;
                        successCount++;
                    }
                }
            }

            if (successCount > 0)
            {
                result.Data = allMetrics;
                result.Success = true;
                LastSuccessfulCollection = DateTime.UtcNow;

                _logger.LogInformation(
                    "Collected liquidity metrics for {Count} symbol-exchange pairs",
                    successCount);
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "No liquidity metrics collected from any exchange";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting liquidity metrics");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    private async Task<(string Exchange, Dictionary<string, LiquidityMetricsDto?> Metrics)>
        CollectFromExchangeAsync(
            string exchangeName,
            IExchangeConnector connector,
            List<string> symbols,
            CancellationToken cancellationToken)
    {
        var metricsDict = new Dictionary<string, LiquidityMetricsDto?>();

        try
        {
            _logger.LogDebug("Fetching liquidity metrics from {Exchange}", exchangeName);

            // Initialize connector for public data access ONCE per exchange
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Fetch liquidity metrics for all symbols in parallel (with limit)
            var semaphore = new SemaphoreSlim(Configuration.MaxParallelFetches);
            var tasks = symbols.Select(async symbol =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    var metrics = await connector.GetLiquidityMetricsAsync(symbol);
                    return (symbol, metrics);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to collect liquidity metrics for {Symbol} from {Exchange}", symbol, exchangeName);
                    return (symbol, (LiquidityMetricsDto?)null);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var results = await Task.WhenAll(tasks);

            foreach (var (symbol, metrics) in results)
            {
                metricsDict[symbol] = metrics;
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect liquidity metrics from {Exchange}", exchangeName);
        }

        return (exchangeName, metricsDict);
    }
}
