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
/// Collects funding rates from all configured exchanges
/// </summary>
public class FundingRateCollector : IDataCollector<FundingRateDto, FundingRateCollectorConfiguration>
{
    private readonly ILogger<FundingRateCollector> _logger;
    private readonly IDataRepository<FundingRateDto> _repository;
    private readonly FundingRateCollectorConfiguration _configuration;
    private readonly IServiceProvider _serviceProvider;
    private readonly SymbolDiscoveryService _symbolDiscoveryService;

    public FundingRateCollectorConfiguration Configuration => _configuration;
    public CollectionResult<FundingRateDto>? LastResult { get; private set; }
    public DateTime? LastSuccessfulCollection { get; private set; }

    public FundingRateCollector(
        ILogger<FundingRateCollector> logger,
        IDataRepository<FundingRateDto> repository,
        FundingRateCollectorConfiguration configuration,
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
                result.ErrorMessage = "No symbols available for collection";
                LastResult = result;
                return result;
            }

            _logger.LogDebug("Collecting funding rates for {Count} symbols from exchanges",
                symbols.Count);

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

            // Aggregate results from all exchanges
            var allData = new Dictionary<string, FundingRateDto>();
            foreach (var exchangeData in exchangeResults)
            {
                foreach (var (key, value) in exchangeData)
                {
                    allData[key] = value;
                }
            }

            // Store in repository (memory + database)
            if (allData.Any())
            {
                var ttl = Configuration.CacheTtlMinutes.HasValue
                    ? TimeSpan.FromMinutes(Configuration.CacheTtlMinutes.Value)
                    : TimeSpan.FromHours(1);

                await _repository.StoreBatchAsync(allData, ttl, cancellationToken);
            }

            result.Data = allData;
            result.Success = true;
            LastSuccessfulCollection = DateTime.UtcNow;

            _logger.LogInformation(
                "Collected {Count} funding rates from {Exchanges} exchanges",
                allData.Count,
                connectors.Count(c => c.Connector != null));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting funding rates");
            result.Success = false;
            result.ErrorMessage = ex.Message;
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;
        LastResult = result;

        return result;
    }

    private async Task<Dictionary<string, FundingRateDto>> CollectFromExchangeAsync(
        string exchangeName,
        IExchangeConnector connector,
        List<string> symbols,
        CancellationToken cancellationToken)
    {
        var result = new Dictionary<string, FundingRateDto>();

        try
        {
            _logger.LogDebug("Fetching funding rates from {Exchange}", exchangeName);

            // Initialize connector for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            var fundingRates = await connector.GetFundingRatesAsync(symbols);

            foreach (var rate in fundingRates)
            {
                var key = DataCollectionConstants.CacheKeys.BuildFundingRateKey(exchangeName, rate.Symbol);
                result[key] = rate;
            }

            _logger.LogDebug("Fetched {Count} funding rates from {Exchange}",
                fundingRates.Count, exchangeName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect funding rates from {Exchange}", exchangeName);
        }

        return result;
    }
}
