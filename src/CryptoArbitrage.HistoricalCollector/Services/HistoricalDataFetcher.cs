using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.Exchanges;
using CryptoArbitrage.HistoricalCollector.Config;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Fetches historical market data from exchange APIs
/// Handles pagination, rate limiting, and data caching
/// </summary>
public class HistoricalDataFetcher
{
    private readonly ILogger<HistoricalDataFetcher> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly DataCollectionConfig _config;

    public HistoricalDataFetcher(
        ILogger<HistoricalDataFetcher> logger,
        IServiceProvider serviceProvider,
        DataCollectionConfig config)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
    }

    /// <summary>
    /// Fetch historical funding rates for all symbols from all exchanges
    /// Uses parallel requests with rate limiting for optimal performance
    /// </summary>
    public async Task<Dictionary<string, List<FundingRateDto>>> FetchAllFundingRates(
        DateTime startDate,
        DateTime endDate,
        List<string> exchanges,
        List<string> symbols)
    {
        var result = new Dictionary<string, List<FundingRateDto>>();

        // Process exchanges in parallel
        var exchangeTasks = exchanges.Select(async exchange =>
        {
            _logger.LogInformation("Fetching funding rates from {Exchange}...", exchange);

            // Create connector once per exchange (not per symbol!)
            using var scope = _serviceProvider.CreateScope();
            IExchangeConnector? connector = exchange switch
            {
                "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
                "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
                _ => null
            };

            if (connector == null)
            {
                _logger.LogWarning("No connector available for {Exchange}", exchange);
                return (exchange, new List<FundingRateDto>());
            }

            // Connect once for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Get rate limiting configuration for this exchange
            var exchangeConfig = _config.GetConfigForExchange(exchange);
            var maxConcurrent = exchangeConfig.MaxConcurrentRequests;
            var rateLimitDelay = exchangeConfig.RateLimitDelayMs;

            _logger.LogDebug("Using config for {Exchange}: MaxConcurrent={MaxConcurrent}, DelayMs={DelayMs}",
                exchange, maxConcurrent, rateLimitDelay);

            using var semaphore = new SemaphoreSlim(maxConcurrent);
            var rates = new List<FundingRateDto>();
            var ratesLock = new object();

            var symbolTasks = symbols.Select(async symbol =>
            {
                await semaphore.WaitAsync();
                try
                {
                    var symbolRates = await FetchFundingRatesForSymbol(
                        connector,
                        symbol,
                        startDate,
                        endDate);

                    lock (ratesLock)
                    {
                        rates.AddRange(symbolRates);
                    }

                    _logger.LogDebug("Fetched {Count} funding rates for {Symbol} from {Exchange}",
                        symbolRates.Count, symbol, exchange);

                    // Rate limiting delay
                    await Task.Delay(rateLimitDelay);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to fetch funding rates for {Symbol} from {Exchange}",
                        symbol, exchange);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            await Task.WhenAll(symbolTasks);

            _logger.LogInformation("Total funding rates from {Exchange}: {Count}", exchange, rates.Count);
            return (exchange, rates);
        });

        var exchangeResults = await Task.WhenAll(exchangeTasks);

        foreach (var (exchange, rates) in exchangeResults)
        {
            result[exchange] = rates;
        }

        return result;
    }

    /// <summary>
    /// Fetch historical 1-minute price klines for all symbols
    /// Uses parallel requests with rate limiting for optimal performance
    /// </summary>
    public async Task<Dictionary<string, Dictionary<string, List<PriceDto>>>> FetchAllPriceKlines(
        DateTime startDate,
        DateTime endDate,
        List<string> exchanges,
        List<string> symbols,
        string interval = "1m")
    {
        var result = new Dictionary<string, Dictionary<string, List<PriceDto>>>();

        // Process exchanges in parallel
        var exchangeTasks = exchanges.Select(async exchange =>
        {
            _logger.LogInformation("Fetching {Interval} klines from {Exchange}...", interval, exchange);

            // Create connector once per exchange (not per symbol!)
            using var scope = _serviceProvider.CreateScope();
            IExchangeConnector? connector = exchange switch
            {
                "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
                "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
                _ => null
            };

            if (connector == null)
            {
                _logger.LogWarning("No connector available for {Exchange}", exchange);
                return (exchange, new Dictionary<string, List<PriceDto>>());
            }

            // Connect once for public data access
            await connector.ConnectAsync(string.Empty, string.Empty);

            // Get rate limiting configuration for this exchange
            var exchangeConfig = _config.GetConfigForExchange(exchange);
            var maxConcurrent = exchangeConfig.MaxConcurrentRequests;
            var rateLimitDelay = exchangeConfig.RateLimitDelayMs;

            _logger.LogDebug("Using config for {Exchange}: MaxConcurrent={MaxConcurrent}, DelayMs={DelayMs}",
                exchange, maxConcurrent, rateLimitDelay);

            using var semaphore = new SemaphoreSlim(maxConcurrent);
            var exchangePrices = new Dictionary<string, List<PriceDto>>();
            var pricesLock = new object();

            var symbolTasks = symbols.Select(async symbol =>
            {
                await semaphore.WaitAsync();
                try
                {
                    var prices = await FetchPriceKlinesForSymbol(
                        connector,
                        symbol,
                        startDate,
                        endDate,
                        interval,
                        isPerpetual: true);

                    lock (pricesLock)
                    {
                        exchangePrices[symbol] = prices;
                    }

                    _logger.LogDebug("Fetched {Count} klines for {Symbol} from {Exchange}",
                        prices.Count, symbol, exchange);

                    // Rate limiting delay
                    await Task.Delay(rateLimitDelay);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to fetch klines for {Symbol} from {Exchange}",
                        symbol, exchange);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            await Task.WhenAll(symbolTasks);

            _logger.LogInformation("Total klines from {Exchange}: {Count}",
                exchange, exchangePrices.Sum(x => x.Value.Count));
            return (exchange, exchangePrices);
        });

        var exchangeResults = await Task.WhenAll(exchangeTasks);

        foreach (var (exchange, exchangePrices) in exchangeResults)
        {
            result[exchange] = exchangePrices;
        }

        return result;
    }

    /// <summary>
    /// Fetch CURRENT funding rates using bulk endpoint (1 call per exchange)
    /// Use for snapshot/real-time data collection - avoids rate limiting
    /// </summary>
    public async Task<Dictionary<string, List<FundingRateDto>>> FetchCurrentFundingRates(
        List<string> exchanges,
        List<string> symbols)
    {
        var result = new Dictionary<string, List<FundingRateDto>>();

        using var scope = _serviceProvider.CreateScope();

        foreach (var exchange in exchanges)
        {
            _logger.LogInformation("Fetching current funding rates from {Exchange}...", exchange);

            try
            {
                IExchangeConnector? connector = exchange switch
                {
                    "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
                    "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
                    _ => null
                };

                if (connector == null)
                {
                    _logger.LogWarning("No connector available for {Exchange}", exchange);
                    continue;
                }

                // Connect for public data access
                await connector.ConnectAsync(string.Empty, string.Empty);

                // Get funding rates for all symbols in ONE call
                var fundingRates = await connector.GetFundingRatesAsync(symbols);

                result[exchange] = fundingRates;
                _logger.LogInformation("Fetched {Count} current funding rates from {Exchange}",
                    fundingRates.Count, exchange);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to fetch current funding rates from {Exchange}", exchange);
                result[exchange] = new List<FundingRateDto>();
            }
        }

        return result;
    }

    /// <summary>
    /// Fetch CURRENT prices using bulk endpoint (1 call per exchange)
    /// Use for snapshot/real-time data collection - avoids rate limiting
    /// </summary>
    public async Task<Dictionary<string, Dictionary<string, List<PriceDto>>>> FetchCurrentPrices(
        List<string> exchanges,
        List<string> symbols)
    {
        var result = new Dictionary<string, Dictionary<string, List<PriceDto>>>();

        using var scope = _serviceProvider.CreateScope();

        foreach (var exchange in exchanges)
        {
            _logger.LogInformation("Fetching current prices from {Exchange}...", exchange);

            try
            {
                IExchangeConnector? connector = exchange switch
                {
                    "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
                    "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
                    _ => null
                };

                if (connector == null)
                {
                    _logger.LogWarning("No connector available for {Exchange}", exchange);
                    continue;
                }

                // Connect for public data access
                await connector.ConnectAsync(string.Empty, string.Empty);

                // Get perpetual prices for all symbols in ONE call
                var perpPrices = await connector.GetPerpetualPricesAsync(symbols);

                // Convert to List<PriceDto> format for compatibility with reconstructor
                var exchangePrices = new Dictionary<string, List<PriceDto>>();
                foreach (var (symbol, priceDto) in perpPrices)
                {
                    exchangePrices[symbol] = new List<PriceDto> { priceDto };
                }

                result[exchange] = exchangePrices;
                _logger.LogInformation("Fetched {Count} current prices from {Exchange}",
                    perpPrices.Count, exchange);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to fetch current prices from {Exchange}", exchange);
                result[exchange] = new Dictionary<string, List<PriceDto>>();
            }
        }

        return result;
    }

    private async Task<List<FundingRateDto>> FetchFundingRatesForSymbol(
        IExchangeConnector connector,
        string symbol,
        DateTime startDate,
        DateTime endDate)
    {
        try
        {
            // Use the connector's GetFundingRateHistoryAsync method
            // Note: The connector methods use the Binance.Net and Bybit.Net libraries which handle
            // rate limiting, retries, and proper API formatting automatically
            var rates = await connector.GetFundingRateHistoryAsync(symbol, startDate, endDate);

            return rates;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rates for {Symbol} from {Exchange}",
                symbol, connector.ExchangeName);
            return new List<FundingRateDto>();
        }
    }


    private async Task<List<PriceDto>> FetchPriceKlinesForSymbol(
        IExchangeConnector connector,
        string symbol,
        DateTime startDate,
        DateTime endDate,
        string interval,
        bool isPerpetual)
    {
        try
        {
            // Convert string interval to KlineInterval enum
            var klineInterval = ParseInterval(interval);

            // Use the connector's GetKlinesAsync method
            // Note: The connector methods use the Binance.Net and Bybit.Net libraries which handle
            // rate limiting, retries, and proper API formatting automatically
            var klines = await connector.GetKlinesAsync(symbol, startDate, endDate, klineInterval);

            // Convert KlineDto to PriceDto format
            var prices = klines.Select(k => new PriceDto
            {
                Symbol = symbol,
                Price = k.Close,
                Volume24h = k.Volume,
                Timestamp = k.OpenTime
            }).ToList();

            return prices;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching klines for {Symbol} from {Exchange}",
                symbol, connector.ExchangeName);
            return new List<PriceDto>();
        }
    }

    /// <summary>
    /// Parse interval string (e.g., "1m", "5m", "1h") to KlineInterval enum
    /// </summary>
    private KlineInterval ParseInterval(string interval)
    {
        return interval.ToLower() switch
        {
            "1m" => KlineInterval.OneMinute,
            "5m" => KlineInterval.FiveMinutes,
            "15m" => KlineInterval.FifteenMinutes,
            "30m" => KlineInterval.ThirtyMinutes,
            "1h" => KlineInterval.OneHour,
            "4h" => KlineInterval.FourHours,
            "1d" => KlineInterval.OneDay,
            _ => KlineInterval.OneMinute // default
        };
    }
}
