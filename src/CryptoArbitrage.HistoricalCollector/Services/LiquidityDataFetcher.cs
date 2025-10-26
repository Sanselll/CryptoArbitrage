using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.Exchanges;
using CryptoArbitrage.HistoricalCollector.Config;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Fetches current liquidity metrics (orderbook snapshots) to use as proxy for historical liquidity
/// Assumption: Liquidity patterns are relatively stable over short time periods (weeks/months)
/// </summary>
public class LiquidityDataFetcher
{
    private readonly ILogger<LiquidityDataFetcher> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly DataCollectionConfig _config;

    public LiquidityDataFetcher(
        ILogger<LiquidityDataFetcher> logger,
        IServiceProvider serviceProvider,
        DataCollectionConfig config)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
    }

    /// <summary>
    /// Fetch current liquidity metrics for all symbols from all exchanges
    /// This will be used as a proxy for historical liquidity
    /// Uses parallel processing for optimal performance
    /// </summary>
    public async Task<Dictionary<string, Dictionary<string, LiquidityMetricsDto>>> FetchCurrentLiquidityMetrics(
        List<string> exchanges,
        List<string> symbols)
    {
        _logger.LogInformation("Fetching current liquidity metrics from {Count} exchanges for {SymbolCount} symbols",
            exchanges.Count, symbols.Count);

        var result = new Dictionary<string, Dictionary<string, LiquidityMetricsDto>>();

        // Process exchanges in parallel
        var exchangeTasks = exchanges.Select(async exchange =>
        {
            _logger.LogInformation("Fetching liquidity from {Exchange}...", exchange);

            try
            {
                var exchangeLiquidity = await FetchFromExchangeAsync(exchange, symbols);

                _logger.LogInformation("Fetched liquidity for {Count} symbols from {Exchange}",
                    exchangeLiquidity.Count, exchange);

                return (exchange, exchangeLiquidity);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to fetch liquidity from {Exchange}", exchange);
                return (exchange, new Dictionary<string, LiquidityMetricsDto>());
            }
        });

        var exchangeResults = await Task.WhenAll(exchangeTasks);

        foreach (var (exchange, exchangeLiquidity) in exchangeResults)
        {
            result[exchange] = exchangeLiquidity;
        }

        var totalCount = result.Sum(x => x.Value.Count);
        _logger.LogInformation("Total liquidity metrics fetched: {Count}", totalCount);

        return result;
    }

    /// <summary>
    /// Fetch liquidity metrics from a specific exchange
    /// Uses parallel processing with rate limiting
    /// </summary>
    private async Task<Dictionary<string, LiquidityMetricsDto>> FetchFromExchangeAsync(
        string exchange,
        List<string> symbols)
    {
        var metrics = new Dictionary<string, LiquidityMetricsDto>();
        var metricsLock = new object();

        // Create a scope to get exchange connector
        using var scope = _serviceProvider.CreateScope();

        IExchangeConnector? connector = exchange switch
        {
            "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
            "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
            _ => null
        };

        if (connector == null)
        {
            _logger.LogWarning("No connector available for exchange {Exchange}", exchange);
            return metrics;
        }

        // Connect to exchange (public API, no auth needed)
        await connector.ConnectAsync(string.Empty, string.Empty);

        // Get rate limiting configuration for this exchange
        var exchangeConfig = _config.GetConfigForExchange(exchange);
        var maxConcurrent = exchangeConfig.MaxConcurrentRequests;
        var rateLimitDelay = exchangeConfig.RateLimitDelayMs;

        _logger.LogDebug("Using config for {Exchange}: MaxConcurrent={MaxConcurrent}, DelayMs={DelayMs}",
            exchange, maxConcurrent, rateLimitDelay);

        using var semaphore = new SemaphoreSlim(maxConcurrent);

        var symbolTasks = symbols.Select(async symbol =>
        {
            await semaphore.WaitAsync();
            try
            {
                var liquidityMetric = await connector.GetLiquidityMetricsAsync(symbol);

                if (liquidityMetric != null)
                {
                    lock (metricsLock)
                    {
                        metrics[symbol] = liquidityMetric;
                    }

                    _logger.LogDebug("Fetched liquidity for {Symbol} from {Exchange}: Spread={Spread}%, Depth=${Depth}",
                        symbol, exchange, liquidityMetric.BidAskSpreadPercent * 100, liquidityMetric.OrderbookDepthUsd);
                }
                else
                {
                    _logger.LogWarning("No liquidity metrics returned for {Symbol} from {Exchange}", symbol, exchange);
                }

                // Rate limiting delay
                await Task.Delay(rateLimitDelay);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to fetch liquidity for {Symbol} from {Exchange}", symbol, exchange);
            }
            finally
            {
                semaphore.Release();
            }
        });

        await Task.WhenAll(symbolTasks);

        return metrics;
    }
}
