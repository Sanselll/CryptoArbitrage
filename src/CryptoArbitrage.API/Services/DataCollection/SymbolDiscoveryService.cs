using CryptoArbitrage.API.Config;
using System.Diagnostics;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Services.DataCollection;

/// <summary>
/// Service responsible for discovering and managing active trading symbols
/// Implements auto-discovery from exchanges with caching
/// </summary>
public class SymbolDiscoveryService
{
    private readonly ILogger<SymbolDiscoveryService> _logger;
    private readonly ArbitrageConfig _config;
    private readonly IServiceProvider _serviceProvider;

    private List<string>? _cachedSymbols;
    private DateTime? _lastDiscovery;
    private readonly SemaphoreSlim _discoveryLock = new(1, 1);

    // Refresh symbols every hour by default
    private readonly TimeSpan _cacheExpiration = TimeSpan.FromHours(1);

    public SymbolDiscoveryService(
        ILogger<SymbolDiscoveryService> logger,
        ArbitrageConfig config,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _config = config;
        _serviceProvider = serviceProvider;
    }

    /// <summary>
    /// Gets the active symbols list. Uses cached symbols if available and not expired.
    /// </summary>
    public async Task<List<string>> GetActiveSymbolsAsync(CancellationToken cancellationToken = default)
    {
        // Return cached symbols if still valid
        if (_cachedSymbols != null &&
            _lastDiscovery.HasValue &&
            DateTime.UtcNow - _lastDiscovery.Value < _cacheExpiration)
        {
            _logger.LogDebug("Returning {Count} cached symbols (age: {Age}s)",
                _cachedSymbols.Count,
                (DateTime.UtcNow - _lastDiscovery.Value).TotalSeconds);
            return _cachedSymbols;
        }

        // Need to discover/refresh symbols
        await _discoveryLock.WaitAsync(cancellationToken);
        try
        {
            // Double-check after acquiring lock
            if (_cachedSymbols != null &&
                _lastDiscovery.HasValue &&
                DateTime.UtcNow - _lastDiscovery.Value < _cacheExpiration)
            {
                return _cachedSymbols;
            }

            // Perform discovery
            var symbols = await DiscoverSymbolsAsync(cancellationToken);

            _cachedSymbols = symbols;
            _lastDiscovery = DateTime.UtcNow;

            return symbols;
        }
        finally
        {
            _discoveryLock.Release();
        }
    }

    /// <summary>
    /// Forces a refresh of the symbols list, bypassing cache
    /// </summary>
    public async Task<List<string>> RefreshSymbolsAsync(CancellationToken cancellationToken = default)
    {
        await _discoveryLock.WaitAsync(cancellationToken);
        try
        {
            var symbols = await DiscoverSymbolsAsync(cancellationToken);
            _cachedSymbols = symbols;
            _lastDiscovery = DateTime.UtcNow;
            return symbols;
        }
        finally
        {
            _discoveryLock.Release();
        }
    }

    private async Task<List<string>> DiscoverSymbolsAsync(CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();

        // If auto-discovery is disabled, use static config
        if (!_config.AutoDiscoverSymbols)
        {
            var configSymbols = _config.WatchedSymbols ?? new List<string>();
            if (!configSymbols.Any())
            {
                _logger.LogWarning("Auto-discovery disabled and no static symbols configured");
                return new List<string>();
            }

            _logger.LogInformation("Using {Count} static symbols from configuration", configSymbols.Count);
            return configSymbols;
        }

        // Auto-discovery enabled
        _logger.LogInformation("Starting symbol auto-discovery...");

        using var scope = _serviceProvider.CreateScope();
        var binanceConnector = scope.ServiceProvider.GetService<BinanceConnector>();
        var bybitConnector = scope.ServiceProvider.GetService<BybitConnector>();

        var connectors = new List<(string Name, IExchangeConnector? Connector)>
        {
            ("Binance", binanceConnector),
            ("Bybit", bybitConnector)
        };

        // Collect symbols from all exchanges
        var symbolSets = new List<List<string>>();
        foreach (var (name, connector) in connectors.Where(c => c.Connector != null))
        {
            try
            {
                // Connect without API credentials for public data
                await connector!.ConnectAsync(string.Empty, string.Empty);

                var exchangeSymbols = await connector.GetActiveSymbolsAsync(
                    _config.MinDailyVolumeUsd,
                    _config.MaxSymbolCount,
                    _config.MinHighPriorityFundingRate);

                symbolSets.Add(exchangeSymbols);

                _logger.LogInformation("Discovered {Count} symbols from {Exchange}",
                    exchangeSymbols.Count, name);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to discover symbols from {Exchange}", name);
            }
        }

        // Merge and deduplicate symbols from all exchanges
        var allSymbols = symbolSets.SelectMany(s => s).Distinct().ToList();

        stopwatch.Stop();

        if (!allSymbols.Any())
        {
            _logger.LogWarning("Auto-discovery found no symbols matching criteria");
            return new List<string>();
        }

        _logger.LogInformation(
            "Symbol discovery completed in {Elapsed}ms. Found {Count} unique symbols",
            stopwatch.ElapsedMilliseconds,
            allSymbols.Count);

        return allSymbols;
    }

    /// <summary>
    /// Clears the cached symbols, forcing next call to re-discover
    /// </summary>
    public void ClearCache()
    {
        _cachedSymbols = null;
        _lastDiscovery = null;
        _logger.LogInformation("Symbol cache cleared");
    }
}
