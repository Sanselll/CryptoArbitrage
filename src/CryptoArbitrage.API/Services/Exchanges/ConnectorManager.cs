using CryptoArbitrage.API.Config;

namespace CryptoArbitrage.API.Services.Exchanges;

/// <summary>
/// Manages exchange connectors based on configuration
/// Provides enabled connectors to data collectors and other services
/// </summary>
public class ConnectorManager
{
    private readonly ArbitrageConfig _config;
    private readonly ILogger<ConnectorManager> _logger;

    public ConnectorManager(
        ArbitrageConfig config,
        ILogger<ConnectorManager> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Gets the list of enabled exchange names from ArbitrageConfig
    /// </summary>
    public List<string> GetEnabledExchangeNames()
    {
        return _config.Exchanges?
            .Where(e => e.IsEnabled)
            .Select(e => e.Name)
            .ToList() ?? new List<string>();
    }

    /// <summary>
    /// Validates if an exchange is configured and enabled
    /// </summary>
    public bool IsExchangeEnabled(string exchangeName)
    {
        return _config.Exchanges?
            .Any(e => e.Name.Equals(exchangeName, StringComparison.OrdinalIgnoreCase) && e.IsEnabled) ?? false;
    }

    /// <summary>
    /// Gets enabled exchange connectors using a provided service scope
    /// Use this when you already have a scope and want to avoid creating a new one
    /// </summary>
    public List<(string Name, IExchangeConnector? Connector)> GetEnabledConnectors(IServiceScope scope)
    {
        var connectors = new List<(string Name, IExchangeConnector? Connector)>();

        if (_config.Exchanges == null || !_config.Exchanges.Any())
        {
            _logger.LogWarning("No exchanges configured in ArbitrageConfig.Exchanges");
            return connectors;
        }

        foreach (var exchangeConfig in _config.Exchanges.Where(e => e.IsEnabled))
        {
            IExchangeConnector? connector = exchangeConfig.Name switch
            {
                "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
                "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
                "Kraken" => scope.ServiceProvider.GetService<KrakenConnector>(),
                _ => null
            };

            if (connector != null)
            {
                connectors.Add((exchangeConfig.Name, connector));
            }
            else
            {
                _logger.LogWarning("Connector not found for configured exchange: {Exchange}", exchangeConfig.Name);
            }
        }

        return connectors;
    }

    /// <summary>
    /// Gets a specific exchange connector by name and initializes it with provided credentials
    /// Use this when you need to test user-provided credentials (e.g., AddApiKey endpoint)
    /// </summary>
    public async Task<IExchangeConnector?> GetConnectorByNameAsync(
        IServiceScope scope,
        string exchangeName,
        string apiKey,
        string apiSecret,
        CancellationToken cancellationToken = default)
    {
        IExchangeConnector? connector = exchangeName switch
        {
            "Binance" => scope.ServiceProvider.GetService<BinanceConnector>(),
            "Bybit" => scope.ServiceProvider.GetService<BybitConnector>(),
            "Kraken" => scope.ServiceProvider.GetService<KrakenConnector>(),
            _ => null
        };

        if (connector == null)
        {
            _logger.LogWarning("Connector not found for exchange: {Exchange}", exchangeName);
            return null;
        }

        try
        {
            // Initialize connector with provided credentials
            var connected = await connector.ConnectAsync(apiKey, apiSecret);

            if (!connected)
            {
                _logger.LogWarning("Failed to connect to {Exchange} with provided credentials", exchangeName);
                return null;
            }

            _logger.LogDebug("Successfully initialized {Exchange} connector with provided credentials", exchangeName);
            return connector;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing {Exchange} connector", exchangeName);
            return null;
        }
    }
}
