using Microsoft.AspNetCore.SignalR;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Config;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Orchestrator service that coordinates data collection, opportunity detection, and real-time streaming.
/// Implements two independent loops:
/// 1. Slow loop: Collect market data and detect opportunities (every 60s by default)
/// 2. Fast loop: Broadcast cached data to UI via SignalR (every 1s by default)
/// </summary>
public class ArbitrageEngineService : BackgroundService
{
    private readonly ILogger<ArbitrageEngineService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly ArbitrageConfig _config;
    private readonly IDataAggregationService _dataAggregationService;
    private readonly IOpportunityDetectionService _opportunityDetectionService;
    private readonly ISignalRStreamingService _signalRStreamingService;
    private readonly IMemoryCache _cache;
    private readonly ConnectionResilienceService _resilienceService;
    private readonly Dictionary<string, IExchangeConnector> _exchangeConnectors = new();
    private readonly SemaphoreSlim _liquiditySemaphore;
    private List<string> _activeSymbols = new();
    private DateTime _lastSymbolRefresh = DateTime.MinValue;

    // Cached data for broadcasting
    private List<FundingRateDto> _cachedFundingRates = new();
    private List<ArbitrageOpportunityDto> _cachedOpportunities = new();
    private readonly object _cacheLock = new();

    public ArbitrageEngineService(
        ILogger<ArbitrageEngineService> logger,
        IServiceProvider serviceProvider,
        ArbitrageConfig config,
        IDataAggregationService dataAggregationService,
        IOpportunityDetectionService opportunityDetectionService,
        ISignalRStreamingService signalRStreamingService,
        IMemoryCache cache,
        ConnectionResilienceService resilienceService)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
        _dataAggregationService = dataAggregationService;
        _opportunityDetectionService = opportunityDetectionService;
        _signalRStreamingService = signalRStreamingService;
        _cache = cache;
        _resilienceService = resilienceService;
        _liquiditySemaphore = new SemaphoreSlim(config.MaxConcurrentLiquidityRequests);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Arbitrage Engine Service starting...");
        _logger.LogInformation("Configuration: OpportunityCollection={OpportunityInterval}s, SignalRBroadcast={BroadcastInterval}s, UserData={UserDataInterval}s",
            _config.OpportunityCollectionIntervalSeconds,
            _config.SignalRBroadcastIntervalSeconds,
            _config.UserDataRefreshIntervalSeconds);

        try
        {
            // Initialize global exchange connectors (for shared funding rate fetching)
            await InitializeExchangesAsync();

            // Start three independent loops in parallel
            var tasks = new[]
            {
                OpportunityCollectionLoopAsync(stoppingToken),    // Slow: Collect data & detect opportunities
                SignalRBroadcastLoopAsync(stoppingToken),         // Fast: Broadcast cached data to UI
                UserDataRefreshLoopAsync(stoppingToken)           // Medium: Refresh user-specific data
            };

            await Task.WhenAll(tasks);
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "Fatal error in Arbitrage Engine Service. Service will attempt to restart.");

            // Don't let the service die - allow BackgroundService to restart it
            throw;
        }
    }

    /// <summary>
    /// Loop 1: Collect market data and detect opportunities (slow, configurable interval)
    /// This is the expensive operation that calls exchange APIs
    /// </summary>
    private async Task OpportunityCollectionLoopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting opportunity collection loop (interval: {Interval}s)",
            _config.OpportunityCollectionIntervalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Periodically refresh symbol list
                await RefreshActiveSymbolsAsync();

                if (!_activeSymbols.Any())
                {
                    _logger.LogWarning("No active symbols to analyze");
                    await Task.Delay(TimeSpan.FromSeconds(_config.OpportunityCollectionIntervalSeconds), stoppingToken);
                    continue;
                }

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // PHASE 1: Fetch and cache market data
                _logger.LogInformation("Fetching market data for {Count} symbols...", _activeSymbols.Count);
                var snapshot = await _dataAggregationService.FetchAndCacheMarketDataAsync(
                    _activeSymbols, _exchangeConnectors, stoppingToken);

                // PHASE 2: Detect opportunities from market data
                _logger.LogInformation("Detecting opportunities from market snapshot...");
                var opportunities = await _opportunityDetectionService.DetectOpportunitiesAsync(snapshot);

                // PHASE 3: Enrich opportunities with volume and liquidity data
                await EnrichOpportunitiesAsync(opportunities, snapshot, stoppingToken);

                // PHASE 4: Update cached data (thread-safe)
                lock (_cacheLock)
                {
                    _cachedFundingRates = snapshot.FundingRates.Values.SelectMany(r => r).ToList();
                    _cachedOpportunities = opportunities;
                }

                sw.Stop();
                _logger.LogInformation(
                    "Opportunity collection completed in {Elapsed}ms: {Rates} rates, {Opps} opportunities",
                    sw.ElapsedMilliseconds, _cachedFundingRates.Count, _cachedOpportunities.Count);

                await Task.Delay(TimeSpan.FromSeconds(_config.OpportunityCollectionIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in opportunity collection loop");
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
        }
    }

    /// <summary>
    /// Loop 2: Broadcast cached data to UI (fast, configurable interval)
    /// This is cheap - just sends already-collected data to SignalR clients
    /// </summary>
    private async Task SignalRBroadcastLoopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting SignalR broadcast loop (interval: {Interval}s)",
            _config.SignalRBroadcastIntervalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                List<FundingRateDto> ratesToBroadcast;
                List<ArbitrageOpportunityDto> oppsToBroadcast;

                // Get snapshot of cached data (thread-safe)
                lock (_cacheLock)
                {
                    ratesToBroadcast = _cachedFundingRates.ToList();
                    oppsToBroadcast = _cachedOpportunities.ToList();
                }

                // Broadcast to all clients
                if (ratesToBroadcast.Any())
                {
                    await _signalRStreamingService.BroadcastFundingRatesAsync(ratesToBroadcast, stoppingToken);
                }

                if (oppsToBroadcast.Any())
                {
                    await _signalRStreamingService.BroadcastOpportunitiesAsync(oppsToBroadcast, stoppingToken);
                }

                await Task.Delay(TimeSpan.FromSeconds(_config.SignalRBroadcastIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in SignalR broadcast loop");
                await Task.Delay(TimeSpan.FromSeconds(1), stoppingToken);
            }
        }
    }

    /// <summary>
    /// Loop 3: Refresh user-specific data (medium frequency, configurable interval)
    /// </summary>
    private async Task UserDataRefreshLoopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting user data refresh loop (interval: {Interval}s)",
            _config.UserDataRefreshIntervalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await ProcessAllUsersDataAsync(stoppingToken);
                await Task.Delay(TimeSpan.FromSeconds(_config.UserDataRefreshIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in user data refresh loop");
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
        }
    }

    private async Task InitializeExchangesAsync()
    {
        // Use exchanges from configuration instead of database
        var exchanges = _config.Exchanges.Where(e => e.IsEnabled).ToList();

        foreach (var exchange in exchanges)
        {
            try
            {
                IExchangeConnector? connector = exchange.Name.ToLower() switch
                {
                    "binance" => new BinanceConnector(
                        _serviceProvider.GetRequiredService<ILogger<BinanceConnector>>(),
                        _serviceProvider.GetRequiredService<IConfiguration>()),
                    "bybit" => new BybitConnector(
                        _serviceProvider.GetRequiredService<ILogger<BybitConnector>>(),
                        _serviceProvider.GetRequiredService<IConfiguration>()),
                    _ => null
                };

                if (connector != null)
                {
                    // For multi-user system: Initialize without API keys to use public APIs
                    // User-specific API keys are stored in database and used for trading operations
                    var connected = await _resilienceService.ExecuteWithRetryBoolAsync(
                        $"Connect to {exchange.Name}",
                        $"global-{exchange.Name}",
                        async () => await connector.ConnectAsync(null, null)
                    );

                    if (connected)
                    {
                        _exchangeConnectors[exchange.Name] = connector;
                        _logger.LogInformation("Connected to {Exchange} public API", exchange.Name);
                    }
                    else
                    {
                        _logger.LogWarning("Failed to connect to {Exchange} after retries. Will retry later.", exchange.Name);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize connector for {Exchange}", exchange.Name);
            }
        }

        // Discover active symbols after connecting to exchanges
        await RefreshActiveSymbolsAsync();
    }

    private async Task RefreshActiveSymbolsAsync()
    {
        if (!_config.AutoDiscoverSymbols)
        {
            // Use manual symbol list from config
            _activeSymbols = _config.WatchedSymbols;
            _logger.LogInformation("Using {Count} manually configured symbols", _activeSymbols.Count);
            return;
        }

        // Check if refresh is needed
        var hoursSinceRefresh = (DateTime.UtcNow - _lastSymbolRefresh).TotalHours;
        if (hoursSinceRefresh < _config.SymbolRefreshIntervalHours && _activeSymbols.Any())
        {
            return; // Don't refresh yet
        }

        try
        {
            var allDiscoveredSymbols = new List<string>();

            // Discover symbols from each connected exchange
            foreach (var (exchangeName, connector) in _exchangeConnectors)
            {
                try
                {
                    var symbols = await connector.GetActiveSymbolsAsync(
                        _config.MinDailyVolumeUsd,
                        _config.MaxSymbolCount,
                        _config.MinHighPriorityFundingRate
                    );
                    allDiscoveredSymbols.AddRange(symbols);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error discovering symbols from {Exchange}", exchangeName);
                }
            }

            // Use UNION of all symbols from all exchanges
            _activeSymbols = allDiscoveredSymbols.Distinct().OrderBy(s => s).ToList();

            _lastSymbolRefresh = DateTime.UtcNow;
            _logger.LogInformation(
                "Refreshed active symbols: {Count} symbols discovered (Volume >= ${MinVolume:N0})",
                _activeSymbols.Count,
                _config.MinDailyVolumeUsd
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error refreshing active symbols");
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Enrich opportunities with volume (copied from funding rates) and liquidity metrics
    /// </summary>
    private async Task EnrichOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities, MarketDataSnapshot snapshot, CancellationToken stoppingToken)
    {
        // PHASE 1: Copy volume from funding rates to opportunities
        try
        {
            foreach (var opp in opportunities)
            {
                if (opp.Strategy == ArbitrageStrategy.SpotPerpetual)
                {
                    // Get volume from funding rate for same exchange
                    if (snapshot.FundingRates.ContainsKey(opp.Exchange))
                    {
                        var fundingRate = snapshot.FundingRates[opp.Exchange]
                            .FirstOrDefault(fr => fr.Symbol == opp.Symbol);
                        if (fundingRate != null)
                        {
                            opp.Volume24h = fundingRate.Volume24h;
                        }
                    }
                }
                else if (opp.Strategy == ArbitrageStrategy.CrossExchange)
                {
                    // Get volume from both exchanges and use the minimum
                    decimal longVolume = 0, shortVolume = 0;

                    if (snapshot.FundingRates.ContainsKey(opp.LongExchange))
                    {
                        var longFundingRate = snapshot.FundingRates[opp.LongExchange]
                            .FirstOrDefault(fr => fr.Symbol == opp.Symbol);
                        if (longFundingRate != null)
                            longVolume = longFundingRate.Volume24h;
                    }

                    if (snapshot.FundingRates.ContainsKey(opp.ShortExchange))
                    {
                        var shortFundingRate = snapshot.FundingRates[opp.ShortExchange]
                            .FirstOrDefault(fr => fr.Symbol == opp.Symbol);
                        if (shortFundingRate != null)
                            shortVolume = shortFundingRate.Volume24h;
                    }

                    // Use the minimum of the two volumes (bottleneck)
                    opp.Volume24h = Math.Min(longVolume > 0 ? longVolume : decimal.MaxValue,
                                             shortVolume > 0 ? shortVolume : decimal.MaxValue);
                    if (opp.Volume24h == decimal.MaxValue)
                        opp.Volume24h = 0;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error copying volume data to opportunities");
        }

        // PHASE 2: Enrich with liquidity metrics
        try
        {
            var liquidityByExchange = new Dictionary<string, Dictionary<string, LiquidityMetricsDto>>();

            foreach (var (exchangeName, connector) in _exchangeConnectors)
            {
                var exchangeLiquidity = new Dictionary<string, LiquidityMetricsDto>();
                var exchangeSymbols = opportunities
                    .Where(o => o.Strategy == ArbitrageStrategy.SpotPerpetual && o.Exchange == exchangeName ||
                                o.Strategy == ArbitrageStrategy.CrossExchange && (o.LongExchange == exchangeName || o.ShortExchange == exchangeName))
                    .Select(o => o.Symbol)
                    .Distinct()
                    .ToList();

                var symbolTasks = exchangeSymbols.Select(async symbol =>
                {
                    await _liquiditySemaphore.WaitAsync(stoppingToken);
                    try
                    {
                        var liquidity = await connector.GetLiquidityMetricsAsync(symbol);
                        if (liquidity != null)
                        {
                            lock (exchangeLiquidity)
                            {
                                exchangeLiquidity[symbol] = liquidity;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to fetch liquidity for {Symbol} from {Exchange}", symbol, exchangeName);
                    }
                    finally
                    {
                        _liquiditySemaphore.Release();
                    }
                });

                await Task.WhenAll(symbolTasks);
                liquidityByExchange[exchangeName] = exchangeLiquidity;
            }

            // Populate liquidity for each opportunity
            foreach (var opp in opportunities)
            {
                LiquidityMetricsDto? primaryLiquidity = null;
                LiquidityMetricsDto? secondaryLiquidity = null;

                if (opp.Strategy == ArbitrageStrategy.SpotPerpetual && liquidityByExchange.ContainsKey(opp.Exchange))
                {
                    liquidityByExchange[opp.Exchange].TryGetValue(opp.Symbol, out primaryLiquidity);
                }
                else if (opp.Strategy == ArbitrageStrategy.CrossExchange)
                {
                    if (liquidityByExchange.ContainsKey(opp.LongExchange))
                    {
                        liquidityByExchange[opp.LongExchange].TryGetValue(opp.Symbol, out primaryLiquidity);
                    }
                    if (liquidityByExchange.ContainsKey(opp.ShortExchange))
                    {
                        liquidityByExchange[opp.ShortExchange].TryGetValue(opp.Symbol, out secondaryLiquidity);
                    }
                }

                if (primaryLiquidity != null)
                {
                    var (status, warning) = EvaluateLiquidity(primaryLiquidity, opp.Volume24h);
                    opp.LiquidityStatus = status;
                    opp.LiquidityWarning = warning;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enriching opportunities with liquidity data");
        }
    }

    private (LiquidityStatus status, string? warning) EvaluateLiquidity(
        LiquidityMetricsDto liquidity, decimal volume24h)
    {
        var warnings = new List<string>();
        var status = LiquidityStatus.Good;

        // Check bid-ask spread (already calculated in DTO)
        if (liquidity.BidAskSpreadPercent > _config.MaxBidAskSpreadPercent)
        {
            warnings.Add($"Wide bid-ask spread ({liquidity.BidAskSpreadPercent * 100:F2}%)");
            status = LiquidityStatus.Low;
        }

        // Check orderbook depth
        if (liquidity.OrderbookDepthUsd < _config.MinOrderbookDepthUsd)
        {
            warnings.Add($"Thin orderbook (${liquidity.OrderbookDepthUsd:N0})");
            status = LiquidityStatus.Low;
        }
        else if (liquidity.OrderbookDepthUsd < _config.MinOrderbookDepthUsd * 2)
        {
            warnings.Add($"Moderate orderbook depth (${liquidity.OrderbookDepthUsd:N0})");
            if (status == LiquidityStatus.Good)
                status = LiquidityStatus.Medium;
        }

        // Check 24h volume
        if (volume24h < _config.MinVolume24hUsd)
        {
            warnings.Add($"Low 24h volume (${volume24h:N0})");
            status = LiquidityStatus.Low;
        }
        else if (volume24h < _config.MinVolume24hUsd * 1.5m)
        {
            warnings.Add($"Moderate 24h volume (${volume24h:N0})");
            if (status == LiquidityStatus.Good)
                status = LiquidityStatus.Medium;
        }

        var warningMessage = warnings.Any()
            ? string.Join("; ", warnings) + ". Consider using limit orders."
            : null;

        return (status, warningMessage);
    }

    /// <summary>
    /// Process user-specific data for all active users.
    /// </summary>
    private async Task ProcessAllUsersDataAsync(CancellationToken stoppingToken)
    {
        try
        {
            using var scope = _serviceProvider.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            var activeUsers = await db.Users
                .Include(u => u.ExchangeApiKeys)
                .Where(u => u.ExchangeApiKeys.Any(k => k.IsEnabled))
                .ToListAsync(stoppingToken);

            foreach (var user in activeUsers)
            {
                user.ExchangeApiKeys = user.ExchangeApiKeys.Where(k => k.IsEnabled).ToList();
            }

            if (!activeUsers.Any())
            {
                _logger.LogDebug("No active users with enabled API keys");
                return;
            }

            _logger.LogInformation("Processing data for {Count} active users", activeUsers.Count);

            foreach (var user in activeUsers)
            {
                try
                {
                    await ProcessUserDataAsync(user, stoppingToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing data for user {UserId}", user.Id);
                }
            }

            // Check for notification triggers for each active user
            var notificationService = scope.ServiceProvider.GetService<INotificationService>();
            if (notificationService != null)
            {
                foreach (var user in activeUsers)
                {
                    try
                    {
                        await notificationService.CheckNegativeFundingForUserAsync(user.Id);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error checking notifications for user {UserId}", user.Id);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ProcessAllUsersDataAsync");
        }
    }

    private async Task ProcessUserDataAsync(ApplicationUser user, CancellationToken stoppingToken)
    {
        var userId = user.Id;
        var email = user.Email;

        // DEBOUNCE: Check if an immediate broadcast happened recently (within last 2 seconds)
        // This prevents race conditions where Stop/Execute methods broadcast fresh data
        // but this loop broadcasts stale data shortly after, causing UI flickering
        if (_signalRStreamingService.ShouldSkipUserRefresh(userId, debounceSeconds: 2.0))
        {
            _logger.LogDebug("Skipping user data refresh for {UserId} - recent broadcast detected", userId);
            return;
        }

        try
        {
            using var scope = _serviceProvider.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
            var encryptionService = scope.ServiceProvider.GetRequiredService<IEncryptionService>();

            var userApiKeys = user.ExchangeApiKeys.Where(k => k.IsEnabled).ToList();
            if (!userApiKeys.Any())
            {
                _logger.LogDebug("User {UserId} has no enabled API keys", userId);
                return;
            }

            // Create user-specific exchange connectors with decrypted API keys
            var userConnectors = new Dictionary<string, IExchangeConnector>();
            foreach (var apiKey in userApiKeys)
            {
                try
                {
                    var decryptedKey = encryptionService.Decrypt(apiKey.EncryptedApiKey);
                    var decryptedSecret = encryptionService.Decrypt(apiKey.EncryptedApiSecret);

                    IExchangeConnector? connector = apiKey.ExchangeName.ToLower() switch
                    {
                        "binance" => new BinanceConnector(
                            scope.ServiceProvider.GetRequiredService<ILogger<BinanceConnector>>(),
                            scope.ServiceProvider.GetRequiredService<IConfiguration>()),
                        "bybit" => new BybitConnector(
                            scope.ServiceProvider.GetRequiredService<ILogger<BybitConnector>>(),
                            scope.ServiceProvider.GetRequiredService<IConfiguration>()),
                        _ => null
                    };

                    if (connector != null)
                    {
                        var connected = await _resilienceService.ExecuteWithRetryBoolAsync(
                            $"Connect user {userId} to {apiKey.ExchangeName}",
                            $"user-{userId}-{apiKey.ExchangeName}",
                            async () => await connector.ConnectAsync(decryptedKey, decryptedSecret)
                        );

                        if (connected)
                        {
                            userConnectors[apiKey.ExchangeName] = connector;
                            _logger.LogDebug("User {UserId} connected to {Exchange}", userId, apiKey.ExchangeName);
                        }
                        else
                        {
                            _logger.LogWarning("User {UserId} failed to connect to {Exchange} after retries",
                                userId, apiKey.ExchangeName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error creating connector for user {UserId}, exchange {Exchange}",
                        userId, apiKey.ExchangeName);
                }
            }

            if (!userConnectors.Any())
            {
                _logger.LogWarning("User {UserId} ({Email}) has no working connectors", userId, email);
                return;
            }

            // Fetch and broadcast user positions (only Open positions)
            var userPositions = await db.Positions
                .Where(p => p.UserId == userId && p.Status == PositionStatus.Open)
                .ToListAsync(stoppingToken);

            var positionDtos = new List<PositionDto>();
            foreach (var position in userPositions)
            {
                if (userConnectors.TryGetValue(position.Exchange, out var connector))
                {
                    try
                    {
                        var positionDto = await UpdatePositionWithLiveDataAsync(position, connector);
                        positionDtos.Add(positionDto);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Error updating position {PositionId} for user {UserId}",
                            position.Id, userId);
                    }
                }
            }

            await _signalRStreamingService.BroadcastPositionsToUserAsync(userId, positionDtos, stoppingToken);

            // Fetch and broadcast user balances
            var balances = new List<AccountBalanceDto>();
            foreach (var (exchangeName, connector) in userConnectors)
            {
                try
                {
                    var balance = await connector.GetAccountBalanceAsync();
                    balances.Add(balance);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error fetching balances from {Exchange} for user {UserId}",
                        exchangeName, userId);
                }
            }

            await _signalRStreamingService.BroadcastBalancesToUserAsync(userId, balances, stoppingToken);

            // Calculate and broadcast P&L
            decimal totalPnL = positionDtos.Sum(p => p.UnrealizedPnL + p.RealizedPnL);
            decimal todayPnL = positionDtos
                .Where(p => p.OpenedAt.Date == DateTime.UtcNow.Date)
                .Sum(p => p.UnrealizedPnL + p.RealizedPnL);

            await _signalRStreamingService.BroadcastPnLToUserAsync(userId, totalPnL, todayPnL, stoppingToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing user data for {UserId}", userId);
        }
    }

    private async Task<PositionDto> UpdatePositionWithLiveDataAsync(Position position, IExchangeConnector connector)
    {
        decimal unrealizedPnL = position.UnrealizedPnL;
        decimal realizedPnL = position.RealizedPnL;
        decimal fundingPaid = position.TotalFundingFeePaid;
        decimal fundingReceived = position.TotalFundingFeeReceived;

        if (position.Type == PositionType.Perpetual)
        {
            var livePositions = await connector.GetOpenPositionsAsync();
            var livePosition = livePositions.FirstOrDefault(p =>
                p.Symbol == position.Symbol &&
                p.Side == position.Side &&
                p.Type == position.Type);

            if (livePosition != null)
            {
                unrealizedPnL = livePosition.UnrealizedPnL;
                realizedPnL = livePosition.RealizedPnL;
                fundingPaid = livePosition.TotalFundingFeePaid;
                fundingReceived = livePosition.TotalFundingFeeReceived;
            }
            else
            {
                var perpPrices = await connector.GetPerpetualPricesAsync(new List<string> { position.Symbol });
                if (perpPrices.TryGetValue(position.Symbol, out var currentPrice))
                {
                    decimal priceDiff = currentPrice - position.EntryPrice;
                    unrealizedPnL = position.Side == PositionSide.Long
                        ? priceDiff * position.Quantity
                        : -priceDiff * position.Quantity;
                }
            }
        }
        else if (position.Type == PositionType.Spot)
        {
            var spotPrices = await connector.GetSpotPricesAsync(new List<string> { position.Symbol });
            if (spotPrices.TryGetValue(position.Symbol, out var spotPrice))
            {
                decimal currentPrice = spotPrice.Price;
                decimal priceDiff = currentPrice - position.EntryPrice;
                unrealizedPnL = position.Side == PositionSide.Long
                    ? priceDiff * position.Quantity
                    : -priceDiff * position.Quantity;
            }
        }

        return new PositionDto
        {
            Id = position.Id,
            ExecutionId = position.ExecutionId,
            Exchange = position.Exchange,
            Symbol = position.Symbol,
            Type = position.Type,
            Side = position.Side,
            Status = position.Status,
            EntryPrice = position.EntryPrice,
            Quantity = position.Quantity,
            UnrealizedPnL = unrealizedPnL,
            RealizedPnL = realizedPnL,
            TotalFundingFeePaid = fundingPaid,
            TotalFundingFeeReceived = fundingReceived,
            OpenedAt = position.OpenedAt,
            ClosedAt = position.ClosedAt
        };
    }

}
