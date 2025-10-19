using Microsoft.AspNetCore.SignalR;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Config;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.API.Services;

public class ArbitrageEngineService : BackgroundService
{
    private readonly ILogger<ArbitrageEngineService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IHubContext<ArbitrageHub> _hubContext;
    private readonly ArbitrageConfig _config;
    private readonly Dictionary<string, IExchangeConnector> _exchangeConnectors = new();
    private List<string> _activeSymbols = new();
    private DateTime _lastSymbolRefresh = DateTime.MinValue;

    public ArbitrageEngineService(
        ILogger<ArbitrageEngineService> logger,
        IServiceProvider serviceProvider,
        IHubContext<ArbitrageHub> hubContext,
        ArbitrageConfig config)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _hubContext = hubContext;
        _config = config;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Arbitrage Engine Service starting...");

        // Initialize global exchange connectors (for shared funding rate fetching)
        await InitializeExchangesAsync();

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await RunArbitrageAnalysisAsync(stoppingToken);
                await Task.Delay(TimeSpan.FromSeconds(_config.DataRefreshIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in arbitrage engine loop");
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
                    "binance" => new BinanceConnector(_serviceProvider.GetRequiredService<ILogger<BinanceConnector>>()),
                    "bybit" => new BybitConnector(_serviceProvider.GetRequiredService<ILogger<BybitConnector>>()),
                    _ => null
                };

                if (connector != null)
                {
                    // For multi-user system: Initialize without API keys to use public APIs
                    // User-specific API keys are stored in database and used for trading operations
                    var connected = await connector.ConnectAsync(null, null, exchange.UseDemoTrading);
                    if (connected)
                    {
                        _exchangeConnectors[exchange.Name] = connector;
                        var environment = exchange.UseDemoTrading ? "Demo/Testnet" : "Live";
                        _logger.LogInformation("Connected to {Exchange} public API ({Environment})", exchange.Name, environment);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to connect to {Exchange}", exchange.Name);
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
            // This allows us to:
            // 1. Detect cross-exchange opportunities for symbols on multiple exchanges (intersection)
            // 2. Detect spot-perpetual opportunities for ALL symbols (union)
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
            // Fall back to manual config if auto-discovery fails
            if (!_activeSymbols.Any())
            {
                _activeSymbols = _config.WatchedSymbols;
            }
        }
    }

    private async Task RunArbitrageAnalysisAsync(CancellationToken stoppingToken)
    {
        // Periodically refresh symbol list
        await RefreshActiveSymbolsAsync();

        if (!_activeSymbols.Any())
        {
            _logger.LogWarning("No active symbols to analyze");
            return;
        }

        var fundingRates = new Dictionary<string, List<FundingRateDto>>();
        var spotPrices = new Dictionary<string, Dictionary<string, SpotPriceDto>>();
        var perpPrices = new Dictionary<string, Dictionary<string, decimal>>();

        // PHASE 1: Fetch GLOBAL funding rates (shared across all users)
        var fetchStopwatch = System.Diagnostics.Stopwatch.StartNew();
        foreach (var (exchangeName, connector) in _exchangeConnectors)
        {
            try
            {
                // Load previous rates from database for enrichment
                using var scope1 = _serviceProvider.CreateScope();
                var dbContext1 = scope1.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                var dbLoadSw = System.Diagnostics.Stopwatch.StartNew();
                var previousRates = await dbContext1.FundingRates
                    .Where(f => f.Exchange == exchangeName)
                    .GroupBy(f => f.Symbol)
                    .Select(g => g.OrderByDescending(f => f.RecordedAt).FirstOrDefault())
                    .ToDictionaryAsync(f => f!.Symbol, f => f!, stoppingToken);
                dbLoadSw.Stop();
                _logger.LogInformation("📖 Loaded {Count} previous rates from database for {Exchange} in {ElapsedMs}ms",
                    previousRates.Count, exchangeName, dbLoadSw.ElapsedMilliseconds);

                // Fetch current rates from API
                var exchangeFetchSw = System.Diagnostics.Stopwatch.StartNew();
                var rates = await connector.GetFundingRatesAsync(_activeSymbols);
                exchangeFetchSw.Stop();
                _logger.LogInformation("⏱️  Fetched {Count} funding rates from {Exchange} in {ElapsedMs}ms",
                    rates.Count, exchangeName, exchangeFetchSw.ElapsedMilliseconds);

                // Enrich current rates with previous rate data from database
                foreach (var rate in rates)
                {
                    if (previousRates.TryGetValue(rate.Symbol, out var prevRate))
                    {
                        rate.PreviousRate = prevRate.Rate;
                        rate.PreviousAnnualizedRate = prevRate.AnnualizedRate;
                    }
                }

                fundingRates[exchangeName] = rates;

                // Fetch spot prices for spot-perpetual arbitrage
                var prices = await connector.GetSpotPricesAsync(_activeSymbols);
                spotPrices[exchangeName] = prices;

                // Fetch perpetual prices
                var perps = await connector.GetPerpetualPricesAsync(_activeSymbols);
                perpPrices[exchangeName] = perps;

                // Save to database using efficient bulk upsert
                using var scope2 = _serviceProvider.CreateScope();
                var dbContext2 = scope2.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                var dbSaveSw = System.Diagnostics.Stopwatch.StartNew();
                await UpsertFundingRatesAsync(dbContext2, exchangeName, rates, stoppingToken);
                dbSaveSw.Stop();
                _logger.LogInformation("💾 Saved {Count} funding rates for {Exchange} to database in {ElapsedMs}ms",
                    rates.Count, exchangeName, dbSaveSw.ElapsedMilliseconds);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching funding rates from {Exchange}", exchangeName);
            }
        }
        fetchStopwatch.Stop();
        _logger.LogInformation("⏱️  Total funding rate fetch and save time: {ElapsedMs}ms", fetchStopwatch.ElapsedMilliseconds);

        // Broadcast funding rates to ALL users (shared global data)
        var allRates = fundingRates.Values.SelectMany(r => r).ToList();
        await _hubContext.Clients.All.SendAsync("ReceiveFundingRates", allRates, stoppingToken);

        // Detect all types of arbitrage opportunities
        var opportunities = new List<ArbitrageOpportunityDto>();

        // Cross-exchange arbitrage (requires 2+ exchanges)
        if (fundingRates.Count >= 2)
        {
            // Futures/Futures cross-exchange
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeFuturesFutures))
            {
                var crossExchangeFuturesOpps = await DetectCrossExchangeFuturesOpportunitiesAsync(fundingRates, perpPrices);
                opportunities.AddRange(crossExchangeFuturesOpps);
            }

            // Spot/Futures cross-exchange
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeSpotFutures))
            {
                var crossExchangeSpotFuturesOpps = await DetectCrossExchangeSpotFuturesOpportunitiesAsync(fundingRates, spotPrices, perpPrices);
                opportunities.AddRange(crossExchangeSpotFuturesOpps);
            }
        }

        // Spot-perpetual arbitrage (same exchange)
        if (_config.IsStrategyEnabled(StrategySubType.SpotPerpetualSameExchange))
        {
            var spotPerpOpps = await DetectSpotPerpetualOpportunitiesAsync(fundingRates, spotPrices, perpPrices);
            opportunities.AddRange(spotPerpOpps);
        }

        if (opportunities.Any())
        {
            // Enrich opportunities with 24h volume data
            try
            {
                var uniqueSymbols = opportunities.Select(o => o.Symbol).Distinct().ToList();
                var volumeByExchange = new Dictionary<string, Dictionary<string, decimal>>();

                foreach (var (exchangeName, connector) in _exchangeConnectors)
                {
                    try
                    {
                        var volumes = await connector.Get24hVolumeAsync(uniqueSymbols);
                        volumeByExchange[exchangeName] = volumes;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to fetch 24h volumes from {Exchange}", exchangeName);
                    }
                }

                // Populate volume for each opportunity
                foreach (var opp in opportunities)
                {
                    // For spot-perp, use the exchange's volume
                    if (opp.Strategy == ArbitrageStrategy.SpotPerpetual && volumeByExchange.ContainsKey(opp.Exchange))
                    {
                        if (volumeByExchange[opp.Exchange].TryGetValue(opp.Symbol, out var volume))
                        {
                            opp.Volume24h = volume;
                        }
                    }
                    // For cross-exchange, use the average or max volume from both exchanges
                    else if (opp.Strategy == ArbitrageStrategy.CrossExchange)
                    {
                        decimal longVolume = 0, shortVolume = 0;
                        if (volumeByExchange.ContainsKey(opp.LongExchange))
                        {
                            volumeByExchange[opp.LongExchange].TryGetValue(opp.Symbol, out longVolume);
                        }
                        if (volumeByExchange.ContainsKey(opp.ShortExchange))
                        {
                            volumeByExchange[opp.ShortExchange].TryGetValue(opp.Symbol, out shortVolume);
                        }
                        // Use the minimum of the two volumes (limiting factor for arbitrage)
                        opp.Volume24h = Math.Min(longVolume > 0 ? longVolume : decimal.MaxValue,
                                                 shortVolume > 0 ? shortVolume : decimal.MaxValue);
                        if (opp.Volume24h == decimal.MaxValue)
                            opp.Volume24h = 0; // No volume data available
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error enriching opportunities with volume data");
            }

            // Enrich opportunities with execution information from Executions table
            using (var scope = _serviceProvider.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                var runningExecutions = await dbContext.Executions
                    .Where(e => e.State == ExecutionState.Running)
                    .ToListAsync(stoppingToken);

                foreach (var opp in opportunities)
                {
                    // Try to find matching running execution
                    Execution? matchingExecution = null;

                    if (opp.Strategy == ArbitrageStrategy.SpotPerpetual)
                    {
                        // Match by symbol and exchange for spot-perp
                        matchingExecution = runningExecutions.FirstOrDefault(e =>
                            e.Exchange == opp.Exchange &&
                            e.Symbol == opp.Symbol);
                    }

                    if (matchingExecution != null)
                    {
                        // Merge execution data into opportunity DTO
                        opp.ExecutionId = matchingExecution.Id;
                        opp.ExecutionState = matchingExecution.State;
                        opp.ExecutionStartedAt = matchingExecution.StartedAt;
                        opp.ExecutionFundingEarned = matchingExecution.FundingEarned;
                        opp.ActiveOpportunityExecutedAt = matchingExecution.StartedAt; // For backward compatibility
                    }
                }
            }

            _logger.LogInformation("Found {Count} arbitrage opportunities", opportunities.Count);

            // Broadcast opportunities via SignalR (in-memory only, not persisted to database)
            await _hubContext.Clients.All.SendAsync("ReceiveOpportunities", opportunities, stoppingToken);

            // Auto-execute if enabled
            if (_config.AutoExecute)
            {
                await ExecuteOpportunitiesAsync(opportunities);
            }
        }

        // PHASE 2: Process each user individually for user-specific data (positions, balances, P&L)
        await ProcessAllUsersDataAsync(stoppingToken);
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (fundingRates.Count < 2)
            return opportunities;

        var exchangeNames = fundingRates.Keys.ToList();

        // Get common symbols across all exchanges
        var commonSymbols = fundingRates.Values
            .Select(rates => rates.Select(r => r.Symbol).ToHashSet())
            .Aggregate((a, b) => { a.IntersectWith(b); return a; });

        foreach (var symbol in commonSymbols)
        {
            // Compare funding rates between all exchange pairs
            for (int i = 0; i < exchangeNames.Count; i++)
            {
                for (int j = i + 1; j < exchangeNames.Count; j++)
                {
                    var exchange1 = exchangeNames[i];
                    var exchange2 = exchangeNames[j];

                    var rate1 = fundingRates[exchange1].First(r => r.Symbol == symbol);
                    var rate2 = fundingRates[exchange2].First(r => r.Symbol == symbol);

                    // Get perpetual prices for both exchanges
                    decimal price1 = 0;
                    decimal price2 = 0;
                    if (perpPrices.ContainsKey(exchange1) && perpPrices[exchange1].ContainsKey(symbol))
                        price1 = perpPrices[exchange1][symbol];
                    if (perpPrices.ContainsKey(exchange2) && perpPrices[exchange2].ContainsKey(symbol))
                        price2 = perpPrices[exchange2][symbol];

                    // CRITICAL: Funding rate calculation for Futures/Futures cross-exchange
                    // Strategy: LONG on exchange with LOWER funding rate, SHORT on exchange with HIGHER funding rate
                    //
                    // Net funding earnings = |fundingRateLong| + |fundingRateShort| when signs differ
                    //                      = |fundingRateHigh - fundingRateLow| when signs are same
                    //
                    // Example 1: Exchange1: +0.05%, Exchange2: -0.03%
                    //   -> LONG on Exchange2 (lower/negative), SHORT on Exchange1 (higher/positive)
                    //   -> Earn: receive 0.03% from long position + receive 0.05% from short position = 0.08% per funding
                    //
                    // Example 2: Exchange1: +0.10%, Exchange2: +0.03%
                    //   -> LONG on Exchange2 (lower), SHORT on Exchange1 (higher)
                    //   -> Pay 0.03% on long, Receive 0.10% on short -> Net: +0.07%
                    //
                    // Example 3: Exchange1: -0.05%, Exchange2: -0.10%
                    //   -> LONG on Exchange2 (lower/more negative), SHORT on Exchange1 (higher/less negative)
                    //   -> Receive 0.10% on long, Pay 0.05% on short -> Net: +0.05%

                    decimal netFundingRate;
                    string longExchange, shortExchange;
                    decimal longRate, shortRate;

                    if (rate1.Rate < rate2.Rate)
                    {
                        // LONG on exchange1 (lower rate), SHORT on exchange2 (higher rate)
                        longExchange = exchange1;
                        shortExchange = exchange2;
                        longRate = rate1.Rate;
                        shortRate = rate2.Rate;
                    }
                    else
                    {
                        // LONG on exchange2 (lower rate), SHORT on exchange1 (higher rate)
                        longExchange = exchange2;
                        shortExchange = exchange1;
                        longRate = rate2.Rate;
                        shortRate = rate1.Rate;
                    }

                    // Net funding = -longRate (we pay if positive, receive if negative)
                    //              + shortRate (we receive if positive, pay if negative)
                    // Simplifies to: shortRate - longRate
                    netFundingRate = shortRate - longRate;
                    var annualizedNetFunding = netFundingRate * 3 * 365; // 3 fundings per day, 365 days

                    // Only profitable if net funding is positive
                    if (annualizedNetFunding * 100 >= _config.MinSpreadPercentage)
                    {
                        // Determine which price goes to which field based on long/short exchanges
                        decimal longExchangePrice = (longExchange == exchange1) ? price1 : price2;
                        decimal shortExchangePrice = (shortExchange == exchange1) ? price1 : price2;

                        opportunities.Add(new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeFuturesFutures,
                            Symbol = symbol,
                            LongExchange = longExchange,
                            ShortExchange = shortExchange,
                            LongFundingRate = longRate,
                            ShortFundingRate = shortRate,
                            // Use SpotPrice for longExchange perp price, PerpetualPrice for shortExchange perp price
                            SpotPrice = longExchangePrice,
                            PerpetualPrice = shortExchangePrice,
                            SpreadRate = netFundingRate,
                            AnnualizedSpread = annualizedNetFunding,
                            EstimatedProfitPercentage = annualizedNetFunding * 100,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.AnnualizedSpread).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectSpotPerpetualOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, SpotPriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        foreach (var (exchangeName, rates) in fundingRates)
        {
            // Skip if no spot prices or perpetual prices available for this exchange
            if (!spotPrices.ContainsKey(exchangeName) || !spotPrices[exchangeName].Any())
                continue;

            if (!perpPrices.ContainsKey(exchangeName) || !perpPrices[exchangeName].Any())
                continue;

            var exchangeSpotPrices = spotPrices[exchangeName];
            var exchangePerpPrices = perpPrices[exchangeName];

            foreach (var fundingRate in rates)
            {
                // Check if we have both spot and perpetual price for this symbol
                if (!exchangeSpotPrices.ContainsKey(fundingRate.Symbol))
                    continue;

                if (!exchangePerpPrices.ContainsKey(fundingRate.Symbol))
                    continue;

                var spotPrice = exchangeSpotPrices[fundingRate.Symbol];
                decimal perpetualPrice = exchangePerpPrices[fundingRate.Symbol];

                // Skip if spot price is zero (invalid data)
                if (spotPrice.Price == 0 || perpetualPrice == 0)
                    continue;

                // Calculate price premium: (Perp - Spot) / Spot
                decimal pricePremium = (perpetualPrice - spotPrice.Price) / spotPrice.Price;

                // Annualized funding rate (already calculated in funding rate)
                decimal annualizedFundingRate = fundingRate.AnnualizedRate;

                // Estimated trading fees (0.1% for spot buy + 0.05% for futures short = 0.15%)
                decimal estimatedTradingFees = 0.0015m;

                // Calculate net profit for positive funding rate strategy
                // Positive funding = shorts pay longs -> buy spot + short perp -> collect funding
                // Note: Negative funding would require selling spot (which we cannot do with USDT-only capital)
                decimal netProfit = annualizedFundingRate - Math.Abs(pricePremium) - estimatedTradingFees;
                decimal netProfitPercentage = netProfit * 100;

                // Only create opportunity if net profit exceeds minimum threshold AND funding is positive
                if (netProfitPercentage >= _config.MinSpreadPercentage && annualizedFundingRate > 0)
                {
                    opportunities.Add(new ArbitrageOpportunityDto
                    {
                        Strategy = ArbitrageStrategy.SpotPerpetual,
                        SubType = StrategySubType.SpotPerpetualSameExchange,
                        Symbol = fundingRate.Symbol,
                        Exchange = exchangeName,
                        SpotPrice = spotPrice.Price,
                        PerpetualPrice = perpetualPrice,
                        FundingRate = fundingRate.Rate,
                        AnnualizedFundingRate = annualizedFundingRate,
                        PricePremium = pricePremium,
                        SpreadRate = netProfit,
                        AnnualizedSpread = netProfit,
                        EstimatedProfitPercentage = netProfitPercentage,
                        Status = OpportunityStatus.Detected,
                        DetectedAt = DateTime.UtcNow
                    });
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeSpotFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, SpotPriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (fundingRates.Count < 2)
            return opportunities;

        var exchangeNames = fundingRates.Keys.ToList();

        // Strategy: Buy spot on one exchange (Exchange A) + Short perpetual on another exchange (Exchange B)
        // This works ACROSS exchanges, combining spot from one and futures from another
        //
        // Profitability: Funding rate on Exchange B - price premium - trading fees
        // Note: We need positive funding on the SHORT exchange to make this profitable

        foreach (var spotExchange in exchangeNames)
        {
            if (!spotPrices.ContainsKey(spotExchange) || !spotPrices[spotExchange].Any())
                continue;

            var spotExchangePrices = spotPrices[spotExchange];

            foreach (var futuresExchange in exchangeNames)
            {
                if (spotExchange == futuresExchange)
                    continue; // Skip same exchange (that's handled by spot-perpetual strategy)

                if (!fundingRates.ContainsKey(futuresExchange) || !fundingRates[futuresExchange].Any())
                    continue;

                if (!perpPrices.ContainsKey(futuresExchange) || !perpPrices[futuresExchange].Any())
                    continue;

                var futuresRates = fundingRates[futuresExchange];
                var futuresPrices = perpPrices[futuresExchange];

                // Find symbols that exist on BOTH exchanges (spot on one, futures on other)
                var commonSymbols = spotExchangePrices.Keys
                    .Intersect(futuresRates.Select(r => r.Symbol))
                    .Intersect(futuresPrices.Keys)
                    .ToList();

                foreach (var symbol in commonSymbols)
                {
                    var spotPrice = spotExchangePrices[symbol].Price;
                    var futuresPrice = futuresPrices[symbol];
                    var fundingRate = futuresRates.First(r => r.Symbol == symbol);

                    if (spotPrice == 0 || futuresPrice == 0)
                        continue;

                    // Calculate price premium between exchanges: (Futures - Spot) / Spot
                    decimal pricePremium = (futuresPrice - spotPrice) / spotPrice;

                    // Annualized funding rate
                    decimal annualizedFundingRate = fundingRate.AnnualizedRate;

                    // Estimated trading fees (0.1% spot + 0.05% futures = 0.15%)
                    decimal estimatedTradingFees = 0.0015m;

                    // Net profit = funding earned - price premium - trading fees
                    // Only profitable if funding rate is POSITIVE (shorts pay longs -> we collect funding)
                    decimal netProfit = annualizedFundingRate - Math.Abs(pricePremium) - estimatedTradingFees;
                    decimal netProfitPercentage = netProfit * 100;

                    // Only create opportunity if:
                    // 1. Net profit exceeds minimum threshold
                    // 2. Funding rate is POSITIVE (we receive funding as short position)
                    if (netProfitPercentage >= _config.MinSpreadPercentage && annualizedFundingRate > 0)
                    {
                        opportunities.Add(new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeSpotFutures,
                            Symbol = symbol,
                            LongExchange = spotExchange,      // Buy spot here
                            ShortExchange = futuresExchange,  // Short futures here
                            Exchange = spotExchange,          // For UI compatibility
                            SpotPrice = spotPrice,
                            PerpetualPrice = futuresPrice,
                            FundingRate = fundingRate.Rate,
                            AnnualizedFundingRate = annualizedFundingRate,
                            PricePremium = pricePremium,
                            SpreadRate = netProfit,
                            AnnualizedSpread = netProfit,
                            EstimatedProfitPercentage = netProfitPercentage,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }


    /// <summary>
    /// Process user-specific data for all active users.
    /// Each user gets their own exchange connectors with decrypted API keys,
    /// and user-specific data (positions, balances, P&L) is broadcast only to that user.
    /// </summary>
    private async Task ProcessAllUsersDataAsync(CancellationToken stoppingToken)
    {
        try
        {
            using var scope = _serviceProvider.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            // Get all users who have enabled API keys
            var activeUsers = await db.Users
                .Include(u => u.ExchangeApiKeys)
                .Where(u => u.ExchangeApiKeys.Any(k => k.IsEnabled))
                .ToListAsync(stoppingToken);

            // Filter keys in memory after loading
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

            // Process each user individually
            foreach (var user in activeUsers)
            {
                try
                {
                    await ProcessUserDataAsync(user, stoppingToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing data for user {UserId}", user.Id);
                    // Continue with next user on error
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ProcessAllUsersDataAsync");
        }
    }

    /// <summary>
    /// Process user-specific data for a single user.
    /// Gets user's exchange API keys, creates user-specific connectors,
    /// fetches positions/balances, and broadcasts to user's SignalR group only.
    /// </summary>
    private async Task ProcessUserDataAsync(ApplicationUser user, CancellationToken stoppingToken)
    {
        var userId = user.Id;
        var email = user.Email;

        try
        {
            using var scope = _serviceProvider.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
            var encryptionService = scope.ServiceProvider.GetRequiredService<IEncryptionService>();

            // Get user's enabled API keys
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
                            scope.ServiceProvider.GetRequiredService<ILogger<BinanceConnector>>()),
                        "bybit" => new BybitConnector(
                            scope.ServiceProvider.GetRequiredService<ILogger<BybitConnector>>()),
                        _ => null
                    };

                    if (connector != null)
                    {
                        var connected = await connector.ConnectAsync(
                            decryptedKey,
                            decryptedSecret,
                            apiKey.UseDemoTrading);

                        if (connected)
                        {
                            userConnectors[apiKey.ExchangeName] = connector;
                            _logger.LogDebug(
                                "User {UserId} connected to {Exchange} ({Mode})",
                                userId, apiKey.ExchangeName,
                                apiKey.UseDemoTrading ? "Demo" : "Live");
                        }
                        else
                        {
                            _logger.LogWarning(
                                "Failed to connect user {UserId} to {Exchange}",
                                userId, apiKey.ExchangeName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex,
                        "Error creating connector for user {UserId}, exchange {Exchange}",
                        userId, apiKey.ExchangeName);
                }
            }

            if (!userConnectors.Any())
            {
                _logger.LogWarning("User {UserId} ({Email}) has no working connectors", userId, email);
                return;
            }

            // Fetch user-specific positions (filtered by UserId)
            var userPositions = await db.Positions
                .Where(p => p.UserId == userId)
                .ToListAsync(stoppingToken);

            var positionDtos = new List<PositionDto>();

            // Update positions with live data from user's exchanges
            foreach (var position in userPositions)
            {
                if (userConnectors.TryGetValue(position.Exchange, out var connector))
                {
                    try
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
                                // Fallback: calculate P&L manually
                                var perpPrices = await connector.GetPerpetualPricesAsync(
                                    new List<string> { position.Symbol });

                                if (perpPrices.TryGetValue(position.Symbol, out var currentPrice))
                                {
                                    decimal priceDiff = currentPrice - position.EntryPrice;
                                    if (position.Side == PositionSide.Long)
                                    {
                                        unrealizedPnL = priceDiff * position.Quantity;
                                    }
                                    else
                                    {
                                        unrealizedPnL = -priceDiff * position.Quantity;
                                    }
                                }
                            }
                        }
                        else if (position.Type == PositionType.Spot)
                        {
                            var spotPrices = await connector.GetSpotPricesAsync(
                                new List<string> { position.Symbol });

                            if (spotPrices.TryGetValue(position.Symbol, out var spotPrice))
                            {
                                decimal currentPrice = spotPrice.Price;
                                decimal priceDiff = currentPrice - position.EntryPrice;

                                if (position.Side == PositionSide.Long)
                                {
                                    unrealizedPnL = priceDiff * position.Quantity;
                                }
                                else
                                {
                                    unrealizedPnL = -priceDiff * position.Quantity;
                                }
                            }
                        }

                        // Create DTO with live data
                        var positionDto = new PositionDto
                        {
                            Id = position.Id,
                            ExecutionId = position.ExecutionId,
                            Exchange = position.Exchange,
                            Symbol = position.Symbol,
                            Type = position.Type,
                            Side = position.Side,
                            Status = position.Status,
                            EntryPrice = position.EntryPrice,
                            ExitPrice = position.ExitPrice ?? 0,
                            Quantity = position.Quantity,
                            Leverage = position.Leverage,
                            InitialMargin = position.InitialMargin,
                            RealizedPnL = realizedPnL,
                            UnrealizedPnL = unrealizedPnL,
                            TotalFundingFeePaid = fundingPaid,
                            TotalFundingFeeReceived = fundingReceived,
                            OpenedAt = position.OpenedAt,
                            ClosedAt = position.ClosedAt,
                            ActiveOpportunityId = position.ExecutionId
                        };

                        positionDtos.Add(positionDto);

                        // Update database with latest values
                        position.UnrealizedPnL = unrealizedPnL;
                        position.RealizedPnL = realizedPnL;
                        position.TotalFundingFeePaid = fundingPaid;
                        position.TotalFundingFeeReceived = fundingReceived;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex,
                            "Error enriching position {PositionId} for user {UserId}",
                            position.Id, userId);
                    }
                }
            }

            // Save position updates
            if (userPositions.Any())
            {
                await db.SaveChangesAsync(stoppingToken);
            }

            // Fetch user-specific balances
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
                    _logger.LogError(ex,
                        "Error fetching balance for user {UserId} from {Exchange}",
                        userId, exchangeName);
                }
            }

            // Calculate user-specific P&L
            var totalPnL = userPositions.Sum(p => p.UnrealizedPnL + p.RealizedPnL);
            var today = DateTime.UtcNow.Date;
            var todayMetric = await db.PerformanceMetrics
                .Where(m => m.UserId == userId && m.Date == today)
                .FirstOrDefaultAsync(stoppingToken);
            var todayPnL = todayMetric?.TotalPnL ?? 0;

            // BROADCAST USER-SPECIFIC DATA - only to this user's group
            _logger.LogDebug("Broadcasting data for user {UserId} to group user_{UserId}", userId, userId);

            // Send positions to user's group only
            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceivePositions", positionDtos, stoppingToken);

            // Send balances to user's group only
            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceiveBalances", balances, stoppingToken);

            // Send P&L to user's group only
            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceivePnLUpdate", new { totalPnL, todayPnL, timestamp = DateTime.UtcNow }, stoppingToken);

            // Disconnect user-specific connectors
            foreach (var connector in userConnectors.Values)
            {
                try
                {
                    await connector.DisconnectAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disconnecting user {UserId} connector", userId);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing data for user {UserId}", userId);
        }
    }

    private async Task ExecuteOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        // Auto-execution logic would go here
        // This should be implemented carefully with proper risk management
        await Task.CompletedTask;
    }

    private async Task UpdatePositionsAndBalancesAsync()
    {
        var allBalances = new List<AccountBalanceDto>();
        var allPositions = new List<PositionDto>();

        // Query open spot positions first (needed for operational balance calculation)
        Dictionary<string, Dictionary<string, decimal>> activeSpotPositionsByExchange;
        using (var scope = _serviceProvider.CreateScope())
        {
            var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            var openSpotPositions = await dbContext.Positions
                .Where(p => p.Status == PositionStatus.Open && p.Type == PositionType.Spot)
                .ToListAsync();

            // Group by exchange and aggregate quantities by symbol
            activeSpotPositionsByExchange = openSpotPositions
                .GroupBy(p => p.Exchange)
                .ToDictionary(
                    g => g.Key,
                    g => g.GroupBy(p => p.Symbol.Replace("USDT", "")) // Extract base asset (e.g., BTC from BTCUSDT)
                          .ToDictionary(
                              sg => sg.Key,
                              sg => sg.Sum(p => p.Quantity)
                          )
                );
        }

        // Get balances from exchange connectors
        foreach (var (exchangeName, connector) in _exchangeConnectors)
        {
            try
            {
                // Get active spot positions for this exchange
                var activePositions = activeSpotPositionsByExchange.ContainsKey(exchangeName)
                    ? activeSpotPositionsByExchange[exchangeName]
                    : new Dictionary<string, decimal>();

                // Fetch balance with active positions
                var balance = await connector.GetAccountBalanceAsync(activePositions);
                allBalances.Add(balance);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating balance for {Exchange}", exchangeName);
            }
        }

        // Get positions from database and update with live PnL data
        using (var scope = _serviceProvider.CreateScope())
        {
            var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            var dbPositions = await dbContext.Positions
                .Where(p => p.Status == PositionStatus.Open)
                .ToListAsync();

            // Convert database positions to DTOs and enrich with live PnL
            foreach (var dbPosition in dbPositions)
            {
                if (_exchangeConnectors.TryGetValue(dbPosition.Exchange, out var connector))
                {
                    try
                    {
                        decimal unrealizedPnL = dbPosition.UnrealizedPnL;
                        decimal realizedPnL = dbPosition.RealizedPnL;
                        decimal fundingPaid = dbPosition.TotalFundingFeePaid;
                        decimal fundingReceived = dbPosition.TotalFundingFeeReceived;

                        if (dbPosition.Type == PositionType.Perpetual)
                        {
                            // For perpetual positions, get live data from exchange
                            var livePositions = await connector.GetOpenPositionsAsync();
                            var livePosition = livePositions.FirstOrDefault(p =>
                                p.Symbol == dbPosition.Symbol &&
                                p.Side == dbPosition.Side &&
                                p.Type == dbPosition.Type);

                            if (livePosition != null)
                            {
                                // Position found on exchange - use exchange data
                                unrealizedPnL = livePosition.UnrealizedPnL;
                                realizedPnL = livePosition.RealizedPnL;
                                fundingPaid = livePosition.TotalFundingFeePaid;
                                fundingReceived = livePosition.TotalFundingFeeReceived;
                            }
                            else
                            {
                                // Position not found on exchange (e.g., demo/testnet accounts)
                                // Calculate P&L manually using current perpetual price
                                _logger.LogWarning(
                                    "Position not found on {Exchange} for {Symbol} {Side} - calculating P&L from current price (demo mode)",
                                    dbPosition.Exchange, dbPosition.Symbol, dbPosition.Side);

                                var perpPrices = await connector.GetPerpetualPricesAsync(new List<string> { dbPosition.Symbol });
                                if (perpPrices.TryGetValue(dbPosition.Symbol, out var currentPrice))
                                {
                                    decimal priceDiff = currentPrice - dbPosition.EntryPrice;

                                    // Calculate unrealized P&L: (CurrentPrice - EntryPrice) * Quantity
                                    // For LONG: profit when price goes up (currentPrice > entryPrice)
                                    // For SHORT: profit when price goes down (currentPrice < entryPrice)
                                    if (dbPosition.Side == PositionSide.Long)
                                    {
                                        unrealizedPnL = priceDiff * dbPosition.Quantity;
                                    }
                                    else // Short
                                    {
                                        unrealizedPnL = -priceDiff * dbPosition.Quantity;
                                    }

                                    _logger.LogInformation(
                                        "Calculated P&L for {Exchange} {Symbol} {Side}: Entry=${Entry}, Current=${Current}, P&L=${PnL}",
                                        dbPosition.Exchange, dbPosition.Symbol, dbPosition.Side,
                                        dbPosition.EntryPrice, currentPrice, unrealizedPnL);
                                }
                            }
                        }
                        else if (dbPosition.Type == PositionType.Spot)
                        {
                            // For spot positions, calculate PnL based on current price
                            var spotPrices = await connector.GetSpotPricesAsync(new List<string> { dbPosition.Symbol });
                            if (spotPrices.TryGetValue(dbPosition.Symbol, out var spotPrice))
                            {
                                decimal currentPrice = spotPrice.Price;
                                decimal priceDiff = currentPrice - dbPosition.EntryPrice;

                                // Calculate unrealized P&L: (CurrentPrice - EntryPrice) * Quantity
                                // For long positions, profit when price goes up
                                // For short positions, profit when price goes down
                                if (dbPosition.Side == PositionSide.Long)
                                {
                                    unrealizedPnL = priceDiff * dbPosition.Quantity;
                                }
                                else // Short
                                {
                                    unrealizedPnL = -priceDiff * dbPosition.Quantity;
                                }
                            }
                        }

                        // Create DTO with live/calculated data
                        var positionDto = new PositionDto
                        {
                            Id = dbPosition.Id,
                            ExecutionId = dbPosition.ExecutionId,
                            Exchange = dbPosition.Exchange,
                            Symbol = dbPosition.Symbol,
                            Type = dbPosition.Type,
                            Side = dbPosition.Side,
                            Status = dbPosition.Status,
                            EntryPrice = dbPosition.EntryPrice,
                            ExitPrice = dbPosition.ExitPrice ?? 0,
                            Quantity = dbPosition.Quantity,
                            Leverage = dbPosition.Leverage,
                            InitialMargin = dbPosition.InitialMargin,
                            RealizedPnL = realizedPnL,
                            UnrealizedPnL = unrealizedPnL,
                            TotalFundingFeePaid = fundingPaid,
                            TotalFundingFeeReceived = fundingReceived,
                            OpenedAt = dbPosition.OpenedAt,
                            ClosedAt = dbPosition.ClosedAt,
                            ActiveOpportunityId = dbPosition.ExecutionId
                        };

                        allPositions.Add(positionDto);

                        // Update database position with latest PnL values
                        dbPosition.UnrealizedPnL = unrealizedPnL;
                        dbPosition.RealizedPnL = realizedPnL;
                        dbPosition.TotalFundingFeePaid = fundingPaid;
                        dbPosition.TotalFundingFeeReceived = fundingReceived;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error enriching position {Id} with live data", dbPosition.Id);

                        // Fallback: use database values only
                        var positionDto = new PositionDto
                        {
                            Id = dbPosition.Id,
                            ExecutionId = dbPosition.ExecutionId,
                            Exchange = dbPosition.Exchange,
                            Symbol = dbPosition.Symbol,
                            Type = dbPosition.Type,
                            Side = dbPosition.Side,
                            Status = dbPosition.Status,
                            EntryPrice = dbPosition.EntryPrice,
                            ExitPrice = dbPosition.ExitPrice ?? 0,
                            Quantity = dbPosition.Quantity,
                            Leverage = dbPosition.Leverage,
                            InitialMargin = dbPosition.InitialMargin,
                            RealizedPnL = dbPosition.RealizedPnL,
                            UnrealizedPnL = dbPosition.UnrealizedPnL,
                            TotalFundingFeePaid = dbPosition.TotalFundingFeePaid,
                            TotalFundingFeeReceived = dbPosition.TotalFundingFeeReceived,
                            OpenedAt = dbPosition.OpenedAt,
                            ClosedAt = dbPosition.ClosedAt,
                            ActiveOpportunityId = dbPosition.ExecutionId
                        };

                        allPositions.Add(positionDto);
                    }
                }
            }

            // Save updated PnL values to database
            await dbContext.SaveChangesAsync();
        }

        // Broadcast balances and positions
        await _hubContext.Clients.All.SendAsync("ReceiveBalances", allBalances);
        await _hubContext.Clients.All.SendAsync("ReceivePositions", allPositions);
    }

    private async Task SendDashboardUpdateAsync()
    {
        using var scope = _serviceProvider.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        var today = DateTime.UtcNow.Date;
        var todayMetric = await dbContext.PerformanceMetrics
            .Where(pm => pm.Date == today)
            .FirstOrDefaultAsync();

        // Total PnL is now tracked in PerformanceMetrics table only
        // (closed ActiveOpportunities are deleted, so we can't query them)
        var totalPnL = await dbContext.PerformanceMetrics
            .SumAsync(pm => pm.TotalPnL);

        await _hubContext.Clients.All.SendAsync("ReceivePnLUpdate", new
        {
            totalPnL,
            todayPnL = todayMetric?.TotalPnL ?? 0
        });
    }

    /// <summary>
    /// Efficiently upsert funding rates using PostgreSQL's ON CONFLICT clause.
    /// Updates existing records or inserts new ones based on Exchange + Symbol uniqueness.
    /// This is optimized for bulk operations (500+ records).
    /// </summary>
    private async Task UpsertFundingRatesAsync(
        ArbitrageDbContext dbContext,
        string exchangeName,
        List<FundingRateDto> rates,
        CancellationToken cancellationToken)
    {
        if (!rates.Any())
            return;

        // Process in batches of 100 to avoid parameter limit (PostgreSQL has a 65535 param limit)
        const int batchSize = 100;
        for (int i = 0; i < rates.Count; i += batchSize)
        {
            var batch = rates.Skip(i).Take(batchSize).ToList();

            // Build VALUES clause for batch
            var valuesClauses = new List<string>();

            for (int j = 0; j < batch.Count; j++)
            {
                valuesClauses.Add($@"
                    (@Exchange{j}, @Symbol{j}, @Rate{j}, @AnnualizedRate{j},
                     @FundingIntervalHours{j}, @Average3DayRate{j}, @Direction{j},
                     @PreviousRate{j}, @PreviousAnnualizedRate{j}, @FundingCap{j}, @FundingFloor{j},
                     @FundingTime{j}, @NextFundingTime{j}, @RecordedAt{j})");
            }

            // Construct final SQL with all VALUES
            var batchSql = $@"
                INSERT INTO ""FundingRates""
                    (""Exchange"", ""Symbol"", ""Rate"", ""AnnualizedRate"", ""FundingIntervalHours"",
                     ""Average3DayRate"", ""Direction"", ""PreviousRate"", ""PreviousAnnualizedRate"",
                     ""FundingCap"", ""FundingFloor"", ""FundingTime"", ""NextFundingTime"", ""RecordedAt"")
                VALUES
                    {string.Join(",", valuesClauses)}
                ON CONFLICT (""Exchange"", ""Symbol"")
                DO UPDATE SET
                    ""Rate"" = EXCLUDED.""Rate"",
                    ""AnnualizedRate"" = EXCLUDED.""AnnualizedRate"",
                    ""FundingIntervalHours"" = EXCLUDED.""FundingIntervalHours"",
                    ""Average3DayRate"" = EXCLUDED.""Average3DayRate"",
                    ""Direction"" = EXCLUDED.""Direction"",
                    ""PreviousRate"" = EXCLUDED.""PreviousRate"",
                    ""PreviousAnnualizedRate"" = EXCLUDED.""PreviousAnnualizedRate"",
                    ""FundingCap"" = EXCLUDED.""FundingCap"",
                    ""FundingFloor"" = EXCLUDED.""FundingFloor"",
                    ""FundingTime"" = EXCLUDED.""FundingTime"",
                    ""NextFundingTime"" = EXCLUDED.""NextFundingTime"",
                    ""RecordedAt"" = EXCLUDED.""RecordedAt""";

            // Create parameter objects for Npgsql
            var npgsqlParams = new List<Npgsql.NpgsqlParameter>();
            for (int j = 0; j < batch.Count; j++)
            {
                var rate = batch[j];
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@Exchange{j}", exchangeName));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@Symbol{j}", rate.Symbol));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@Rate{j}", rate.Rate));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@AnnualizedRate{j}", rate.AnnualizedRate));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@FundingIntervalHours{j}", rate.FundingIntervalHours));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@Average3DayRate{j}",
                    rate.Average3DayRate.HasValue ? (object)rate.Average3DayRate.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@Direction{j}",
                    rate.Direction.HasValue ? (object)(int)rate.Direction.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@PreviousRate{j}",
                    rate.PreviousRate.HasValue ? (object)rate.PreviousRate.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@PreviousAnnualizedRate{j}",
                    rate.PreviousAnnualizedRate.HasValue ? (object)rate.PreviousAnnualizedRate.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@FundingCap{j}",
                    rate.FundingCap.HasValue ? (object)rate.FundingCap.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@FundingFloor{j}",
                    rate.FundingFloor.HasValue ? (object)rate.FundingFloor.Value : DBNull.Value));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@FundingTime{j}", rate.FundingTime));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@NextFundingTime{j}", rate.NextFundingTime));
                npgsqlParams.Add(new Npgsql.NpgsqlParameter($"@RecordedAt{j}", rate.RecordedAt));
            }

            await dbContext.Database.ExecuteSqlRawAsync(batchSql, npgsqlParams.ToArray(), cancellationToken);
        }

        _logger.LogDebug("Upserted {Count} funding rates for {Exchange}", rates.Count, exchangeName);
    }

    public override async Task StopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Arbitrage Engine Service stopping...");

        foreach (var connector in _exchangeConnectors.Values)
        {
            await connector.DisconnectAsync();
        }

        await base.StopAsync(stoppingToken);
    }
}
