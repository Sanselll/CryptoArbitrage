using Microsoft.AspNetCore.SignalR;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Config;
using Microsoft.EntityFrameworkCore;

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

        // Initialize exchange connectors
        await InitializeExchangesAsync();

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await RunArbitrageAnalysisAsync();
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
        using var scope = _serviceProvider.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        var exchanges = await dbContext.Exchanges
            .Where(e => e.IsEnabled)
            .ToListAsync();

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
                    var connected = await connector.ConnectAsync(exchange.ApiKey, exchange.ApiSecret);
                    if (connected)
                    {
                        _exchangeConnectors[exchange.Name] = connector;
                        _logger.LogInformation("Connected to {Exchange}", exchange.Name);
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
                        _config.MaxSymbolCount
                    );
                    allDiscoveredSymbols.AddRange(symbols);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error discovering symbols from {Exchange}", exchangeName);
                }
            }

            // Get unique symbols (intersection if multiple exchanges, union otherwise)
            if (_exchangeConnectors.Count > 1)
            {
                // For cross-exchange arbitrage, only include symbols available on all exchanges
                var symbolGroups = _exchangeConnectors.Values
                    .Select(async c => await c.GetActiveSymbolsAsync(_config.MinDailyVolumeUsd, _config.MaxSymbolCount))
                    .Select(t => t.Result.ToHashSet());

                _activeSymbols = symbolGroups
                    .Aggregate((a, b) => { a.IntersectWith(b); return a; })
                    .OrderBy(s => s)
                    .ToList();
            }
            else
            {
                // Single exchange, use all discovered symbols
                _activeSymbols = allDiscoveredSymbols.Distinct().OrderBy(s => s).ToList();
            }

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

    private async Task RunArbitrageAnalysisAsync()
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

        // Fetch funding rates, spot prices, and perpetual prices from all exchanges
        foreach (var (exchangeName, connector) in _exchangeConnectors)
        {
            try
            {
                var rates = await connector.GetFundingRatesAsync(_activeSymbols);
                fundingRates[exchangeName] = rates;

                // Fetch spot prices for spot-perpetual arbitrage
                var prices = await connector.GetSpotPricesAsync(_activeSymbols);
                spotPrices[exchangeName] = prices;

                // Fetch perpetual prices
                var perps = await connector.GetPerpetualPricesAsync(_activeSymbols);
                perpPrices[exchangeName] = perps;

                // Save to database
                using var scope = _serviceProvider.CreateScope();
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                var exchange = await dbContext.Exchanges.FirstAsync(e => e.Name == exchangeName);

                foreach (var rate in rates)
                {
                    dbContext.FundingRates.Add(new FundingRate
                    {
                        ExchangeId = exchange.Id,
                        Symbol = rate.Symbol,
                        Rate = rate.Rate,
                        AnnualizedRate = rate.AnnualizedRate,
                        FundingTime = rate.FundingTime,
                        NextFundingTime = rate.NextFundingTime,
                        RecordedAt = rate.RecordedAt
                    });
                }

                await dbContext.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching funding rates from {Exchange}", exchangeName);
            }
        }

        // Broadcast funding rates via SignalR
        var allRates = fundingRates.Values.SelectMany(r => r).ToList();
        await _hubContext.Clients.All.SendAsync("ReceiveFundingRates", allRates);

        // Detect both types of arbitrage opportunities
        var opportunities = new List<ArbitrageOpportunityDto>();

        // Cross-exchange arbitrage (requires 2+ exchanges)
        if (fundingRates.Count >= 2)
        {
            var crossExchangeOpps = await DetectCrossExchangeOpportunitiesAsync(fundingRates);
            opportunities.AddRange(crossExchangeOpps);
        }

        // Spot-perpetual arbitrage (works with single exchange)
        var spotPerpOpps = await DetectSpotPerpetualOpportunitiesAsync(fundingRates, spotPrices, perpPrices);
        opportunities.AddRange(spotPerpOpps);

        if (opportunities.Any())
        {
            // Enrich opportunities with execution information from Executions table
            using (var scope = _serviceProvider.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                var runningExecutions = await dbContext.Executions
                    .Where(e => e.State == ExecutionState.Running)
                    .ToListAsync();

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
            await _hubContext.Clients.All.SendAsync("ReceiveOpportunities", opportunities);

            // Auto-execute if enabled
            if (_config.AutoExecute)
            {
                await ExecuteOpportunitiesAsync(opportunities);
            }
        }

        // Update positions and balances
        await UpdatePositionsAndBalancesAsync();

        // Send dashboard update
        await SendDashboardUpdateAsync();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates)
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

                    var spread = Math.Abs(rate1.Rate - rate2.Rate);
                    var annualizedSpread = Math.Abs(rate1.AnnualizedRate - rate2.AnnualizedRate);

                    if (annualizedSpread * 100 >= _config.MinSpreadPercentage)
                    {
                        // Determine which exchange should be long and which should be short
                        var (longExchange, longRate, shortExchange, shortRate) =
                            rate1.Rate < rate2.Rate
                            ? (exchange1, rate1.Rate, exchange2, rate2.Rate)
                            : (exchange2, rate2.Rate, exchange1, rate1.Rate);

                        opportunities.Add(new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            Symbol = symbol,
                            LongExchange = longExchange,
                            ShortExchange = shortExchange,
                            LongFundingRate = longRate,
                            ShortFundingRate = shortRate,
                            SpreadRate = spread,
                            AnnualizedSpread = annualizedSpread,
                            EstimatedProfitPercentage = annualizedSpread * 100,
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

                // Calculate net profit based on funding rate direction
                // Positive funding = shorts pay longs -> buy spot + short perp -> collect funding
                // Negative funding = longs pay shorts -> sell spot (borrow) + long perp -> collect funding
                // In both cases, we collect the absolute value of funding rate
                decimal netProfit = Math.Abs(annualizedFundingRate) - Math.Abs(pricePremium) - estimatedTradingFees;
                decimal netProfitPercentage = netProfit * 100;

                // Only create opportunity if net profit exceeds minimum threshold
                if (netProfitPercentage >= _config.MinSpreadPercentage)
                {
                    opportunities.Add(new ArbitrageOpportunityDto
                    {
                        Strategy = ArbitrageStrategy.SpotPerpetual,
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

        // Get balances from exchange connectors
        foreach (var (exchangeName, connector) in _exchangeConnectors)
        {
            try
            {
                // Fetch balance
                var balance = await connector.GetAccountBalanceAsync();
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
                                unrealizedPnL = livePosition.UnrealizedPnL;
                                realizedPnL = livePosition.RealizedPnL;
                                fundingPaid = livePosition.TotalFundingFeePaid;
                                fundingReceived = livePosition.TotalFundingFeeReceived;
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
