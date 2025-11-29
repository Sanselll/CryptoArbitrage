using Bybit.Net.Clients;
using Bybit.Net.Enums;
using CryptoExchange.Net.Authentication;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;
using Microsoft.Extensions.Configuration;
using PositionSide = CryptoArbitrage.API.Data.Entities.PositionSide;
using PositionStatus = CryptoArbitrage.API.Data.Entities.PositionStatus;
using BybitOrderSide = Bybit.Net.Enums.OrderSide;
using ModelOrderSide = CryptoArbitrage.API.Models.OrderSide;

namespace CryptoArbitrage.API.Services.Exchanges;

public class BybitConnector : IExchangeConnector
{
    private readonly ILogger<BybitConnector> _logger;
    private readonly IConfiguration _configuration;
    private BybitRestClient? _restClient;
    private BybitSocketClient? _socketClient;

    public string ExchangeName => "Bybit";

    public BybitConnector(ILogger<BybitConnector> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
    }

    public async Task<bool> ConnectAsync(string? apiKey, string? apiSecret)
    {
        try
        {
            // Read environment configuration to determine live vs demo mode
            var isLive = _configuration.GetValue<bool>("Environment:IsLive");
            var environment = isLive ? Bybit.Net.BybitEnvironment.Live : Bybit.Net.BybitEnvironment.DemoTrading;
            
            // For public API access (funding rates, prices), credentials are optional
            var hasCredentials = !string.IsNullOrEmpty(apiKey) && !string.IsNullOrEmpty(apiSecret);

            _restClient = new BybitRestClient(options =>
            {
                if (hasCredentials)
                {
                    options.ApiCredentials = new ApiCredentials(apiKey!, apiSecret!);
                }
                options.Environment = environment;
                // Configure request timeout to prevent hanging requests
                options.RequestTimeout = TimeSpan.FromSeconds(30);
            });

            _socketClient = new BybitSocketClient(options =>
            {
                if (hasCredentials)
                {
                    options.ApiCredentials = new ApiCredentials(apiKey!, apiSecret!);
                }
                options.Environment = environment;
            });

            // Test connection - use public API if no credentials
            if (hasCredentials)
            {
                var accountInfo = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

                if (accountInfo.Success)
                {
                    return true;
                }

                _logger.LogError("Failed to connect to Bybit: {Error}", accountInfo.Error);
                return false;
            }
            else
            {
                // For public API access, test with a funding rates query
                var fundingTest = await _restClient.V5Api.ExchangeData.GetFundingRateHistoryAsync(Category.Linear, "BTCUSDT", limit: 1);
                if (fundingTest.Success)
                {
                    return true;
                }

                _logger.LogError("Failed to connect to Bybit public API: {Error}", fundingTest.Error);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting to Bybit");
            return false;
        }
    }

    public async Task DisconnectAsync()
    {
        _restClient?.Dispose();
        await (_socketClient?.UnsubscribeAllAsync() ?? Task.CompletedTask);
        _socketClient?.Dispose();
    }

    public async Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols, decimal minHighPriorityFundingRate = 0)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            // Get all linear perpetual tickers
            var perpTickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);
            if (!perpTickers.Success || perpTickers.Data == null)
            {
                _logger.LogError("Failed to get perpetual tickers from Bybit");
                return new List<string>();
            }

            // Get all spot tickers
            var spotTickers = await _restClient.V5Api.ExchangeData.GetSpotTickersAsync();
            if (!spotTickers.Success || spotTickers.Data == null)
            {
                _logger.LogError("Failed to get spot tickers from Bybit");
                return new List<string>();
            }

            // === STEP 1: Find high-priority symbols (high funding rate) ===
            var highPrioritySymbols = new HashSet<string>();

            if (minHighPriorityFundingRate > 0)
            {
                // Find symbols with high absolute funding rates
                var highFundingSymbols = perpTickers.Data.List
                    .Where(t =>
                        t.Symbol.EndsWith("USDT") &&
                        t.FundingRate.HasValue &&
                        Math.Abs(t.FundingRate.Value) >= minHighPriorityFundingRate)
                    .Select(t => t.Symbol)
                    .ToHashSet();

                // Filter to only include symbols that have BOTH perp AND spot markets
                var spotSymbolSet = spotTickers.Data.List
                    .Where(t => t.Symbol.EndsWith("USDT"))
                    .Select(t => t.Symbol)
                    .ToHashSet();

                foreach (var symbol in highFundingSymbols)
                {
                    if (spotSymbolSet.Contains(symbol))
                    {
                        highPrioritySymbols.Add(symbol);
                    }
                }

                _logger.LogInformation(
                    "Found {Count} high-priority symbols with |funding| >= {MinFunding:P4} and both perp+spot markets: {Symbols}",
                    highPrioritySymbols.Count,
                    minHighPriorityFundingRate,
                    string.Join(", ", highPrioritySymbols)
                );
            }

            // === STEP 2: Get volume-based symbols ===
            // Filter for USDT perpetuals with sufficient volume
            var perpSymbols = perpTickers.Data.List
                .Where(t => t.Symbol.EndsWith("USDT") && t.Volume24h > 0 && t.LastPrice > 0)
                .Select(t => new
                {
                    Symbol = t.Symbol,
                    VolumeUsd = t.Volume24h * t.LastPrice
                })
                .Where(s => s.VolumeUsd >= minDailyVolumeUsd)
                .OrderByDescending(s => s.VolumeUsd)
                .Take(maxSymbols)
                .ToList();

            // Filter for USDT spot markets with sufficient volume
            var spotSymbols = spotTickers.Data.List
                .Where(t => t.Symbol.EndsWith("USDT") && t.Volume24h > 0 && t.LastPrice > 0)
                .Select(t => new
                {
                    Symbol = t.Symbol,
                    VolumeUsd = t.Volume24h * t.LastPrice
                })
                .Where(s => s.VolumeUsd >= minDailyVolumeUsd)
                .OrderByDescending(s => s.VolumeUsd)
                .Take(maxSymbols)
                .ToList();

            // Merge both lists, taking unique symbols
            var volumeBasedSymbols = perpSymbols
                .Concat(spotSymbols)
                .GroupBy(s => s.Symbol)
                .Select(g => new { Symbol = g.Key, VolumeUsd = g.Max(x => x.VolumeUsd) })
                .OrderByDescending(s => s.VolumeUsd)
                .Take(maxSymbols)
                .Select(s => s.Symbol)
                .ToList();

            // === STEP 3: Combine high-priority and volume-based symbols ===
            // Prioritize high-funding symbols, then fill with volume-based
            var finalSymbols = highPrioritySymbols.ToList();

            foreach (var symbol in volumeBasedSymbols)
            {
                if (!finalSymbols.Contains(symbol))
                {
                    finalSymbols.Add(symbol);
                }

                if (finalSymbols.Count >= maxSymbols)
                {
                    break;
                }
            }

            _logger.LogInformation(
                "Discovered {TotalCount} active symbols from Bybit ({HighPriorityCount} high-priority, {PerpCount} perp, {SpotCount} spot, min volume: ${MinVolume:N0}, max symbols: {MaxSymbols})",
                finalSymbols.Count,
                highPrioritySymbols.Count,
                perpSymbols.Count,
                spotSymbols.Count,
                minDailyVolumeUsd,
                maxSymbols
            );

            return finalSymbols;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering active symbols from Bybit");
            return new List<string>();
        }
    }

    public async Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var fundingRates = new List<FundingRateDto>();

        try
        {
            // Fetch ALL instruments info with pagination to get funding intervals
            var fundingIntervalMap = new Dictionary<string, int>();
            string? cursor = null;
            int pageCount = 0;
            int totalSymbols = 0;

            do
            {
                pageCount++;
                var instrumentsInfo = await _restClient.V5Api.ExchangeData.GetLinearInverseSymbolsAsync(
                    Category.Linear,
                    cursor: cursor,
                    limit: 1000 // Max limit per page
                );

                if (instrumentsInfo.Success && instrumentsInfo.Data != null)
                {
                    foreach (var info in instrumentsInfo.Data.List)
                    {
                        // FundingInterval is in minutes, convert to hours
                        int intervalMinutes = info.FundingInterval > 0 ? info.FundingInterval : 480; // Default: 480 minutes = 8 hours
                        fundingIntervalMap[info.Name] = intervalMinutes / 60;

                        // DEBUG: Log TNSRUSDT specifically
                        if (info.Name == "TNSRUSDT")
                        {
                            _logger.LogWarning("✅ Found TNSRUSDT: FundingInterval={Minutes}min ({Hours}h) on page {Page}",
                                info.FundingInterval, intervalMinutes / 60, pageCount);
                        }
                    }

                    totalSymbols += instrumentsInfo.Data.List.Count();
                    cursor = instrumentsInfo.Data.NextPageCursor;

                    _logger.LogDebug("Loaded page {Page}: {Count} symbols (cursor: {Cursor})",
                        pageCount, instrumentsInfo.Data.List.Count(), cursor ?? "END");
                }
                else
                {
                    _logger.LogWarning("Failed to fetch Bybit instruments page {Page}: {Error}",
                        pageCount, instrumentsInfo.Error?.Message ?? "Unknown error");
                    break;
                }

            } while (!string.IsNullOrEmpty(cursor));

            _logger.LogInformation("Loaded funding intervals for {Total} symbols from Bybit across {Pages} pages",
                totalSymbols, pageCount);

            // Get all linear perpetual contracts
            var instruments = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);

            if (instruments.Success && instruments.Data != null)
            {
                var relevantInstruments = instruments.Data.List
                    .Where(i => symbols.Count == 0 || symbols.Contains(i.Symbol))
                    .ToList();
                
                foreach (var instrument in relevantInstruments)
                {
                    if (instrument.FundingRate.HasValue && instrument.NextFundingTime.HasValue)
                    {
                        var currentRate = instrument.FundingRate.Value;

                        // Get funding interval from map, fallback to 8 hours
                        int fundingIntervalHours = 8; // Default
                        if (fundingIntervalMap.TryGetValue(instrument.Symbol, out var intervalHours))
                        {
                            fundingIntervalHours = intervalHours;
                        }

                        // Note: We skip fetching historical rates per-symbol as it requires 228+ sequential API calls
                        // which is too slow. Previous rate will be populated from database on next fetch cycle.
                        decimal? previousRate = null;
                        decimal? previousAnnualizedRate = null;

                        // Calculate annualized rate: Rate × (24 / interval) × 365 × 100
                        var annualizedRate = currentRate * (24m / fundingIntervalHours) * 365 * 100;

                        // Determine direction
                        var direction = currentRate < 0
                            ? FundingDirection.ShortPaysLong
                            : FundingDirection.LongPaysShort;

                        // Bybit doesn't expose funding cap/floor in API
                        decimal? fundingCap = null;
                        decimal? fundingFloor = null;

                        fundingRates.Add(new FundingRateDto
                        {
                            Exchange = ExchangeName,
                            Symbol = instrument.Symbol,
                            Rate = currentRate,
                            AnnualizedRate = annualizedRate,
                            FundingIntervalHours = fundingIntervalHours,
                            Direction = direction,
                            PreviousRate = previousRate,
                            PreviousAnnualizedRate = previousAnnualizedRate,
                            FundingCap = fundingCap,
                            FundingFloor = fundingFloor,
                            FundingTime = DateTime.UtcNow,
                            NextFundingTime = instrument.NextFundingTime.Value,
                            RecordedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rates from Bybit");
        }

        return fundingRates;
    }

    public async Task<Dictionary<string, PriceDto>> GetSpotPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var spotPrices = new Dictionary<string, PriceDto>();

        try
        {
            // Get spot prices from Bybit Spot API
            var tickers = await _restClient.V5Api.ExchangeData.GetSpotTickersAsync();

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.List.Where(t => symbols.Contains(t.Symbol)))
                {
                    spotPrices[ticker.Symbol] = new PriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Symbol,
                        Price = ticker.LastPrice,
                        Volume24h = ticker.Volume24h,
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot prices from Bybit");
        }

        return spotPrices;
    }

    public async Task<Dictionary<string, PriceDto>> GetPerpetualPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var perpPrices = new Dictionary<string, PriceDto>();

        try
        {
            // Get perpetual prices from Bybit linear tickers
            var tickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.List.Where(t => symbols.Contains(t.Symbol)))
                {
                    perpPrices[ticker.Symbol] = new PriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Symbol,
                        Price = ticker.LastPrice,
                        Volume24h = ticker.Volume24h,
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching perpetual prices from Bybit");
        }

        return perpPrices;
    }

    public async Task<Dictionary<string, decimal>> Get24hVolumeAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var volumes = new Dictionary<string, decimal>();

        try
        {
            // Get 24h ticker data from Bybit linear tickers
            var tickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.List.Where(t => symbols.Contains(t.Symbol)))
                {
                    // Turnover24h is the 24h volume in USDT
                    volumes[ticker.Symbol] = ticker.Turnover24h;
                }
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching 24h volumes from Bybit");
        }

        return volumes;
    }

    public async Task<LiquidityMetricsDto?> GetLiquidityMetricsAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            // Get orderbook depth for linear perpetuals
            // Note: Bybit.Net V5 API - GetOrderbookAsync may have different signature
            // Adjust based on your Bybit.Net library version
            var orderbook = await _restClient.V5Api.ExchangeData.GetOrderbookAsync(Category.Linear, symbol, 100);

            if (!orderbook.Success || orderbook.Data == null)
            {
                _logger.LogWarning("Failed to fetch orderbook for {Symbol} from Bybit", symbol);
                return null;
            }

            var bids = orderbook.Data.Bids.ToList();
            var asks = orderbook.Data.Asks.ToList();

            if (bids.Count() == 0 || asks.Count() == 0)
            {
                _logger.LogWarning("Empty orderbook for {Symbol} from Bybit", symbol);
                return null;
            }

            // Calculate best bid and ask
            var bestBid = bids.First().Price;
            var bestAsk = asks.First().Price;

            // Calculate bid/ask spread percentage
            var midPrice = (bestBid + bestAsk) / 2;
            var bidAskSpread = ((bestAsk - bestBid) / midPrice) * 100;

            // Calculate orderbook depth within 1% of mid price
            var lowerBound = midPrice * 0.99m;
            var upperBound = midPrice * 1.01m;

            var depthUsd = 0m;

            // Sum bid depth within 1% below mid price
            foreach (var bid in bids.Where(b => b.Price >= lowerBound))
            {
                depthUsd += bid.Quantity * bid.Price;
            }

            // Sum ask depth within 1% above mid price
            foreach (var ask in asks.Where(a => a.Price <= upperBound))
            {
                depthUsd += ask.Quantity * ask.Price;
            }
            
            return new LiquidityMetricsDto
            {
                BidAskSpreadPercent = bidAskSpread,
                OrderbookDepthUsd = depthUsd,
                Status = LiquidityStatus.Good, // Will be evaluated by ArbitrageEngineService
                WarningMessage = null
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching liquidity metrics for {Symbol} from Bybit", symbol);
            return null;
        }
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync()
    {
        // Call the overload with empty active positions dictionary
        return await GetAccountBalanceAsync(new Dictionary<string, decimal>());
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync(Dictionary<string, decimal> activeSpotPositions)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            var balance = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

            if (balance.Success && balance.Data != null && balance.Data.List.Any())
            {
                var account = balance.Data.List.First();

                // Get all asset balances
                var spotBalances = await GetSpotBalancesAsync();
                decimal spotTotalUsd = 0;
                decimal spotAvailableUsd = 0;
                decimal spotUsdtOnly = 0; // Only USDT

                // Convert spot assets to USD
                if (spotBalances.Any())
                {
                    var tickers = await _restClient.V5Api.ExchangeData.GetSpotTickersAsync();
                    if (tickers.Success && tickers.Data != null)
                    {
                        foreach (var asset in spotBalances)
                        {
                            if (asset.Key == "USDT")
                            {
                                spotTotalUsd += asset.Value;
                                spotAvailableUsd = asset.Value; // Only USDT for available
                                spotUsdtOnly = asset.Value; // Track USDT separately
                            }
                            else
                            {
                                var symbol = $"{asset.Key}USDT";
                                var price = tickers.Data.List.FirstOrDefault(t => t.Symbol == symbol);
                                if (price != null)
                                {
                                    var usdValue = asset.Value * price.LastPrice;
                                    spotTotalUsd += usdValue; // Add to total only
                                }
                            }
                        }
                    }
                }

                // Bybit Unified Trading Account: All assets share the same collateral pool
                // Use account-level balances (not individual asset balances)
                var usdtBalance = account.Assets.FirstOrDefault(a => a.Asset == "USDT");

                // For Bybit Unified account:
                // - totalWalletBalance = total wallet balance (all assets in USD)
                // - totalAvailableBalance = available balance (not locked in positions/orders)
                // - totalEquity = totalWalletBalance + unrealized P&L
                // - totalInitialMargin = margin used in positions
                decimal futuresTotal = account.TotalWalletBalance ?? 0;
                decimal futuresAvailable = account.TotalAvailableBalance ?? 0;
                decimal marginUsed = (account.TotalInitialMargin ?? 0);
                decimal unrealizedPnL = usdtBalance?.UnrealizedPnl ?? 0;
                
                // Calculate coins in active positions value
                decimal coinsInActivePositionsUsd = 0;
                if (activeSpotPositions.Any())
                {
                    var tickers = await _restClient.V5Api.ExchangeData.GetSpotTickersAsync();
                    if (tickers.Success && tickers.Data != null)
                    {
                        foreach (var position in activeSpotPositions)
                        {
                            var symbol = $"{position.Key}USDT";
                            var price = tickers.Data.List.FirstOrDefault(t => t.Symbol == symbol);
                            if (price != null)
                            {
                                coinsInActivePositionsUsd += position.Value * price.LastPrice;
                            }
                        }
                    }
                }

                // Bybit Unified Account: totalWalletBalance already includes all assets (USDT + coins)
                // spotTotalUsd may be slightly different due to price calculation timing
                // Use account-level totals which are authoritative
                decimal totalBalance = futuresTotal; // This already includes all assets in USD
                
                // For frontend calculations (Bybit Unified Account):
                // In Bybit's unified account, all assets are in one pool
                // For the frontend display:
                // - SpotBalanceUsd = USDT + other assets (total "spot-like" balance)
                // - SpotAvailableUsd = USDT only (the liquid stablecoin)
                // - FuturesBalanceUsd = total wallet balance
                // - FuturesAvailableUsd = available for trading
                // - MarginUsed = what's locked in positions
                decimal spotAssetsOnly = spotTotalUsd - spotUsdtOnly; // Non-USDT assets value

                return new AccountBalanceDto
                {
                    Exchange = ExchangeName,
                    // Total balance (all assets in USD)
                    TotalBalance = totalBalance,
                    AvailableBalance = futuresAvailable,
                    OperationalBalanceUsd = totalBalance,
                    // Spot balances: For unified account, show USDT + other assets
                    SpotBalanceUsd = spotTotalUsd, // USDT + other assets
                    SpotAvailableUsd = spotUsdtOnly, // Only USDT (the stablecoin)
                    SpotAssets = spotBalances,
                    // Futures/Unified account totals
                    FuturesBalanceUsd = totalBalance,
                    FuturesAvailableUsd = futuresAvailable,
                    MarginUsed = marginUsed,
                    UnrealizedPnL = unrealizedPnL,
                    UpdatedAt = DateTime.UtcNow
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching account balance from Bybit");
        }

        return new AccountBalanceDto { Exchange = ExchangeName };
    }

    public async Task<string> PlaceMarketOrderAsync(string symbol, PositionSide side, decimal quantity, decimal leverage)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            _logger.LogInformation("Placing perpetual {Side} order on Bybit: {Symbol}, Quantity: {Quantity}, Leverage: {Leverage}",
                side, symbol, quantity, leverage);

            // Set leverage BEFORE placing the order (Bybit requires this)
            try
            {
                var leverageResult = await _restClient.V5Api.Account.SetLeverageAsync(
                    Category.Linear,
                    symbol,
                    leverage,
                    leverage  // buyLeverage and sellLeverage (same for both sides in unified margin)
                );

                if (!leverageResult.Success)
                {
                    _logger.LogWarning("Failed to set leverage for {Symbol} to {Leverage}x: {Error}. Continuing with current leverage setting.",
                        symbol, leverage, leverageResult.Error);
                }
                else
                {
                    _logger.LogInformation("Set leverage for {Symbol} to {Leverage}x", symbol, leverage);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error setting leverage for {Symbol}, continuing with current leverage setting", symbol);
            }

            var orderSide = side == PositionSide.Long ? BybitOrderSide.Buy : BybitOrderSide.Sell;

            var order = await _restClient.V5Api.Trading.PlaceOrderAsync(
                Category.Linear,
                symbol,
                orderSide,
                NewOrderType.Market,
                quantity,
                null, // No price for market orders
                isLeverage: true
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Perpetual {Side} order placed on Bybit: {OrderId}, Quantity: {Quantity}",
                    side, order.Data.OrderId, quantity);
                return order.Data.OrderId;
            }

            _logger.LogError("Failed to place perpetual order on Bybit: {Error}", order.Error);
            throw new Exception($"Failed to place order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing perpetual order on Bybit");
            throw;
        }
    }

    public async Task<ClosePositionResult> ClosePositionAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var result = new ClosePositionResult();

        try
        {
            var positions = await _restClient.V5Api.Trading.GetPositionsAsync(Category.Linear, symbol);

            if (positions.Success && positions.Data != null)
            {
                decimal totalFilledQty = 0;
                decimal weightedPrice = 0;
                decimal totalFee = 0;

                foreach (var position in positions.Data.List.Where(p => p.Quantity > 0))
                {
                    var orderSide = position.Side == Bybit.Net.Enums.PositionSide.Buy ? BybitOrderSide.Sell : BybitOrderSide.Buy;

                    var orderResult = await _restClient.V5Api.Trading.PlaceOrderAsync(
                        Category.Linear,
                        symbol,
                        orderSide,
                        NewOrderType.Market,
                        position.Quantity,
                        null
                    );

                    if (orderResult.Success && orderResult.Data != null)
                    {
                        result.OrderId = orderResult.Data.OrderId;

                        // Get order details to retrieve fill price
                        await Task.Delay(500); // Small delay for order to settle
                        var orderDetails = await _restClient.V5Api.Trading.GetOrdersAsync(
                            Category.Linear,
                            symbol: symbol,
                            orderId: orderResult.Data.OrderId
                        );

                        if (orderDetails.Success && orderDetails.Data?.List?.Any() == true)
                        {
                            var order = orderDetails.Data.List.First();
                            var filledQty = order.QuantityFilled ?? position.Quantity;
                            var avgPrice = order.AveragePrice ?? position.MarkPrice ?? 0;

                            totalFilledQty += filledQty;
                            weightedPrice += avgPrice * filledQty;

                            // Always calculate exit fee explicitly (API returns deprecated/unreliable values)
                            var calculatedFee = filledQty * avgPrice * 0.00055m;
                            var apiFee = order.ExecutedFee ?? 0;

                            // Use calculated fee (more reliable than deprecated API field)
                            var orderFee = calculatedFee;
                            totalFee += orderFee;

                            _logger.LogInformation(
                                "Bybit ClosePosition: Filled {Qty} @ {Price}, CalculatedFee: {CalcFee}, ApiFee: {ApiFee}, UsingFee: {Fee}",
                                filledQty, avgPrice, calculatedFee, apiFee, orderFee);
                        }
                    }
                }

                if (totalFilledQty > 0)
                {
                    result.Success = true;
                    result.ExitPrice = weightedPrice / totalFilledQty;
                    result.FilledQuantity = totalFilledQty;
                    result.TradingFee = totalFee;
                }
                else
                {
                    result.Success = true; // Position might have already been closed
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error closing position on Bybit");
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    public async Task<List<PositionDto>> GetOpenPositionsAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var positions = new List<PositionDto>();

        try
        {
           
            // Bybit V5 API - use settleAsset parameter (4th param) to get all USDT-settled positions
            // Parameters: (category, symbol, baseAsset, settleAsset, limit, cursor, ct)
            var result = await _restClient.V5Api.Trading.GetPositionsAsync(Category.Linear, settleAsset: "USDT");
            
            if (result.Success && result.Data != null)
            {
                
                // Get current mark prices for manual P&L calculation fallback
                var markPrices = new Dictionary<string, decimal>();
                try
                {
                    var tickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);
                    if (tickers.Success && tickers.Data != null)
                    {
                        markPrices = tickers.Data.List.ToDictionary(t => t.Symbol, t => t.MarkPrice > 0 ? t.MarkPrice : t.LastPrice);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to fetch mark prices for P&L calculation");
                }

                positions = result.Data.List
                    .Where(p => p.Quantity > 0)
                    .Select(p =>
                    {
                        decimal unrealizedPnl = p.UnrealizedPnl ?? 0;

                        // FALLBACK: If API returns 0 or null for unrealized P&L, calculate manually
                        // This is common in demo accounts where P&L might not be calculated server-side
                        if (unrealizedPnl == 0 && p.AveragePrice.HasValue && p.AveragePrice.Value > 0)
                        {
                            decimal currentPrice = markPrices.GetValueOrDefault(p.Symbol, 0);
                            if (currentPrice > 0)
                            {
                                // Calculate P&L: (Current Price - Entry Price) * Quantity * Direction
                                // For Long: positive when price goes up
                                // For Short: positive when price goes down
                                decimal priceDiff = currentPrice - p.AveragePrice.Value;
                                if (p.Side == Bybit.Net.Enums.PositionSide.Sell) // Short position
                                {
                                    priceDiff = -priceDiff; // Invert for shorts
                                }
                                unrealizedPnl = priceDiff * p.Quantity;

                                _logger.LogInformation("📊 Calculated unrealized P&L for {Symbol}: Entry={Entry}, Mark={Mark}, Qty={Qty}, Side={Side} → P&L={PnL}",
                                    p.Symbol, p.AveragePrice.Value, currentPrice, p.Quantity, p.Side, unrealizedPnl);
                            }
                        }

                        return new PositionDto
                        {
                            Exchange = ExchangeName,
                            Symbol = p.Symbol,
                            Side = p.Side == Bybit.Net.Enums.PositionSide.Buy ? PositionSide.Long : PositionSide.Short,
                            Status = PositionStatus.Open,
                            EntryPrice = p.AveragePrice ?? 0,
                            Quantity = p.Quantity,
                            Leverage = p.Leverage ?? 1,
                            UnrealizedPnL = unrealizedPnl,
                            FundingEarnedUsd = 0,
                TradingFeesUsd = 0,
                PricePnLUsd = 0,
                RealizedPnLUsd = 0,
                RealizedPnLPct = 0, // Realized P&L not available in position list API
                            OpenedAt = p.CreateTime ?? DateTime.UtcNow
                        };
                    })
                    .ToList();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching positions from Bybit");
        }

        return positions;
    }

    // Spot trading methods for cash-and-carry arbitrage
    public async Task<(string orderId, decimal filledQuantity)> PlaceSpotBuyOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            // Extract base asset to check balance before and after
            var baseAsset = symbol.Replace("USDT", "").Replace("BUSD", "").Replace("USDC", "");

            // Get balance before order
            var balanceBefore = await GetSpotBalanceAsync(baseAsset);

            _logger.LogInformation("Placing spot BUY order on Bybit: {Symbol}, Quantity: {Quantity} (base asset), Balance before: {BalanceBefore}",
                symbol, quantity, balanceBefore);

            var order = await _restClient.V5Api.Trading.PlaceOrderAsync(
                Category.Spot,
                symbol,
                BybitOrderSide.Buy,
                NewOrderType.Market,
                quantity,
                marketUnit: MarketUnit.BaseAsset
            );

            if (order.Success && order.Data != null)
            {
                // Wait a moment for the order to settle
                await Task.Delay(500);

                // Get balance after order to determine filled quantity
                var balanceAfter = await GetSpotBalanceAsync(baseAsset);
                var filledQuantity = balanceAfter - balanceBefore;

                _logger.LogInformation("Spot BUY order placed on Bybit: {OrderId}, Requested: {Requested}, Filled: {Filled}",
                    order.Data.OrderId, quantity, filledQuantity);

                return (order.Data.OrderId, filledQuantity);
            }

            _logger.LogError("Failed to place spot BUY order on Bybit: {Error}", order.Error);
            throw new Exception($"Failed to place spot buy order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot BUY order on Bybit");
            throw;
        }
    }

    public async Task<string> PlaceSpotSellOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            _logger.LogInformation("Placing spot SELL order on Bybit: {Symbol}, Quantity: {Quantity}",
                symbol, quantity);

            var order = await _restClient.V5Api.Trading.PlaceOrderAsync(
                Category.Spot,
                symbol,
                BybitOrderSide.Sell,
                NewOrderType.Market,
                quantity
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Spot SELL order placed on Bybit: {OrderId}", order.Data.OrderId);
                return order.Data.OrderId;
            }

            _logger.LogError("Failed to place spot SELL order on Bybit: {Error}", order.Error);
            throw new Exception($"Failed to place spot sell order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot SELL order on Bybit");
            throw;
        }
    }

    public async Task<decimal> GetSpotBalanceAsync(string asset)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            var balance = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

            if (balance.Success && balance.Data != null && balance.Data.List.Any())
            {
                var assetBalance = balance.Data.List.First().Assets.FirstOrDefault(a => a.Asset == asset);
                if (assetBalance != null)
                {
                    // For Bybit Unified account, use WalletBalance (total including locked)
                    // AvailableToWithdraw is often null for Unified accounts
                    var assetBalanceAmount = assetBalance.WalletBalance ?? assetBalance.AvailableToWithdraw ?? 0;
                    return assetBalanceAmount;
                }
            }

            _logger.LogWarning("No balance found for asset {Asset} on Bybit", asset);
            return 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot balance for {Asset} from Bybit", asset);
            throw;
        }
    }

    public async Task<Dictionary<string, decimal>> GetSpotBalancesAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var balances = new Dictionary<string, decimal>();

        try
        {
            // Bybit Unified account includes both spot and derivatives
            var balance = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

            if (balance.Success && balance.Data != null && balance.Data.List.Any())
            {
                var assets = balance.Data.List.First().Assets;
                foreach (var asset in assets.Where(a => a.WalletBalance.HasValue && a.WalletBalance.Value > 0))
                {
                    balances[asset.Asset] = asset.WalletBalance.Value;
                }

            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching balances from Bybit");
        }

        return balances;
    }

    public async Task<FeeInfoDto> GetTradingFeesAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            // Bybit API: GET /v5/account/fee-rate with category=linear
            // Note: This returns fee rates for a specific symbol, so we'll use a common one like BTCUSDT
            var feeRateResult = await _restClient.V5Api.Account.GetFeeRateAsync(Category.Linear, symbol: "BTCUSDT");

            if (feeRateResult.Success && feeRateResult.Data != null && feeRateResult.Data.List.Any())
            {
                var feeData = feeRateResult.Data.List.First();

                // Bybit returns fee rates as decimals (e.g., 0.0006 for 0.06%)
                return new FeeInfoDto
                {
                    Exchange = ExchangeName,
                    MakerFeeRate = feeData.MakerFeeRate,
                    TakerFeeRate = feeData.TakerFeeRate,
                    FeeTier = feeData.Symbol, // Use symbol as tier indicator
                    CollectedAt = DateTime.UtcNow
                };
            }

            // Fallback to default fees if API call fails
            _logger.LogWarning("Failed to get fee info from Bybit, using defaults");
            return new FeeInfoDto
            {
                Exchange = ExchangeName,
                MakerFeeRate = 0.0002m, // 0.02% default maker
                TakerFeeRate = 0.0006m, // 0.06% default taker (Bybit default)
                FeeTier = "Unknown",
                CollectedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching trading fees from Bybit");

            // Return default fees on error
            return new FeeInfoDto
            {
                Exchange = ExchangeName,
                MakerFeeRate = 0.0002m, // 0.02% default maker
                TakerFeeRate = 0.0006m, // 0.06% default taker
                FeeTier = "Error",
                CollectedAt = DateTime.UtcNow
            };
        }
    }

    public async Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate)
    {
        if (_socketClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        // Subscribe to ticker updates which include funding rates
        await Task.CompletedTask;
    }

    // Get instrument specifications for quantity validation
    public async Task<InstrumentInfo?> GetInstrumentInfoAsync(string symbol, bool isSpot = false)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            if (isSpot)
            {
                // Get spot instrument info
                var spotInfo = await _restClient.V5Api.ExchangeData.GetSpotSymbolsAsync(symbol: symbol);

                if (spotInfo.Success && spotInfo.Data?.List != null && spotInfo.Data.List.Any())
                {
                    var instrument = spotInfo.Data.List.First();
                    var lotSizeFilter = instrument.LotSizeFilter;

                    // For Bybit spot, use BasePrecision from API to determine the step size
                    // BasePrecision is a string like "0.000001" which means 6 decimal places
                    decimal qtyStep = 0.00001m; // Default to 5 decimals
                    int decimals = 5; // Default

                    if (lotSizeFilter != null && lotSizeFilter.BasePrecision > 0)
                    {
                        try
                        {
                            // Parse BasePrecision value (e.g., 0.000001m) to string, then get decimal count
                            var precisionString = lotSizeFilter.BasePrecision.ToString("0.########");
                            decimals = ConvertPrecisionStringToDecimals(precisionString);
                            qtyStep = lotSizeFilter.BasePrecision;

                            _logger.LogInformation(
                                "Spot instrument info for {Symbol}: BasePrecision={BasePrecision} ({Decimals} decimals), QtyStep={Step}",
                                symbol, qtyStep, decimals, qtyStep);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Failed to parse BasePrecision for {Symbol}, using default", symbol);
                        }
                    }
                    else
                    {
                        _logger.LogWarning("No BasePrecision found for {Symbol}, using default {Decimals} decimals",
                            symbol, decimals);
                    }

                    return new InstrumentInfo
                    {
                        Symbol = symbol,
                        MinOrderQty = lotSizeFilter?.MinOrderQuantity ?? 0,
                        MaxOrderQty = lotSizeFilter?.MaxOrderQuantity ?? decimal.MaxValue,
                        QtyStep = qtyStep,
                        IsSpot = true
                    };
                }
            }
            else
            {
                // Get linear perpetual instrument info
                var perpInfo = await _restClient.V5Api.ExchangeData.GetLinearInverseSymbolsAsync(Category.Linear, symbol: symbol);

                if (perpInfo.Success && perpInfo.Data?.List != null && perpInfo.Data.List.Any())
                {
                    var instrument = perpInfo.Data.List.First();
                    var lotSizeFilter = instrument.LotSizeFilter;

                    var qtyStep = lotSizeFilter?.QuantityStep ?? 0.001m;

                    _logger.LogInformation(
                        "Perpetual instrument info for {Symbol}: MinQty={Min}, MaxQty={Max}, QtyStep={Step}",
                        symbol, lotSizeFilter?.MinOrderQuantity, lotSizeFilter?.MaxOrderQuantity, qtyStep);

                    return new InstrumentInfo
                    {
                        Symbol = symbol,
                        MinOrderQty = lotSizeFilter?.MinOrderQuantity ?? 0,
                        MaxOrderQty = lotSizeFilter?.MaxOrderQuantity ?? decimal.MaxValue,
                        QtyStep = qtyStep,
                        IsSpot = false
                    };
                }
            }

            _logger.LogWarning("No instrument info found for {Symbol} ({Type})", symbol, isSpot ? "Spot" : "Perpetual");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching instrument info for {Symbol} from Bybit", symbol);
            return null;
        }
    }

    // Validate and adjust quantity to meet exchange requirements
    public decimal ValidateAndAdjustQuantity(decimal quantity, InstrumentInfo instrumentInfo)
    {
        if (instrumentInfo == null)
        {
            _logger.LogWarning("No instrument info provided for quantity validation, using original quantity");
            return quantity;
        }

        // Check if quantity is within limits
        if (quantity < instrumentInfo.MinOrderQty)
        {
            _logger.LogWarning(
                "Quantity {Quantity} is below minimum {MinQty} for {Symbol}, adjusting to minimum",
                quantity, instrumentInfo.MinOrderQty, instrumentInfo.Symbol);
            quantity = instrumentInfo.MinOrderQty;
        }

        if (quantity > instrumentInfo.MaxOrderQty)
        {
            _logger.LogWarning(
                "Quantity {Quantity} exceeds maximum {MaxQty} for {Symbol}, adjusting to maximum",
                quantity, instrumentInfo.MaxOrderQty, instrumentInfo.Symbol);
            quantity = instrumentInfo.MaxOrderQty;
        }

        // Round to the correct step size
        var steps = Math.Floor(quantity / instrumentInfo.QtyStep);
        var adjustedQuantity = steps * instrumentInfo.QtyStep;

        if (adjustedQuantity != quantity)
        {
            _logger.LogInformation(
                "Adjusted quantity from {OriginalQty} to {AdjustedQty} (step: {QtyStep}) for {Symbol}",
                quantity, adjustedQuantity, instrumentInfo.QtyStep, instrumentInfo.Symbol);
        }

        return adjustedQuantity;
    }

    public async Task<List<FundingRateDto>> GetFundingRateHistoryAsync(string symbol, DateTime startTime, DateTime endTime)
    {
        if (_restClient == null)
        {
            _logger.LogWarning("Cannot get funding rate history for {Exchange} - not connected", ExchangeName);
            return new List<FundingRateDto>();
        }

        try
        {
            var results = new List<FundingRateDto>();

            // Bybit uses variable funding intervals: 1h, 4h, or 8h
            // Calculate required limit based on time range and smallest interval (1h)
            // Add buffer to account for varying intervals
            var timeRangeHours = (endTime - startTime).TotalHours;
            var maxRecordsNeeded = (int)Math.Ceiling(timeRangeHours) + 10; // +10 buffer
            var limit = Math.Min(maxRecordsNeeded, 200); // Bybit max is 200

            _logger.LogDebug("Fetching funding rates for {Symbol} from {Start} to {End} (limit: {Limit})",
                symbol, startTime, endTime, limit);

            // Use startTime and endTime to get data for the specific date range
            // Note: Bybit expects millisecond timestamps
            var historicalRates = await _restClient.V5Api.ExchangeData.GetFundingRateHistoryAsync(
                Category.Linear,
                symbol,
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (!historicalRates.Success || historicalRates.Data?.List == null || !historicalRates.Data.List.Any())
            {
                if (!historicalRates.Success)
                {
                    _logger.LogWarning("Failed to fetch funding rate history for {Symbol} on {Exchange}: {Error}",
                        symbol, ExchangeName, historicalRates.Error?.Message ?? "Unknown error");
                }
                else if (historicalRates.Data?.List == null)
                {
                    _logger.LogWarning("Funding rate history returned null data for {Symbol} on {Exchange}", symbol, ExchangeName);
                }
                else
                {
                    _logger.LogWarning("Funding rate history returned empty list for {Symbol} on {Exchange}", symbol, ExchangeName);
                }

                return results;
            }

            // Filter to only include rates within the requested time range
            // Keep rates where timestamp is within [startTime, endTime]
            var filteredRates = historicalRates.Data.List
                .Where(r => r.Timestamp >= startTime && r.Timestamp <= endTime)
                .OrderBy(r => r.Timestamp)
                .ToList();

            _logger.LogDebug("Received {Total} rates, {Filtered} within range [{Start}, {End}] for {Symbol}",
                historicalRates.Data.List.Count(), filteredRates.Count, startTime, endTime, symbol);

            if (!filteredRates.Any())
            {
                _logger.LogWarning("No funding rates found within requested range [{Start}, {End}] for {Symbol} on {Exchange}",
                    startTime, endTime, symbol, ExchangeName);
                return results;
            }

            // Detect interval for EACH rate individually to handle interval changes mid-period
            for (int i = 0; i < filteredRates.Count; i++)
            {
                var rate = filteredRates[i];
                int fundingIntervalHours = 8; // Default to 8h

                // Detect interval by comparing with next rate (if available)
                if (i < filteredRates.Count - 1)
                {
                    var timeDiff = (filteredRates[i + 1].Timestamp - rate.Timestamp).TotalHours;

                    // Round to nearest hour and determine interval (1h, 4h, or 8h)
                    if (Math.Abs(timeDiff - 1) < 0.1) fundingIntervalHours = 1;
                    else if (Math.Abs(timeDiff - 4) < 0.1) fundingIntervalHours = 4;
                    else if (Math.Abs(timeDiff - 8) < 0.1) fundingIntervalHours = 8;
                }
                else if (i > 0)
                {
                    // For the last rate, use the interval from the previous rate
                    var timeDiff = (rate.Timestamp - filteredRates[i - 1].Timestamp).TotalHours;

                    if (Math.Abs(timeDiff - 1) < 0.1) fundingIntervalHours = 1;
                    else if (Math.Abs(timeDiff - 4) < 0.1) fundingIntervalHours = 4;
                    else if (Math.Abs(timeDiff - 8) < 0.1) fundingIntervalHours = 8;
                }

                // Calculate annualized rate
                var periodsPerYear = (365.0m * 24.0m) / fundingIntervalHours;
                var annualizedRate = rate.FundingRate * periodsPerYear;

                results.Add(new FundingRateDto
                {
                    Exchange = ExchangeName,
                    Symbol = rate.Symbol,
                    Rate = rate.FundingRate,
                    AnnualizedRate = annualizedRate,
                    FundingIntervalHours = fundingIntervalHours,
                    FundingTime = rate.Timestamp,
                    NextFundingTime = rate.Timestamp.AddHours(fundingIntervalHours),
                    RecordedAt = DateTime.UtcNow
                });
            }

            _logger.LogDebug("Fetched {Count} funding rates for {Symbol} on {Exchange} (intervals detected per-rate)",
                results.Count, symbol, ExchangeName);
            

            return results.OrderBy(r => r.FundingTime).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rate history for {Symbol} on {Exchange}", symbol, ExchangeName);
            return new List<FundingRateDto>();
        }
    }

    public async Task<List<KlineDto>> GetKlinesAsync(string symbol, DateTime startTime, DateTime endTime, Models.KlineInterval interval)
    {
        if (_restClient == null)
        {
            _logger.LogWarning("Cannot get klines for {Exchange} - not connected", ExchangeName);
            return new List<KlineDto>();
        }

        try
        {
            // Convert our KlineInterval enum to Bybit.Net enum
            var bybitInterval = interval switch
            {
                Models.KlineInterval.OneMinute => Bybit.Net.Enums.KlineInterval.OneMinute,
                Models.KlineInterval.FiveMinutes => Bybit.Net.Enums.KlineInterval.FiveMinutes,
                Models.KlineInterval.FifteenMinutes => Bybit.Net.Enums.KlineInterval.FifteenMinutes,
                Models.KlineInterval.ThirtyMinutes => Bybit.Net.Enums.KlineInterval.ThirtyMinutes,
                Models.KlineInterval.OneHour => Bybit.Net.Enums.KlineInterval.OneHour,
                Models.KlineInterval.FourHours => Bybit.Net.Enums.KlineInterval.FourHours,
                Models.KlineInterval.OneDay => Bybit.Net.Enums.KlineInterval.OneDay,
                _ => Bybit.Net.Enums.KlineInterval.OneHour
            };

            // Bybit API limit is 1000 records per request
            var klines = await _restClient.V5Api.ExchangeData.GetKlinesAsync(
                Category.Linear,
                symbol,
                bybitInterval,
                startTime,
                endTime,
                limit: 1000);

            if (!klines.Success || klines.Data?.List == null || !klines.Data.List.Any())
            {
                if (!klines.Success)
                {
                    _logger.LogWarning("Failed to fetch klines for {Symbol} on {Exchange}: {Error}",
                        symbol, ExchangeName, klines.Error?.Message ?? "Unknown error");
                }
                return new List<KlineDto>();
            }

            // Bybit returns klines in reverse chronological order (newest first), so reverse it
            var results = klines.Data.List.Reverse().Select(k => new KlineDto
            {
                Exchange = ExchangeName,
                Symbol = symbol,
                OpenTime = k.StartTime,
                CloseTime = k.StartTime.AddMinutes(GetIntervalMinutes(bybitInterval)), // Bybit doesn't provide CloseTime, calculate it
                Open = k.OpenPrice,
                High = k.HighPrice,
                Low = k.LowPrice,
                Close = k.ClosePrice,
                Volume = k.Volume
            }).ToList();

            _logger.LogDebug("Fetched {Count} klines for {Symbol} on {Exchange}", results.Count, symbol, ExchangeName);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching klines for {Symbol} on {Exchange}", symbol, ExchangeName);
            return new List<KlineDto>();
        }
    }

    private int GetIntervalMinutes(Bybit.Net.Enums.KlineInterval interval)
    {
        return interval switch
        {
            Bybit.Net.Enums.KlineInterval.OneMinute => 1,
            Bybit.Net.Enums.KlineInterval.FiveMinutes => 5,
            Bybit.Net.Enums.KlineInterval.FifteenMinutes => 15,
            Bybit.Net.Enums.KlineInterval.ThirtyMinutes => 30,
            Bybit.Net.Enums.KlineInterval.OneHour => 60,
            Bybit.Net.Enums.KlineInterval.FourHours => 240,
            Bybit.Net.Enums.KlineInterval.OneDay => 1440,
            _ => 60
        };
    }

    /// <summary>
    /// Convert Bybit precision string to decimal count
    /// Examples: "0.000001" → 6, "0.01" → 2, "0.1" → 1, "1" → 0
    /// </summary>
    private int ConvertPrecisionStringToDecimals(string precisionString)
    {
        if (string.IsNullOrEmpty(precisionString))
            return 0;

        // Parse the precision value
        if (!decimal.TryParse(precisionString, out decimal precisionValue))
            return 0;

        // If precision is >= 1, no decimals
        if (precisionValue >= 1m)
            return 0;

        // Count decimal places by converting to string and finding position of first non-zero digit after decimal
        var parts = precisionString.Split('.');
        if (parts.Length != 2)
            return 0;

        // Count leading zeros + 1 for the '1'
        // "0.000001" → "000001" → 6 decimals
        var decimalPart = parts[1];
        int decimals = decimalPart.Length;

        return decimals;
    }

    // Trading data methods
    public async Task<List<OrderDto>> GetOpenOrdersAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var orders = new List<OrderDto>();

        try
        {
            var result = await _restClient.V5Api.Trading.GetOrdersAsync(Category.Linear);

            if (result.Success && result.Data?.List != null)
            {
                orders = result.Data.List
                    .Where(o => o.Status == Bybit.Net.Enums.OrderStatus.New ||
                               o.Status == Bybit.Net.Enums.OrderStatus.PartiallyFilled)
                    .Select(o => new OrderDto
                    {
                        Exchange = ExchangeName,
                        OrderId = o.OrderId,
                        ClientOrderId = o.ClientOrderId,
                        Symbol = o.Symbol,
                        Side = o.Side == Bybit.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                        Type = MapBybitOrderType(o.OrderType),
                        Status = MapBybitOrderStatus(o.Status),
                        Price = o.Price ?? 0,
                        AveragePrice = o.AveragePrice ?? 0,
                        Quantity = o.Quantity,
                        FilledQuantity = o.QuantityFilled ?? 0,
                        Fee = 0, // Not available in open orders
                        FeeAsset = null,
                        ReduceOnly = o.ReduceOnly?.ToString(),
                        TimeInForce = o.TimeInForce.ToString(),
                        CreatedAt = o.CreateTime,
                        UpdatedAt = o.UpdateTime
                    })
                    .ToList();

     
            }
            else
            {
                _logger.LogWarning("Failed to fetch open orders from Bybit: {Error}", result.Error?.Message);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching open orders from Bybit");
        }

        return orders;
    }

    public async Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var orders = new List<OrderDto>();

        try
        {
            var result = await _restClient.V5Api.Trading.GetOrderHistoryAsync(
                Category.Linear,
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (result.Success && result.Data?.List != null)
            {
                orders = result.Data.List.Select(o => new OrderDto
                {
                    Exchange = ExchangeName,
                    OrderId = o.OrderId,
                    ClientOrderId = o.ClientOrderId,
                    Symbol = o.Symbol,
                    Side = o.Side == Bybit.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                    Type = MapBybitOrderType(o.OrderType),
                    Status = MapBybitOrderStatus(o.Status),
                    Price = o.Price ?? 0,
                    AveragePrice = o.AveragePrice ?? 0,
                    Quantity = o.Quantity,
                    FilledQuantity = o.QuantityFilled ?? 0,
                    Fee = 0, // Fee details not in order history, see trades
                    FeeAsset = null,
                    // PositionSide and RejectReason not in OrderDto
                    ReduceOnly = o.ReduceOnly?.ToString(),
                    TimeInForce = o.TimeInForce.ToString(),
                    CreatedAt = o.CreateTime,
                    UpdatedAt = o.UpdateTime
                }).ToList();

    
            }
            else
            {
                _logger.LogWarning("Failed to fetch order history from Bybit: {Error}", result.Error?.Message);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching order history from Bybit");
        }

        return orders;
    }

    public async Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var trades = new List<TradeDto>();

        try
        {
            var result = await _restClient.V5Api.Trading.GetUserTradesAsync(
                Category.Linear,
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (result.Success && result.Data?.List != null)
            {
                trades = result.Data.List.Select(t => new TradeDto
                {
                    Exchange = ExchangeName,
                    TradeId = t.TradeId,
                    OrderId = t.OrderId,
                    Symbol = t.Symbol,
                    Side = t.Side == Bybit.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                    Price = t.Price,
                    Quantity = t.Quantity,
                    QuoteQuantity = t.Price * t.Quantity,
                    Fee = t.Fee ?? 0,
                    FeeAsset = t.FeeAsset,
                    IsMaker = t.IsMaker,
                    IsBuyer = t.Side == Bybit.Net.Enums.OrderSide.Buy,
                    ExecutedAt = t.Timestamp,
                    OrderType = t.OrderType?.ToString(),
                    PositionSide = null // BybitUserTrade does not have PositionSide property
                }).ToList();

                _logger.LogInformation("Fetched {Count} user trades from Bybit", trades.Count);
            }
            else
            {
                _logger.LogWarning("Failed to fetch user trades from Bybit: {Error}", result.Error?.Message);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching user trades from Bybit");
        }

        return trades;
    }

    public async Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var transactions = new List<TransactionDto>();

        try
        {
            _logger.LogDebug("Fetching Bybit transaction history from {StartTime} to {EndTime}, limit {Limit}",
                startTime, endTime, limit);

            var result = await _restClient.V5Api.Account.GetTransactionHistoryAsync(
                AccountType.Unified,
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (result.Success && result.Data?.List != null)
            {
                transactions = result.Data.List.Select(t =>
                {
                    // Funding fee amount (negative = paid, positive = received)
                    var funding = t.Funding ?? 0m;
                    // Use cashflow for the amount when available (more accurate for PnL)
                    // IMPORTANT: Use funding if all cashflow fields are 0 (not just null)
                    var cashflowAmount = t.Cashflow ?? t.Change ?? t.Quantity ?? t.Size ?? 0;
                    var amount = cashflowAmount != 0 ? cashflowAmount : funding;
                    // Fee is negative in the API, so take absolute value
                    var fee = t.Fee.HasValue ? Math.Abs(t.Fee.Value) : 0m;

                    var transactionType = MapBybitTransactionType(t.Type, t.SubType, funding, fee, t.TradeId);

                    // Calculate signed fee: negative for costs, positive for income
                    decimal? signedFee = transactionType switch
                    {
                        Models.TransactionType.Trade => fee > 0 ? -fee : 0m, // Trading fee (commission) - always negative
                        Models.TransactionType.Commission => fee > 0 ? -fee : 0m, // Commission - always negative
                        Models.TransactionType.FundingFee => funding, // Funding fee - can be +/- (income/cost)
                        Models.TransactionType.Rebate => fee > 0 ? fee : 0m, // Rebate - always positive
                        Models.TransactionType.RealizedPnL => 0m, // P&L is in amount, not fee
                        Models.TransactionType.Settlement => 0m, // Settlement is in amount
                        Models.TransactionType.Liquidation => fee > 0 ? -fee : 0m, // Liquidation fee
                        Models.TransactionType.Deposit => null, // No fee concept for deposits
                        Models.TransactionType.Withdrawal => fee > 0 ? -fee : 0m, // Withdrawal fee
                        Models.TransactionType.Transfer => null, // No fee for transfers
                        _ => null // For other types where fee doesn't apply
                    };

                    return new TransactionDto
                    {
                        Exchange = ExchangeName,
                        TransactionId = t.Id ?? string.Empty,
                        Type = transactionType,
                        Asset = t.Asset ?? string.Empty,
                        Amount = amount,
                        Status = TransactionStatus.Confirmed, // Bybit transaction history only shows confirmed
                        Symbol = t.Symbol,
                        TradeId = t.TradeId,
                        OrderId = t.OrderId,
                        ClientOrderId = t.ClientOrderId,
                        Side = t.Side.ToString(),
                        TradePrice = t.TradePrice,
                        Fee = fee,
                        FeeAsset = fee > 0 ? t.Asset : null,
                        SignedFee = signedFee,
                        SubType = t.SubType,
                        Info = $"{t.Type} - {t.SubType}" + (funding != 0 ? $" (Funding: {funding})" : ""),
                        CreatedAt = t.TransactionTime,
                        ConfirmedAt = t.TransactionTime
                    };
                })
                .Where(tx => tx.Amount != 0 || tx.Fee > 0) // Filter out empty transactions
                .ToList();

                // Log detailed breakdown
                var tradeCount = transactions.Count(t => t.Type == Models.TransactionType.Trade);
                var pnlCount = transactions.Count(t => t.Type == Models.TransactionType.RealizedPnL);
                var fundingCount = transactions.Count(t => t.Type == Models.TransactionType.FundingFee);
                var commissionCount = transactions.Count(t => t.Type == Models.TransactionType.Commission);
                var otherCount = transactions.Count(t => t.Type == Models.TransactionType.Other);

                _logger.LogInformation("Fetched {Count} transactions from Bybit (trades: {TradeCount}, realized PnL: {PnLCount}, funding fees: {FundingCount}, commissions: {CommissionCount}, other: {OtherCount})",
                    transactions.Count,
                    tradeCount,
                    pnlCount,
                    fundingCount,
                    commissionCount,
                    otherCount);

                // Debug logging: log a few sample transactions to verify field-based detection
                if (transactions.Any())
                {
                    _logger.LogDebug("Sample Bybit transactions for verification:");
                    foreach (var sample in transactions.Take(3))
                    {
                        _logger.LogDebug("  [{Type}] Symbol: {Symbol}, Amount: {Amount}, Fee: {Fee}, Info: {Info}",
                            sample.Type, sample.Symbol, sample.Amount, sample.Fee, sample.Info);
                    }
                }
            }
            else
            {
                _logger.LogWarning("Failed to fetch transactions from Bybit: {Error}", result.Error?.Message ?? "Unknown error");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching transactions from Bybit");
        }

        return transactions;
    }

    // Helper methods to map Bybit enums to application enums
    private Models.OrderType MapBybitOrderType(Bybit.Net.Enums.OrderType bybitType)
    {
        return bybitType switch
        {
            Bybit.Net.Enums.OrderType.Market => Models.OrderType.Market,
            Bybit.Net.Enums.OrderType.Limit => Models.OrderType.Limit,
            _ => Models.OrderType.Market
        };
    }

    private Models.OrderStatus MapBybitOrderStatus(Bybit.Net.Enums.OrderStatus bybitStatus)
    {
        return bybitStatus switch
        {
            Bybit.Net.Enums.OrderStatus.New => Models.OrderStatus.New,
            Bybit.Net.Enums.OrderStatus.PartiallyFilled => Models.OrderStatus.PartiallyFilled,
            Bybit.Net.Enums.OrderStatus.Filled => Models.OrderStatus.Filled,
            Bybit.Net.Enums.OrderStatus.Cancelled => Models.OrderStatus.Canceled,
            Bybit.Net.Enums.OrderStatus.Rejected => Models.OrderStatus.Rejected,
            _ => Models.OrderStatus.New
        };
    }

    private Models.TransactionType MapBybitTransactionType(
        Bybit.Net.Enums.TransactionLogType bybitType,
        string? subType = null,
        decimal funding = 0,
        decimal fee = 0,
        string? tradeId = null)
    {
        // CRITICAL: Bybit returns Type=TRADE or Type=SETTLEMENT for most transactions
        // We must use field-based detection instead of relying on Type/SubType alone

        // Priority 1: Check Funding field - if it has a value, it's a funding fee
        if (funding != 0)
        {
            return Models.TransactionType.FundingFee;
        }

        // Priority 2: For Type=SETTLEMENT without funding, it's realized PnL
        if (bybitType == Bybit.Net.Enums.TransactionLogType.Settlement)
        {
            // Settlement transactions are for realized PnL from closed positions
            return Models.TransactionType.RealizedPnL;
        }

        // Priority 3: For Type=TRADE, differentiate between actual trades and commissions
        if (bybitType == Bybit.Net.Enums.TransactionLogType.Trade)
        {
            // If it has a TradeId, it's an actual trade (buy/sell execution)
            if (!string.IsNullOrEmpty(tradeId))
            {
                return Models.TransactionType.Trade;
            }

            // If it has a fee but no TradeId, it's a commission-only entry
            if (fee > 0)
            {
                return Models.TransactionType.Commission;
            }

            // Default to Trade for Type=TRADE
            return Models.TransactionType.Trade;
        }

        // Priority 4: Check SubType if available (though usually empty)
        if (!string.IsNullOrEmpty(subType))
        {
            if (subType.Contains("FUNDING", StringComparison.OrdinalIgnoreCase))
                return Models.TransactionType.FundingFee;
            if (subType.Contains("FEE", StringComparison.OrdinalIgnoreCase) ||
                subType.Contains("COMMISSION", StringComparison.OrdinalIgnoreCase))
                return Models.TransactionType.Commission;
        }

        // Priority 5: Map based on primary Type for all other cases
        return bybitType switch
        {
            // Asset Movement
            Bybit.Net.Enums.TransactionLogType.TransferIn => Models.TransactionType.Transfer,
            Bybit.Net.Enums.TransactionLogType.TransferOut => Models.TransactionType.Transfer,

            // Trading & Settlement (fallback)
            Bybit.Net.Enums.TransactionLogType.Delivery => Models.TransactionType.Delivery,
            Bybit.Net.Enums.TransactionLogType.Liquidation => Models.TransactionType.Liquidation,
            Bybit.Net.Enums.TransactionLogType.Adl => Models.TransactionType.Adl,
            Bybit.Net.Enums.TransactionLogType.FeeRefund => Models.TransactionType.Rebate,

            // Rewards & Bonuses
            Bybit.Net.Enums.TransactionLogType.Airdrop => Models.TransactionType.Airdrop,
            Bybit.Net.Enums.TransactionLogType.Bonus => Models.TransactionType.Bonus,
            Bybit.Net.Enums.TransactionLogType.BonusTransferIn => Models.TransactionType.Bonus,
            Bybit.Net.Enums.TransactionLogType.BonusTransferOut => Models.TransactionType.Bonus,
            Bybit.Net.Enums.TransactionLogType.BonusRecollect => Models.TransactionType.Bonus,

            // Currency Operations
            Bybit.Net.Enums.TransactionLogType.CurrencyBuy => Models.TransactionType.Trade,
            Bybit.Net.Enums.TransactionLogType.CurrencySell => Models.TransactionType.Trade,
            Bybit.Net.Enums.TransactionLogType.Interest => Models.TransactionType.Commission,

            // Investment Products
            Bybit.Net.Enums.TransactionLogType.TokensSubscription => Models.TransactionType.Transfer,
            Bybit.Net.Enums.TransactionLogType.TokensRedemption => Models.TransactionType.Transfer,
            Bybit.Net.Enums.TransactionLogType.FlexibleStakingSubscription => Models.TransactionType.Transfer,
            Bybit.Net.Enums.TransactionLogType.FlexibleStakingRedemption => Models.TransactionType.Transfer,
            Bybit.Net.Enums.TransactionLogType.FixedStakingSubscription => Models.TransactionType.Transfer,

            // Internal Transfers
            Bybit.Net.Enums.TransactionLogType.TransferInInsLoan => Models.TransactionType.InternalTransfer,
            Bybit.Net.Enums.TransactionLogType.TransferOutInsLoan => Models.TransactionType.InternalTransfer,

            // Log unmapped types for monitoring
            _ => LogUnmappedType(bybitType, subType, funding, fee, tradeId)
        };
    }

    private Models.TransactionType LogUnmappedType(
        Bybit.Net.Enums.TransactionLogType bybitType,
        string? subType,
        decimal funding,
        decimal fee,
        string? tradeId)
    {
        _logger.LogWarning("Unmapped Bybit transaction - Type: {Type}, SubType: {SubType}, Funding: {Funding}, Fee: {Fee}, TradeId: {TradeId}",
            bybitType, subType ?? "null", funding, fee, tradeId ?? "null");
        return Models.TransactionType.Other;
    }
}

// Helper class for instrument specifications
public class InstrumentInfo
{
    public string Symbol { get; set; } = string.Empty;
    public decimal MinOrderQty { get; set; }
    public decimal MaxOrderQty { get; set; }
    public decimal QtyStep { get; set; }
    public bool IsSpot { get; set; }
}
