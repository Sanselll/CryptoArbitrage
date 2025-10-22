using Binance.Net.Clients;
using Binance.Net.Objects;
using Binance.Net.Objects.Options;
using CryptoExchange.Net.Authentication;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;
using Binance.Net.Enums;
using CryptoExchange.Net.Objects;
using CryptoExchange.Net.Objects.Options;
using Binance.Net;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using PositionSide = CryptoArbitrage.API.Data.Entities.PositionSide;
using BinanceOrderSide = Binance.Net.Enums.OrderSide;
using ModelOrderSide = CryptoArbitrage.API.Models.OrderSide;

namespace CryptoArbitrage.API.Services.Exchanges;

public class BinanceConnector : IExchangeConnector
{
    private readonly ILogger<BinanceConnector> _logger;
    private readonly IConfiguration _configuration;
    private BinanceRestClient? _restClient;
    private BinanceSocketClient? _socketClient;

    public string ExchangeName => "Binance";

    public BinanceConnector(ILogger<BinanceConnector> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
    }

    public async Task<bool> ConnectAsync(string apiKey, string apiSecret)
    {
        try
        {
            // Read IsLive setting from configuration
            var isLive = _configuration.GetValue<bool>("Environment:IsLive");

            // Check if API credentials are provided
            bool hasCredentials = !string.IsNullOrWhiteSpace(apiKey) && !string.IsNullOrWhiteSpace(apiSecret);

            if (hasCredentials)
            {
                var credentials = new ApiCredentials(apiKey, apiSecret);
                var environment = isLive ? BinanceEnvironment.Live : BinanceEnvironment.Demo;

                _restClient = new BinanceRestClient(options =>
                {
                    options.ApiCredentials = credentials;
                    options.Environment = environment;
                    // Configure request timeout to prevent hanging requests
                    options.RequestTimeout = TimeSpan.FromSeconds(30);
                });

                _socketClient = new BinanceSocketClient(options =>
                {
                    options.ApiCredentials = credentials;
                    options.Environment = environment;
                });

                // Test connection with authenticated endpoint
                var accountInfo = await _restClient.UsdFuturesApi.Account.GetBalancesAsync();

                if (accountInfo.Success)
                {
                    return true;
                }

                _logger.LogError("Failed to connect to Binance: {Error}", accountInfo.Error);
                return false;
            }
            else
            {
                // Create client without credentials for public data only
                var environment = isLive ? BinanceEnvironment.Live : BinanceEnvironment.Demo;

                _restClient = new BinanceRestClient(options =>
                {
                    // No credentials - public data only
                    options.Environment = environment;
                    // Configure request timeout to prevent hanging requests
                    options.RequestTimeout = TimeSpan.FromSeconds(30);
                });

                _socketClient = new BinanceSocketClient(options => { options.Environment = environment; });

                // Test connection with public endpoint
                var exchangeInfo = await _restClient.UsdFuturesApi.ExchangeData.GetExchangeInfoAsync();

                if (exchangeInfo.Success)
                {
                    var mode = isLive ? "Live" : "Demo";
                    _logger.LogInformation(
                        "Successfully connected to Binance {Mode} (public data only - no API credentials)", mode);
                    return true;
                }

                _logger.LogError("Failed to connect to Binance: {Error}", exchangeInfo.Error);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting to Binance");
            return false;
        }
    }

    public async Task DisconnectAsync()
    {
        _restClient?.Dispose();
        await (_socketClient?.UnsubscribeAllAsync() ?? Task.CompletedTask);
        _socketClient?.Dispose();
    }

    public async Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols,
        decimal minHighPriorityFundingRate = 0)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            // Get all perpetual futures symbols
            var exchangeInfo = await _restClient.UsdFuturesApi.ExchangeData.GetExchangeInfoAsync();
            if (!exchangeInfo.Success || exchangeInfo.Data == null)
            {
                _logger.LogError("Failed to get exchange info from Binance");
                return new List<string>();
            }

            // Filter for active USDT perpetuals only
            var tradingSymbols = exchangeInfo.Data.Symbols
                .Where(s => s.Status == Binance.Net.Enums.SymbolStatus.Trading &&
                            s.Name.EndsWith("USDT") &&
                            s.ContractType == Binance.Net.Enums.ContractType.Perpetual)
                .Select(s => s.Name)
                .ToList();

            // Get 24h tickers for volume data
            var tickers = await _restClient.UsdFuturesApi.ExchangeData.GetTickersAsync();
            if (!tickers.Success || tickers.Data == null)
            {
                _logger.LogWarning("Failed to get ticker data, using all symbols");
                return tradingSymbols.Take(maxSymbols).ToList();
            }

            // Calculate 24h volume in USD and filter
            var symbolsWithVolume = tickers.Data
                .Where(t => tradingSymbols.Contains(t.Symbol))
                .Select(t => new
                {
                    Symbol = t.Symbol,
                    VolumeUsd = t.QuoteVolume // Already in USDT
                })
                .Where(s => s.VolumeUsd >= minDailyVolumeUsd)
                .OrderByDescending(s => s.VolumeUsd)
                .Take(maxSymbols)
                .Select(s => s.Symbol)
                .ToList();

            _logger.LogInformation(
                "Discovered {Count} active symbols (min volume: ${MinVolume:N0}, max symbols: {MaxSymbols})",
                symbolsWithVolume.Count,
                minDailyVolumeUsd,
                maxSymbols
            );

            return symbolsWithVolume;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering active symbols from Binance");
            return new List<string>();
        }
    }

    public async Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var fundingRates = new List<FundingRateDto>();

        try
        {
            // First, fetch funding info for all symbols to get interval hours and caps/floors
            var fundingInfo = await _restClient.UsdFuturesApi.ExchangeData.GetFundingInfoAsync();
            var fundingInfoMap = new Dictionary<string, (int intervalHours, decimal cap, decimal floor)>();

            if (fundingInfo.Success && fundingInfo.Data != null)
            {
                foreach (var info in fundingInfo.Data)
                {
                    fundingInfoMap[info.Symbol] = (
                        info.FundingIntervalHours,
                        info.AdjustedFundingRateCap,
                        info.AdjustedFundingRateFloor
                    );
                }

                _logger.LogInformation("Loaded funding info for {Count} symbols from Binance", fundingInfoMap.Count);
            }
            

            // Use premium index to get current funding rate (shown on Binance website)
            // This is the rate that will be applied at the next funding time
            var premiumIndex = await _restClient.UsdFuturesApi.ExchangeData.GetMarkPricesAsync();

            if (premiumIndex.Success && premiumIndex.Data != null)
            {
                var symbolsToFetch = symbols.Count > 0
                    ? symbols
                    : new List<string> {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"};

                // Filter mark prices to only requested symbols to avoid processing all 621 symbols
                var relevantMarkPrices = premiumIndex.Data.Where(p => symbolsToFetch.Contains(p.Symbol)).ToList();

                _logger.LogInformation("Fetching funding rates for {Count} symbols from Binance",
                    relevantMarkPrices.Count);

                foreach (var markPrice in relevantMarkPrices)
                {
                    var currentRate = markPrice.FundingRate ?? 0;

                    // Get funding interval and caps from API, fallback to defaults
                    int fundingIntervalHours = 8; // Default: 8h (3 times per day)
                    decimal? fundingCap = 0.00375m; // Default: 0.375%
                    decimal? fundingFloor = -0.00375m; // Default: -0.375%

                    if (fundingInfoMap.TryGetValue(markPrice.Symbol, out var info))
                    {
                        fundingIntervalHours = info.intervalHours;
                        fundingCap = info.cap;
                        fundingFloor = info.floor;
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

                    fundingRates.Add(new FundingRateDto
                    {
                        Exchange = ExchangeName,
                        Symbol = markPrice.Symbol,
                        Rate = currentRate,
                        AnnualizedRate = annualizedRate,
                        FundingIntervalHours = fundingIntervalHours,
                        Direction = direction,
                        PreviousRate = previousRate,
                        PreviousAnnualizedRate = previousAnnualizedRate,
                        FundingCap = fundingCap,
                        FundingFloor = fundingFloor,
                        FundingTime = DateTime.UtcNow,
                        NextFundingTime = markPrice.NextFundingTime,
                        RecordedAt = DateTime.UtcNow
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rates from Binance");
        }

        return fundingRates;
    }

    public async Task<Dictionary<string, PriceDto>> GetSpotPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var spotPrices = new Dictionary<string, PriceDto>();

        try
        {
            // Get 24h ticker data (contains both price and volume)
            var tickers = await _restClient.SpotApi.ExchangeData.GetTickersAsync();

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.Where(t => symbols.Contains(t.Symbol)))
                {
                    spotPrices[ticker.Symbol] = new PriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Symbol,
                        Price = ticker.LastPrice,
                        Volume24h = ticker.QuoteVolume, // 24h volume in quote currency
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

         
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot prices from Binance");
        }

        return spotPrices;
    }

    public async Task<Dictionary<string, PriceDto>> GetPerpetualPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var perpPrices = new Dictionary<string, PriceDto>();

        try
        {
            // Get 24h ticker data (contains both price and volume)
            var tickers = await _restClient.UsdFuturesApi.ExchangeData.GetTickersAsync();

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.Where(t => symbols.Contains(t.Symbol)))
                {
                    perpPrices[ticker.Symbol] = new PriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Symbol,
                        Price = ticker.LastPrice,
                        Volume24h = ticker.QuoteVolume, // 24h volume in quote currency
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching perpetual prices from Binance");
        }

        return perpPrices;
    }
    
    public async Task<LiquidityMetricsDto?> GetLiquidityMetricsAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            // Get orderbook depth
            var orderbook = await _restClient.UsdFuturesApi.ExchangeData.GetOrderBookAsync(symbol, limit: 100);

            if (!orderbook.Success || orderbook.Data == null)
            {
                _logger.LogWarning("Failed to fetch orderbook for {Symbol} from Binance", symbol);
                return null;
            }

            var bids = orderbook.Data.Bids.ToList();
            var asks = orderbook.Data.Asks.ToList();

            if (bids.Count == 0 || asks.Count == 0)
            {
                _logger.LogWarning("Empty orderbook for {Symbol} from Binance", symbol);
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
            _logger.LogError(ex, "Error fetching liquidity metrics for {Symbol} from Binance", symbol);
            return null;
        }
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");
        

        decimal futuresTotal = 0;
        decimal futuresAvailable = 0;
        decimal marginUsed = 0;
        decimal unrealizedPnL = 0;
        bool futuresSuccess = false;

        // Try to fetch futures balances
        try
        {
            var futuresBalances = await _restClient.UsdFuturesApi.Account.GetBalancesAsync();
            

            if (futuresBalances.Data != null && futuresBalances.Data.Any())
            {
                var usdtBalance = futuresBalances.Data.FirstOrDefault(b => b.Asset == "USDT");
                if (usdtBalance != null)
                {
                    futuresTotal = usdtBalance.WalletBalance;
                    futuresAvailable = usdtBalance.AvailableBalance;
                    marginUsed = usdtBalance.WalletBalance - usdtBalance.AvailableBalance;
                    unrealizedPnL = usdtBalance.CrossUnrealizedPnl ?? 0;
                    futuresSuccess = true;
                }
                else
                {
                    _logger.LogWarning("Futures API succeeded but no USDT balance found");
                }
            }
            else
            {
                _logger.LogWarning("Futures API succeeded but returned no data");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "✗ Exception fetching futures balances - will continue with spot only");
        }

        decimal spotTotalUsd = 0;
        decimal spotAvailableUsd = 0;
        decimal spotUsdtOnly = 0;
        Dictionary<string, decimal> spotBalances = new Dictionary<string, decimal>();
        bool spotSuccess = false;

        // Try to fetch spot balances
        try
        {
            spotBalances = await GetSpotBalancesAsync();
            

            // Convert spot assets to USD equivalent
            if (spotBalances.Any())
            {
                var prices = await _restClient.SpotApi.ExchangeData.GetPricesAsync();

                if (prices.Success && prices.Data != null)
                {
                    foreach (var asset in spotBalances)
                    {
                        if (asset.Key == "USDT")
                        {
                            // USDT is already in USD
                            spotTotalUsd += asset.Value;
                            spotAvailableUsd = asset.Value; // Only USDT for available
                            spotUsdtOnly = asset.Value;
                        }
                        else
                        {
                            // Find price in USDT for other assets
                            var symbol = $"{asset.Key}USDT";
                            var price = prices.Data.FirstOrDefault(p => p.Symbol == symbol);
                            if (price != null)
                            {
                                var usdValue = asset.Value * price.Price;
                                spotTotalUsd += usdValue; // Add to total only
                            }
                        }
                    }

                    spotSuccess = true;
                }
                else
                {
                    _logger.LogWarning("✗ Price fetch failed");
                    _logger.LogWarning("Error Code: {Code}", prices.Error?.Code);
                    _logger.LogWarning("Error Message: {Message}", prices.Error?.Message);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "✗ Exception fetching spot balances - will continue with futures only");
        }
        

        if (!futuresSuccess && !spotSuccess)
        {
            _logger.LogError("CRITICAL: Both futures and spot balance fetches FAILED - returning DTO with zeros");
        }
        
        // Calculate operational balance
        decimal operationalBalance = spotTotalUsd + futuresTotal;

        var result = new AccountBalanceDto
        {
            Exchange = ExchangeName,
            // Combined totals
            TotalBalance = spotTotalUsd + futuresTotal,
            AvailableBalance = spotAvailableUsd + futuresAvailable,
            OperationalBalanceUsd = operationalBalance,
            // Spot specific
            SpotBalanceUsd = spotTotalUsd,
            SpotAvailableUsd = spotAvailableUsd,
            SpotAssets = spotBalances,
            // Futures specific
            FuturesBalanceUsd = futuresTotal,
            FuturesAvailableUsd = futuresAvailable,
            MarginUsed = marginUsed,
            UnrealizedPnL = unrealizedPnL,
            UpdatedAt = DateTime.UtcNow
        };


        return result;
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync(Dictionary<string, decimal> activeSpotPositions)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            // Fetch futures balances
            var futuresBalances = await _restClient.UsdFuturesApi.Account.GetBalancesAsync();
            decimal futuresTotal = 0;
            decimal futuresAvailable = 0;
            decimal marginUsed = 0;
            decimal unrealizedPnL = 0;

            if (futuresBalances.Success && futuresBalances.Data != null && futuresBalances.Data.Any())
            {
                var usdtBalance = futuresBalances.Data.FirstOrDefault(b => b.Asset == "USDT");
                if (usdtBalance != null)
                {
                    futuresTotal = usdtBalance.WalletBalance;
                    futuresAvailable = usdtBalance.AvailableBalance;
                    marginUsed = usdtBalance.WalletBalance - usdtBalance.AvailableBalance;
                    unrealizedPnL = usdtBalance.CrossUnrealizedPnl ?? 0;
                }
            }

            // Fetch spot balances
            var spotBalances = await GetSpotBalancesAsync();
            decimal spotTotalUsd = 0;
            decimal spotAvailableUsd = 0;
            decimal spotUsdtOnly = 0; // Only USDT

            // Convert spot assets to USD equivalent
            if (spotBalances.Any())
            {
                // Get prices for all spot assets
                var prices = await _restClient.SpotApi.ExchangeData.GetPricesAsync();
                if (prices.Success && prices.Data != null)
                {
                    foreach (var asset in spotBalances)
                    {
                        if (asset.Key == "USDT")
                        {
                            // USDT is already in USD
                            spotTotalUsd += asset.Value;
                            spotAvailableUsd = asset.Value; // Only USDT for available
                            spotUsdtOnly = asset.Value; // Track USDT separately
                        }
                        else
                        {
                            // Find price in USDT
                            var symbol = $"{asset.Key}USDT";
                            var price = prices.Data.FirstOrDefault(p => p.Symbol == symbol);
                            if (price != null)
                            {
                                var usdValue = asset.Value * price.Price;
                                spotTotalUsd += usdValue; // Add to total only
                            }
                        }
                    }
                }
            }

            // Calculate coins in active positions value
            decimal coinsInActivePositionsUsd = 0;
            if (activeSpotPositions.Any())
            {
                var prices = await _restClient.SpotApi.ExchangeData.GetPricesAsync();
                if (prices.Success && prices.Data != null)
                {
                    foreach (var position in activeSpotPositions)
                    {
                        var symbol = $"{position.Key}USDT";
                        var price = prices.Data.FirstOrDefault(p => p.Symbol == symbol);
                        if (price != null)
                        {
                            coinsInActivePositionsUsd += position.Value * price.Price;
                        }
                    }
                }
            }

            // Calculate operational balance: USDT + coins in active positions + futures
            decimal operationalBalance = spotUsdtOnly + coinsInActivePositionsUsd + futuresTotal;

            return new AccountBalanceDto
            {
                Exchange = ExchangeName,
                // Combined totals
                TotalBalance = spotTotalUsd + futuresTotal,
                AvailableBalance = spotAvailableUsd + futuresAvailable,
                OperationalBalanceUsd = operationalBalance,
                // Spot specific
                SpotBalanceUsd = spotTotalUsd,
                SpotAvailableUsd = spotAvailableUsd,
                SpotAssets = spotBalances,
                // Futures specific
                FuturesBalanceUsd = futuresTotal,
                FuturesAvailableUsd = futuresAvailable,
                MarginUsed = marginUsed,
                UnrealizedPnL = unrealizedPnL,
                UpdatedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching account balance from Binance");
        }

        return new AccountBalanceDto {Exchange = ExchangeName};
    }

    public async Task<string> PlaceMarketOrderAsync(string symbol, PositionSide side, decimal quantity,
        decimal leverage)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            // Enable Hedge Mode (required for simultaneous long/short positions)
            // This call is idempotent - safe to call even if already enabled
            try
            {
                await _restClient.UsdFuturesApi.Account.ModifyPositionModeAsync(true);
                _logger.LogInformation("Hedge mode enabled for Binance account");
            }
            catch (Exception ex)
            {
                // If already in hedge mode, this will fail with -4059, which is fine
                _logger.LogDebug("Hedge mode setting: {Error}", ex.Message);
            }

            // Validate symbol exists and is trading
            var exchangeInfo = await _restClient.UsdFuturesApi.ExchangeData.GetExchangeInfoAsync();
            var symbolInfo = exchangeInfo.Data?.Symbols.FirstOrDefault(s => s.Name == symbol);

            if (symbolInfo == null)
            {
                throw new Exception($"Symbol {symbol} not found on Binance Futures");
            }

            if (symbolInfo.Status != Binance.Net.Enums.SymbolStatus.Trading)
            {
                throw new Exception($"Symbol {symbol} is not currently trading (Status: {symbolInfo.Status})");
            }

            // Log all filters for debugging (without serialization to avoid circular references)
            if (symbolInfo.Filters != null && symbolInfo.Filters.Any())
            {
                var filterNames = string.Join(", ", symbolInfo.Filters.Select(f => f.GetType().Name));
                _logger.LogInformation("Symbol {Symbol} has {Count} filters: {Filters}", symbol,
                    symbolInfo.Filters.Count(), filterNames);

                // Get current mark price for validation
                var markPriceResult = await _restClient.UsdFuturesApi.ExchangeData.GetMarkPricesAsync();
                if (markPriceResult.Success && markPriceResult.Data != null)
                {
                    var currentPrice = markPriceResult.Data.FirstOrDefault(p => p.Symbol == symbol)?.MarkPrice ?? 0;
                    _logger.LogInformation("Current mark price for {Symbol}: ${Price}", symbol, currentPrice);

                    // Log expected order value to help diagnose PERCENT_PRICE issues
                    var estimatedOrderValue = quantity * currentPrice;
                    _logger.LogInformation("Order details: Quantity={Quantity}, EstimatedValue=${EstimatedValue}",
                        quantity, estimatedOrderValue);
                }
            }
            else
            {
                _logger.LogWarning("Symbol {Symbol} has no filters defined", symbol);
            }

            // Set leverage first
            await _restClient.UsdFuturesApi.Account.ChangeInitialLeverageAsync(symbol, (int) leverage);

            var orderSide = side == PositionSide.Long ? BinanceOrderSide.Buy : BinanceOrderSide.Sell;
            var positionSide = side == PositionSide.Long
                ? Binance.Net.Enums.PositionSide.Long
                : Binance.Net.Enums.PositionSide.Short;

            _logger.LogInformation("Placing market order on Binance: {Symbol} {Side} {Quantity} @ Leverage {Leverage}",
                symbol, side, quantity, leverage);

            var order = await _restClient.UsdFuturesApi.Trading.PlaceOrderAsync(
                symbol,
                orderSide,
                FuturesOrderType.Market,
                quantity,
                positionSide: positionSide
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Market order placed on Binance: {OrderId}", order.Data.Id);
                return order.Data.Id.ToString();
            }

            // Log detailed error information
            var errorCode = order.Error?.Code;
            var errorMessage = order.Error?.Message;

            _logger.LogError("Failed to place order on Binance - Code: {Code}, Message: {Message}",
                errorCode, errorMessage);

            throw new Exception($"Failed to place order - Code: {errorCode}, Message: {errorMessage}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order on Binance");
            throw;
        }
    }

    public async Task<bool> ClosePositionAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            var positions = await _restClient.UsdFuturesApi.Account.GetPositionInformationAsync(symbol);

            if (positions.Success && positions.Data != null)
            {
                foreach (var position in positions.Data.Where(p => p.Quantity != 0))
                {
                    var orderSide = position.Quantity > 0 ? BinanceOrderSide.Sell : BinanceOrderSide.Buy;
                    var quantity = Math.Abs(position.Quantity);

                    await _restClient.UsdFuturesApi.Trading.PlaceOrderAsync(
                        symbol,
                        orderSide,
                        FuturesOrderType.Market,
                        quantity,
                        positionSide: position.PositionSide
                    );
                }

                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error closing position on Binance");
        }

        return false;
    }

    public async Task<List<PositionDto>> GetOpenPositionsAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var positions = new List<PositionDto>();

        try
        {
            var result = await _restClient.UsdFuturesApi.Account.GetPositionInformationAsync();

            if (result.Success && result.Data != null)
            {
                positions = result.Data
                    .Where(p => p.Quantity != 0)
                    .Select(p => new PositionDto
                    {
                        Exchange = ExchangeName,
                        Symbol = p.Symbol,
                        Side = p.Quantity > 0 ? PositionSide.Long : PositionSide.Short,
                        Status = PositionStatus.Open,
                        EntryPrice = p.EntryPrice,
                        Quantity = Math.Abs(p.Quantity),
                        Leverage = p.Leverage,
                        UnrealizedPnL = p.UnrealizedPnl,
                        OpenedAt = DateTime.UtcNow // Binance doesn't provide this directly
                    })
                    .ToList();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching positions from Binance");
        }

        return positions;
    }

    // Spot trading methods for cash-and-carry arbitrage
    public async Task<(string orderId, decimal filledQuantity)> PlaceSpotBuyOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            _logger.LogInformation("Placing spot BUY order on Binance: {Symbol}, Quantity: {Quantity}",
                symbol, quantity);

            var order = await _restClient.SpotApi.Trading.PlaceOrderAsync(
                symbol,
                Binance.Net.Enums.OrderSide.Buy,
                Binance.Net.Enums.SpotOrderType.Market,
                quantity: quantity
            );

            if (order.Success && order.Data != null)
            {
                // Binance returns filled quantity in the order response
                var filledQuantity = order.Data.QuantityFilled > 0 ? order.Data.QuantityFilled : quantity;

                _logger.LogInformation(
                    "Spot BUY order placed on Binance: {OrderId}, Requested: {Requested}, Filled: {Filled}",
                    order.Data.Id, quantity, filledQuantity);

                return (order.Data.Id.ToString(), filledQuantity);
            }

            _logger.LogError("Failed to place spot BUY order on Binance: {Error}", order.Error);
            throw new Exception($"Failed to place spot buy order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot BUY order on Binance");
            throw;
        }
    }

    public async Task<string> PlaceSpotSellOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            _logger.LogInformation("Placing spot SELL order on Binance: {Symbol}, Quantity: {Quantity}",
                symbol, quantity);

            var order = await _restClient.SpotApi.Trading.PlaceOrderAsync(
                symbol,
                Binance.Net.Enums.OrderSide.Sell,
                Binance.Net.Enums.SpotOrderType.Market,
                quantity: quantity
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Spot SELL order placed on Binance: {OrderId}", order.Data.Id);
                return order.Data.Id.ToString();
            }

            _logger.LogError("Failed to place spot SELL order on Binance: {Error}", order.Error);
            throw new Exception($"Failed to place spot sell order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot SELL order on Binance");
            throw;
        }
    }

    public async Task<decimal> GetSpotBalanceAsync(string asset)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            var accountInfo = await _restClient.SpotApi.Account.GetAccountInfoAsync();

            if (accountInfo.Success && accountInfo.Data != null)
            {
                var balance = accountInfo.Data.Balances.FirstOrDefault(b => b.Asset == asset);
                if (balance != null)
                {
                    _logger.LogInformation("Spot balance for {Asset}: {Available} (Total: {Total})",
                        asset, balance.Available, balance.Total);
                    return balance.Available;
                }
            }

            _logger.LogWarning("No balance found for asset {Asset} on Binance", asset);
            return 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot balance for {Asset} from Binance", asset);
            throw;
        }
    }

    public async Task<Dictionary<string, decimal>> GetSpotBalancesAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var balances = new Dictionary<string, decimal>();

        try
        {
            var accountInfo = await _restClient.SpotApi.Account.GetAccountInfoAsync();

            if (accountInfo.Success && accountInfo.Data != null)
            {
                foreach (var balance in accountInfo.Data.Balances.Where(b => b.Available > 0 || b.Locked > 0))
                {
                    balances[balance.Asset] = balance.Total;
                }
                
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot balances from Binance");
        }

        return balances;
    }

    public async Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate)
    {
        if (_socketClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        // Binance doesn't have a direct funding rate WebSocket stream
        // We'll need to poll periodically instead
        await Task.CompletedTask;
    }

    // Get instrument specifications for quantity validation
    public async Task<InstrumentInfo?> GetInstrumentInfoAsync(string symbol, bool isSpot = false)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        try
        {
            if (isSpot)
            {
                // Get spot exchange info
                var exchangeInfo = await _restClient.SpotApi.ExchangeData.GetExchangeInfoAsync(symbol);

                if (exchangeInfo.Success && exchangeInfo.Data?.Symbols != null && exchangeInfo.Data.Symbols.Any())
                {
                    var symbolInfo = exchangeInfo.Data.Symbols.First();

                    // Find LOT_SIZE filter
                    var lotSizeFilter = symbolInfo.LotSizeFilter;

                    if (lotSizeFilter != null)
                    {
                        _logger.LogInformation(
                            "Spot instrument info for {Symbol}: MinQty={Min}, MaxQty={Max}, StepSize={Step}",
                            symbol, lotSizeFilter.MinQuantity, lotSizeFilter.MaxQuantity, lotSizeFilter.StepSize);

                        return new InstrumentInfo
                        {
                            Symbol = symbol,
                            MinOrderQty = lotSizeFilter.MinQuantity,
                            MaxOrderQty = lotSizeFilter.MaxQuantity,
                            QtyStep = lotSizeFilter.StepSize,
                            IsSpot = true
                        };
                    }
                }
            }
            else
            {
                // Get futures exchange info
                var exchangeInfo = await _restClient.UsdFuturesApi.ExchangeData.GetExchangeInfoAsync();

                if (exchangeInfo.Success && exchangeInfo.Data?.Symbols != null)
                {
                    var symbolInfo = exchangeInfo.Data.Symbols.FirstOrDefault(s => s.Name == symbol);

                    if (symbolInfo != null)
                    {
                        // Find LOT_SIZE filter
                        var lotSizeFilter = symbolInfo.LotSizeFilter;

                        if (lotSizeFilter != null)
                        {
                            _logger.LogInformation(
                                "Futures instrument info for {Symbol}: MinQty={Min}, MaxQty={Max}, StepSize={Step}",
                                symbol, lotSizeFilter.MinQuantity, lotSizeFilter.MaxQuantity, lotSizeFilter.StepSize);

                            return new InstrumentInfo
                            {
                                Symbol = symbol,
                                MinOrderQty = lotSizeFilter.MinQuantity,
                                MaxOrderQty = lotSizeFilter.MaxQuantity,
                                QtyStep = lotSizeFilter.StepSize,
                                IsSpot = false
                            };
                        }
                    }
                }
            }

            _logger.LogWarning("No instrument info found for {Symbol} ({Type})", symbol, isSpot ? "Spot" : "Futures");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching instrument info for {Symbol} from Binance", symbol);
            return null;
        }
    }

    // Validate and adjust quantity to meet exchange requirements
    public async Task<List<FundingRateDto>> GetFundingRateHistoryAsync(string symbol, DateTime startTime,
        DateTime endTime)
    {
        if (_restClient == null)
        {
            _logger.LogWarning("Cannot get funding rate history for {Exchange} - not connected", ExchangeName);
            return new List<FundingRateDto>();
        }

        try
        {
            var results = new List<FundingRateDto>();

            // Binance uses variable funding intervals: 1h, 4h, or 8h
            // To cover 3 days (72 hours) for all intervals:
            // - 1h interval: 72 records needed
            // - 4h interval: 18 records needed
            // - 8h interval: 9 records needed
            // Using limit without startTime/endTime to avoid 403 errors on public API
            const int limit = 75; // Fetch last 75 records to cover 3+ days for all interval types

            var historicalRates = await _restClient.UsdFuturesApi.ExchangeData.GetFundingRatesAsync(
                symbol,
                limit: limit);

            if (!historicalRates.Success || historicalRates.Data == null || !historicalRates.Data.Any())
            {
                if (!historicalRates.Success)
                {
                    _logger.LogWarning("Failed to fetch funding rate history for {Symbol} on {Exchange}: Code={ErrorCode}, Message={ErrorMsg}",
                        symbol, ExchangeName, historicalRates.Error?.Code, historicalRates.Error?.Message ?? "Unknown error");
                }
                else if (historicalRates.Data == null)
                {
                    _logger.LogWarning("Funding rate history returned null data for {Symbol} on {Exchange}", symbol, ExchangeName);
                }
                else
                {
                    _logger.LogWarning("Funding rate history returned empty list for {Symbol} on {Exchange}", symbol, ExchangeName);
                }

                return results;
            }

            // Filter to only include rates from the last 3 days
            var threeDaysAgo = DateTime.UtcNow.AddDays(-3);
            var filteredRates = historicalRates.Data
                .Where(r => r.FundingTime >= threeDaysAgo)
                .OrderBy(r => r.FundingTime)
                .ToList();

            if (!filteredRates.Any())
            {
                _logger.LogWarning("No funding rates found within last 3 days for {Symbol} on {Exchange}",
                    symbol, ExchangeName);
                return results;
            }

            // Determine the actual funding interval by comparing timestamps
            int fundingIntervalHours = 8; // Default to 8h
            if (filteredRates.Count >= 2)
            {
                var timeDiff = (filteredRates[1].FundingTime - filteredRates[0].FundingTime).TotalHours;

                // Round to nearest hour and determine interval (1h, 4h, or 8h)
                if (Math.Abs(timeDiff - 1) < 0.1) fundingIntervalHours = 1;
                else if (Math.Abs(timeDiff - 4) < 0.1) fundingIntervalHours = 4;
                else if (Math.Abs(timeDiff - 8) < 0.1) fundingIntervalHours = 8;

                _logger.LogDebug("Detected {Interval}h funding interval for {Symbol} on {Exchange}",
                    fundingIntervalHours, symbol, ExchangeName);
            }

            foreach (var rate in filteredRates)
            {
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
                    FundingTime = rate.FundingTime,
                    NextFundingTime = rate.FundingTime.AddHours(fundingIntervalHours),
                    RecordedAt = DateTime.UtcNow
                });
            }
            

            return results.OrderBy(r => r.FundingTime).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rate history for {Symbol} on {Exchange}", symbol,
                ExchangeName);
            return new List<FundingRateDto>();
        }
    }

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

    public async Task<List<OrderDto>> GetOpenOrdersAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var orders = new List<OrderDto>();

        try
        {
            var result = await _restClient.UsdFuturesApi.Trading.GetOpenOrdersAsync();

            if (result.Success && result.Data != null)
            {
                orders = result.Data.Select(o => new OrderDto
                {
                    Exchange = ExchangeName,
                    OrderId = o.Id.ToString(),
                    ClientOrderId = o.ClientOrderId,
                    Symbol = o.Symbol,
                    Side = o.Side == Binance.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                    Type = MapBinanceOrderType(o.Type),
                    Status = MapBinanceOrderStatus(o.Status),
                    TimeInForce = o.TimeInForce.ToString(),
                    Price = o.Price,
                    AveragePrice = o.AveragePrice,
                    StopPrice = o.StopPrice,
                    Quantity = o.Quantity,
                    FilledQuantity = o.QuantityFilled,
                    Fee = 0,
                    FeeAsset = null,
                    CreatedAt = o.CreateTime,
                    UpdatedAt = o.UpdateTime > DateTime.MinValue ? o.UpdateTime : o.CreateTime,
                    WorkingTime = null,
                    ReduceOnly = o.ReduceOnly.ToString(),
                    PostOnly = null
                }).ToList();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching open orders from Binance");
        }

        return orders;
    }

    public async Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var orders = new List<OrderDto>();

        try
        {
            _logger.LogDebug("Fetching order history from Binance (startTime: {StartTime}, endTime: {EndTime}, limit: {Limit})",
                startTime, endTime, limit);

            var result = await _restClient.UsdFuturesApi.Trading.GetOrdersAsync(
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (!result.Success)
            {
                _logger.LogWarning("Failed to fetch orders from Binance: {Error} (Code: {Code})",
                    result.Error?.Message ?? "Unknown error",
                    result.Error?.Code);
                return orders;
            }

            if (result.Data == null)
            {
                _logger.LogWarning("Binance GetOrdersAsync returned null data");
                return orders;
            }

            if (!result.Data.Any())
            {
                _logger.LogDebug("No orders found in Binance order history");
                return orders;
            }

            orders = result.Data.Select(o => new OrderDto
            {
                Exchange = ExchangeName,
                OrderId = o.Id.ToString(),
                ClientOrderId = o.ClientOrderId,
                Symbol = o.Symbol,
                Side = o.Side == Binance.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                Type = MapBinanceOrderType(o.Type),
                Status = MapBinanceOrderStatus(o.Status),
                TimeInForce = o.TimeInForce.ToString(),
                Price = o.Price,
                AveragePrice = o.AveragePrice,
                StopPrice = o.StopPrice,
                Quantity = o.Quantity,
                FilledQuantity = o.QuantityFilled,
                Fee = 0,
                FeeAsset = null,
                CreatedAt = o.CreateTime,
                UpdatedAt = o.UpdateTime > DateTime.MinValue ? o.UpdateTime : o.CreateTime,
                WorkingTime = null,
                ReduceOnly = o.ReduceOnly.ToString(),
                PostOnly = null
            }).ToList();

            _logger.LogInformation("Fetched {Count} orders from Binance", orders.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching order history from Binance");
        }

        return orders;
    }

    public async Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var trades = new List<TradeDto>();

        try
        {
            // Binance API requires symbol parameter for trades. First, get all orders to determine symbols with activity
            var ordersResult = await _restClient.UsdFuturesApi.Trading.GetOrdersAsync(
                startTime: startTime,
                endTime: endTime,
                limit: 1000); // Get more orders to find all active symbols

            if (!ordersResult.Success || ordersResult.Data == null)
            {
                _logger.LogWarning("Could not fetch orders to determine symbols for trade history: {Error}",
                    ordersResult.Error?.Message ?? "Unknown error");
                return trades;
            }

            // Get unique symbols from orders
            var symbols = ordersResult.Data
                .Select(o => o.Symbol)
                .Distinct()
                .ToList();

            if (!symbols.Any())
            {
                _logger.LogDebug("No symbols found in order history for Binance");
                return trades;
            }

            _logger.LogDebug("Fetching trades for {Count} symbols on Binance", symbols.Count);

            // Fetch trades for each symbol
            foreach (var symbol in symbols)
            {
                try
                {
                    var tradesResult = await _restClient.UsdFuturesApi.Trading.GetUserTradesAsync(
                        symbol: symbol,
                        startTime: startTime,
                        endTime: endTime,
                        limit: limit);

                    if (tradesResult.Success && tradesResult.Data != null)
                    {
                        var symbolTrades = tradesResult.Data.Select(t => new TradeDto
                        {
                            Exchange = ExchangeName,
                            TradeId = t.Id.ToString(),
                            OrderId = t.OrderId.ToString(),
                            Symbol = t.Symbol,
                            Side = t.Side == Binance.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                            Price = t.Price,
                            Quantity = t.Quantity,
                            QuoteQuantity = t.QuoteQuantity,
                            Commission = t.Fee,
                            CommissionAsset = t.FeeAsset,
                            IsMaker = t.Maker,
                            IsBuyer = t.Side == Binance.Net.Enums.OrderSide.Buy,
                            ExecutedAt = t.Timestamp
                        }).ToList();

                        trades.AddRange(symbolTrades);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error fetching trades for symbol {Symbol} from Binance", symbol);
                }
            }

            _logger.LogInformation("Fetched {Count} trades across {SymbolCount} symbols from Binance",
                trades.Count, symbols.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching user trades from Binance");
        }

        return trades;
    }

    public async Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var transactions = new List<TransactionDto>();

        try
        {
            _logger.LogDebug("Fetching Binance income history from {StartTime} to {EndTime}, limit {Limit}",
                startTime, endTime, limit);

            var result = await _restClient.UsdFuturesApi.Account.GetIncomeHistoryAsync(
                startTime: startTime,
                endTime: endTime,
                limit: limit);

            if (result.Success && result.Data != null)
            {
                transactions = result.Data.Select(t =>
                {
                    var incomeType = t.IncomeType ?? Binance.Net.Enums.IncomeType.Transfer;
                    // For commission transactions, the income value (negative) represents the fee paid
                    var fee = incomeType == Binance.Net.Enums.IncomeType.Commission
                        ? Math.Abs(t.Income)
                        : 0m;

                    return new TransactionDto
                    {
                        Exchange = ExchangeName,
                        TransactionId = t.TransactionId?.ToString() ?? t.Timestamp.Ticks.ToString(),
                        TxHash = t.TransactionId?.ToString(),
                        Type = MapBinanceIncomeType(incomeType),
                        Asset = t.Asset ?? string.Empty,
                        Amount = t.Income,
                        Status = Models.TransactionStatus.Confirmed,
                        FromAddress = null,
                        ToAddress = null,
                        Network = null,
                        Info = t.Info,
                        Symbol = t.Symbol,
                        TradeId = t.TradeId?.ToString(),
                        Fee = fee,
                        FeeAsset = fee > 0 ? t.Asset : null,
                        CreatedAt = t.Timestamp,
                        ConfirmedAt = t.Timestamp
                    };
                }).ToList();

                _logger.LogInformation("Fetched {Count} income transactions from Binance (commissions: {CommissionCount}, PnL: {PnLCount})",
                    transactions.Count,
                    transactions.Count(t => t.Type == Models.TransactionType.Commission),
                    transactions.Count(t => t.Type == Models.TransactionType.RealizedPnL));
            }
            else
            {
                _logger.LogWarning("Failed to fetch transactions from Binance: {Error}", result.Error?.Message);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching transactions from Binance");
        }

        return transactions;
    }

    private Models.OrderType MapBinanceOrderType(Binance.Net.Enums.FuturesOrderType type)
    {
        return type switch
        {
            Binance.Net.Enums.FuturesOrderType.Market => Models.OrderType.Market,
            Binance.Net.Enums.FuturesOrderType.Limit => Models.OrderType.Limit,
            Binance.Net.Enums.FuturesOrderType.Stop => Models.OrderType.StopMarket,
            Binance.Net.Enums.FuturesOrderType.StopMarket => Models.OrderType.StopMarket,
            Binance.Net.Enums.FuturesOrderType.TakeProfit => Models.OrderType.TakeProfitMarket,
            Binance.Net.Enums.FuturesOrderType.TakeProfitMarket => Models.OrderType.TakeProfitMarket,
            _ => Models.OrderType.Market
        };
    }

    private Models.OrderStatus MapBinanceOrderStatus(Binance.Net.Enums.OrderStatus status)
    {
        return status switch
        {
            Binance.Net.Enums.OrderStatus.New => Models.OrderStatus.New,
            Binance.Net.Enums.OrderStatus.PartiallyFilled => Models.OrderStatus.PartiallyFilled,
            Binance.Net.Enums.OrderStatus.Filled => Models.OrderStatus.Filled,
            Binance.Net.Enums.OrderStatus.Canceled => Models.OrderStatus.Canceled,
            Binance.Net.Enums.OrderStatus.Rejected => Models.OrderStatus.Rejected,
            Binance.Net.Enums.OrderStatus.Expired => Models.OrderStatus.Expired,
            _ => Models.OrderStatus.New
        };
    }

    private Models.TransactionType MapBinanceIncomeType(Binance.Net.Enums.IncomeType type)
    {
        return type switch
        {
            Binance.Net.Enums.IncomeType.RealizedPnl => Models.TransactionType.RealizedPnL,
            Binance.Net.Enums.IncomeType.FundingFee => Models.TransactionType.FundingFee,
            Binance.Net.Enums.IncomeType.Commission => Models.TransactionType.Commission,
            Binance.Net.Enums.IncomeType.ReferralKickback => Models.TransactionType.ReferralKickback,
            Binance.Net.Enums.IncomeType.CommissionRebate => Models.TransactionType.CommissionRebate,
            Binance.Net.Enums.IncomeType.ApiRebate => Models.TransactionType.Rebate,
            Binance.Net.Enums.IncomeType.ContestReward => Models.TransactionType.ContestReward,
            Binance.Net.Enums.IncomeType.InsuranceClear => Models.TransactionType.InsuranceClear,
            Binance.Net.Enums.IncomeType.Transfer => Models.TransactionType.Transfer,
            Binance.Net.Enums.IncomeType.InternalTransfer => Models.TransactionType.InternalTransfer,
            Binance.Net.Enums.IncomeType.WelcomeBonus => Models.TransactionType.WelcomeBonus,
            Binance.Net.Enums.IncomeType.DeliveredSettlement => Models.TransactionType.Settlement,
            _ => Models.TransactionType.Other
        };
    }
}