using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoExchange.Net.Authentication;
using Kraken.Net;
using Kraken.Net.Clients;
using Kraken.Net.Objects.Models.Futures;
using KrakenOrderSide = Kraken.Net.Enums.OrderSide;
using PositionSide = CryptoArbitrage.API.Data.Entities.PositionSide;

namespace CryptoArbitrage.API.Services.Exchanges;

/// <summary>
/// Kraken Exchange Connector - Currently supports Spot trading only
/// Note: Kraken Futures API requires separate integration and is not yet supported
/// </summary>
public class KrakenConnector : IExchangeConnector
{
    private readonly ILogger<KrakenConnector> _logger;
    private readonly IConfiguration _configuration;
    private KrakenRestClient? _restClient;
    private KrakenSocketClient? _socketClient;

    public string ExchangeName => "Kraken";

    public KrakenConnector(ILogger<KrakenConnector> logger, IConfiguration configuration)
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

            var environment = isLive
                ? KrakenEnvironment.Live
                : KrakenEnvironment.CreateCustom("Demo", "https://api.kraken.com", "wss://ws.kraken.com", "wss://ws-auth.kraken.com/", "https://demo-futures.kraken.com", "wss://futures.kraken.com/");

            bool hasCredentials = !string.IsNullOrWhiteSpace(apiKey) && !string.IsNullOrWhiteSpace(apiSecret);

            if (hasCredentials)
            {
                _restClient = new KrakenRestClient(options =>
                {
                    options.Environment = environment;
                    options.ApiCredentials = new ApiCredentials(apiKey, apiSecret);
                    options.RequestTimeout = TimeSpan.FromSeconds(30);
                });

                _socketClient = new KrakenSocketClient(options =>
                {
                    options.ApiCredentials = new ApiCredentials(apiKey, apiSecret);
                });

                var accountInfo = await _restClient.FuturesApi.Account.GetBalancesAsync();
                if (accountInfo.Success)
                {
                    _logger.LogInformation("Successfully connected to Kraken with API credentials");
                    return true;
                }

                _logger.LogError("Failed to connect to Kraken: {Error}", accountInfo.Error);
                return false;
            }
            else
            {
                _restClient = new KrakenRestClient(options =>
                {
                    options.Environment = environment;
                    options.RequestTimeout = TimeSpan.FromSeconds(30);
                });

                _socketClient = new KrakenSocketClient();

                var serverTime = await _restClient.SpotApi.ExchangeData.GetServerTimeAsync();
                if (serverTime.Success)
                {
                    _logger.LogInformation("Successfully connected to Kraken (public data only)");
                    return true;
                }

                _logger.LogError("Failed to connect to Kraken: {Error}", serverTime.Error);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting to Kraken");
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
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var response = await _restClient.FuturesApi.ExchangeData.GetTickersAsync();
            if (!response.Success || response.Data == null)
            {
                _logger.LogError("Failed to get symbols from Kraken");
                return new List<string>();
            }

            var symbols = response.Data.AsQueryable();

            if (minHighPriorityFundingRate > 0 && false)
            {
                symbols = symbols.Where(x => x.FundingRate.HasValue && Math.Abs(x.FundingRate.Value) >= minHighPriorityFundingRate);
            }

            // Return USDT and USD pairs with normalized symbols (BTC:USD -> BTCUSD)
            var usdtPairs = symbols
                .Where(x => x.Volume24h >= minDailyVolumeUsd)
                .Where(x => x.Tag == "perpetual" && x.Symbol.StartsWith("PF_"))
                .Where(s => s.Pair.EndsWith("USDT") || s.Pair.EndsWith("USD"))
                .Select(s => NormalizeKrakenSymbol(s.Pair))
                .Take(maxSymbols)
                .ToList();

            _logger.LogInformation("Discovered {Count} active symbols from Kraken", usdtPairs.Count);
            return usdtPairs;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering active symbols from Kraken");
            return new List<string>();
        }
    }

    public async Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var fundingRates = new List<FundingRateDto>();

        try
        {
            // Get all tickers from Futures API which includes funding rates
            var response = await _restClient.FuturesApi.ExchangeData.GetTickersAsync();

            if (!response.Success || response.Data == null)
            {
                _logger.LogError("Failed to get funding rates from Kraken Futures: {Error}", response.Error?.Message);
                return fundingRates;
            }

            // Filter for perpetual futures only (they have funding rates)
            var perpetuals = response.Data
                .Where(x => x.Tag == "perpetual" && x.Symbol.StartsWith("PF_"))
                .Where(x => x.FundingRate.HasValue && x.FundingRatePrediction.HasValue);

            // If specific symbols are requested, filter by them
            if (symbols != null && symbols.Count > 0)
            {
                // Convert normalized symbols back to Kraken format for matching
                perpetuals = perpetuals.Where(x => symbols.Contains(NormalizeKrakenSymbol(x.Pair)));
            }

            foreach (var ticker in perpetuals)
            {
                var normalizedSymbol = NormalizeKrakenSymbol(ticker.Pair);

                // Perpetual Derivatives are a type of Derivatives contract that have no expiration date and an auto-rolling feature every hour.
                // https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications
                var fundingIntervalHours = 1;
                var intervalsPerYear = 365 * 24 / fundingIntervalHours; // 2190 intervals per year

                // Calculate annualized rate
                var annualizedRate = ticker.FundingRate.Value * intervalsPerYear;

                // Calculate next funding time (Kraken typically has funding every 4 hours)
                var now = DateTime.UtcNow;
                var nextFundingHour = ((now.Hour / fundingIntervalHours) + 1) * fundingIntervalHours;
                var nextFundingTime = now.Date.AddHours(nextFundingHour);
                if (nextFundingHour >= 24)
                {
                    nextFundingTime = now.Date.AddDays(1).AddHours(nextFundingHour % 24);
                }

                // Determine direction
                var direction = ticker.FundingRate.Value < 0
                    ? FundingDirection.ShortPaysLong
                    : FundingDirection.LongPaysShort;

                fundingRates.Add(new FundingRateDto
                {
                    Exchange = ExchangeName,
                    Symbol = normalizedSymbol,
                    Rate = ticker.FundingRate.Value,
                    AnnualizedRate = annualizedRate,
                    FundingIntervalHours = fundingIntervalHours,
                    PreviousRate = ticker.FundingRatePrediction, // Using prediction as previous rate
                    PreviousAnnualizedRate = ticker.FundingRatePrediction.HasValue
                        ? ticker.FundingRatePrediction.Value * intervalsPerYear
                        : null,
                    FundingTime = DateTime.UtcNow, // Current funding rate
                    NextFundingTime = nextFundingTime,
                    RecordedAt = DateTime.UtcNow,
                    Direction = direction
                });
            }

            _logger.LogInformation("Fetched {Count} funding rates from Kraken Futures", fundingRates.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rates from Kraken Futures");
        }

        return fundingRates;
    }

    public async Task<Dictionary<string, PriceDto>> GetSpotPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var spotPrices = new Dictionary<string, PriceDto>();

        try
        {
            var tickers = await _restClient.SpotApi.ExchangeData.GetTickersAsync();
            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.Where(t => symbols.Count == 0 || symbols.Contains(t.Key)))
                {
                    var lastPrice = ticker.Value.LastTrade?.Price ?? 0;
                    spotPrices[ticker.Key] = new PriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Key,
                        Price = lastPrice,
                        Volume24h = ticker.Value.Volume.Value24H,
                        Timestamp = DateTime.UtcNow
                    };
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot prices from Kraken");
        }

        return spotPrices;
    }

    public async Task<Dictionary<string, PriceDto>> GetPerpetualPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var perpPrices = new Dictionary<string, PriceDto>();

        try
        {
            var response = await _restClient.FuturesApi.ExchangeData.GetTickersAsync();
            if (!response.Success || response.Data == null)
            {
                _logger.LogError("Failed to get perpetual prices from Kraken Futures");
                return perpPrices;
            }

            // Filter for perpetual futures only
            var perpetuals = response.Data
                .Where(x => x.Tag == "perpetual" && x.Symbol.StartsWith("PF_"))
                .Where(x => x.Pair.EndsWith("USDT") || x.Pair.EndsWith("USD"));

            // If specific symbols are requested, filter by them
            if (symbols != null && symbols.Count > 0)
            {
                // Convert input symbols to Kraken format for matching
                perpetuals = perpetuals.Where(x => symbols.Contains(NormalizeKrakenSymbol(x.Pair)));
            }

            foreach (var ticker in perpetuals)
            {
                var normalizedSymbol = NormalizeKrakenSymbol(ticker.Pair);
                perpPrices[normalizedSymbol] = new PriceDto
                {
                    Exchange = ExchangeName,
                    Symbol = normalizedSymbol,
                    Price = ticker.LastPrice.GetValueOrDefault(),
                    Volume24h = ticker.Volume24h,
                    Timestamp = DateTime.UtcNow
                };
            }

            _logger.LogInformation("Fetched {Count} perpetual prices from Kraken Futures", perpPrices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching perpetual prices from Kraken Futures");
        }

        return perpPrices;
    }

    public async Task<LiquidityMetricsDto?> GetLiquidityMetricsAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var orderbook = await _restClient.SpotApi.ExchangeData.GetOrderBookAsync(symbol, 100);
            if (!orderbook.Success || orderbook.Data == null)
            {
                return null;
            }

            var bids = orderbook.Data.Bids.ToList();
            var asks = orderbook.Data.Asks.ToList();

            if (bids.Count == 0 || asks.Count == 0)
                return null;

            var bestBid = bids.First().Price;
            var bestAsk = asks.First().Price;
            var midPrice = (bestBid + bestAsk) / 2;
            var bidAskSpread = ((bestAsk - bestBid) / midPrice) * 100;

            var depthUsd = 0m;
            foreach (var bid in bids) depthUsd += bid.Quantity * bid.Price;
            foreach (var ask in asks) depthUsd += ask.Quantity * ask.Price;

            return new LiquidityMetricsDto
            {
                BidAskSpreadPercent = bidAskSpread,
                OrderbookDepthUsd = depthUsd,
                Status = LiquidityStatus.Good,
                WarningMessage = null
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching liquidity metrics from Kraken");
            return null;
        }
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            // Fetch Kraken Futures balances
            var balancesResponse = await _restClient.FuturesApi.Account.GetBalancesAsync();

            if (!balancesResponse.Success || balancesResponse.Data == null)
            {
                _logger.LogError("Failed to fetch Kraken Futures balances: {Error}",
                    balancesResponse.Error?.Message ?? "Unknown error");
                return new AccountBalanceDto
                {
                    Exchange = ExchangeName,
                    UpdatedAt = DateTime.UtcNow
                };
            }

            var balance = balancesResponse.Data;

            // Initialize with default values
            decimal futuresTotal = 0;
            decimal futuresAvailable = 0;
            decimal marginUsed = 0;
            decimal unrealizedPnL = 0;
            decimal operationalBalance = 0;

            // Check if MultiCollateralMarginAccount exists and extract values
            if (balance.MultiCollateralMarginAccount != null)
            {
                futuresTotal = balance.MultiCollateralMarginAccount.BalanceValue;
                futuresAvailable = balance.MultiCollateralMarginAccount.AvailableMargin;
                marginUsed = balance.MultiCollateralMarginAccount.InitialMarginWithOrders;
                unrealizedPnL = balance.MultiCollateralMarginAccount.ProfitAndLoss;
                operationalBalance = balance.MultiCollateralMarginAccount.MarginEquity;

                _logger.LogDebug("Kraken Futures balance - Total: {Total}, Available: {Available}, Margin: {Margin}, PnL: {PnL}",
                    futuresTotal, futuresAvailable, marginUsed, unrealizedPnL);
            }
            else
            {
                _logger.LogWarning("MultiCollateralMarginAccount is null in Kraken Futures balance response");
            }

            // Kraken implementation focuses on Futures
            decimal spotBalanceUsd = 0;
            decimal spotAvailableUsd = 0;
            var spotAssets = new Dictionary<string, decimal>();

            return new AccountBalanceDto
            {
                Exchange = ExchangeName,
                TotalBalance = futuresTotal + spotBalanceUsd,
                AvailableBalance = futuresAvailable,
                OperationalBalanceUsd = operationalBalance,
                SpotBalanceUsd = spotBalanceUsd,
                SpotAvailableUsd = spotAvailableUsd,
                SpotAssets = spotAssets,
                FuturesBalanceUsd = futuresTotal,
                FuturesAvailableUsd = futuresAvailable,
                MarginUsed = marginUsed,
                UnrealizedPnL = unrealizedPnL,
                UpdatedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching account balance from Kraken Futures");
            return new AccountBalanceDto
            {
                Exchange = ExchangeName,
                UpdatedAt = DateTime.UtcNow
            };
        }
    }

    public Task<AccountBalanceDto> GetAccountBalanceAsync(Dictionary<string, decimal> activeSpotPositions)
    {
        return GetAccountBalanceAsync();
    }

    public async Task<Dictionary<string, decimal>> GetSpotBalancesAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var balances = new Dictionary<string, decimal>();
        try
        {
            var result = await _restClient.SpotApi.Account.GetBalancesAsync();
            if (result.Success && result.Data != null)
            {
                foreach (var balance in result.Data.Where(b => b.Value > 0))
                {
                    balances[balance.Key] = balance.Value;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot balances from Kraken");
        }

        return balances;
    }

    public async Task<string> PlaceMarketOrderAsync(string symbol, PositionSide side, decimal quantity, decimal leverage)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var orderSide = side == PositionSide.Long ? KrakenOrderSide.Buy : KrakenOrderSide.Sell;
            var order = await _restClient.SpotApi.Trading.PlaceOrderAsync(
                symbol, orderSide, Kraken.Net.Enums.OrderType.Market, quantity);

            if (order.Success && order.Data != null)
            {
                var orderId = order.Data.OrderIds.FirstOrDefault() ?? string.Empty;
                _logger.LogInformation("Market order placed on Kraken: {OrderId}", orderId);
                return orderId;
            }

            throw new Exception($"Failed to place order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order on Kraken");
            throw;
        }
    }

    public async Task<ClosePositionResult> ClosePositionAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var closeResult = new ClosePositionResult();

        try
        {
            // Get current positions for this symbol
            var positions = await GetOpenPositionsAsync();
            var position = positions.FirstOrDefault(p => p.Symbol == symbol);

            if (position == null)
            {
                _logger.LogWarning("No open position found for {Symbol} on Kraken", symbol);
                closeResult.Success = true; // Already closed
                return closeResult;
            }

            // Determine the opposite side to close the position
            var closeSide = position.Side == PositionSide.Long
                ? Kraken.Net.Enums.OrderSide.Sell
                : Kraken.Net.Enums.OrderSide.Buy;

            // Convert symbol to Kraken format
            var krakenSymbol = symbol;
            if (!krakenSymbol.StartsWith("PF_"))
            {
                krakenSymbol = $"PF_{symbol}";
            }
            if (krakenSymbol.EndsWith("USDT"))
            {
                krakenSymbol = krakenSymbol[..^1]; // Remove 'T' from USDT
            }

            _logger.LogInformation("Closing {Side} position for {Symbol} (quantity: {Quantity}) on Kraken",
                position.Side, symbol, position.Quantity);

            // Place a market order to close the position with reduceOnly flag
            var result = await _restClient.FuturesApi.Trading.PlaceOrderAsync(
                symbol: krakenSymbol,
                side: closeSide,
                type: Kraken.Net.Enums.FuturesOrderType.Market,
                quantity: position.Quantity,
                price: null,
                stopPrice: null,
                reduceOnly: true); // This ensures we only close, not reverse the position

            if (result.Success && result.Data != null)
            {
                closeResult.OrderId = result.Data.OrderId;
                closeResult.Success = true;

                // Try to get the fill price from the order
                await Task.Delay(500); // Small delay for order to settle

                // Try to get order details - Kraken API might not support this directly
                // Use mark price as fallback
                var tickerResult = await _restClient.FuturesApi.ExchangeData.GetTickersAsync();
                if (tickerResult.Success && tickerResult.Data != null)
                {
                    var ticker = tickerResult.Data.FirstOrDefault(t => t.Symbol == krakenSymbol);
                    if (ticker != null)
                    {
                        closeResult.ExitPrice = ticker.LastPrice;
                        closeResult.FilledQuantity = position.Quantity;
                        // Estimate fee: Kraken futures taker fee is typically 0.05%
                        closeResult.TradingFee = position.Quantity * ticker.LastPrice * 0.0005m;
                    }
                }

                _logger.LogInformation("Successfully closed position for {Symbol} on Kraken. Order ID: {OrderId}, Status: {Status}, ExitPrice: {ExitPrice}",
                    symbol, result.Data.OrderId, result.Data.Status, closeResult.ExitPrice);
            }
            else
            {
                _logger.LogError("Failed to close position for {Symbol} on Kraken: {Error}",
                    symbol, result.Error?.Message ?? "Unknown error");
                closeResult.ErrorMessage = result.Error?.Message ?? "Unknown error";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error closing position for {Symbol} on Kraken", symbol);
            closeResult.ErrorMessage = ex.Message;
        }

        return closeResult;
    }

    public async Task<List<PositionDto>> GetOpenPositionsAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var positions = new List<PositionDto>();

        try
        {
            var result = await _restClient.FuturesApi.Trading.GetOpenPositionsAsync();

            if (!result.Success)
            {
                _logger.LogWarning("Failed to fetch open positions from Kraken: {Error}",
                    result.Error?.Message ?? "Unknown error");
                return positions;
            }

            if (result.Data == null || !result.Data.Any())
            {
                _logger.LogDebug("No open positions found on Kraken");
                return positions;
            }

            positions = result.Data
                .Select(p => new PositionDto
                {
                    Exchange = ExchangeName,
                    Symbol = NormalizeKrakenSymbol(p.Symbol),
                    Type = PositionType.Perpetual, // Kraken Futures are perpetual contracts
                    Side = MapKrakenPositionSide(p.Side),
                    Status = PositionStatus.Open,
                    EntryPrice = p.Price,
                    Quantity = p.Quantity,
                    Leverage = p.MaxFixedLeverage ?? 1m, // Default to 1x if not specified
                    InitialMargin = 0, // Not directly available, would need calculation
                    FundingEarnedUsd = 0,
                TradingFeesUsd = 0,
                PricePnLUsd = 0,
                RealizedPnLUsd = 0,
                RealizedPnLPct = 0, // Not available in position data
                    UnrealizedPnL = 0, // Would need current price to calculate
                    TotalFundingFeePaid = p.UnrealizedFunding < 0 ? Math.Abs(p.UnrealizedFunding ?? 0) : 0,
                    TotalFundingFeeReceived = p.UnrealizedFunding > 0 ? p.UnrealizedFunding ?? 0 : 0,
                    OpenedAt = p.FillTime
                })
                .ToList();

            _logger.LogInformation("Fetched {Count} open positions from Kraken", positions.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching positions from Kraken");
        }

        return positions;
    }

    /// <summary>
    /// Maps Kraken position side to application position side
    /// </summary>
    private PositionSide MapKrakenPositionSide(Kraken.Net.Enums.PositionSide krakenSide)
    {
        return krakenSide switch
        {
            Kraken.Net.Enums.PositionSide.Long => PositionSide.Long,
            Kraken.Net.Enums.PositionSide.Short => PositionSide.Short,
            _ => PositionSide.Long
        };
    }

    public async Task<(string orderId, decimal filledQuantity)> PlaceSpotBuyOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var order = await _restClient.SpotApi.Trading.PlaceOrderAsync(
                symbol, KrakenOrderSide.Buy, Kraken.Net.Enums.OrderType.Market, quantity);

            if (order.Success && order.Data != null)
            {
                var orderId = order.Data.OrderIds.FirstOrDefault() ?? string.Empty;
                return (orderId, quantity);
            }

            throw new Exception($"Failed to place spot buy order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot BUY order on Kraken");
            throw;
        }
    }

    public async Task<string> PlaceSpotSellOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var order = await _restClient.SpotApi.Trading.PlaceOrderAsync(
                symbol, KrakenOrderSide.Sell, Kraken.Net.Enums.OrderType.Market, quantity);

            if (order.Success && order.Data != null)
            {
                return order.Data.OrderIds.FirstOrDefault() ?? string.Empty;
            }

            throw new Exception($"Failed to place spot sell order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing spot SELL order on Kraken");
            throw;
        }
    }

    public async Task<decimal> GetSpotBalanceAsync(string asset)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            var balances = await _restClient.SpotApi.Account.GetBalancesAsync();
            if (balances.Success && balances.Data != null && balances.Data.TryGetValue(asset, out var balance))
            {
                return balance;
            }

            return 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot balance from Kraken");
            throw;
        }
    }

    public async Task<FeeInfoDto> GetTradingFeesAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        try
        {
            // Note: Kraken.Net library may not expose commission rates directly via the API
            // The Kraken Futures API does have fee schedule endpoints, but they may not be
            // available in the current version of Kraken.Net library

            // Attempt to get account info which might include fee information
            var accountInfo = await _restClient.FuturesApi.Account.GetBalancesAsync();

            if (accountInfo.Success)
            {
                // Note: The Kraken.Net library doesn't currently expose fee tier information
                // in the account balance response. We'll use default values based on Kraken's
                // published fee schedule.

                // Kraken Futures standard fees (as of 2024):
                // Volume-based tiers:
                // - Standard (< $100k/month): Maker 0.02%, Taker 0.05%
                // - VIP 1 ($100k-$1M): Maker 0.015%, Taker 0.04%
                // - VIP 2+ (> $1M): Lower fees

                // Default to standard tier fees
                var makerRate = 0.0002m; // 0.02%
                var takerRate = 0.0005m; // 0.05%
                var feeTier = "Standard";

                return new FeeInfoDto
                {
                    Exchange = ExchangeName,
                    MakerFeeRate = makerRate,
                    TakerFeeRate = takerRate,
                    FeeTier = feeTier,
                    CollectedAt = DateTime.UtcNow
                };
            }

            // Fallback to default Kraken Futures fees if API call fails
            _logger.LogWarning("Failed to get account info from Kraken Futures, using default fees");
            return new FeeInfoDto
            {
                Exchange = ExchangeName,
                MakerFeeRate = 0.0002m, // 0.02% default maker
                TakerFeeRate = 0.0005m, // 0.05% default taker
                FeeTier = "Unknown",
                CollectedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching trading fees from Kraken Futures");

            // Return default fees on error
            return new FeeInfoDto
            {
                Exchange = ExchangeName,
                MakerFeeRate = 0.0002m, // 0.02% default maker
                TakerFeeRate = 0.0005m, // 0.05% default taker
                FeeTier = "Error",
                CollectedAt = DateTime.UtcNow
            };
        }
    }

    public Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate)
    {
        _logger.LogWarning("Funding rate subscription not implemented for Kraken - requires Futures API");
        return Task.CompletedTask;
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

            // Kraken uses 1-hour funding intervals for perpetual futures
            // To cover 3 days (72 hours): 72 records needed
            const int limit = 75; // Fetch last 75 records to cover 3+ days

            // Convert symbol format to Kraken format if needed
            var krakenSymbol = symbol;
            if (!krakenSymbol.StartsWith("PF_"))
            {
                krakenSymbol = $"PF_{symbol}";
            }

            // Remove trailing 'T' from USDT symbols (Kraken uses USD not USDT)
            if (krakenSymbol.EndsWith("USDT"))
            {
                krakenSymbol = krakenSymbol[..^1];
            }

            _logger.LogDebug("Fetching funding rate history for {Symbol} (Kraken: {KrakenSymbol}) on {Exchange}",
                symbol, krakenSymbol, ExchangeName);

            // Fetch historical funding rates from Kraken Futures API
            var historicalRates = await _restClient.FuturesApi.ExchangeData.GetHistoricalFundingRatesAsync(krakenSymbol);

            if (!historicalRates.Success || historicalRates.Data == null || !historicalRates.Data.Any())
            {
                if (!historicalRates.Success)
                {
                    _logger.LogWarning("Failed to fetch funding rate history for {Symbol} on {Exchange}: {Error}",
                        symbol, ExchangeName, historicalRates.Error?.Message ?? "Unknown error");
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
                .Where(r => r.Timestamp >= threeDaysAgo)
                .OrderBy(r => r.Timestamp)
                .ToList();

            if (!filteredRates.Any())
            {
                _logger.LogWarning("No funding rates found within last 3 days for {Symbol} on {Exchange}",
                    symbol, ExchangeName);
                return results;
            }

            // Kraken typically uses 1-hour funding intervals for perpetual futures
            // Detect interval for EACH rate individually to handle any interval changes
            for (int i = 0; i < filteredRates.Count; i++)
            {
                var rate = filteredRates[i];
                int fundingIntervalHours = 1; // Default to 1h for Kraken

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

                // Determine direction
                var direction = rate.FundingRate < 0
                    ? FundingDirection.ShortPaysLong
                    : FundingDirection.LongPaysShort;

                results.Add(new FundingRateDto
                {
                    Exchange = ExchangeName,
                    Symbol = NormalizeKrakenSymbol(symbol), // Use the normalized input symbol
                    Rate = rate.FundingRate,
                    AnnualizedRate = annualizedRate,
                    FundingIntervalHours = fundingIntervalHours,
                    Direction = direction,
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

    public async Task<List<OrderDto>> GetOpenOrdersAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var orders = new List<OrderDto>();

        try
        {
            // Kraken Futures API: Get open orders
            var result = await _restClient.FuturesApi.Trading.GetOpenOrdersAsync();

            if (result.Success && result.Data != null)
            {
                orders = result.Data.Select(o => new OrderDto
                {
                    Exchange = ExchangeName,
                    OrderId = o.OrderId,
                    ClientOrderId = o.ClientOrderId,
                    Symbol = NormalizeKrakenSymbol(o.Symbol),
                    Side = o.Side == Kraken.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                    Type = MapKrakenFuturesOrderType(o.Type),
                    Status = MapKrakenOpenOrderStatus(o.Status),
                    TimeInForce = null, // Not available in Kraken open orders
                    Price = o.Price,
                    AveragePrice = o.QuantityFilled > 0
                        ? (o.Price.HasValue ? o.Price.Value : null)
                        : null,
                    StopPrice = o.StopPrice,
                    Quantity = o.Quantity,
                    FilledQuantity = o.QuantityFilled,
                    Fee = 0, // Fee not available in open orders
                    FeeAsset = null,
                    CreatedAt = o.Timestamp,
                    UpdatedAt = o.LastUpdateTime ?? o.Timestamp,
                    WorkingTime = null,
                    ReduceOnly = o.ReduceOnly.ToString(),
                    PostOnly = o.Type == Kraken.Net.Enums.FuturesOrderType.PostOnlyLimit ? "true" : null
                }).ToList();

                _logger.LogInformation("Fetched {Count} open orders from Kraken", orders.Count);
            }
            else
            {
                if (!result.Success)
                {
                    _logger.LogWarning("Failed to fetch open orders from Kraken: {Error}",
                        result.Error?.Message ?? "Unknown error");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching open orders from Kraken");
        }

        return orders;
    }

    public async Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var orders = new List<OrderDto>();

        try
        {
            _logger.LogDebug("Fetching order history from Kraken (startTime: {StartTime}, endTime: {EndTime}, limit: {Limit})",
                startTime, endTime, limit);

            // Note: Kraken Futures GetExecutionEventsAsync returns execution events with order information
            // This means we only get orders that had executions (fills)
            // Unfilled/canceled-before-fill orders won't appear
            // This is a limitation of the Kraken Futures API
            var result = await _restClient.FuturesApi.Trading.GetExecutionEventsAsync(
                startTime: startTime,
                endTime: endTime);

            if (!result.Success)
            {
                _logger.LogWarning("Failed to fetch execution events from Kraken: {Error}",
                    result.Error?.Message ?? "Unknown error");
                return orders;
            }

            if (result.Data == null || result.Data.Elements == null || !result.Data.Elements.Any())
            {
                _logger.LogDebug("No execution events found in Kraken order history");
                return orders;
            }

            _logger.LogDebug("Retrieved {Count} execution events from Kraken", result.Data.Elements.Length);

            // Extract orders from execution events
            // Each execution event may contain MakerOrder and/or TakerOrder
            // We need to deduplicate by order ID and keep the most recent state
            var orderDict = new Dictionary<string, (OrderDto Order, DateTime UpdateTime)>();

            foreach (var element in result.Data.Elements)
            {
                if (element.Event?.Execution?.Execution == null)
                    continue;

                var execution = element.Event.Execution.Execution;

                // Process maker order if present
                if (execution.MakerOrder != null)
                {
                    ProcessExecutionOrder(execution.MakerOrder, execution.MakerOrderData,
                        element.Timestamp, orderDict);
                }

                // Process taker order if present
                if (execution.TakerOrder != null)
                {
                    ProcessExecutionOrder(execution.TakerOrder, execution.TakerOrderData,
                        element.Timestamp, orderDict);
                }
            }

            // Convert dictionary to list and apply limit
            orders = orderDict.Values
                .OrderByDescending(o => o.UpdateTime)
                .Take(limit)
                .Select(o => o.Order)
                .ToList();

            _logger.LogInformation("Fetched {Count} unique orders from {EventCount} execution events on Kraken",
                orders.Count, result.Data.Elements.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching order history from Kraken");
        }

        return orders;
    }

    /// <summary>
    /// Processes an order from an execution event and updates the order dictionary
    /// </summary>
    private void ProcessExecutionOrder(
        KrakenFuturesExecutionOrder order,
        KrakenFuturesOrderData? orderData,
        DateTime eventTimestamp,
        Dictionary<string, (OrderDto Order, DateTime UpdateTime)> orderDict)
    {
        var orderId = order.OrderId;

        // Determine if this is a newer version of the order
        var updateTime = order.LastUpdateTime ?? eventTimestamp;

        if (orderDict.TryGetValue(orderId, out var existing))
        {
            // Only update if this event is newer
            if (updateTime <= existing.UpdateTime)
                return;
        }

        // Determine order status based on filled quantity
        var status = Models.OrderStatus.New;
        if (order.QuantityFilled >= order.Quantity)
        {
            status = Models.OrderStatus.Filled;
        }
        else if (order.QuantityFilled > 0)
        {
            status = Models.OrderStatus.PartiallyFilled;
        }

        var orderDto = new OrderDto
        {
            Exchange = ExchangeName,
            OrderId = order.OrderId,
            ClientOrderId = order.ClientOrderId,
            Symbol = NormalizeKrakenSymbol(order.Tradeable),
            Side = order.Side == Kraken.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
            Type = MapKrakenFuturesOrderType(order.Type),
            Status = status,
            TimeInForce = null, // Not available in execution events
            Price = order.Price,
            AveragePrice = order.QuantityFilled > 0 && order.Price.HasValue
                ? order.Price.Value
                : null,
            StopPrice = null, // Not available in execution events
            Quantity = order.Quantity,
            FilledQuantity = order.QuantityFilled,
            Fee = orderData?.Fee ?? 0,
            FeeAsset = null, // Fee asset not specified in execution events
            CreatedAt = order.Timestamp,
            UpdatedAt = updateTime,
            WorkingTime = null,
            ReduceOnly = order.ReduceOnly.ToString(),
            PostOnly = order.Type == Kraken.Net.Enums.FuturesOrderType.PostOnlyLimit ? "true" : null
        };

        orderDict[orderId] = (orderDto, updateTime);
    }

    public async Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var trades = new List<TradeDto>();

        try
        {
            _logger.LogDebug("Fetching user trades from Kraken (startTime: {StartTime}, endTime: {EndTime}, limit: {Limit})",
                startTime, endTime, limit);

            // Kraken Futures API supports time-based filtering directly
            var result = await _restClient.FuturesApi.Trading.GetUserTradesAsync(
                startTime: startTime);

            if (!result.Success)
            {
                _logger.LogWarning("Failed to fetch user trades from Kraken: {Error}",
                    result.Error?.Message ?? "Unknown error");
                return trades;
            }

            if (result.Data == null)
            {
                _logger.LogWarning("Kraken GetUserTradesAsync returned null data");
                return trades;
            }

            if (!result.Data.Any())
            {
                _logger.LogDebug("No trades found in Kraken trade history");
                return trades;
            }

            // Filter by endTime if specified (Kraken API only supports startTime filter)
            var filteredData = result.Data.AsEnumerable();
            if (endTime.HasValue)
            {
                filteredData = filteredData.Where(t => t.FillTime <= endTime.Value);
            }

            // Apply limit
            filteredData = filteredData.Take(limit);

            trades = filteredData.Select(t => new TradeDto
            {
                Exchange = ExchangeName,
                TradeId = t.Id,
                OrderId = t.OrderId,
                Symbol = NormalizeKrakenSymbol(t.Symbol),
                Side = t.Side == Kraken.Net.Enums.OrderSide.Buy ? Models.OrderSide.Buy : Models.OrderSide.Sell,
                Price = t.Price,
                Quantity = t.Quantity,
                QuoteQuantity = t.Price * t.Quantity, // Calculate quote quantity
                Fee = 0, // Fee information not available in Kraken Futures user trades
                FeeAsset = null,
                Commission = null,
                CommissionAsset = null,
                IsMaker = t.Type == Kraken.Net.Enums.TradeType.Maker,
                IsBuyer = t.Side == Kraken.Net.Enums.OrderSide.Buy,
                ExecutedAt = t.FillTime,
                OrderType = null, // Order type not available in trade data
                PositionSide = null // Position side not available in trade data
            }).ToList();

            _logger.LogInformation("Fetched {Count} trades from Kraken", trades.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching user trades from Kraken");
        }

        return trades;
    }

    public async Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Kraken");

        var transactions = new List<TransactionDto>();

        try
        {
            _logger.LogDebug("Fetching transactions from Kraken Spot (startTime: {StartTime}, endTime: {EndTime}, limit: {Limit})",
                startTime, endTime, limit);

            // Note: Using Kraken Spot API ledger for transaction history
            // This covers deposits, withdrawals, and transfers for Spot account
            // Futures transactions are not included
            // Filter for transaction-relevant ledger entry types
            var entryTypes = new[]
            {
                Kraken.Net.Enums.LedgerEntryType.Deposit,
                Kraken.Net.Enums.LedgerEntryType.Withdrawal,
                Kraken.Net.Enums.LedgerEntryType.Transfer
            };

            var result = await _restClient.SpotApi.Account.GetLedgerInfoAsync(
                assets: null, // All assets
                entryTypes: entryTypes,
                startTime: startTime,
                endTime: endTime,
                resultOffset: null);

            if (!result.Success)
            {
                _logger.LogWarning("Failed to fetch ledger info from Kraken Spot: {Error}",
                    result.Error?.Message ?? "Unknown error");
                return transactions;
            }

            if (result.Data?.Ledger == null || !result.Data.Ledger.Any())
            {
                _logger.LogDebug("No ledger entries found in Kraken transaction history");
                return transactions;
            }

            // Convert ledger entries to transactions
            transactions = result.Data.Ledger.Values
                .OrderByDescending(l => l.Timestamp)
                .Take(limit)
                .Select(l => new TransactionDto
                {
                    Exchange = ExchangeName,
                    TransactionId = l.Id,
                    TxHash = l.ReferenceId, // Note: This is Kraken's internal reference, not blockchain tx hash
                    Type = MapKrakenLedgerTypeToTransactionType(l.Type),
                    Asset = l.Asset,
                    Amount = l.Quantity,
                    Status = TransactionStatus.Completed, // Ledger only shows completed transactions
                    FromAddress = null, // Not available in Kraken Spot API ledger
                    ToAddress = null, // Not available in Kraken Spot API ledger
                    Network = null, // Not available in Kraken Spot API ledger
                    Info = l.SubType,
                    Symbol = null, // Not available in ledger entries
                    TradeId = l.Type == Kraken.Net.Enums.LedgerEntryType.Trade ? l.ReferenceId : null,
                    Fee = l.Fee,
                    FeeAsset = l.Asset, // Assume fee is in the same asset
                    CreatedAt = l.Timestamp,
                    ConfirmedAt = l.Timestamp // No separate confirmation time in ledger
                })
                .ToList();

            _logger.LogInformation("Fetched {Count} transactions from Kraken Spot ledger", transactions.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching transactions from Kraken");
        }

        return transactions;
    }

    /// <summary>
    /// Maps Kraken ledger entry type to application transaction type
    /// </summary>
    private Models.TransactionType MapKrakenLedgerTypeToTransactionType(Kraken.Net.Enums.LedgerEntryType ledgerType)
    {
        return ledgerType switch
        {
            Kraken.Net.Enums.LedgerEntryType.Deposit => Models.TransactionType.Deposit,
            Kraken.Net.Enums.LedgerEntryType.Withdrawal => Models.TransactionType.Withdrawal,
            Kraken.Net.Enums.LedgerEntryType.Transfer => Models.TransactionType.InternalTransfer,
            Kraken.Net.Enums.LedgerEntryType.Trade => Models.TransactionType.Trade,
            Kraken.Net.Enums.LedgerEntryType.Margin => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Rollover => Models.TransactionType.FundingFee,
            Kraken.Net.Enums.LedgerEntryType.Staking => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Reward => Models.TransactionType.Rebate,
            Kraken.Net.Enums.LedgerEntryType.Dividend => Models.TransactionType.Rebate,
            Kraken.Net.Enums.LedgerEntryType.Credit => Models.TransactionType.Bonus,
            Kraken.Net.Enums.LedgerEntryType.Adjustment => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Settled => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Spend => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Receive => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.Sale => Models.TransactionType.Trade,
            Kraken.Net.Enums.LedgerEntryType.Conversion => Models.TransactionType.Other,
            Kraken.Net.Enums.LedgerEntryType.NftTrade => Models.TransactionType.Trade,
            Kraken.Net.Enums.LedgerEntryType.NftCreatorFee => Models.TransactionType.Commission,
            Kraken.Net.Enums.LedgerEntryType.NftRebate => Models.TransactionType.Rebate,
            Kraken.Net.Enums.LedgerEntryType.CustodyTransfer => Models.TransactionType.InternalTransfer,
            _ => Models.TransactionType.Other
        };
    }

    /// <summary>
    /// Normalizes Kraken symbol format from "BTC:USD" to "BTCUSDT"
    /// </summary>
    /// <param name="krakenSymbol">Kraken symbol in format "BASE:QUOTE"</param>
    /// <returns>Normalized symbol without separator</returns>
    private string NormalizeKrakenSymbol(string krakenSymbol)
    {
        krakenSymbol = krakenSymbol.Replace(":", "");

        if (krakenSymbol.EndsWith("USD"))
        {
            krakenSymbol = $"{krakenSymbol}T";
        }

        return krakenSymbol;
    }

    public async Task<List<KlineDto>> GetKlinesAsync(string symbol, DateTime startTime, DateTime endTime, KlineInterval interval)
    {
        if (_restClient == null)
        {
            _logger.LogWarning("Cannot get klines for {Exchange} - not connected", ExchangeName);
            return new List<KlineDto>();
        }

        try
        {
            // Convert our KlineInterval enum to Kraken futures interval
            var krakenInterval = interval switch
            {
                KlineInterval.OneMinute => Kraken.Net.Enums.FuturesKlineInterval.OneMinute,
                KlineInterval.FiveMinutes => Kraken.Net.Enums.FuturesKlineInterval.FiveMinutes,
                KlineInterval.FifteenMinutes => Kraken.Net.Enums.FuturesKlineInterval.FifteenMinutes,
                KlineInterval.ThirtyMinutes => Kraken.Net.Enums.FuturesKlineInterval.ThirtyMinutes,
                KlineInterval.OneHour => Kraken.Net.Enums.FuturesKlineInterval.OneHour,
                KlineInterval.FourHours => Kraken.Net.Enums.FuturesKlineInterval.FourHours,
                KlineInterval.OneDay => Kraken.Net.Enums.FuturesKlineInterval.OneDay,
                _ => Kraken.Net.Enums.FuturesKlineInterval.OneHour
            };

            _logger.LogDebug("Fetching klines for {Symbol} on {Exchange} from {StartTime} to {EndTime}",
                symbol, ExchangeName, startTime, endTime);

            if (symbol.StartsWith("PF_") == false)
            {
                symbol = $"PF_{symbol}";
            }

            if (symbol.EndsWith("USDT"))
            {
                symbol = symbol[..^1];
            }

            // Fetch klines from Kraken Futures API
            // Using Trade tick type for actual trade prices
            var klines = await _restClient.FuturesApi.ExchangeData.GetKlinesAsync(
                Kraken.Net.Enums.TickType.Trade,
                symbol,
                krakenInterval,
                startTime,
                endTime);

            if (!klines.Success || klines.Data == null || klines.Data.Klines == null || !klines.Data.Klines.Any())
            {
                if (!klines.Success)
                {
                    _logger.LogWarning("Failed to fetch klines for {Symbol} on {Exchange}: {Error}",
                        symbol, ExchangeName, klines.Error?.Message ?? "Unknown error");
                }
                return new List<KlineDto>();
            }

            var results = klines.Data.Klines.Select(k => new KlineDto
            {
                Exchange = ExchangeName,
                Symbol = symbol,
                OpenTime = k.Timestamp,
                CloseTime = k.Timestamp, // Kraken returns timestamp for each candle
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

    /// <summary>
    /// Maps Kraken Futures order type to application OrderType enum
    /// </summary>
    private Models.OrderType MapKrakenFuturesOrderType(Kraken.Net.Enums.FuturesOrderType krakenType)
    {
        return krakenType switch
        {
            Kraken.Net.Enums.FuturesOrderType.Market => Models.OrderType.Market,
            Kraken.Net.Enums.FuturesOrderType.Limit => Models.OrderType.Limit,
            Kraken.Net.Enums.FuturesOrderType.PostOnlyLimit => Models.OrderType.Limit,
            Kraken.Net.Enums.FuturesOrderType.ImmediateOrCancel => Models.OrderType.Market,
            Kraken.Net.Enums.FuturesOrderType.Stop => Models.OrderType.StopMarket,
            Kraken.Net.Enums.FuturesOrderType.TakeProfit => Models.OrderType.TakeProfitMarket,
            _ => Models.OrderType.Market
        };
    }

    /// <summary>
    /// Maps Kraken open order status to application OrderStatus enum
    /// </summary>
    private Models.OrderStatus MapKrakenOpenOrderStatus(Kraken.Net.Enums.OpenOrderStatus krakenStatus)
    {
        return krakenStatus switch
        {
            Kraken.Net.Enums.OpenOrderStatus.Untouched => Models.OrderStatus.New,
            Kraken.Net.Enums.OpenOrderStatus.PartiallyFilled => Models.OrderStatus.PartiallyFilled,
            _ => Models.OrderStatus.New
        };
    }
}
