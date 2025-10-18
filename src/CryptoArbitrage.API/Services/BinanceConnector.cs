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

namespace CryptoArbitrage.API.Services;

public class BinanceConnector : IExchangeConnector
{
    private readonly ILogger<BinanceConnector> _logger;
    private BinanceRestClient? _restClient;
    private BinanceSocketClient? _socketClient;

    public string ExchangeName => "Binance";

    public BinanceConnector(ILogger<BinanceConnector> logger)
    {
        _logger = logger;
    }

    public async Task<bool> ConnectAsync(string apiKey, string apiSecret, bool useDemoTrading = false)
    {
        try
        {
            // Check if API credentials are provided
            bool hasCredentials = !string.IsNullOrWhiteSpace(apiKey) && !string.IsNullOrWhiteSpace(apiSecret);

            if (hasCredentials)
            {
                var credentials = new ApiCredentials(apiKey, apiSecret);
                var environment = useDemoTrading ? BinanceEnvironment.Demo : BinanceEnvironment.Live;

                _restClient = new BinanceRestClient(options =>
                {
                    options.ApiCredentials = credentials;
                    options.Environment = environment;
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
                    _logger.LogInformation("Successfully connected to Binance with API credentials");
                    return true;
                }

                _logger.LogError("Failed to connect to Binance: {Error}", accountInfo.Error);
                return false;
            }
            else
            {
                // Create client without credentials for public data only
                _restClient = new BinanceRestClient(options =>
                {
                    // No credentials - public data only
                    options.Environment = BinanceEnvironment.Live; // Use live for public data
                });

                _socketClient = new BinanceSocketClient(options =>
                {
                    options.Environment = BinanceEnvironment.Live;
                });

                // Test connection with public endpoint
                var exchangeInfo = await _restClient.UsdFuturesApi.ExchangeData.GetExchangeInfoAsync();

                if (exchangeInfo.Success)
                {
                    _logger.LogInformation("Successfully connected to Binance (public data only - no API credentials)");
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

    public async Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols, decimal minHighPriorityFundingRate = 0)
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
            // Use premium index to get current funding rate (shown on Binance website)
            // This is the rate that will be applied at the next funding time
            var premiumIndex = await _restClient.UsdFuturesApi.ExchangeData.GetMarkPricesAsync();

            if (premiumIndex.Success && premiumIndex.Data != null)
            {
                var symbolsToFetch = symbols.Count > 0 ? symbols : new List<string> { "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT" };

                foreach (var symbol in symbolsToFetch)
                {
                    var markPrice = premiumIndex.Data.FirstOrDefault(p => p.Symbol == symbol);
                    if (markPrice != null)
                    {
                        fundingRates.Add(new FundingRateDto
                        {
                            Exchange = ExchangeName,
                            Symbol = markPrice.Symbol,
                            Rate = markPrice.FundingRate ?? 0, // Current predicted funding rate for next funding time
                            AnnualizedRate = (markPrice.FundingRate ?? 0) * 3 * 365, // 3 times per day
                            FundingTime = DateTime.UtcNow,
                            NextFundingTime = markPrice.NextFundingTime,
                            RecordedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching funding rates from Binance");
        }

        return fundingRates;
    }

    public async Task<Dictionary<string, SpotPriceDto>> GetSpotPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var spotPrices = new Dictionary<string, SpotPriceDto>();

        try
        {
            // Get spot prices from Binance Spot API
            var prices = await _restClient.SpotApi.ExchangeData.GetPricesAsync();

            if (prices.Success && prices.Data != null)
            {
                foreach (var price in prices.Data.Where(p => symbols.Contains(p.Symbol)))
                {
                    spotPrices[price.Symbol] = new SpotPriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = price.Symbol,
                        Price = price.Price,
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

            var spotSymbolsList = string.Join(", ", spotPrices.Keys.OrderBy(s => s));
            _logger.LogInformation("Fetched {Count} spot prices from Binance: {Symbols}", spotPrices.Count, spotSymbolsList);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot prices from Binance");
        }

        return spotPrices;
    }

    public async Task<Dictionary<string, decimal>> GetPerpetualPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Binance");

        var perpPrices = new Dictionary<string, decimal>();

        try
        {
            // Get perpetual futures prices from Binance USD-M Futures API
            var prices = await _restClient.UsdFuturesApi.ExchangeData.GetPricesAsync();

            if (prices.Success && prices.Data != null)
            {
                foreach (var price in prices.Data.Where(p => symbols.Contains(p.Symbol)))
                {
                    perpPrices[price.Symbol] = price.Price;
                }
            }

            _logger.LogInformation("Fetched {Count} perpetual prices from Binance", perpPrices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching perpetual prices from Binance");
        }

        return perpPrices;
    }

    public async Task<AccountBalanceDto> GetAccountBalanceAsync()
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
            decimal spotUsdtOnly = 0; // Only USDT for operational balance

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
                            spotAvailableUsd += asset.Value;
                            spotUsdtOnly = asset.Value; // Track USDT separately for operational balance
                        }
                        else
                        {
                            // Find price in USDT
                            var symbol = $"{asset.Key}USDT";
                            var price = prices.Data.FirstOrDefault(p => p.Symbol == symbol);
                            if (price != null)
                            {
                                var usdValue = asset.Value * price.Price;
                                spotTotalUsd += usdValue;
                                spotAvailableUsd += usdValue;
                            }
                        }
                    }
                }
            }

            // Calculate operational balance: USDT + coins in active positions + futures
            // In cash-and-carry arbitrage, coins (BTC, ETH, etc.) ARE the positions
            decimal operationalBalance = spotTotalUsd + futuresTotal;

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

        return new AccountBalanceDto { Exchange = ExchangeName };
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
                            spotAvailableUsd += asset.Value;
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
                                spotTotalUsd += usdValue;
                                spotAvailableUsd += usdValue;
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

        return new AccountBalanceDto { Exchange = ExchangeName };
    }

    public async Task<string> PlaceMarketOrderAsync(string symbol, Data.Entities.PositionSide side, decimal quantity, decimal leverage)
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

            // Set leverage first
            await _restClient.UsdFuturesApi.Account.ChangeInitialLeverageAsync(symbol, (int)leverage);

            var orderSide = side == Data.Entities.PositionSide.Long ? OrderSide.Buy : OrderSide.Sell;
            var positionSide = side == Data.Entities.PositionSide.Long ? Binance.Net.Enums.PositionSide.Long : Binance.Net.Enums.PositionSide.Short;

            var order = await _restClient.UsdFuturesApi.Trading.PlaceOrderAsync(
                symbol,
                orderSide,
                FuturesOrderType.Market,
                quantity,
                positionSide: positionSide
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Order placed on Binance: {OrderId}", order.Data.Id);
                return order.Data.Id.ToString();
            }

            _logger.LogError("Failed to place order on Binance: {Error}", order.Error);
            throw new Exception($"Failed to place order: {order.Error}");
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
                    var orderSide = position.Quantity > 0 ? OrderSide.Sell : OrderSide.Buy;
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
                        Side = p.Quantity > 0 ? Data.Entities.PositionSide.Long : Data.Entities.PositionSide.Short,
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

                _logger.LogInformation("Spot BUY order placed on Binance: {OrderId}, Requested: {Requested}, Filled: {Filled}",
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

                _logger.LogInformation("Fetched {Count} spot asset balances from Binance", balances.Count);
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
}
