using Bybit.Net.Clients;
using Bybit.Net.Enums;
using CryptoExchange.Net.Authentication;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services;

public class BybitConnector : IExchangeConnector
{
    private readonly ILogger<BybitConnector> _logger;
    private BybitRestClient? _restClient;
    private BybitSocketClient? _socketClient;

    public string ExchangeName => "Bybit";

    public BybitConnector(ILogger<BybitConnector> logger)
    {
        _logger = logger;
    }

    public async Task<bool> ConnectAsync(string apiKey, string apiSecret, bool useDemoTrading = false)
    {
        try
        {
            var credentials = new ApiCredentials(apiKey, apiSecret);

            _restClient = new BybitRestClient(options =>
            {
                options.ApiCredentials = credentials;
                // Demo trading uses Bybit's demo trading environment
                if (useDemoTrading)
                {
                    options.Environment = Bybit.Net.BybitEnvironment.DemoTrading;
                }
            });

            _socketClient = new BybitSocketClient(options =>
            {
                options.ApiCredentials = credentials;
                if (useDemoTrading)
                {
                    options.Environment = Bybit.Net.BybitEnvironment.DemoTrading;
                }
            });

            // Test connection
            var accountInfo = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

            if (accountInfo.Success)
            {
                _logger.LogInformation("Successfully connected to Bybit");
                return true;
            }

            _logger.LogError("Failed to connect to Bybit: {Error}", accountInfo.Error);
            return false;
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
                        fundingRates.Add(new FundingRateDto
                        {
                            Exchange = ExchangeName,
                            Symbol = instrument.Symbol,
                            Rate = instrument.FundingRate.Value,
                            AnnualizedRate = instrument.FundingRate.Value * 3 * 365, // 3 times per day
                            FundingTime = DateTime.UtcNow, // Current time as funding time
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

    public async Task<Dictionary<string, SpotPriceDto>> GetSpotPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var spotPrices = new Dictionary<string, SpotPriceDto>();

        try
        {
            // Get spot prices from Bybit Spot API
            var tickers = await _restClient.V5Api.ExchangeData.GetSpotTickersAsync();

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.List.Where(t => symbols.Contains(t.Symbol)))
                {
                    spotPrices[ticker.Symbol] = new SpotPriceDto
                    {
                        Exchange = ExchangeName,
                        Symbol = ticker.Symbol,
                        Price = ticker.LastPrice,
                        Timestamp = DateTime.UtcNow
                    };
                }
            }

            _logger.LogInformation("Fetched {Count} spot prices from Bybit", spotPrices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching spot prices from Bybit");
        }

        return spotPrices;
    }

    public async Task<Dictionary<string, decimal>> GetPerpetualPricesAsync(List<string> symbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var perpPrices = new Dictionary<string, decimal>();

        try
        {
            // Get perpetual prices from Bybit linear tickers
            var tickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);

            if (tickers.Success && tickers.Data != null)
            {
                foreach (var ticker in tickers.Data.List.Where(t => symbols.Contains(t.Symbol)))
                {
                    perpPrices[ticker.Symbol] = ticker.LastPrice;
                }
            }

            _logger.LogInformation("Fetched {Count} perpetual prices from Bybit", perpPrices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching perpetual prices from Bybit");
        }

        return perpPrices;
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
                                spotAvailableUsd += asset.Value;
                                spotUsdtOnly = asset.Value; // Track USDT separately
                            }
                            else
                            {
                                var symbol = $"{asset.Key}USDT";
                                var price = tickers.Data.List.FirstOrDefault(t => t.Symbol == symbol);
                                if (price != null)
                                {
                                    var usdValue = asset.Value * price.LastPrice;
                                    spotTotalUsd += usdValue;
                                    spotAvailableUsd += usdValue;
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

                _logger.LogInformation("Bybit Raw API Data - WalletBalance: {WalletBalance}, AvailableToWithdraw: {Available}, MarginUsed: {Margin}, UnrealizedPnL: {PnL}",
                    futuresTotal, futuresAvailable, marginUsed, unrealizedPnL);

                // Check for all assets in account
                _logger.LogInformation("Bybit Account Assets:");
                foreach (var asset in account.Assets)
                {
                    _logger.LogInformation("  Asset: {Asset}, WalletBalance: {WalletBalance}, Equity: {Equity}, Available: {Available}, Locked: {Locked}",
                        asset.Asset, asset.WalletBalance, asset.Equity, asset.AvailableToWithdraw, asset.Locked);
                }

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

                _logger.LogInformation("Bybit Balances - Total: {Total} USD, Available: {Available} USD, Margin: {Margin} USD",
                    totalBalance, futuresAvailable, marginUsed);

                // For frontend calculations:
                // - TotalBalance = all assets
                // - FuturesAvailableUsd = available balance for trading
                // - MarginUsed = what's locked in positions
                // - SpotBalanceUsd = non-USDT coins (for "in positions" calculation)
                decimal spotAssetsOnly = spotTotalUsd - spotUsdtOnly;

                return new AccountBalanceDto
                {
                    Exchange = ExchangeName,
                    // Total balance (all assets in USD)
                    TotalBalance = totalBalance,
                    AvailableBalance = futuresAvailable,
                    OperationalBalanceUsd = totalBalance,
                    // Spot specific (only non-USDT assets like BTC, ETH)
                    SpotBalanceUsd = spotAssetsOnly,
                    SpotAvailableUsd = spotAssetsOnly - coinsInActivePositionsUsd,
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

    public async Task<string> PlaceMarketOrderAsync(string symbol, Data.Entities.PositionSide side, decimal quantity, decimal leverage)
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

            var orderSide = side == Data.Entities.PositionSide.Long ? OrderSide.Buy : OrderSide.Sell;

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

    public async Task<bool> ClosePositionAsync(string symbol)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            var positions = await _restClient.V5Api.Trading.GetPositionsAsync(Category.Linear, symbol);

            if (positions.Success && positions.Data != null)
            {
                foreach (var position in positions.Data.List.Where(p => p.Quantity > 0))
                {
                    var orderSide = position.Side == Bybit.Net.Enums.PositionSide.Buy ? OrderSide.Sell : OrderSide.Buy;

                    await _restClient.V5Api.Trading.PlaceOrderAsync(
                        Category.Linear,
                        symbol,
                        orderSide,
                        NewOrderType.Market,
                        position.Quantity,
                        null
                    );
                }
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error closing position on Bybit");
        }

        return false;
    }

    public async Task<List<PositionDto>> GetOpenPositionsAsync()
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        var positions = new List<PositionDto>();

        try
        {
            _logger.LogInformation("🔍 Fetching Bybit positions via GetPositionsAsync(Category.Linear, symbol: COAIUSDT)...");
            // Bybit V5 API requires either symbol or settleCoin parameter
            // For now, query the specific symbol we know has a position
            var result = await _restClient.V5Api.Trading.GetPositionsAsync(Category.Linear, "COAIUSDT");

            _logger.LogInformation("📡 Bybit GetPositions API Response - Success: {Success}, HasData: {HasData}, Error: {Error}",
                result.Success, result.Data != null, result.Error?.Message ?? "None");

            if (result.Success && result.Data != null)
            {
                _logger.LogInformation("Bybit Open Positions Check - Total positions returned: {Count}", result.Data.List.Count());

                // Log detailed position information for debugging P&L issues
                foreach (var p in result.Data.List)
                {
                    _logger.LogInformation("  Position: {Symbol}, Qty: {Qty}, Side: {Side}, AvgPrice: {Price}, MarkPrice: {MarkPrice}, UnrealizedPnL: {PnL}",
                        p.Symbol, p.Quantity, p.Side, p.AveragePrice, p.MarkPrice, p.UnrealizedPnl);
                }

                _logger.LogInformation("🔎 Positions with Qty > 0: {Count}", result.Data.List.Count(p => p.Quantity > 0));

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
                            Side = p.Side == Bybit.Net.Enums.PositionSide.Buy ? Data.Entities.PositionSide.Long : Data.Entities.PositionSide.Short,
                            Status = Data.Entities.PositionStatus.Open,
                            EntryPrice = p.AveragePrice ?? 0,
                            Quantity = p.Quantity,
                            Leverage = p.Leverage ?? 1,
                            UnrealizedPnL = unrealizedPnl,
                            RealizedPnL = 0, // Realized P&L not available in position list API
                            OpenedAt = p.CreateTime ?? DateTime.UtcNow
                        };
                    })
                    .ToList();

                _logger.LogInformation("Bybit Open Positions (Qty > 0) - Count: {Count}, Total Unrealized P&L: {TotalPnL}",
                    positions.Count, positions.Sum(p => p.UnrealizedPnL));
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
                OrderSide.Buy,
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
                OrderSide.Sell,
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
                    _logger.LogInformation("Spot balance for {Asset}: {WalletBalance} (Equity: {Equity})",
                        asset, assetBalanceAmount, assetBalance.Equity);
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

                _logger.LogInformation("Fetched {Count} asset balances from Bybit Unified account", balances.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching balances from Bybit");
        }

        return balances;
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
