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

    public async Task<bool> ConnectAsync(string apiKey, string apiSecret)
    {
        try
        {
            var credentials = new ApiCredentials(apiKey, apiSecret);

            _restClient = new BybitRestClient(options =>
            {
                options.ApiCredentials = credentials;
            });

            _socketClient = new BybitSocketClient(options =>
            {
                options.ApiCredentials = credentials;
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

    public async Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            // Get all linear perpetual tickers
            var tickers = await _restClient.V5Api.ExchangeData.GetLinearInverseTickersAsync(Category.Linear);
            if (!tickers.Success || tickers.Data == null)
            {
                _logger.LogError("Failed to get tickers from Bybit");
                return new List<string>();
            }

            // Filter for USDT perpetuals with sufficient volume
            var symbolsWithVolume = tickers.Data.List
                .Where(t => t.Symbol.EndsWith("USDT") && t.Volume24h > 0 && t.LastPrice > 0)
                .Select(t => new
                {
                    Symbol = t.Symbol,
                    // Volume24h is in contracts, multiply by price to get USD volume
                    VolumeUsd = t.Volume24h * t.LastPrice
                })
                .Where(s => s.VolumeUsd >= minDailyVolumeUsd)
                .OrderByDescending(s => s.VolumeUsd)
                .Take(maxSymbols)
                .Select(s => s.Symbol)
                .ToList();

            _logger.LogInformation(
                "Discovered {Count} active symbols from Bybit (min volume: ${MinVolume:N0}, max symbols: {MaxSymbols})",
                symbolsWithVolume.Count,
                minDailyVolumeUsd,
                maxSymbols
            );

            return symbolsWithVolume;
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
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            var balance = await _restClient.V5Api.Account.GetBalancesAsync(AccountType.Unified);

            if (balance.Success && balance.Data != null && balance.Data.List.Any())
            {
                var usdtBalance = balance.Data.List.First().Assets.FirstOrDefault(a => a.Asset == "USDT");

                if (usdtBalance != null)
                {
                    return new AccountBalanceDto
                    {
                        Exchange = ExchangeName,
                        TotalBalance = usdtBalance.Equity ?? 0,
                        AvailableBalance = usdtBalance.AvailableToWithdraw ?? 0,
                        MarginUsed = (usdtBalance.Equity ?? 0) - (usdtBalance.AvailableToWithdraw ?? 0),
                        UnrealizedPnL = usdtBalance.UnrealizedPnl ?? 0,
                        UpdatedAt = DateTime.UtcNow
                    };
                }
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
                _logger.LogInformation("Order placed on Bybit: {OrderId}", order.Data.OrderId);
                return order.Data.OrderId;
            }

            _logger.LogError("Failed to place order on Bybit: {Error}", order.Error);
            throw new Exception($"Failed to place order: {order.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order on Bybit");
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
            var result = await _restClient.V5Api.Trading.GetPositionsAsync(Category.Linear);

            if (result.Success && result.Data != null)
            {
                positions = result.Data.List
                    .Where(p => p.Quantity > 0)
                    .Select(p => new PositionDto
                    {
                        Exchange = ExchangeName,
                        Symbol = p.Symbol,
                        Side = p.Side == Bybit.Net.Enums.PositionSide.Buy ? Data.Entities.PositionSide.Long : Data.Entities.PositionSide.Short,
                        Status = Data.Entities.PositionStatus.Open,
                        EntryPrice = p.AveragePrice ?? 0,
                        Quantity = p.Quantity,
                        Leverage = p.Leverage ?? 1,
                        UnrealizedPnL = p.UnrealizedPnl ?? 0,
                        OpenedAt = p.CreateTime ?? DateTime.UtcNow
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
    public async Task<string> PlaceSpotBuyOrderAsync(string symbol, decimal quantity)
    {
        if (_restClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        try
        {
            _logger.LogInformation("Placing spot BUY order on Bybit: {Symbol}, Quantity: {Quantity}",
                symbol, quantity);

            var order = await _restClient.V5Api.Trading.PlaceOrderAsync(
                Category.Spot,
                symbol,
                OrderSide.Buy,
                NewOrderType.Market,
                quantity
            );

            if (order.Success && order.Data != null)
            {
                _logger.LogInformation("Spot BUY order placed on Bybit: {OrderId}", order.Data.OrderId);
                return order.Data.OrderId;
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
                    _logger.LogInformation("Spot balance for {Asset}: {Available} (Total: {Total})",
                        asset, assetBalance.AvailableToWithdraw, assetBalance.Equity);
                    return assetBalance.AvailableToWithdraw ?? 0;
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

    public async Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate)
    {
        if (_socketClient == null)
            throw new InvalidOperationException("Not connected to Bybit");

        // Subscribe to ticker updates which include funding rates
        await Task.CompletedTask;
    }
}
