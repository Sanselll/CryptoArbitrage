using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services.Exchanges;

/// <summary>
/// Result of closing a position, including exit price information
/// </summary>
public class ClosePositionResult
{
    public bool Success { get; set; }
    public string? OrderId { get; set; }
    public decimal? ExitPrice { get; set; }
    public decimal? FilledQuantity { get; set; }
    public decimal? TradingFee { get; set; }
    public string? ErrorMessage { get; set; }
}

public interface IExchangeConnector
{
    string ExchangeName { get; }
    Task<bool> ConnectAsync(string apiKey, string apiSecret);
    Task DisconnectAsync();
    Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols, decimal minHighPriorityFundingRate = 0);
    Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols);
    Task<Dictionary<string, PriceDto>> GetSpotPricesAsync(List<string> symbols);
    Task<Dictionary<string, PriceDto>> GetPerpetualPricesAsync(List<string> symbols);
    Task<LiquidityMetricsDto?> GetLiquidityMetricsAsync(string symbol);
    Task<AccountBalanceDto> GetAccountBalanceAsync();
    Task<AccountBalanceDto> GetAccountBalanceAsync(Dictionary<string, decimal> activeSpotPositions);
    Task<Dictionary<string, decimal>> GetSpotBalancesAsync();
    Task<FeeInfoDto> GetTradingFeesAsync();

    // Perpetual futures trading
    Task<string> PlaceMarketOrderAsync(string symbol, PositionSide side, decimal quantity, decimal leverage);
    Task<ClosePositionResult> ClosePositionAsync(string symbol);
    Task<List<PositionDto>> GetOpenPositionsAsync();

    // Spot trading (for cash-and-carry arbitrage)
    Task<(string orderId, decimal filledQuantity)> PlaceSpotBuyOrderAsync(string symbol, decimal quantity);
    Task<string> PlaceSpotSellOrderAsync(string symbol, decimal quantity);
    Task<decimal> GetSpotBalanceAsync(string asset);

    Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate);
    Task<List<FundingRateDto>> GetFundingRateHistoryAsync(string symbol, DateTime startTime, DateTime endTime);

    // Historical price data
    Task<List<KlineDto>> GetKlinesAsync(string symbol, DateTime startTime, DateTime endTime, KlineInterval interval);

    // Trading data methods
    Task<List<OrderDto>> GetOpenOrdersAsync();
    Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100);
    Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100);
    Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100);
}
