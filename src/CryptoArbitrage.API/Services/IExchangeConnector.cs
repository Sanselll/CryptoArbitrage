using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services;

public interface IExchangeConnector
{
    string ExchangeName { get; }
    Task<bool> ConnectAsync(string apiKey, string apiSecret);
    Task DisconnectAsync();
    Task<List<string>> GetActiveSymbolsAsync(decimal minDailyVolumeUsd, int maxSymbols);
    Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols);
    Task<Dictionary<string, SpotPriceDto>> GetSpotPricesAsync(List<string> symbols);
    Task<Dictionary<string, decimal>> GetPerpetualPricesAsync(List<string> symbols);
    Task<AccountBalanceDto> GetAccountBalanceAsync();
    Task<AccountBalanceDto> GetAccountBalanceAsync(Dictionary<string, decimal> activeSpotPositions);
    Task<Dictionary<string, decimal>> GetSpotBalancesAsync();

    // Perpetual futures trading
    Task<string> PlaceMarketOrderAsync(string symbol, Data.Entities.PositionSide side, decimal quantity, decimal leverage);
    Task<bool> ClosePositionAsync(string symbol);
    Task<List<PositionDto>> GetOpenPositionsAsync();

    // Spot trading (for cash-and-carry arbitrage)
    Task<string> PlaceSpotBuyOrderAsync(string symbol, decimal quantity);
    Task<string> PlaceSpotSellOrderAsync(string symbol, decimal quantity);
    Task<decimal> GetSpotBalanceAsync(string asset);

    Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate);
}
