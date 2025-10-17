using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public class ExecuteOpportunityRequest
{
    public required string Symbol { get; set; }
    public ArbitrageStrategy Strategy { get; set; }
    public required string Exchange { get; set; }
    public string? LongExchange { get; set; } // For cross-exchange arbitrage
    public string? ShortExchange { get; set; } // For cross-exchange arbitrage
    public decimal PositionSizeUsd { get; set; }
    public decimal Leverage { get; set; } = 1m;
    public decimal? StopLossPercentage { get; set; }
    public decimal? TakeProfitPercentage { get; set; }

    // Funding rate information for creating ArbitrageOpportunity record
    public decimal FundingRate { get; set; } // For SpotPerpetual: perp funding rate. For CrossExchange: not used directly
    public decimal? LongFundingRate { get; set; } // For CrossExchange arbitrage
    public decimal? ShortFundingRate { get; set; } // For CrossExchange arbitrage
    public decimal SpreadRate { get; set; } // The actual spread/difference
    public decimal AnnualizedSpread { get; set; } // Annualized spread percentage
    public decimal EstimatedProfitPercentage { get; set; } // Estimated profit %
}
