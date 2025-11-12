using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public class ExecuteOpportunityRequest
{
    public required string Symbol { get; set; }
    public ArbitrageStrategy Strategy { get; set; }
    public StrategySubType SubType { get; set; } = StrategySubType.SpotPerpetualSameExchange;
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

    // Funding interval information (needed for accurate APR calculations)
    public decimal? FundingIntervalHours { get; set; } // For SpotPerpetual: perp funding interval. Default: 8h
    public decimal? LongFundingIntervalHours { get; set; } // For CrossExchange: long side interval. Default: 8h
    public decimal? ShortFundingIntervalHours { get; set; } // For CrossExchange: short side interval. Default: 8h

    // Pre-calculated FundApr from opportunity (ensures consistency with opportunity APR calculation)
    public decimal? FundApr { get; set; } // Annualized funding rate difference (correctly calculated)
}
