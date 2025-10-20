using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public enum ArbitrageStrategy
{
    CrossExchange,      // Arbitrage between two exchanges
    SpotPerpetual      // Cash-and-carry arbitrage (spot vs perpetual on same exchange)
}

public enum StrategySubType
{
    SpotPerpetualSameExchange,       // Buy spot + short perp on same exchange
    CrossExchangeFuturesFutures,     // Long perp on one exchange + short perp on another
    CrossExchangeSpotFutures         // Buy spot on one exchange + short perp on another
}

public class ArbitrageOpportunityDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public ArbitrageStrategy Strategy { get; set; } = ArbitrageStrategy.CrossExchange;
    public StrategySubType SubType { get; set; } = StrategySubType.CrossExchangeFuturesFutures;

    // For cross-exchange arbitrage
    public string LongExchange { get; set; } = string.Empty;
    public string ShortExchange { get; set; } = string.Empty;
    public decimal LongFundingRate { get; set; }
    public decimal ShortFundingRate { get; set; }
    public int? LongFundingIntervalHours { get; set; }  // Funding interval for long exchange (1h, 4h, 8h, etc.)
    public int? ShortFundingIntervalHours { get; set; } // Funding interval for short exchange

    // For spot-perpetual arbitrage
    public string Exchange { get; set; } = string.Empty;
    public decimal SpotPrice { get; set; }
    public decimal PerpetualPrice { get; set; }
    public decimal FundingRate { get; set; }
    public decimal AnnualizedFundingRate { get; set; }
    public decimal PricePremium { get; set; } // (Perp - Spot) / Spot

    // Common fields
    public decimal SpreadRate { get; set; }
    public decimal AnnualizedSpread { get; set; }
    public decimal EstimatedProfitPercentage { get; set; }
    public decimal Volume24h { get; set; }  // 24-hour trading volume in USDT

    // Liquidity metrics
    public decimal? BidAskSpreadPercent { get; set; }
    public decimal? OrderbookDepthUsd { get; set; }
    public LiquidityStatus? LiquidityStatus { get; set; }
    public string? LiquidityWarning { get; set; }

    public OpportunityStatus Status { get; set; }
    public DateTime DetectedAt { get; set; }
    public DateTime? ExecutedAt { get; set; }

    // Execution fields (merged from Execution table)
    public int? ExecutionId { get; set; }
    public ExecutionState? ExecutionState { get; set; }
    public DateTime? ExecutionStartedAt { get; set; }
    public decimal? ExecutionFundingEarned { get; set; }

    // Computed unique key for frontend tracking (not stored in DB)
    public string UniqueKey => Strategy == ArbitrageStrategy.SpotPerpetual
        ? $"{Symbol}-{Exchange}-SpotPerp"
        : $"{Symbol}-{LongExchange}-{ShortExchange}-CrossEx";
}
