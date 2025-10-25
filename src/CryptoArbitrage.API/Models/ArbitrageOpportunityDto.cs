using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public enum ArbitrageStrategy
{
    CrossExchange,      // Arbitrage between two exchanges
    SpotPerpetual      // Cash-and-carry arbitrage (spot vs perpetual on same exchange)
}

public enum StrategySubType
{
    SpotPerpetualSameExchange,         // Buy spot + short perp on same exchange
    CrossExchangeFuturesFutures,       // Long perp on one exchange + short perp on another (FUNDING arbitrage)
    CrossExchangeSpotFutures,          // Buy spot on one exchange + short perp on another
    CrossExchangeFuturesPriceSpread    // Long perp on cheaper exchange + short perp on expensive exchange (PRICE arbitrage)
}

public class ArbitrageOpportunityDto
{
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
    public DateTime? LongNextFundingTime { get; set; }  // When long funding rate expires/renews
    public DateTime? ShortNextFundingTime { get; set; } // When short funding rate expires/renews

    // For cross-exchange arbitrage - Price fields
    public decimal? LongExchangePrice { get; set; }   // Price on long exchange (perpetual)
    public decimal? ShortExchangePrice { get; set; }  // Price on short exchange (perpetual)
    public decimal? CurrentPriceSpreadPercent { get; set; }  // Current instant price spread: (Short - Long) / Long * 100

    // For spot-perpetual arbitrage
    public string Exchange { get; set; } = string.Empty;

    // DEPRECATED: These fields have confusing names for cross-exchange arbitrage
    // For cross-exchange: SpotPrice = LongExchangePrice, PerpetualPrice = ShortExchangePrice
    // For spot-perp: SpotPrice = actual spot, PerpetualPrice = actual perpetual
    // Use LongExchangePrice/ShortExchangePrice for cross-exchange instead
    public decimal SpotPrice { get; set; }
    public decimal PerpetualPrice { get; set; }


    // Common fields
    public decimal SpreadRate { get; set; }
    public decimal AnnualizedSpread { get; set; }
    public decimal EstimatedProfitPercentage { get; set; }
    public decimal PositionCostPercent { get; set; } = 0.2m; // Trading fees for open/close (0.1% each = 0.2% total)
    public decimal? BreakEvenTimeHours { get; set; } // Time to recover position cost from funding fees
    public decimal Volume24h { get; set; }  // 24-hour trading volume in USDT (min of long/short for cross-exchange)

    // Calculated metrics (current funding rate)
    public decimal FundProfit8h { get; set; }  // 8-hour profit percentage using current funding rate
    public decimal FundApr { get; set; }       // Annualized percentage rate using current funding rate

    // Projected metrics (24-hour average)
    public decimal? FundProfit8h24hProj { get; set; }      // 8-hour profit % using 24h average funding rate
    public decimal? FundApr24hProj { get; set; }           // APR % using 24h average funding rate
    public decimal? FundBreakEvenTime24hProj { get; set; } // Break-even hours using 24h average funding rate

    // Projected metrics (3-day average)
    public decimal? FundProfit8h3dProj { get; set; }       // 8-hour profit % using 3D average funding rate
    public decimal? FundApr3dProj { get; set; }            // APR % using 3D average funding rate
    public decimal? FundBreakEvenTime3dProj { get; set; }  // Break-even hours using 3D average funding rate

    // Price spread projection metrics (for CFPS - CrossExchangeFuturesPriceSpread)
    public decimal? PriceSpread24hAvg { get; set; }        // 24-hour average price spread %
    public decimal? PriceSpread3dAvg { get; set; }         // 3-day average price spread %

    // Spread history metrics (30 samples) - for Cross-Exchange only
    public decimal? Spread30SampleAvg { get; set; }        // Average spread based on last 30 price samples
    public decimal? SpreadVolatilityStdDev { get; set; }   // Standard deviation of spread samples
    public decimal? SpreadVolatilityCv { get; set; }       // Coefficient of variation (StdDev/Mean)

    // Per-exchange volumes for cross-exchange arbitrage
    public decimal? LongVolume24h { get; set; }  // 24h volume on long exchange
    public decimal? ShortVolume24h { get; set; } // 24h volume on short exchange

    // Liquidity metrics
    public decimal? BidAskSpreadPercent { get; set; }
    public decimal? OrderbookDepthUsd { get; set; }
    public LiquidityStatus? LiquidityStatus { get; set; }
    public string? LiquidityWarning { get; set; }

    public OpportunityStatus Status { get; set; }
    public DateTime DetectedAt { get; set; }
    
    // Computed unique key for frontend tracking (not stored in DB)
    public string UniqueKey => Strategy == ArbitrageStrategy.SpotPerpetual
        ? $"{Symbol}-{Exchange}-{SubType}"
        : $"{Symbol}-{LongExchange}-{ShortExchange}-{SubType}";
}
