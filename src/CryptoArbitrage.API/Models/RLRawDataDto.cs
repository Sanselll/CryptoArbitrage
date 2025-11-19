using System.Text.Json.Serialization;

namespace CryptoArbitrage.API.Models;

/// <summary>
/// Raw data DTOs for V2 ML API - Unified Feature Builder Architecture
///
/// These DTOs represent RAW data sent from backend to ML API.
/// All feature engineering is done by the UnifiedFeatureBuilder in Python.
///
/// Key principle: Backend sends complete raw data, Python calculates ALL features.
/// This eliminates code duplication between C# and Python.
/// </summary>

/// <summary>
/// Trading configuration raw data (5 scalar values)
/// </summary>
public class TradingConfigRawData
{
    [JsonPropertyName("max_leverage")]
    public decimal MaxLeverage { get; set; } = 1.0m;

    [JsonPropertyName("target_utilization")]
    public decimal TargetUtilization { get; set; } = 0.5m;

    [JsonPropertyName("max_positions")]
    public int MaxPositions { get; set; } = 3;

    [JsonPropertyName("stop_loss_threshold")]
    public decimal StopLossThreshold { get; set; } = -0.02m;

    [JsonPropertyName("liquidation_buffer")]
    public decimal LiquidationBuffer { get; set; } = 0.15m;
}

/// <summary>
/// Position raw data - Contains ALL raw fields needed for feature calculation
/// Python will calculate the 17 features from these raw values
/// </summary>
public class PositionRawData
{
    // Identity
    [JsonPropertyName("is_active")]
    public bool IsActive { get; set; }

    [JsonPropertyName("symbol")]
    public string Symbol { get; set; } = string.Empty;

    // Position basics
    [JsonPropertyName("position_size_usd")]
    public decimal PositionSizeUsd { get; set; }

    [JsonPropertyName("position_age_hours")]
    public decimal PositionAgeHours { get; set; }

    [JsonPropertyName("leverage")]
    public decimal Leverage { get; set; } = 1.0m;

    // Prices (for price P&L calculation)
    [JsonPropertyName("entry_long_price")]
    public decimal EntryLongPrice { get; set; }

    [JsonPropertyName("entry_short_price")]
    public decimal EntryShortPrice { get; set; }

    [JsonPropertyName("current_long_price")]
    public decimal CurrentLongPrice { get; set; }

    [JsonPropertyName("current_short_price")]
    public decimal CurrentShortPrice { get; set; }

    [JsonPropertyName("slippage_pct")]
    public decimal SlippagePct { get; set; }

    // P&L (pre-calculated for convenience)
    [JsonPropertyName("unrealized_pnl_pct")]
    public decimal UnrealizedPnlPct { get; set; }

    [JsonPropertyName("long_pnl_pct")]
    public decimal LongPnlPct { get; set; }

    [JsonPropertyName("short_pnl_pct")]
    public decimal ShortPnlPct { get; set; }

    // Funding rates (for funding profit calculation)
    [JsonPropertyName("long_funding_rate")]
    public decimal LongFundingRate { get; set; }

    [JsonPropertyName("short_funding_rate")]
    public decimal ShortFundingRate { get; set; }

    [JsonPropertyName("long_funding_interval_hours")]
    public int LongFundingIntervalHours { get; set; } = 8;

    [JsonPropertyName("short_funding_interval_hours")]
    public int ShortFundingIntervalHours { get; set; } = 8;

    // APR (for APR comparison features)
    [JsonPropertyName("entry_apr")]
    public decimal EntryApr { get; set; }

    [JsonPropertyName("current_position_apr")]
    public decimal CurrentPositionApr { get; set; }

    // Risk
    [JsonPropertyName("liquidation_distance")]
    public decimal LiquidationDistance { get; set; } = 1.0m;
}

/// <summary>
/// Opportunity raw data - Contains ALL raw opportunity fields
/// Python will extract the 11 opportunity features from these
/// </summary>
public class OpportunityRawData
{
    // Identity
    [JsonPropertyName("symbol")]
    public string Symbol { get; set; } = string.Empty;

    [JsonPropertyName("long_exchange")]
    public string LongExchange { get; set; } = string.Empty;

    [JsonPropertyName("short_exchange")]
    public string ShortExchange { get; set; } = string.Empty;

    // Funding profit projections (6 features)
    [JsonPropertyName("fund_profit_8h")]
    public decimal FundProfit8h { get; set; }

    [JsonPropertyName("fund_profit_8h_24h_proj")]
    public decimal FundProfit8h24hProj { get; set; }

    [JsonPropertyName("fund_profit_8h_3d_proj")]
    public decimal FundProfit8h3dProj { get; set; }

    [JsonPropertyName("fund_apr")]
    public decimal FundApr { get; set; }

    [JsonPropertyName("fund_apr_24h_proj")]
    public decimal FundApr24hProj { get; set; }

    [JsonPropertyName("fund_apr_3d_proj")]
    public decimal FundApr3dProj { get; set; }

    // Spread metrics (4 features)
    [JsonPropertyName("spread_30_sample_avg")]
    public decimal Spread30SampleAvg { get; set; }

    [JsonPropertyName("price_spread_24h_avg")]
    public decimal PriceSpread24hAvg { get; set; }

    [JsonPropertyName("price_spread_3d_avg")]
    public decimal PriceSpread3dAvg { get; set; }

    [JsonPropertyName("spread_volatility_stddev")]
    public decimal SpreadVolatilityStddev { get; set; }

    // Position tracking
    [JsonPropertyName("has_existing_position")]
    public bool HasExistingPosition { get; set; }
}

/// <summary>
/// Portfolio raw data - Contains positions and portfolio state
/// </summary>
public class PortfolioRawData
{
    [JsonPropertyName("positions")]
    public List<PositionRawData> Positions { get; set; } = new();

    [JsonPropertyName("total_capital")]
    public decimal TotalCapital { get; set; } = 10000m;

    [JsonPropertyName("capital_utilization")]
    public decimal CapitalUtilization { get; set; }
}

/// <summary>
/// Complete raw data request for ML API V2
/// </summary>
public class RLRawDataRequest
{
    [JsonPropertyName("trading_config")]
    public TradingConfigRawData TradingConfig { get; set; } = new();

    [JsonPropertyName("portfolio")]
    public PortfolioRawData Portfolio { get; set; } = new();

    [JsonPropertyName("opportunities")]
    public List<OpportunityRawData> Opportunities { get; set; } = new();
}

/// <summary>
/// Response from ML API V2 (same as V1 for backward compatibility)
/// </summary>
public class RLPredictionResponseV2
{
    [JsonPropertyName("action")]
    public string Action { get; set; } = string.Empty;  // "HOLD", "ENTER", "EXIT"

    [JsonPropertyName("action_id")]
    public int ActionId { get; set; }

    [JsonPropertyName("confidence")]
    public decimal Confidence { get; set; }

    [JsonPropertyName("state_value")]
    public decimal StateValue { get; set; }

    [JsonPropertyName("action_probabilities")]
    public List<decimal> ActionProbabilities { get; set; } = new();

    // ENTER action fields
    [JsonPropertyName("opportunity_index")]
    public int? OpportunityIndex { get; set; }

    [JsonPropertyName("opportunity_symbol")]
    public string? OpportunitySymbol { get; set; }

    [JsonPropertyName("opportunity_long_exchange")]
    public string? OpportunityLongExchange { get; set; }

    [JsonPropertyName("opportunity_short_exchange")]
    public string? OpportunityShortExchange { get; set; }

    [JsonPropertyName("opportunity_fund_apr")]
    public decimal? OpportunityFundApr { get; set; }

    [JsonPropertyName("position_size")]
    public string? PositionSize { get; set; }  // "SMALL", "MEDIUM", "LARGE"

    [JsonPropertyName("size_multiplier")]
    public decimal? SizeMultiplier { get; set; }

    // EXIT action fields
    [JsonPropertyName("position_index")]
    public int? PositionIndex { get; set; }

    // Mask info
    [JsonPropertyName("valid_actions")]
    public int ValidActions { get; set; }

    [JsonPropertyName("masked_actions")]
    public int MaskedActions { get; set; }

    // Model info
    [JsonPropertyName("model_info")]
    public Dictionary<string, object>? ModelInfo { get; set; }
}
