# Feature Mapping for C# ONNX Inference

Total Features: 54

## Feature Order

**IMPORTANT**: Features must be provided in this exact order for ONNX inference.

| Index | Feature Name | Description |
|-------|--------------|-------------|
| 0 | `hour_of_day` |  |
| 1 | `day_of_week` |  |
| 2 | `long_funding_rate` |  |
| 3 | `short_funding_rate` |  |
| 4 | `long_funding_interval_hours` |  |
| 5 | `short_funding_interval_hours` |  |
| 6 | `long_next_funding_minutes` |  |
| 7 | `short_next_funding_minutes` |  |
| 8 | `current_price_spread_pct` |  |
| 9 | `fund_profit_8h` | 8-hour profit % (current rate) |
| 10 | `fund_apr` | Annualized percentage rate |
| 11 | `fund_profit_8h_24h_proj` | 8h profit using 24h avg rate |
| 12 | `fund_apr_24h_proj` |  |
| 13 | `fund_break_even_24h_proj` |  |
| 14 | `fund_profit_8h_3d_proj` | 8h profit using 3D avg rate |
| 15 | `fund_apr_3d_proj` |  |
| 16 | `fund_break_even_3d_proj` |  |
| 17 | `break_even_hours` | Hours to recover position cost |
| 18 | `price_spread_24h_avg` |  |
| 19 | `price_spread_3d_avg` |  |
| 20 | `spread_30sample_avg` |  |
| 21 | `spread_volatility_stddev` |  |
| 22 | `spread_volatility_cv` | Spread volatility (CV) |
| 23 | `volume_24h` | 24-hour trading volume |
| 24 | `hit_profit_target` |  |
| 25 | `hit_stop_loss` |  |
| 26 | `is_asian_session` |  |
| 27 | `is_european_session` |  |
| 28 | `is_us_session` |  |
| 29 | `is_weekend` |  |
| 30 | `is_weekday` |  |
| 31 | `hour_sin` | Hour (sine encoding) |
| 32 | `hour_cos` | Hour (cosine encoding) |
| 33 | `day_sin` | Day of week (sine encoding) |
| 34 | `day_cos` | Day of week (cosine encoding) |
| 35 | `hours_until_long_funding` |  |
| 36 | `hours_until_short_funding` |  |
| 37 | `hours_until_next_funding` |  |
| 38 | `rate_momentum_24h` | Rate change vs 24h avg |
| 39 | `rate_momentum_3d` | Rate change vs 3D avg |
| 40 | `rate_stability` |  |
| 41 | `rate_trend_score` |  |
| 42 | `rate_acceleration` |  |
| 43 | `funding_differential` |  |
| 44 | `funding_direction` |  |
| 45 | `funding_asymmetry` |  |
| 46 | `expected_profit_signal` |  |
| 47 | `sharpe_proxy` |  |
| 48 | `profit_consistency` |  |
| 49 | `volatility_risk_score` | Volatility risk (0-1) |
| 50 | `break_even_feasibility` |  |
| 51 | `risk_adjusted_return` | Profit / (1 + volatility) |
| 52 | `breakeven_efficiency` |  |
| 53 | `spread_consistency` |  |

## C# Example

```csharp
// Create feature array (must match order above)
var features = new float[54];

features[0] = (float)opportunity.HourOfDay;
features[1] = (float)opportunity.DayOfWeek;
features[2] = (float)opportunity.LongFundingRate;
features[3] = (float)opportunity.ShortFundingRate;
features[4] = (float)opportunity.LongFundingIntervalHours;
// ... (continue for all features)

// Create ONNX tensor
var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });
```
