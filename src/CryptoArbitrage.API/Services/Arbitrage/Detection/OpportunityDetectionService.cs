using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Services.Arbitrage.Detection;

/// <summary>
/// Service responsible for detecting arbitrage opportunities from market data
/// </summary>
public class OpportunityDetectionService : IOpportunityDetectionService
{
    private readonly ArbitrageConfig _config;
    private readonly ILogger<OpportunityDetectionService> _logger;
    private readonly IDataRepository<Dictionary<string, List<HistoricalPriceDto>>> _historicalPriceRepository;

    // Trading fees: 0.05% taker fee per trade × 4 trades (open long, open short, close long, close short) = 0.2%
    private const decimal POSITION_COST_PERCENT = 0.2m;

    public OpportunityDetectionService(
        ArbitrageConfig config,
        ILogger<OpportunityDetectionService> logger,
        IDataRepository<Dictionary<string, List<HistoricalPriceDto>>> historicalPriceRepository)
    {
        _config = config;
        _logger = logger;
        _historicalPriceRepository = historicalPriceRepository;
    }

    /// <summary>
    /// Detect all types of arbitrage opportunities from the provided market data
    /// NEW ARCHITECTURE: Calculate all combinations once, then filter by strategy
    /// </summary>
    public async Task<List<ArbitrageOpportunityDto>> DetectOpportunitiesAsync(MarketDataSnapshot snapshot)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        // === CROSS-EXCHANGE STRATEGIES (requires 2+ exchanges) ===
        if (snapshot.FundingRates.Count >= 2)
        {
            // **CALCULATE ALL COMBINATIONS ONCE** - this is the key optimization!
            var combinations = CalculateAllCrossExchangeCombinations(
                snapshot.FundingRates,
                snapshot.PerpPrices);

            _logger.LogInformation("Calculated {Count} cross-exchange combinations", combinations.Count);

            // Filter for CFFF (Futures Funding arbitrage)
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeFuturesFutures))
            {
                var cfffOpps = FilterCrossExchangeFuturesFunding(combinations);
                opportunities.AddRange(cfffOpps);
                _logger.LogInformation("Detected {Count} CFFF opportunities", cfffOpps.Count);
            }

            // Filter for CFPS (Futures Price Spread arbitrage) - EXCLUDES CFFF pairs!
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeFuturesPriceSpread))
            {
                var cfffOpps = opportunities
                    .Where(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures)
                    .ToList();

                var cfpsOpps = FilterCrossExchangePriceSpread(combinations, cfffOpps);
                opportunities.AddRange(cfpsOpps);
                _logger.LogInformation("Detected {Count} CFPS opportunities (after excluding CFFF pairs)", cfpsOpps.Count);
            }

            // NOTE: Spot/Futures cross-exchange is COMMENTED OUT (not used)
            // NOTE: Spot-perpetual same exchange is COMMENTED OUT (not used)
        }

        _logger.LogInformation("Total {Count} opportunities detected from market snapshot", opportunities.Count);

        return await Task.FromResult(opportunities);
    }

    /// <summary>
    /// Calculates all metrics for an opportunity: current rate metrics and projected metrics (24h and 3D averages)
    /// </summary>
    private void CalculateOpportunityMetrics(
        ArbitrageOpportunityDto opportunity,
        FundingRateDto? longFundingData,
        FundingRateDto? shortFundingData)
    {
        bool isSpotPerp = opportunity.SubType == StrategySubType.SpotPerpetualSameExchange;
        bool isCrossFut = opportunity.SubType == StrategySubType.CrossExchangeFuturesFutures;
        bool isCrossSpotFut = opportunity.SubType == StrategySubType.CrossExchangeSpotFutures;
        bool isPriceSpread = opportunity.SubType == StrategySubType.CrossExchangeFuturesPriceSpread;

        // === CURRENT RATE METRICS ===

        if (isSpotPerp)
        {
            // Spot-Perpetual: Use EstimatedProfitPercentage as APR
            opportunity.FundApr = opportunity.EstimatedProfitPercentage;

            int fundingIntervalHours = longFundingData?.FundingIntervalHours ?? 8;
            decimal periodsIn8Hours = 8m / fundingIntervalHours;
            opportunity.FundProfit8h = (opportunity.FundApr / 365m) * periodsIn8Hours;
        }
        else if (isCrossFut || isPriceSpread)
        {
            // Cross-Exchange Futures (funding & price spread): Calculate based on funding rate difference
            int longInterval = opportunity.LongFundingIntervalHours ?? longFundingData?.FundingIntervalHours ?? 8;
            int shortInterval = opportunity.ShortFundingIntervalHours ?? shortFundingData?.FundingIntervalHours ?? 8;

            decimal longDailyRate = (opportunity.LongFundingRate * 100m) * (24m / longInterval);
            decimal shortDailyRate = (opportunity.ShortFundingRate * 100m) * (24m / shortInterval);
            decimal netDailyRate = shortDailyRate - longDailyRate;

            opportunity.FundApr = netDailyRate * 365m;
            opportunity.FundProfit8h = netDailyRate / 3m; // Daily rate / 3 for 8h period
        }
        else // CrossExchangeSpotFutures
        {
            // Cross-Exchange Spot-Futures: Similar to spot-perp but using short exchange rate
            opportunity.FundApr = opportunity.EstimatedProfitPercentage;

            int shortInterval = opportunity.ShortFundingIntervalHours ?? shortFundingData?.FundingIntervalHours ?? 8;
            decimal periodsIn8Hours = 8m / shortInterval;
            opportunity.FundProfit8h = (opportunity.FundApr / 365m) * periodsIn8Hours;
        }

        // === 24H PROJECTION METRICS ===

        if (longFundingData?.Average24hRate != null)
        {
            if (isSpotPerp)
            {
                int fundingIntervalHours = longFundingData.FundingIntervalHours;
                decimal periodsPerYear = (365m * 24m) / fundingIntervalHours;
                opportunity.FundApr24hProj = (longFundingData.Average24hRate * 100m) * periodsPerYear;

                decimal periodsIn8Hours = 8m / fundingIntervalHours;
                opportunity.FundProfit8h24hProj = (opportunity.FundApr24hProj.Value / 365m) * periodsIn8Hours;

                // Calculate break even
                opportunity.FundBreakEvenTime24hProj = opportunity.FundProfit8h24hProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h24hProj.Value) * 8m, fundingIntervalHours)
                    : null;
            }
            else if ((isCrossFut || isPriceSpread) && shortFundingData?.Average24hRate != null)
            {
                int longInterval = longFundingData.FundingIntervalHours;
                int shortInterval = shortFundingData.FundingIntervalHours;

                decimal longDailyRate = (longFundingData.Average24hRate.Value * 100m) * (24m / longInterval);
                decimal shortDailyRate = (shortFundingData.Average24hRate.Value * 100m) * (24m / shortInterval);
                decimal netDailyRate = shortDailyRate - longDailyRate;

                opportunity.FundApr24hProj = netDailyRate * 365m;
                opportunity.FundProfit8h24hProj = netDailyRate / 3m;

                // Calculate break even
                int maxInterval = Math.Max(longInterval, shortInterval);
                opportunity.FundBreakEvenTime24hProj = opportunity.FundProfit8h24hProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h24hProj.Value) * 8m, maxInterval)
                    : null;
            }
            else if (isCrossSpotFut && shortFundingData?.Average24hRate != null)
            {
                int shortInterval = shortFundingData.FundingIntervalHours;
                decimal periodsPerYear = (365m * 24m) / shortInterval;
                opportunity.FundApr24hProj = (shortFundingData.Average24hRate * 100m) * periodsPerYear;

                decimal periodsIn8Hours = 8m / shortInterval;
                opportunity.FundProfit8h24hProj = (opportunity.FundApr24hProj.Value / 365m) * periodsIn8Hours;

                // Calculate break even
                opportunity.FundBreakEvenTime24hProj = opportunity.FundProfit8h24hProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h24hProj.Value) * 8m, shortInterval)
                    : null;
            }
        }

        // === 3D PROJECTION METRICS ===

        if (longFundingData?.Average3DayRate != null)
        {
            if (isSpotPerp)
            {
                int fundingIntervalHours = longFundingData.FundingIntervalHours;
                decimal periodsPerYear = (365m * 24m) / fundingIntervalHours;
                opportunity.FundApr3dProj = (longFundingData.Average3DayRate * 100m) * periodsPerYear;

                decimal periodsIn8Hours = 8m / fundingIntervalHours;
                opportunity.FundProfit8h3dProj = (opportunity.FundApr3dProj.Value / 365m) * periodsIn8Hours;

                // Calculate break even
                opportunity.FundBreakEvenTime3dProj = opportunity.FundProfit8h3dProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h3dProj.Value) * 8m, fundingIntervalHours)
                    : null;
            }
            else if ((isCrossFut || isPriceSpread) && shortFundingData?.Average3DayRate != null)
            {
                int longInterval = longFundingData.FundingIntervalHours;
                int shortInterval = shortFundingData.FundingIntervalHours;

                decimal longDailyRate = (longFundingData.Average3DayRate.Value * 100m) * (24m / longInterval);
                decimal shortDailyRate = (shortFundingData.Average3DayRate.Value * 100m) * (24m / shortInterval);
                decimal netDailyRate = shortDailyRate - longDailyRate;

                opportunity.FundApr3dProj = netDailyRate * 365m;
                opportunity.FundProfit8h3dProj = netDailyRate / 3m;

                // Calculate break even
                int maxInterval = Math.Max(longInterval, shortInterval);
                opportunity.FundBreakEvenTime3dProj = opportunity.FundProfit8h3dProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h3dProj.Value) * 8m, maxInterval)
                    : null;
            }
            else if (isCrossSpotFut && shortFundingData?.Average3DayRate != null)
            {
                int shortInterval = shortFundingData.FundingIntervalHours;
                decimal periodsPerYear = (365m * 24m) / shortInterval;
                opportunity.FundApr3dProj = (shortFundingData.Average3DayRate * 100m) * periodsPerYear;

                decimal periodsIn8Hours = 8m / shortInterval;
                opportunity.FundProfit8h3dProj = (opportunity.FundApr3dProj.Value / 365m) * periodsIn8Hours;

                // Calculate break even
                opportunity.FundBreakEvenTime3dProj = opportunity.FundProfit8h3dProj.Value > 0
                    ? Math.Max((POSITION_COST_PERCENT / opportunity.FundProfit8h3dProj.Value) * 8m, shortInterval)
                    : null;
            }
        }
    }

    /// <summary>
    /// Calculate ALL possible cross-exchange combinations with their metrics
    /// This is done ONCE, then different strategies filter the results
    /// </summary>
    private List<ArbitrageOpportunityDto> CalculateAllCrossExchangeCombinations(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, PriceDto>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (fundingRates.Count < 2)
            return opportunities;

        var exchangeNames = fundingRates.Keys.ToList();

        // Find common symbols across all exchanges
        var commonSymbols = fundingRates.Values
            .Select(rates => rates.Select(r => r.Symbol).ToHashSet())
            .Aggregate((a, b) => { a.IntersectWith(b); return a; });

        foreach (var symbol in commonSymbols)
        {
            // Compare all exchange pairs
            for (int i = 0; i < exchangeNames.Count; i++)
            {
                for (int j = i + 1; j < exchangeNames.Count; j++)
                {
                    var exchange1 = exchangeNames[i];
                    var exchange2 = exchangeNames[j];

                    var rate1 = fundingRates[exchange1].First(r => r.Symbol == symbol);
                    var rate2 = fundingRates[exchange2].First(r => r.Symbol == symbol);

                    // Get perpetual prices and volumes for both exchanges
                    decimal price1 = 0;
                    decimal price2 = 0;
                    decimal? volume1 = null;
                    decimal? volume2 = null;

                    if (perpPrices.ContainsKey(exchange1) && perpPrices[exchange1].ContainsKey(symbol))
                    {
                        price1 = perpPrices[exchange1][symbol].Price;
                        volume1 = perpPrices[exchange1][symbol].Volume24h;
                    }
                    if (perpPrices.ContainsKey(exchange2) && perpPrices[exchange2].ContainsKey(symbol))
                    {
                        price2 = perpPrices[exchange2][symbol].Price;
                        volume2 = perpPrices[exchange2][symbol].Volume24h;
                    }

                    // CREATE BOTH DIRECTIONAL COMBINATIONS
                    // This allows CFFF to pick based on funding rates, CFPS to pick based on prices

                    // Combination 1: Exchange1 = LONG, Exchange2 = SHORT
                    var opp1 = new ArbitrageOpportunityDto
                    {
                        Strategy = ArbitrageStrategy.CrossExchange,
                        SubType = StrategySubType.CrossExchangeFuturesFutures, // Temporary, will be set by filter
                        Symbol = symbol,
                        LongExchange = exchange1,
                        ShortExchange = exchange2,
                        LongFundingRate = rate1.Rate,
                        ShortFundingRate = rate2.Rate,
                        LongFundingIntervalHours = rate1.FundingIntervalHours,
                        ShortFundingIntervalHours = rate2.FundingIntervalHours,
                        SpotPrice = price1,      // Long exchange price
                        PerpetualPrice = price2, // Short exchange price
                        LongVolume24h = volume1,
                        ShortVolume24h = volume2,
                        Volume24h = volume1.HasValue && volume2.HasValue ? Math.Min(volume1.Value, volume2.Value) : (volume1 ?? volume2 ?? 0),
                        PositionCostPercent = POSITION_COST_PERCENT,
                        Status = OpportunityStatus.Detected,
                        DetectedAt = DateTime.UtcNow
                    };
                    CalculateOpportunityMetrics(opp1, rate1, rate2);
                    opportunities.Add(opp1);

                    // Combination 2: Exchange2 = LONG, Exchange1 = SHORT
                    var opp2 = new ArbitrageOpportunityDto
                    {
                        Strategy = ArbitrageStrategy.CrossExchange,
                        SubType = StrategySubType.CrossExchangeFuturesFutures, // Temporary, will be set by filter
                        Symbol = symbol,
                        LongExchange = exchange2,
                        ShortExchange = exchange1,
                        LongFundingRate = rate2.Rate,
                        ShortFundingRate = rate1.Rate,
                        LongFundingIntervalHours = rate2.FundingIntervalHours,
                        ShortFundingIntervalHours = rate1.FundingIntervalHours,
                        SpotPrice = price2,      // Long exchange price
                        PerpetualPrice = price1, // Short exchange price
                        LongVolume24h = volume2,
                        ShortVolume24h = volume1,
                        Volume24h = volume1.HasValue && volume2.HasValue ? Math.Min(volume1.Value, volume2.Value) : (volume1 ?? volume2 ?? 0),
                        PositionCostPercent = POSITION_COST_PERCENT,
                        Status = OpportunityStatus.Detected,
                        DetectedAt = DateTime.UtcNow
                    };
                    CalculateOpportunityMetrics(opp2, rate2, rate1);
                    opportunities.Add(opp2);
                }
            }
        }

        return opportunities;
    }

    /// <summary>
    /// Filter combinations for Cross-Exchange Futures Funding arbitrage
    /// Threshold: ANY 8h profit metric >= MinSpreadPercentage (0.1%)
    /// BIDIRECTIONAL: Detects when different time horizons favor opposite directions
    /// </summary>
    private List<ArbitrageOpportunityDto> FilterCrossExchangeFuturesFunding(
        List<ArbitrageOpportunityDto> allOpportunities)
    {
        // Group by symbol to detect bidirectional opportunities
        var groupedBySymbol = allOpportunities
            .GroupBy(o => o.Symbol)
            .ToList();

        var filteredOpportunities = new List<ArbitrageOpportunityDto>();

        foreach (var symbolGroup in groupedBySymbol)
        {
            // For each symbol, we have 2 combinations (direction 1 and direction 2)
            var opportunities = symbolGroup.ToList();

            // Calculate price spread averages for each opportunity
            foreach (var opp in opportunities)
            {
                var (avg24h, avg3d) = CalculateHistoricalPriceSpreadAsync(
                    opp.Symbol,
                    opp.LongExchange,
                    opp.ShortExchange).Result;

                opp.PriceSpread24hAvg = avg24h;
                opp.PriceSpread3dAvg = avg3d;
            }

            // Check which opportunities meet the threshold
            var qualifyingOpportunities = opportunities
                .Where(o =>
                    o.FundProfit8h >= _config.MinSpreadPercentage ||
                    (o.FundProfit8h24hProj.HasValue && o.FundProfit8h24hProj.Value >= _config.MinSpreadPercentage) ||
                    (o.FundProfit8h3dProj.HasValue && o.FundProfit8h3dProj.Value >= _config.MinSpreadPercentage)
                )
                .ToList();

            if (!qualifyingOpportunities.Any())
                continue;

            // BIDIRECTIONAL DETECTION:
            // Check if different time horizons favor different directions
            if (qualifyingOpportunities.Count == 2)
            {
                // Both directions qualify - check if they favor different time horizons
                var opp1 = qualifyingOpportunities[0];
                var opp2 = qualifyingOpportunities[1];

                // Determine which metrics are profitable for each direction
                bool opp1HasCurrent = opp1.FundProfit8h >= _config.MinSpreadPercentage;
                bool opp1Has24h = opp1.FundProfit8h24hProj.HasValue && opp1.FundProfit8h24hProj.Value >= _config.MinSpreadPercentage;
                bool opp1Has3d = opp1.FundProfit8h3dProj.HasValue && opp1.FundProfit8h3dProj.Value >= _config.MinSpreadPercentage;

                bool opp2HasCurrent = opp2.FundProfit8h >= _config.MinSpreadPercentage;
                bool opp2Has24h = opp2.FundProfit8h24hProj.HasValue && opp2.FundProfit8h24hProj.Value >= _config.MinSpreadPercentage;
                bool opp2Has3d = opp2.FundProfit8h3dProj.HasValue && opp2.FundProfit8h3dProj.Value >= _config.MinSpreadPercentage;

                // Check if metrics are split across directions (bidirectional scenario)
                bool isBidirectional =
                    (opp1HasCurrent || opp1Has24h || opp1Has3d) &&
                    (opp2HasCurrent || opp2Has24h || opp2Has3d);

                if (isBidirectional)
                {
                    // Include BOTH directions
                    _logger.LogInformation(
                        "Bidirectional CFF opportunity detected for {Symbol}: {Ex1}<->{Ex2} " +
                        "(Direction 1: cur={Cur1:F4}%, 24h={H24_1:F4}%, 3d={D3_1:F4}%; " +
                        "Direction 2: cur={Cur2:F4}%, 24h={H24_2:F4}%, 3d={D3_2:F4}%)",
                        opp1.Symbol,
                        opp1.LongExchange,
                        opp1.ShortExchange,
                        opp1.FundProfit8h,
                        opp1.FundProfit8h24hProj ?? 0,
                        opp1.FundProfit8h3dProj ?? 0,
                        opp2.FundProfit8h,
                        opp2.FundProfit8h24hProj ?? 0,
                        opp2.FundProfit8h3dProj ?? 0);

                    filteredOpportunities.AddRange(qualifyingOpportunities);
                }
                else
                {
                    // Only one direction is truly profitable across all metrics
                    // Pick the one with highest current rate
                    var best = qualifyingOpportunities.OrderByDescending(o => o.FundApr).First();
                    filteredOpportunities.Add(best);
                }
            }
            else
            {
                // Only one direction qualifies
                filteredOpportunities.AddRange(qualifyingOpportunities);
            }
        }

        // Set SubType and return sorted results
        return filteredOpportunities
            .Select(o => { o.SubType = StrategySubType.CrossExchangeFuturesFutures; return o; })
            .OrderByDescending(o => o.FundApr)
            .ToList();
    }

    /// <summary>
    /// Filter combinations for Cross-Exchange Futures Price Spread arbitrage
    /// Excludes pairs already detected by CFFF
    /// Threshold: Net price spread (after costs) >= MinPriceSpreadPercentage (0.3%)
    /// </summary>
    private List<ArbitrageOpportunityDto> FilterCrossExchangePriceSpread(
        List<ArbitrageOpportunityDto> allOpportunities,
        List<ArbitrageOpportunityDto> cfffOpportunities)
    {
        // Create exclusion set from CFFF opportunities
        var excludedPairs = cfffOpportunities
            .Select(o => (o.Symbol, o.LongExchange, o.ShortExchange))
            .ToHashSet();

        // Trading costs for price arbitrage
        const decimal TRADING_FEES_PERCENT = 0.1m;  // 0.1% (0.05% x 2 sides)
        const decimal SLIPPAGE_PERCENT = 0.05m;     // 0.05% estimated slippage

        return allOpportunities
            .Where(o => !excludedPairs.Contains((o.Symbol, o.LongExchange, o.ShortExchange))) // Deduplicate!
            .Select(o =>
            {
                // Calculate gross price spread: (Short - Long) / Long * 100
                decimal grossPriceSpread = o.SpotPrice > 0
                    ? ((o.PerpetualPrice - o.SpotPrice) / o.SpotPrice) * 100m
                    : 0;

                // Net profit = gross spread - trading fees - slippage
                decimal netProfitPercent = grossPriceSpread - TRADING_FEES_PERCENT - SLIPPAGE_PERCENT;

                // Calculate historical price spread averages (24h and 3D)
                var (avg24h, avg3d) = CalculateHistoricalPriceSpreadAsync(o.Symbol, o.LongExchange, o.ShortExchange).Result;

                // Update opportunity fields for CFPS
                o.SubType = StrategySubType.CrossExchangeFuturesPriceSpread;
                o.SpreadRate = netProfitPercent / 100m;
                o.AnnualizedSpread = netProfitPercent / 100m; // Not annualized (one-time profit)
                o.EstimatedProfitPercentage = netProfitPercent;
                o.BreakEvenTimeHours = null; // Price arbitrage is one-time, no break-even concept
                o.PriceSpread24hAvg = avg24h;
                o.PriceSpread3dAvg = avg3d;

                return o;
            })
            .Where(o => o.EstimatedProfitPercentage >= _config.MinPriceSpreadPercentage) // Filter after calculating net profit
            .OrderByDescending(o => o.EstimatedProfitPercentage)
            .ToList();
    }

    /// <summary>
    /// Calculate 24h and 3D average price spreads from historical data
    /// </summary>
    private async Task<(decimal? avg24h, decimal? avg3d)> CalculateHistoricalPriceSpreadAsync(
        string symbol,
        string longExchange,
        string shortExchange)
    {
        try
        {
            // Build cache keys for both exchanges
            var longKey = $"{longExchange}:{symbol}";
            var shortKey = $"{shortExchange}:{symbol}";

            // Fetch historical prices from repository
            var longPricesDict = await _historicalPriceRepository.GetAsync(longKey);
            var shortPricesDict = await _historicalPriceRepository.GetAsync(shortKey);

            if (longPricesDict == null || shortPricesDict == null)
            {
                return (null, null);
            }

            // Extract price lists from dictionaries
            if (!longPricesDict.TryGetValue(longKey, out var longPrices) ||
                !shortPricesDict.TryGetValue(shortKey, out var shortPrices))
            {
                return (null, null);
            }

            if (!longPrices.Any() || !shortPrices.Any())
            {
                return (null, null);
            }

            // Calculate spreads for each hour where we have data from both exchanges
            var spreads = new List<(DateTime timestamp, decimal spread)>();

            foreach (var longPrice in longPrices)
            {
                // Find corresponding short price at the same timestamp (within 1 minute tolerance)
                var shortPrice = shortPrices.FirstOrDefault(sp =>
                    Math.Abs((sp.Timestamp - longPrice.Timestamp).TotalMinutes) < 1);

                if (shortPrice != null && longPrice.Price > 0)
                {
                    // Calculate spread: (Short - Long) / Long * 100
                    decimal spread = ((shortPrice.Price - longPrice.Price) / longPrice.Price) * 100m;
                    spreads.Add((longPrice.Timestamp, spread));
                }
            }

            if (!spreads.Any())
            {
                return (null, null);
            }

            // Calculate 24h average (last 24 hours)
            var oneDayAgo = DateTime.UtcNow.AddHours(-24);
            var last24hSpreads = spreads.Where(s => s.timestamp >= oneDayAgo).ToList();
            decimal? avg24h = last24hSpreads.Any()
                ? last24hSpreads.Average(s => s.spread)
                : null;

            // Calculate 3D average (last 72 hours)
            var threeDaysAgo = DateTime.UtcNow.AddDays(-3);
            var last3dSpreads = spreads.Where(s => s.timestamp >= threeDaysAgo).ToList();
            decimal? avg3d = last3dSpreads.Any()
                ? last3dSpreads.Average(s => s.spread)
                : null;

            return (avg24h, avg3d);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating historical price spreads for {Symbol} ({LongEx} vs {ShortEx})",
                symbol, longExchange, shortExchange);
            return (null, null);
        }
    }

    // ==================================================================================
    // DEPRECATED METHODS - NO LONGER USED
    // The following methods have been replaced by the new architecture:
    // - CalculateAllCrossExchangeCombinations() calculates ALL metrics once
    // - FilterCrossExchangeFuturesFunding() and FilterCrossExchangePriceSpread() filter
    //
    // These old methods are kept for reference but should NOT be used
    // ==================================================================================

    /*
    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, PriceDto>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (fundingRates.Count < 2)
            return opportunities;

        var exchangeNames = fundingRates.Keys.ToList();

        // Find common symbols across all exchanges
        var commonSymbols = fundingRates.Values
            .Select(rates => rates.Select(r => r.Symbol).ToHashSet())
            .Aggregate((a, b) => { a.IntersectWith(b); return a; });

        foreach (var symbol in commonSymbols)
        {
            // Compare funding rates between all exchange pairs
            for (int i = 0; i < exchangeNames.Count; i++)
            {
                for (int j = i + 1; j < exchangeNames.Count; j++)
                {
                    var exchange1 = exchangeNames[i];
                    var exchange2 = exchangeNames[j];

                    var rate1 = fundingRates[exchange1].First(r => r.Symbol == symbol);
                    var rate2 = fundingRates[exchange2].First(r => r.Symbol == symbol);

                    // Get perpetual prices for both exchanges
                    decimal price1 = 0;
                    decimal price2 = 0;
                    if (perpPrices.ContainsKey(exchange1) && perpPrices[exchange1].ContainsKey(symbol))
                        price1 = perpPrices[exchange1][symbol].Price;
                    if (perpPrices.ContainsKey(exchange2) && perpPrices[exchange2].ContainsKey(symbol))
                        price2 = perpPrices[exchange2][symbol].Price;

                    // CRITICAL: Funding rate calculation for Futures/Futures cross-exchange
                    // Strategy: LONG on exchange with LOWER funding rate, SHORT on exchange with HIGHER funding rate
                    //
                    // Net funding earnings = |fundingRateLong| + |fundingRateShort| when signs differ
                    //                      = |fundingRateHigh - fundingRateLow| when signs are same
                    //
                    // Example 1: Exchange1: +0.05%, Exchange2: -0.03%
                    //   -> LONG on Exchange2 (lower/negative), SHORT on Exchange1 (higher/positive)
                    //   -> Earn: receive 0.03% from long position + receive 0.05% from short position = 0.08% per funding
                    //
                    // Example 2: Exchange1: +0.10%, Exchange2: +0.03%
                    //   -> LONG on Exchange2 (lower), SHORT on Exchange1 (higher)
                    //   -> Pay 0.03% on long, Receive 0.10% on short -> Net: +0.07%
                    //
                    // Example 3: Exchange1: -0.05%, Exchange2: -0.10%
                    //   -> LONG on Exchange2 (lower/more negative), SHORT on Exchange1 (higher/less negative)
                    //   -> Receive 0.10% on long, Pay 0.05% on short -> Net: +0.05%

                    decimal netFundingRate;
                    string longExchange, shortExchange;
                    decimal longRate, shortRate;

                    if (rate1.Rate < rate2.Rate)
                    {
                        // LONG on exchange1 (lower rate), SHORT on exchange2 (higher rate)
                        longExchange = exchange1;
                        shortExchange = exchange2;
                        longRate = rate1.Rate;
                        shortRate = rate2.Rate;
                    }
                    else
                    {
                        // LONG on exchange2 (lower rate), SHORT on exchange1 (higher rate)
                        longExchange = exchange2;
                        shortExchange = exchange1;
                        longRate = rate2.Rate;
                        shortRate = rate1.Rate;
                    }

                    // Net funding = -longRate (we pay if positive, receive if negative)
                    //              + shortRate (we receive if positive, pay if negative)
                    // Simplifies to: shortRate - longRate
                    netFundingRate = shortRate - longRate;

                    // Get funding intervals from the rate objects
                    var longRateObj = (longExchange == exchange1) ? rate1 : rate2;
                    var shortRateObj = (shortExchange == exchange1) ? rate1 : rate2;
                    int longFundingIntervalHours = longRateObj.FundingIntervalHours; // Already defaults to 8
                    int shortFundingIntervalHours = shortRateObj.FundingIntervalHours;

                    // Calculate periods per day for each exchange
                    decimal longPeriodsPerDay = 24m / longFundingIntervalHours;
                    decimal shortPeriodsPerDay = 24m / shortFundingIntervalHours;

                    // For cross-exchange, we earn funding on the short position and pay/receive on long position
                    // Annualized return = (shortRate * shortPeriodsPerDay - longRate * longPeriodsPerDay) * 365
                    var annualizedNetFunding = (shortRate * shortPeriodsPerDay - longRate * longPeriodsPerDay) * 365;

                    // Calculate current 8h profit percentage
                    var dailyFundingRate = shortRate * shortPeriodsPerDay - longRate * longPeriodsPerDay;
                    var fundProfit8h = (dailyFundingRate / 3m) * 100m; // Daily rate / 3 for 8h period

                    // Calculate 24h projection if historical data available
                    decimal? fundProfit8h24hProj = null;
                    if (longRateObj.Average24hRate != null && shortRateObj.Average24hRate != null)
                    {
                        var longDailyRate24h = (longRateObj.Average24hRate.Value * 100m) * (24m / longFundingIntervalHours);
                        var shortDailyRate24h = (shortRateObj.Average24hRate.Value * 100m) * (24m / shortFundingIntervalHours);
                        var netDailyRate24h = shortDailyRate24h - longDailyRate24h;
                        fundProfit8h24hProj = netDailyRate24h / 3m;
                    }

                    // Calculate 3D projection if historical data available
                    decimal? fundProfit8h3dProj = null;
                    if (longRateObj.Average3DayRate != null && shortRateObj.Average3DayRate != null)
                    {
                        var longDailyRate3d = (longRateObj.Average3DayRate.Value * 100m) * (24m / longFundingIntervalHours);
                        var shortDailyRate3d = (shortRateObj.Average3DayRate.Value * 100m) * (24m / shortFundingIntervalHours);
                        var netDailyRate3d = shortDailyRate3d - longDailyRate3d;
                        fundProfit8h3dProj = netDailyRate3d / 3m;
                    }

                    // Detect opportunity if ANY 8h profit metric meets threshold (0.1%)
                    if (fundProfit8h >= _config.MinSpreadPercentage
                        || (fundProfit8h24hProj.HasValue && fundProfit8h24hProj.Value >= _config.MinSpreadPercentage)
                        || (fundProfit8h3dProj.HasValue && fundProfit8h3dProj.Value >= _config.MinSpreadPercentage))
                    {
                        var profit8hPercent = fundProfit8h;

                        // Calculate break even time (minimum is the longer funding interval)
                        var maxFundingInterval = Math.Max(longFundingIntervalHours, shortFundingIntervalHours);
                        decimal? breakEvenTime = profit8hPercent > 0
                            ? Math.Max((POSITION_COST_PERCENT / profit8hPercent) * 8m, maxFundingInterval)
                            : null;

                        // Determine which price goes to which field based on long/short exchanges
                        decimal longExchangePrice = (longExchange == exchange1) ? price1 : price2;
                        decimal shortExchangePrice = (shortExchange == exchange1) ? price1 : price2;

                        var opportunity = new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeFuturesFutures,
                            Symbol = symbol,
                            LongExchange = longExchange,
                            ShortExchange = shortExchange,
                            LongFundingRate = longRate,
                            ShortFundingRate = shortRate,
                            LongFundingIntervalHours = longFundingIntervalHours,
                            ShortFundingIntervalHours = shortFundingIntervalHours,
                            // Use SpotPrice for longExchange perp price, PerpetualPrice for shortExchange perp price
                            SpotPrice = longExchangePrice,
                            PerpetualPrice = shortExchangePrice,
                            SpreadRate = netFundingRate,
                            AnnualizedSpread = annualizedNetFunding,
                            EstimatedProfitPercentage = annualizedNetFunding * 100,
                            PositionCostPercent = POSITION_COST_PERCENT,
                            BreakEvenTimeHours = breakEvenTime,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        };

                        // Calculate all metrics (current + projections)
                        CalculateOpportunityMetrics(opportunity, longRateObj, shortRateObj);

                        opportunities.Add(opportunity);
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.AnnualizedSpread).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangePriceSpreadOpportunitiesAsync(
        Dictionary<string, Dictionary<string, PriceDto>> perpPrices,
        Dictionary<string, List<FundingRateDto>> fundingRates)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (perpPrices.Count < 2)
            return opportunities;

        var exchangeNames = perpPrices.Keys.ToList();

        // Find common symbols across all exchanges
        var commonSymbols = perpPrices.Values
            .Select(prices => prices.Keys.ToHashSet())
            .Aggregate((a, b) => { a.IntersectWith(b); return a; });

        foreach (var symbol in commonSymbols)
        {
            // Compare perpetual prices between all exchange pairs
            for (int i = 0; i < exchangeNames.Count; i++)
            {
                for (int j = i + 1; j < exchangeNames.Count; j++)
                {
                    var exchange1 = exchangeNames[i];
                    var exchange2 = exchangeNames[j];

                    decimal price1 = perpPrices[exchange1][symbol].Price;
                    decimal price2 = perpPrices[exchange2][symbol].Price;

                    // Skip if prices are invalid
                    if (price1 == 0 || price2 == 0)
                        continue;

                    // Determine cheaper and expensive exchanges
                    string cheaperExchange, expensiveExchange;
                    decimal cheaperPrice, expensivePrice;

                    if (price1 < price2)
                    {
                        cheaperExchange = exchange1;
                        expensiveExchange = exchange2;
                        cheaperPrice = price1;
                        expensivePrice = price2;
                    }
                    else
                    {
                        cheaperExchange = exchange2;
                        expensiveExchange = exchange1;
                        cheaperPrice = price2;
                        expensivePrice = price1;
                    }

                    // Calculate percentage price spread
                    decimal priceSpread = Math.Abs(expensivePrice - cheaperPrice) / cheaperPrice;
                    decimal priceSpreadPercent = priceSpread * 100;

                    // Estimate trading fees (0.05% maker fee on both sides = 0.1% total)
                    decimal estimatedTradingFees = 0.001m; // 0.1%

                    // Estimate slippage (conservative estimate)
                    decimal estimatedSlippage = 0.0005m; // 0.05%

                    // Net profit = price spread - fees - slippage
                    decimal netProfit = priceSpread - estimatedTradingFees - estimatedSlippage;
                    decimal netProfitPercent = netProfit * 100;

                    // Only profitable if net profit exceeds minimum threshold
                    if (netProfitPercent >= _config.MinPriceSpreadPercentage)
                    {
                        // Get funding rates for both exchanges (optional context)
                        decimal cheaperFundingRate = 0;
                        decimal expensiveFundingRate = 0;
                        int? cheaperFundingIntervalHours = null;
                        int? expensiveFundingIntervalHours = null;

                        if (fundingRates.ContainsKey(cheaperExchange))
                        {
                            var rate = fundingRates[cheaperExchange].FirstOrDefault(r => r.Symbol == symbol);
                            if (rate != null)
                            {
                                cheaperFundingRate = rate.Rate;
                                cheaperFundingIntervalHours = rate.FundingIntervalHours;
                            }
                        }

                        if (fundingRates.ContainsKey(expensiveExchange))
                        {
                            var rate = fundingRates[expensiveExchange].FirstOrDefault(r => r.Symbol == symbol);
                            if (rate != null)
                            {
                                expensiveFundingRate = rate.Rate;
                                expensiveFundingIntervalHours = rate.FundingIntervalHours;
                            }
                        }

                        var opportunity = new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeFuturesPriceSpread,
                            Symbol = symbol,
                            LongExchange = cheaperExchange,    // Long on cheaper exchange
                            ShortExchange = expensiveExchange, // Short on expensive exchange
                            LongFundingRate = cheaperFundingRate,
                            ShortFundingRate = expensiveFundingRate,
                            LongFundingIntervalHours = cheaperFundingIntervalHours,
                            ShortFundingIntervalHours = expensiveFundingIntervalHours,
                            // Use SpotPrice for cheaper exchange price, PerpetualPrice for expensive exchange price
                            SpotPrice = cheaperPrice,
                            PerpetualPrice = expensivePrice,
                            SpreadRate = netProfit,
                            AnnualizedSpread = netProfit, // Not annualized for price arbitrage (one-time profit)
                            EstimatedProfitPercentage = netProfitPercent,
                            PositionCostPercent = POSITION_COST_PERCENT,
                            BreakEvenTimeHours = null, // Price arbitrage is one-time profit, no break even concept
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        };

                        // Calculate all metrics (current + projections) - price spread has no projections
                        CalculateOpportunityMetrics(opportunity, null, null);

                        opportunities.Add(opportunity);
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectSpotPerpetualOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, PriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, PriceDto>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        foreach (var (exchangeName, rates) in fundingRates)
        {
            // Skip if no spot prices or perpetual prices available for this exchange
            if (!spotPrices.ContainsKey(exchangeName) || !spotPrices[exchangeName].Any())
                continue;

            if (!perpPrices.ContainsKey(exchangeName) || !perpPrices[exchangeName].Any())
                continue;

            var exchangeSpotPrices = spotPrices[exchangeName];
            var exchangePerpPrices = perpPrices[exchangeName];

            foreach (var fundingRate in rates)
            {
                // Check if we have both spot and perpetual price for this symbol
                if (!exchangeSpotPrices.ContainsKey(fundingRate.Symbol))
                    continue;

                if (!exchangePerpPrices.ContainsKey(fundingRate.Symbol))
                    continue;

                var spotPrice = exchangeSpotPrices[fundingRate.Symbol];
                decimal perpetualPrice = exchangePerpPrices[fundingRate.Symbol].Price;

                // Skip if spot price is zero (invalid data)
                if (spotPrice.Price == 0 || perpetualPrice == 0)
                    continue;

                // Calculate price premium: (Perp - Spot) / Spot
                decimal pricePremium = (perpetualPrice - spotPrice.Price) / spotPrice.Price;

                // Annualized funding rate (already calculated in funding rate)
                decimal annualizedFundingRate = fundingRate.AnnualizedRate;

                // Estimated trading fees (0.1% for spot buy + 0.05% for futures short = 0.15%)
                decimal estimatedTradingFees = 0.0015m;

                // Calculate net profit for positive funding rate strategy
                // Positive funding = shorts pay longs -> buy spot + short perp -> collect funding
                // Note: Negative funding would require selling spot (which we cannot do with USDT-only capital)
                decimal netProfit = annualizedFundingRate - Math.Abs(pricePremium) - estimatedTradingFees;
                decimal netProfitPercentage = netProfit * 100;

                // Only create opportunity if net profit exceeds minimum threshold AND funding is positive
                if (netProfitPercentage >= _config.MinSpreadPercentage && annualizedFundingRate > 0)
                {
                    // Calculate 8h profit percentage
                    var periodsPerDay = 24m / fundingRate.FundingIntervalHours;
                    var dailyFundingRate = fundingRate.Rate * periodsPerDay;
                    var profit8hPercent = (dailyFundingRate / 3m) * 100m; // Daily rate / 3 for 8h period

                    // Calculate break even time (minimum is the funding interval)
                    decimal? breakEvenTime = profit8hPercent > 0
                        ? Math.Max((POSITION_COST_PERCENT / profit8hPercent) * 8m, fundingRate.FundingIntervalHours)
                        : null;

                    var opportunity = new ArbitrageOpportunityDto
                    {
                        Strategy = ArbitrageStrategy.SpotPerpetual,
                        SubType = StrategySubType.SpotPerpetualSameExchange,
                        Symbol = fundingRate.Symbol,
                        Exchange = exchangeName,
                        SpotPrice = spotPrice.Price,
                        PerpetualPrice = perpetualPrice,
                        FundingRate = fundingRate.Rate,
                        AnnualizedFundingRate = annualizedFundingRate,
                        PricePremium = pricePremium,
                        SpreadRate = netProfit,
                        AnnualizedSpread = netProfit,
                        EstimatedProfitPercentage = netProfitPercentage,
                        PositionCostPercent = POSITION_COST_PERCENT,
                        BreakEvenTimeHours = breakEvenTime,
                        Status = OpportunityStatus.Detected,
                        DetectedAt = DateTime.UtcNow
                    };

                    // Calculate all metrics (current + projections)
                    CalculateOpportunityMetrics(opportunity, fundingRate, null);

                    opportunities.Add(opportunity);
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeSpotFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, PriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, PriceDto>> perpPrices)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        if (fundingRates.Count < 2)
            return opportunities;

        var exchangeNames = fundingRates.Keys.ToList();

        // Strategy: Buy spot on one exchange (Exchange A) + Short perpetual on another exchange (Exchange B)
        // This works ACROSS exchanges, combining spot from one and futures from another
        //
        // Profitability: Funding rate on Exchange B - price premium - trading fees
        // Note: We need positive funding on the SHORT exchange to make this profitable

        foreach (var spotExchange in exchangeNames)
        {
            if (!spotPrices.ContainsKey(spotExchange) || !spotPrices[spotExchange].Any())
                continue;

            var spotExchangePrices = spotPrices[spotExchange];

            foreach (var futuresExchange in exchangeNames)
            {
                if (spotExchange == futuresExchange)
                    continue; // Skip same exchange (that's handled by spot-perpetual strategy)

                if (!fundingRates.ContainsKey(futuresExchange) || !fundingRates[futuresExchange].Any())
                    continue;

                if (!perpPrices.ContainsKey(futuresExchange) || !perpPrices[futuresExchange].Any())
                    continue;

                var futuresRates = fundingRates[futuresExchange];
                var futuresPrices = perpPrices[futuresExchange];

                // Find symbols that exist on BOTH exchanges (spot on one, futures on other)
                var commonSymbols = spotExchangePrices.Keys
                    .Intersect(futuresRates.Select(r => r.Symbol))
                    .Intersect(futuresPrices.Keys)
                    .ToList();

                foreach (var symbol in commonSymbols)
                {
                    var spotPrice = spotExchangePrices[symbol].Price;
                    var futuresPrice = futuresPrices[symbol].Price;
                    var fundingRate = futuresRates.First(r => r.Symbol == symbol);

                    if (spotPrice == 0 || futuresPrice == 0)
                        continue;

                    // Calculate price premium between exchanges: (Futures - Spot) / Spot
                    decimal pricePremium = (futuresPrice - spotPrice) / spotPrice;

                    // Use actual funding interval for calculating periods per day
                    int fundingIntervalHours = fundingRate.FundingIntervalHours; // Already defaults to 8
                    decimal periodsPerDay = 24m / fundingIntervalHours;

                    // Annualized funding rate using actual intervals
                    decimal annualizedFundingRate = fundingRate.Rate * periodsPerDay * 365;

                    // Estimated trading fees (0.1% spot + 0.05% futures = 0.15%)
                    decimal estimatedTradingFees = 0.0015m;

                    // Net profit = funding earned - price premium - trading fees
                    // Only profitable if funding rate is POSITIVE (shorts pay longs -> we collect funding)
                    decimal netProfit = annualizedFundingRate - Math.Abs(pricePremium) - estimatedTradingFees;
                    decimal netProfitPercentage = netProfit * 100;

                    // Only create opportunity if:
                    // 1. Net profit exceeds minimum threshold
                    // 2. Funding rate is POSITIVE (we receive funding as short position)
                    if (netProfitPercentage >= _config.MinSpreadPercentage && annualizedFundingRate > 0)
                    {
                        // Calculate 8h profit percentage (reuse periodsPerDay from line 494)
                        var dailyFundingRate = fundingRate.Rate * periodsPerDay;
                        var profit8hPercent = (dailyFundingRate / 3m) * 100m; // Daily rate / 3 for 8h period

                        // Calculate break even time (minimum is the funding interval)
                        decimal? breakEvenTime = profit8hPercent > 0
                            ? Math.Max((POSITION_COST_PERCENT / profit8hPercent) * 8m, fundingIntervalHours)
                            : null;

                        var opportunity = new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeSpotFutures,
                            Symbol = symbol,
                            LongExchange = spotExchange,      // Buy spot here
                            ShortExchange = futuresExchange,  // Short futures here
                            ShortFundingIntervalHours = fundingIntervalHours,  // Only short position has funding
                            Exchange = spotExchange,          // For UI compatibility
                            SpotPrice = spotPrice,
                            PerpetualPrice = futuresPrice,
                            FundingRate = fundingRate.Rate,
                            AnnualizedFundingRate = annualizedFundingRate,
                            PricePremium = pricePremium,
                            SpreadRate = netProfit,
                            AnnualizedSpread = netProfit,
                            EstimatedProfitPercentage = netProfitPercentage,
                            PositionCostPercent = POSITION_COST_PERCENT,
                            BreakEvenTimeHours = breakEvenTime,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        };

                        // Calculate all metrics (current + projections)
                        CalculateOpportunityMetrics(opportunity, null, fundingRate);

                        opportunities.Add(opportunity);
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }
    */
}
