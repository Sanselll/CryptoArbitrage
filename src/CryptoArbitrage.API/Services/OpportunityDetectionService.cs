using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service responsible for detecting arbitrage opportunities from market data
/// </summary>
public class OpportunityDetectionService : IOpportunityDetectionService
{
    private readonly ArbitrageConfig _config;
    private readonly ILogger<OpportunityDetectionService> _logger;

    public OpportunityDetectionService(
        ArbitrageConfig config,
        ILogger<OpportunityDetectionService> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Detect all types of arbitrage opportunities from the provided market data
    /// </summary>
    public async Task<List<ArbitrageOpportunityDto>> DetectOpportunitiesAsync(MarketDataSnapshot snapshot)
    {
        var opportunities = new List<ArbitrageOpportunityDto>();

        // Cross-exchange arbitrage (requires 2+ exchanges)
        if (snapshot.FundingRates.Count >= 2)
        {
            // Futures/Futures cross-exchange
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeFuturesFutures))
            {
                var crossExchangeFuturesOpps = await DetectCrossExchangeFuturesOpportunitiesAsync(
                    snapshot.FundingRates, snapshot.PerpPrices);
                opportunities.AddRange(crossExchangeFuturesOpps);
            }

            // Spot/Futures cross-exchange
            if (_config.IsStrategyEnabled(StrategySubType.CrossExchangeSpotFutures))
            {
                var crossExchangeSpotFuturesOpps = await DetectCrossExchangeSpotFuturesOpportunitiesAsync(
                    snapshot.FundingRates, snapshot.SpotPrices, snapshot.PerpPrices);
                opportunities.AddRange(crossExchangeSpotFuturesOpps);
            }
        }

        // Spot-perpetual arbitrage (same exchange)
        if (_config.IsStrategyEnabled(StrategySubType.SpotPerpetualSameExchange))
        {
            var spotPerpOpps = await DetectSpotPerpetualOpportunitiesAsync(
                snapshot.FundingRates, snapshot.SpotPrices, snapshot.PerpPrices);
            opportunities.AddRange(spotPerpOpps);
        }

        _logger.LogInformation("Detected {Count} opportunities from market snapshot", opportunities.Count);

        return opportunities;
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
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
                        price1 = perpPrices[exchange1][symbol];
                    if (perpPrices.ContainsKey(exchange2) && perpPrices[exchange2].ContainsKey(symbol))
                        price2 = perpPrices[exchange2][symbol];

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
                    var annualizedNetFunding = netFundingRate * 3 * 365; // 3 fundings per day, 365 days

                    // Only profitable if net funding is positive
                    if (annualizedNetFunding * 100 >= _config.MinSpreadPercentage)
                    {
                        // Determine which price goes to which field based on long/short exchanges
                        decimal longExchangePrice = (longExchange == exchange1) ? price1 : price2;
                        decimal shortExchangePrice = (shortExchange == exchange1) ? price1 : price2;

                        opportunities.Add(new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeFuturesFutures,
                            Symbol = symbol,
                            LongExchange = longExchange,
                            ShortExchange = shortExchange,
                            LongFundingRate = longRate,
                            ShortFundingRate = shortRate,
                            // Use SpotPrice for longExchange perp price, PerpetualPrice for shortExchange perp price
                            SpotPrice = longExchangePrice,
                            PerpetualPrice = shortExchangePrice,
                            SpreadRate = netFundingRate,
                            AnnualizedSpread = annualizedNetFunding,
                            EstimatedProfitPercentage = annualizedNetFunding * 100,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.AnnualizedSpread).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectSpotPerpetualOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, SpotPriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
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
                decimal perpetualPrice = exchangePerpPrices[fundingRate.Symbol];

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
                    opportunities.Add(new ArbitrageOpportunityDto
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
                        Status = OpportunityStatus.Detected,
                        DetectedAt = DateTime.UtcNow
                    });
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }

    private async Task<List<ArbitrageOpportunityDto>> DetectCrossExchangeSpotFuturesOpportunitiesAsync(
        Dictionary<string, List<FundingRateDto>> fundingRates,
        Dictionary<string, Dictionary<string, SpotPriceDto>> spotPrices,
        Dictionary<string, Dictionary<string, decimal>> perpPrices)
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
                    var futuresPrice = futuresPrices[symbol];
                    var fundingRate = futuresRates.First(r => r.Symbol == symbol);

                    if (spotPrice == 0 || futuresPrice == 0)
                        continue;

                    // Calculate price premium between exchanges: (Futures - Spot) / Spot
                    decimal pricePremium = (futuresPrice - spotPrice) / spotPrice;

                    // Annualized funding rate
                    decimal annualizedFundingRate = fundingRate.AnnualizedRate;

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
                        opportunities.Add(new ArbitrageOpportunityDto
                        {
                            Strategy = ArbitrageStrategy.CrossExchange,
                            SubType = StrategySubType.CrossExchangeSpotFutures,
                            Symbol = symbol,
                            LongExchange = spotExchange,      // Buy spot here
                            ShortExchange = futuresExchange,  // Short futures here
                            Exchange = spotExchange,          // For UI compatibility
                            SpotPrice = spotPrice,
                            PerpetualPrice = futuresPrice,
                            FundingRate = fundingRate.Rate,
                            AnnualizedFundingRate = annualizedFundingRate,
                            PricePremium = pricePremium,
                            SpreadRate = netProfit,
                            AnnualizedSpread = netProfit,
                            EstimatedProfitPercentage = netProfitPercentage,
                            Status = OpportunityStatus.Detected,
                            DetectedAt = DateTime.UtcNow
                        });
                    }
                }
            }
        }

        await Task.CompletedTask;
        return opportunities.OrderByDescending(o => o.EstimatedProfitPercentage).ToList();
    }
}
