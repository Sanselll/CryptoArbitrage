using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Arbitrage.Detection;
using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Reconstructs historical market snapshots from fetched data
/// Uses existing OpportunityDetectionService to ensure consistency with production
/// Enriches opportunities with liquidity and spread metrics
/// </summary>
public class SnapshotReconstructor
{
    private readonly IOpportunityDetectionService _opportunityDetector;
    private readonly HistoricalOpportunityEnricher _enricher;
    private readonly ILogger<SnapshotReconstructor> _logger;

    public SnapshotReconstructor(
        IOpportunityDetectionService opportunityDetector,
        HistoricalOpportunityEnricher enricher,
        ILogger<SnapshotReconstructor> logger)
    {
        _opportunityDetector = opportunityDetector;
        _enricher = enricher;
        _logger = logger;
    }

    /// <summary>
    /// Backfill historical snapshots for a date range
    /// Uses parallel processing in batches for optimal performance
    /// </summary>
    public async Task<List<HistoricalMarketSnapshot>> BackfillHistoricalSnapshots(
        DateTime startDate,
        DateTime endDate,
        TimeSpan interval,
        Dictionary<string, List<FundingRateDto>> fundingHistory,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        _logger.LogInformation(
            "Starting snapshot reconstruction from {Start} to {End} with {Interval} interval",
            startDate, endDate, interval);

        // Generate all timestamps to process
        var timestamps = new List<DateTime>();
        var currentTime = startDate;
        while (currentTime <= endDate)
        {
            timestamps.Add(currentTime);
            currentTime += interval;
        }

        var totalSnapshots = timestamps.Count;
        _logger.LogInformation("Will process {Count} snapshots in parallel batches", totalSnapshots);

        var snapshots = new List<HistoricalMarketSnapshot>();
        var snapshotLock = new object();
        const int batchSize = 100; // Process 100 snapshots at a time
        var processedCount = 0;

        // Process in batches to avoid overwhelming memory
        for (int i = 0; i < timestamps.Count; i += batchSize)
        {
            var batchTimestamps = timestamps.Skip(i).Take(batchSize).ToList();
            var batchNumber = (i / batchSize) + 1;
            var totalBatches = (int)Math.Ceiling((double)timestamps.Count / batchSize);

            _logger.LogDebug("Processing batch {Batch}/{Total} ({Count} snapshots)",
                batchNumber, totalBatches, batchTimestamps.Count);

            // Process batch in parallel
            var batchTasks = batchTimestamps.Select(async timestamp =>
            {
                try
                {
                    var snapshot = await ReconstructSnapshotAt(
                        timestamp,
                        fundingHistory,
                        priceHistory,
                        liquidityMetrics);

                    return snapshot;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to reconstruct snapshot at {Time}", timestamp);
                    return null;
                }
            });

            var batchResults = await Task.WhenAll(batchTasks);

            // Add successful snapshots to collection
            var successfulSnapshots = batchResults.Where(s => s != null).Cast<HistoricalMarketSnapshot>().ToList();
            lock (snapshotLock)
            {
                snapshots.AddRange(successfulSnapshots);
                processedCount += batchTimestamps.Count;

                if (processedCount % 1000 == 0 || processedCount == totalSnapshots)
                {
                    _logger.LogInformation("Progress: {Count}/{Total} snapshots processed ({Percent:F1}%)",
                        processedCount, totalSnapshots, (double)processedCount / totalSnapshots * 100);
                }
            }
        }

        _logger.LogInformation("Reconstruction complete: {Count} snapshots created", snapshots.Count);
        return snapshots;
    }

    /// <summary>
    /// Reconstruct a single market snapshot at a specific timestamp (public API)
    /// </summary>
    public async Task<HistoricalMarketSnapshot> ReconstructSnapshotAtTimestamp(
        DateTime timestamp,
        Dictionary<string, List<FundingRateDto>> fundingHistory,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        return await ReconstructSnapshotAt(timestamp, fundingHistory, priceHistory, liquidityMetrics);
    }

    /// <summary>
    /// Reconstruct a single market snapshot at a specific timestamp
    /// </summary>
    private async Task<HistoricalMarketSnapshot> ReconstructSnapshotAt(
        DateTime timestamp,
        Dictionary<string, List<FundingRateDto>> fundingHistory,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        // Build MarketDataSnapshot from historical data
        var marketData = new MarketDataSnapshot
        {
            FetchedAt = timestamp,
            FundingRates = GetFundingRatesAt(timestamp, fundingHistory),
            PerpPrices = GetPricesAt(timestamp, priceHistory),
            PerpPriceHistory = GetPriceHistoryAt(timestamp, priceHistory),
            SpotPrices = new Dictionary<string, Dictionary<string, PriceDto>>() // Not using spot for now
        };

        // === DIAGNOSTIC LOGGING ===
        _logger.LogInformation(
            "[{Time}] MarketData prepared: FundingRates={FRCount} exchanges, PerpPrices={PPCount} exchanges",
            timestamp.ToString("HH:mm"),
            marketData.FundingRates.Count,
            marketData.PerpPrices.Count);

        foreach (var (exchange, rates) in marketData.FundingRates)
        {
            _logger.LogInformation(
                "[{Time}]   {Exchange}: {Count} funding rates",
                timestamp.ToString("HH:mm"),
                exchange,
                rates.Count);
        }

        foreach (var (exchange, prices) in marketData.PerpPrices)
        {
            _logger.LogInformation(
                "[{Time}]   {Exchange}: {Count} perp prices",
                timestamp.ToString("HH:mm"),
                exchange,
                prices.Count);
        }

        // === USE EXISTING OPPORTUNITY DETECTOR ===
        // This ensures 100% consistency with production detection logic
        var opportunities = await _opportunityDetector.DetectOpportunitiesAsync(marketData);

        // === LOG DETECTION RESULTS ===
        var cfffCount = opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures);
        var cfpsCount = opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesPriceSpread);
        _logger.LogInformation(
            "[{Time}] Detected {Total} opportunities (CFFF: {CFFF}, CFPS: {CFPS})",
            timestamp.ToString("HH:mm"),
            opportunities.Count,
            cfffCount,
            cfpsCount);

        // === LOG FUNDING RATE DETAILS FOR DETECTED OPPORTUNITIES ===
        // Show which funding rates were used for each opportunity (first 5)
        var cfffOpps = opportunities.Where(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures).Take(5).ToList();
        foreach (var opp in cfffOpps)
        {
            // Find the funding rates used for this opportunity
            var longRate = marketData.FundingRates.ContainsKey(opp.LongExchange)
                ? marketData.FundingRates[opp.LongExchange].FirstOrDefault(r => r.Symbol == opp.Symbol)
                : null;
            var shortRate = marketData.FundingRates.ContainsKey(opp.ShortExchange)
                ? marketData.FundingRates[opp.ShortExchange].FirstOrDefault(r => r.Symbol == opp.Symbol)
                : null;

            _logger.LogDebug(
                "[{Time}] {Symbol} ({LongEx}->{ShortEx}): Long={LongRate:F8} [{LongWindow}], Short={ShortRate:F8} [{ShortWindow}], FundProfit8h={Profit:F4}%",
                timestamp.ToString("HH:mm"),
                opp.Symbol,
                opp.LongExchange,
                opp.ShortExchange,
                opp.LongFundingRate,
                longRate != null ? $"{longRate.FundingTime:MM-dd HH:mm}->{longRate.NextFundingTime:MM-dd HH:mm}" : "N/A",
                opp.ShortFundingRate,
                shortRate != null ? $"{shortRate.FundingTime:MM-dd HH:mm}->{shortRate.NextFundingTime:MM-dd HH:mm}" : "N/A",
                opp.FundProfit8h);
        }

        // === DIAGNOSTIC LOGGING ===
        // Log spread statistics to understand why opportunities may not be detected
        LogSpreadStatistics(timestamp, marketData);

        // === ENRICH OPPORTUNITIES ===
        // Add volume, liquidity, and spread metrics
        opportunities = await _enricher.EnrichOpportunitiesAsync(opportunities, marketData, liquidityMetrics);

        // === CALCULATE HISTORICAL PRICE SPREAD AVERAGES ===
        // For each cross-exchange opportunity, calculate 24h and 3d price spread averages
        // Uses dictionary lookups for O(n) instead of O(n²) nested loops
        foreach (var opportunity in opportunities)
        {
            if (opportunity.Strategy == ArbitrageStrategy.CrossExchange)
            {
                var (avg24h, avg3d) = CalculateHistoricalPriceSpread(
                    timestamp,
                    opportunity.Symbol,
                    opportunity.LongExchange,
                    opportunity.ShortExchange,
                    priceHistory);

                opportunity.PriceSpread24hAvg = avg24h;
                opportunity.PriceSpread3dAvg = avg3d;
            }
        }

        // Get BTC price for market context
        var btcPrice = 0m;
        if (marketData.PerpPrices.ContainsKey("Binance") &&
            marketData.PerpPrices["Binance"].ContainsKey("BTCUSDT"))
        {
            btcPrice = marketData.PerpPrices["Binance"]["BTCUSDT"].Price;
        }

        // Calculate 24h volume for BTC (sum of last 1440 candles for 1min interval)
        var btcVolume24h = 0m;
        if (priceHistory.ContainsKey("Binance") &&
            priceHistory["Binance"].ContainsKey("BTCUSDT"))
        {
            var btcPrices = priceHistory["Binance"]["BTCUSDT"];
            var last24h = btcPrices
                .Where(p => p.Timestamp >= timestamp.AddHours(-24) && p.Timestamp <= timestamp)
                .ToList();
            btcVolume24h = last24h.Sum(p => p.Volume24h);
        }

        // Classify market regime
        var marketRegime = ClassifyMarketRegime(btcPrice, priceHistory, timestamp);

        return new HistoricalMarketSnapshot
        {
            Timestamp = timestamp,
            Opportunities = opportunities,
            PerpPrices = marketData.PerpPrices,
            FundingRates = marketData.FundingRates,
            SpotPrices = marketData.SpotPrices,
            Liquidity = null, // Not available for historical data
            BtcPrice = btcPrice,
            BtcVolume24h = btcVolume24h,
            MarketRegime = marketRegime
        };
    }

    /// <summary>
    /// Get funding rates at a specific timestamp using fundingTime and nextFundingTime windows
    /// For each symbol, finds the rate where timestamp falls between fundingTime and nextFundingTime
    /// </summary>
    private Dictionary<string, List<FundingRateDto>> GetFundingRatesAt(
        DateTime timestamp,
        Dictionary<string, List<FundingRateDto>> fundingHistory)
    {
        var result = new Dictionary<string, List<FundingRateDto>>();

        // DEBUG: Log timestamp details
        _logger.LogDebug("GetFundingRatesAt called with timestamp: {Timestamp}, Kind: {Kind}, UTC: {Utc}",
            timestamp, timestamp.Kind, timestamp.ToUniversalTime());

        foreach (var (exchange, allRates) in fundingHistory)
        {
            // Group by symbol
            var symbolGroups = allRates.GroupBy(r => r.Symbol);
            var exchangeRates = new List<FundingRateDto>();

            foreach (var symbolGroup in symbolGroups)
            {
                var symbol = symbolGroup.Key;
                var symbolRates = symbolGroup.OrderBy(r => r.FundingTime).ToList();

                // Find the funding rate where timestamp falls within the window [fundingTime, nextFundingTime)
                // The opportunity timestamp must be >= fundingTime AND < nextFundingTime
                // IMPORTANT: When intervals change mid-period, multiple rates can have overlapping windows.
                // In this case, we must use the rate with the LATEST fundingTime (most recent).
                FundingRateDto? currentRate = null;
                foreach (var rate in symbolRates)
                {
                    // Check if timestamp falls within this rate's validity window
                    if (timestamp >= rate.FundingTime && timestamp < rate.NextFundingTime)
                    {
                        // Keep this rate, but continue checking for a later one
                        // (in case interval changed mid-period and windows overlap)
                        if (currentRate == null || rate.FundingTime > currentRate.FundingTime)
                        {
                            currentRate = rate;
                        }
                    }
                }

                if (currentRate == null)
                {
                    // Log for ZORAUSDT debugging
                    if (symbol == "ZORAUSDT" && exchange == "Binance")
                    {
                        _logger.LogDebug("[{Time}] ZORAUSDT on Binance: No rate found. Available windows:", timestamp.ToString("HH:mm"));
                        foreach (var rate in symbolRates.Take(5))
                        {
                            _logger.LogDebug("  {FundingTime} -> {NextFundingTime} (Rate: {Rate:F8})",
                                rate.FundingTime.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                rate.NextFundingTime.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                rate.Rate);
                        }
                        _logger.LogDebug("  Timestamp for comparison: {Timestamp}", timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff"));
                    }
                    // No rate found with a valid window containing this timestamp
                    continue;
                }

                // Calculate 24h average from historical rates based on FundingTime
                var rates24h = symbolRates
                    .Where(r => r.FundingTime <= timestamp && r.FundingTime >= timestamp.AddHours(-24))
                    .ToList();

                decimal? average24h = rates24h.Any()
                    ? rates24h.Average(r => r.Rate)
                    : null;

                // Calculate 3-day average from historical rates based on FundingTime
                var rates3d = symbolRates
                    .Where(r => r.FundingTime <= timestamp && r.FundingTime >= timestamp.AddDays(-3))
                    .ToList();

                decimal? average3d = rates3d.Any()
                    ? rates3d.Average(r => r.Rate)
                    : null;

                // Create funding rate DTO with current rate and calculated averages
                // Do NOT use RecordedAt - it doesn't matter for this logic
                var fundingRate = new FundingRateDto
                {
                    Exchange = exchange,
                    Symbol = symbol,
                    Rate = currentRate.Rate,
                    RecordedAt = currentRate.FundingTime, // Keep for compatibility but not used in logic
                    FundingIntervalHours = currentRate.FundingIntervalHours,
                    FundingTime = currentRate.FundingTime,
                    NextFundingTime = currentRate.NextFundingTime,
                    Average24hRate = average24h,
                    Average3DayRate = average3d
                };

                // Log first few for debugging
                if (exchangeRates.Count < 3)
                {
                    _logger.LogDebug("[{Time}] {Exchange} {Symbol}: Rate={Rate:F8} (Window: {FundingTime} -> {NextFundingTime}), Avg24h={Avg24h:F8}, Avg3d={Avg3d:F8}",
                        timestamp.ToString("HH:mm"), exchange, symbol, currentRate.Rate,
                        currentRate.FundingTime.ToString("yyyy-MM-dd HH:mm"),
                        currentRate.NextFundingTime.ToString("yyyy-MM-dd HH:mm"),
                        average24h ?? 0, average3d ?? 0);
                }

                exchangeRates.Add(fundingRate);
            }

            if (exchangeRates.Any())
            {
                result[exchange] = exchangeRates;
                _logger.LogDebug("[{Time}] {Exchange}: {Count} symbols with funding rates (next rate + averages)",
                    timestamp.ToString("HH:mm"), exchange, exchangeRates.Count);
            }
            else
            {
                _logger.LogWarning("[{Time}] {Exchange}: No funding rates found (likely at edge of dataset)",
                    timestamp.ToString("HH:mm"), exchange);
            }
        }

        return result;
    }

    /// <summary>
    /// Get prices at a specific timestamp
    /// Finds the closest price candle to the timestamp
    /// </summary>
    private Dictionary<string, Dictionary<string, PriceDto>> GetPricesAt(
        DateTime timestamp,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory)
    {
        var result = new Dictionary<string, Dictionary<string, PriceDto>>();

        foreach (var (exchange, symbols) in priceHistory)
        {
            var exchangePrices = new Dictionary<string, PriceDto>();

            foreach (var (symbol, prices) in symbols)
            {
                // Find the closest price to this timestamp (within 1 minute tolerance)
                var closestPrice = prices
                    .Where(p => Math.Abs((p.Timestamp - timestamp).TotalMinutes) <= 1)
                    .OrderBy(p => Math.Abs((p.Timestamp - timestamp).TotalSeconds))
                    .FirstOrDefault();

                if (closestPrice != null)
                {
                    // Calculate 24h volume (sum of last 1440 1-minute candles)
                    var volume24h = prices
                        .Where(p => p.Timestamp >= timestamp.AddHours(-24) && p.Timestamp <= timestamp)
                        .Sum(p => p.Volume24h);

                    exchangePrices[symbol] = new PriceDto
                    {
                        Symbol = symbol,
                        Price = closestPrice.Price,
                        Volume24h = volume24h,
                        Timestamp = closestPrice.Timestamp
                    };
                }
            }

            if (exchangePrices.Any())
            {
                result[exchange] = exchangePrices;
            }
        }

        return result;
    }

    /// <summary>
    /// Get price history (last 30 samples) at a specific timestamp
    /// Used for spread volatility calculations in opportunity enrichment
    /// </summary>
    private Dictionary<string, Dictionary<string, PriceHistoryDto>> GetPriceHistoryAt(
        DateTime timestamp,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory)
    {
        var result = new Dictionary<string, Dictionary<string, PriceHistoryDto>>();

        foreach (var (exchange, symbols) in priceHistory)
        {
            var exchangeHistory = new Dictionary<string, PriceHistoryDto>();

            foreach (var (symbol, prices) in symbols)
            {
                // Get the last 30 prices before (and including) the timestamp
                var historicalPrices = prices
                    .Where(p => p.Timestamp <= timestamp)
                    .OrderByDescending(p => p.Timestamp)
                    .Take(PriceHistoryDto.MaxSamples)
                    .OrderBy(p => p.Timestamp) // Reverse to oldest-first order
                    .ToList();

                if (historicalPrices.Any())
                {
                    var priceHistoryDto = new PriceHistoryDto
                    {
                        Exchange = exchange,
                        Symbol = symbol,
                        PriceHistory = historicalPrices.Select(p => p.Price).ToList(),
                        Timestamps = historicalPrices.Select(p => p.Timestamp).ToList()
                    };

                    exchangeHistory[symbol] = priceHistoryDto;
                }
            }

            if (exchangeHistory.Any())
            {
                result[exchange] = exchangeHistory;
            }
        }

        return result;
    }

    /// <summary>
    /// Classify market regime based on BTC price action
    /// </summary>
    private string ClassifyMarketRegime(
        decimal currentBtcPrice,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory,
        DateTime timestamp)
    {
        if (currentBtcPrice == 0)
            return "Unknown";

        try
        {
            // Get BTC price history
            if (!priceHistory.ContainsKey("Binance") || !priceHistory["Binance"].ContainsKey("BTCUSDT"))
                return "Unknown";

            var btcPrices = priceHistory["Binance"]["BTCUSDT"];

            // Get prices from last 24 hours
            var last24h = btcPrices
                .Where(p => p.Timestamp >= timestamp.AddHours(-24) && p.Timestamp <= timestamp)
                .OrderBy(p => p.Timestamp)
                .ToList();

            if (last24h.Count < 100)
                return "Unknown";

            // Calculate 24h change
            var price24hAgo = last24h.First().Price;
            var priceChange24h = ((currentBtcPrice - price24hAgo) / price24hAgo) * 100;

            // Calculate volatility (standard deviation of returns)
            var returns = new List<decimal>();
            for (int i = 1; i < last24h.Count; i++)
            {
                var ret = (last24h[i].Price - last24h[i - 1].Price) / last24h[i - 1].Price;
                returns.Add(ret);
            }

            var avgReturn = returns.Average();
            var variance = returns.Sum(r => (r - avgReturn) * (r - avgReturn)) / returns.Count;
            var volatility = (decimal)Math.Sqrt((double)variance) * 100;

            // Classify regime
            if (volatility > 0.5m)
                return "HighVolatility";
            else if (priceChange24h > 3)
                return "BullTrending";
            else if (priceChange24h < -3)
                return "BearTrending";
            else
                return "Ranging";
        }
        catch
        {
            return "Unknown";
        }
    }

    /// <summary>
    /// Save snapshots to file for later use
    /// </summary>
    public async Task SaveSnapshotsToFile(List<HistoricalMarketSnapshot> snapshots, string filePath)
    {
        _logger.LogInformation("Saving {Count} snapshots to {Path}...", snapshots.Count, filePath);

        var options = new JsonSerializerOptions
        {
            WriteIndented = false,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };

        await using var fileStream = File.Create(filePath);
        await JsonSerializer.SerializeAsync(fileStream, snapshots, options);

        _logger.LogInformation("Snapshots saved successfully");
    }

    /// <summary>
    /// Load snapshots from file
    /// </summary>
    public async Task<List<HistoricalMarketSnapshot>> LoadSnapshotsFromFile(string filePath)
    {
        _logger.LogInformation("Loading snapshots from {Path}...", filePath);

        await using var fileStream = File.OpenRead(filePath);
        var snapshots = await JsonSerializer.DeserializeAsync<List<HistoricalMarketSnapshot>>(fileStream);

        _logger.LogInformation("Loaded {Count} snapshots", snapshots?.Count ?? 0);
        return snapshots ?? new List<HistoricalMarketSnapshot>();
    }

    /// <summary>
    /// Log spread statistics to help diagnose why opportunities may not be detected
    /// </summary>
    private void LogSpreadStatistics(DateTime timestamp, MarketDataSnapshot marketData)
    {
        try
        {
            if (marketData.FundingRates.Count < 2 || marketData.PerpPrices.Count < 2)
                return;

            var fundingSpreads = new List<decimal>();
            var priceSpreads = new List<decimal>();

            var exchanges = marketData.FundingRates.Keys.ToList();

            // Calculate spreads for all symbol-exchange combinations
            for (int i = 0; i < exchanges.Count; i++)
            {
                for (int j = i + 1; j < exchanges.Count; j++)
                {
                    var ex1 = exchanges[i];
                    var ex2 = exchanges[j];

                    var rates1 = marketData.FundingRates[ex1];
                    var rates2 = marketData.FundingRates[ex2];

                    // Find common symbols
                    var commonSymbols = rates1.Select(r => r.Symbol)
                        .Intersect(rates2.Select(r => r.Symbol))
                        .ToList();

                    foreach (var symbol in commonSymbols)
                    {
                        var rate1 = rates1.First(r => r.Symbol == symbol);
                        var rate2 = rates2.First(r => r.Symbol == symbol);

                        // Calculate funding profit for 8h period
                        var fundingDiff = Math.Abs(rate2.Rate - rate1.Rate);
                        var longRate = Math.Min(rate1.Rate, rate2.Rate);
                        var shortRate = Math.Max(rate1.Rate, rate2.Rate);

                        int longInterval = longRate == rate1.Rate ? rate1.FundingIntervalHours : rate2.FundingIntervalHours;
                        int shortInterval = shortRate == rate1.Rate ? rate1.FundingIntervalHours : rate2.FundingIntervalHours;

                        var longDailyRate = (longRate * 100m) * (24m / longInterval);
                        var shortDailyRate = (shortRate * 100m) * (24m / shortInterval);
                        var netDailyRate = shortDailyRate - longDailyRate;
                        var fundProfit8h = netDailyRate / 3m; // Daily / 3 for 8h

                        fundingSpreads.Add(fundProfit8h);

                        // Calculate price spread
                        if (marketData.PerpPrices.ContainsKey(ex1) &&
                            marketData.PerpPrices[ex1].ContainsKey(symbol) &&
                            marketData.PerpPrices.ContainsKey(ex2) &&
                            marketData.PerpPrices[ex2].ContainsKey(symbol))
                        {
                            var price1 = marketData.PerpPrices[ex1][symbol].Price;
                            var price2 = marketData.PerpPrices[ex2][symbol].Price;

                            if (price1 > 0 && price2 > 0)
                            {
                                var priceSpread = Math.Abs(price2 - price1) / Math.Min(price1, price2) * 100m;
                                priceSpreads.Add(priceSpread);
                            }
                        }
                    }
                }
            }

            if (fundingSpreads.Any())
            {
                var avgFunding = fundingSpreads.Average();
                var maxFunding = fundingSpreads.Max();
                var aboveThreshold = fundingSpreads.Count(s => s >= 0.1m);

                _logger.LogInformation(
                    "[{Time}] Funding Spreads: Avg={Avg:F4}%, Max={Max:F4}%, Above 0.1%: {Count}/{Total}",
                    timestamp.ToString("HH:mm"),
                    avgFunding,
                    maxFunding,
                    aboveThreshold,
                    fundingSpreads.Count);
            }

            if (priceSpreads.Any())
            {
                var avgPrice = priceSpreads.Average();
                var maxPrice = priceSpreads.Max();
                var aboveThreshold = priceSpreads.Count(s => s >= 0.3m);

                _logger.LogInformation(
                    "[{Time}] Price Spreads: Avg={Avg:F4}%, Max={Max:F4}%, Above 0.3%: {Count}/{Total}",
                    timestamp.ToString("HH:mm"),
                    avgPrice,
                    maxPrice,
                    aboveThreshold,
                    priceSpreads.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error logging spread statistics");
        }
    }

    /// <summary>
    /// Calculate 24h and 3D average price spreads using indexed lookups - O(n log n) instead of O(n²)
    /// </summary>
    private (decimal? avg24h, decimal? avg3d) CalculateHistoricalPriceSpread(
        DateTime timestamp,
        string symbol,
        string longExchange,
        string shortExchange,
        Dictionary<string, Dictionary<string, List<PriceDto>>> priceHistory)
    {
        try
        {
            var threeDaysAgo = timestamp.AddDays(-3);
            var oneDayAgo = timestamp.AddHours(-24);

            // Get price lists for both exchanges
            if (!priceHistory.ContainsKey(longExchange) ||
                !priceHistory[longExchange].ContainsKey(symbol) ||
                !priceHistory.ContainsKey(shortExchange) ||
                !priceHistory[shortExchange].ContainsKey(symbol))
            {
                return (null, null);
            }

            // OPTIMIZATION: Pre-filter to only last 3 days instead of processing ALL history
            // This reduces from ~270k records to ~4,320 records per symbol
            var longPrices = priceHistory[longExchange][symbol]
                .Where(p => p.Timestamp >= threeDaysAgo && p.Timestamp <= timestamp)
                .OrderBy(p => p.Timestamp)
                .ToList();

            var shortPrices = priceHistory[shortExchange][symbol]
                .Where(p => p.Timestamp >= threeDaysAgo && p.Timestamp <= timestamp)
                .OrderBy(p => p.Timestamp)
                .ToList();

            if (!longPrices.Any() || !shortPrices.Any())
            {
                return (null, null);
            }

            // OPTIMIZATION: Build dictionary for O(1) lookup instead of O(n) FirstOrDefault
            // This changes the inner loop from O(n²) to O(n)
            var shortPricesByTime = new Dictionary<DateTime, PriceDto>();
            foreach (var sp in shortPrices)
            {
                if (!shortPricesByTime.ContainsKey(sp.Timestamp))
                {
                    shortPricesByTime[sp.Timestamp] = sp;
                }
            }

            // Calculate spreads - now O(n) instead of O(n²)
            var spreads = new List<(DateTime timestamp, decimal spread)>();

            foreach (var longPrice in longPrices)
            {
                // Try exact match first (O(1) lookup)
                if (shortPricesByTime.TryGetValue(longPrice.Timestamp, out var shortPrice))
                {
                    if (longPrice.Price > 0)
                    {
                        decimal spread = ((shortPrice.Price - longPrice.Price) / longPrice.Price) * 100m;
                        spreads.Add((longPrice.Timestamp, spread));
                    }
                }
                else
                {
                    // Fallback: Find closest within 1 minute (still needed for misaligned timestamps)
                    // But now operates on filtered dataset (4k records) not full history (270k records)
                    var shortPrice1Min = shortPrices.FirstOrDefault(sp =>
                        Math.Abs((sp.Timestamp - longPrice.Timestamp).TotalMinutes) < 1);

                    if (shortPrice1Min != null && longPrice.Price > 0)
                    {
                        decimal spread = ((shortPrice1Min.Price - longPrice.Price) / longPrice.Price) * 100m;
                        spreads.Add((longPrice.Timestamp, spread));
                    }
                }
            }

            if (!spreads.Any())
            {
                return (null, null);
            }

            // Calculate 24h average
            var last24hSpreads = spreads.Where(s => s.timestamp >= oneDayAgo && s.timestamp <= timestamp).ToList();
            decimal? avg24h = last24hSpreads.Any()
                ? last24hSpreads.Average(s => s.spread)
                : null;

            // Calculate 3D average
            var last3dSpreads = spreads.Where(s => s.timestamp >= threeDaysAgo && s.timestamp <= timestamp).ToList();
            decimal? avg3d = last3dSpreads.Any()
                ? last3dSpreads.Average(s => s.spread)
                : null;

            return (avg24h, avg3d);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error calculating historical price spreads for {Symbol} ({LongEx} vs {ShortEx})",
                symbol, longExchange, shortExchange);
            return (null, null);
        }
    }
}
