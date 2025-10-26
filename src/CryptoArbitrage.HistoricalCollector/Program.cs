using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Arbitrage.Detection;
using CryptoArbitrage.API.Services.DataCollection;
using CryptoArbitrage.API.Services.Exchanges;
using CryptoArbitrage.HistoricalCollector.Config;
using CryptoArbitrage.HistoricalCollector.Exporters;
using CryptoArbitrage.HistoricalCollector.Infrastructure;
using CryptoArbitrage.HistoricalCollector.Models;
using CryptoArbitrage.HistoricalCollector.Services;
using CryptoArbitrage.HistoricalCollector.Services.Persistence;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.CommandLine;
using System.Globalization;

// Build host with DI container
var builder = Host.CreateApplicationBuilder(args);

// === CONFIGURATION ===
builder.Configuration
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables();

// === REGISTER SERVICES FROM API PROJECT ===
// Register ArbitrageConfig as singleton
builder.Services.AddSingleton(sp =>
{
    var config = new ArbitrageConfig();
    builder.Configuration.GetSection("Arbitrage").Bind(config);
    return config;
});

// Register DataCollectionConfig as singleton
builder.Services.AddSingleton(sp =>
{
    var config = new DataCollectionConfig();
    builder.Configuration.GetSection("DataCollection").Bind(config);
    return config;
});

// Register mock repositories (OpportunityDetectionService needs them but we won't use their features)
builder.Services.AddSingleton<CryptoArbitrage.API.Services.DataCollection.Abstractions.IDataRepository<Dictionary<string, List<HistoricalPriceDto>>>>(
    sp => new MockDataRepository<Dictionary<string, List<HistoricalPriceDto>>>());
builder.Services.AddSingleton<CryptoArbitrage.API.Services.DataCollection.Abstractions.IDataRepository<UserDataSnapshot>>(
    sp => new MockDataRepository<UserDataSnapshot>());

// Register OpportunityDetectionService (reuse from API!)
builder.Services.AddScoped<IOpportunityDetectionService, OpportunityDetectionService>();

// === REGISTER HISTORICAL COLLECTOR SERVICES ===
// Note: HttpClient is no longer needed - we use Binance.Net and Bybit.Net libraries via connectors
builder.Services.AddSingleton<HistoricalDataFetcher>();
builder.Services.AddSingleton<LiquidityDataFetcher>();
builder.Services.AddSingleton<HistoricalOpportunityEnricher>();
builder.Services.AddSingleton<SnapshotReconstructor>();
builder.Services.AddSingleton<PositionSimulator>();
builder.Services.AddSingleton<CsvExporter>();
builder.Services.AddSingleton<SymbolDiscoveryService>();

// === REGISTER PERSISTENCE SERVICES ===
builder.Services.AddSingleton<CryptoArbitrage.HistoricalCollector.Services.Persistence.RawDataPersister>();
builder.Services.AddSingleton<CryptoArbitrage.HistoricalCollector.Services.Persistence.RawDataLoader>();
builder.Services.AddSingleton<CryptoArbitrage.HistoricalCollector.Services.Persistence.OpportunityPersister>();

// === REGISTER EXCHANGE CONNECTORS (needed for liquidity fetching and symbol discovery) ===
builder.Services.AddSingleton<ConnectorManager>();
builder.Services.AddScoped<BinanceConnector>();
builder.Services.AddScoped<BybitConnector>();

// === LOGGING ===
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.SetMinimumLevel(LogLevel.Information);

var host = builder.Build();

// === COMMAND LINE INTERFACE ===
var rootCommand = new RootCommand("Historical data collector for crypto arbitrage ML training");

// === COMMAND: BACKFILL ===
var backfillCommand = new Command("backfill", "Backfill historical market data and snapshots");

var startDateOption = new Option<DateTime>(
    "--start-date",
    description: "Start date (YYYY-MM-DD)",
    getDefaultValue: () => DateTime.UtcNow.AddMonths(-1));

var endDateOption = new Option<DateTime>(
    "--end-date",
    description: "End date (YYYY-MM-DD)",
    getDefaultValue: () => DateTime.UtcNow);

var intervalOption = new Option<string>(
    "--interval",
    description: "Snapshot interval (1m, 5m, 15m, etc.)",
    getDefaultValue: () => "1m");

var exchangesOption = new Option<string>(
    "--exchanges",
    description: "Comma-separated list of exchanges",
    getDefaultValue: () => "Binance,Bybit");

var symbolsOption = new Option<string?>(
    "--symbols",
    description: "Comma-separated list of symbols (leave empty for auto-discovery)",
    getDefaultValue: () => null);

var outputOption = new Option<string>(
    "--output",
    description: "Output file for snapshots",
    getDefaultValue: () => "snapshots.json");

backfillCommand.AddOption(startDateOption);
backfillCommand.AddOption(endDateOption);
backfillCommand.AddOption(intervalOption);
backfillCommand.AddOption(exchangesOption);
backfillCommand.AddOption(symbolsOption);
backfillCommand.AddOption(outputOption);

backfillCommand.SetHandler(async (startDate, endDate, interval, exchanges, symbols, output) =>
{
    Console.WriteLine("=== Historical Data Backfill ===");
    Console.WriteLine($"Start Date: {startDate:yyyy-MM-dd}");
    Console.WriteLine($"End Date: {endDate:yyyy-MM-dd}");
    Console.WriteLine($"Interval: {interval}");
    Console.WriteLine($"Exchanges: {exchanges}");
    Console.WriteLine();

    var exchangeList = exchanges.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList();
    var symbolList = await GetSymbolsAsync(symbols, exchangeList, host);

    if (symbolList.Count == 0)
    {
        Console.WriteLine("ERROR: No symbols found!");
        return;
    }

    Console.WriteLine($"Processing {symbolList.Count} symbols");
    Console.WriteLine();

    var fetcher = host.Services.GetRequiredService<HistoricalDataFetcher>();
    var liquidityFetcher = host.Services.GetRequiredService<LiquidityDataFetcher>();
    var reconstructor = host.Services.GetRequiredService<SnapshotReconstructor>();

    // Step 1: Fetch current liquidity metrics (as proxy for historical)
    Console.WriteLine("Step 1: Fetching current liquidity metrics...");
    var liquidityMetrics = await liquidityFetcher.FetchCurrentLiquidityMetrics(
        exchangeList,
        symbolList);

    // Step 2: Fetch historical funding rates
    Console.WriteLine("Step 2: Fetching historical funding rates...");
    var fundingHistory = await fetcher.FetchAllFundingRates(
        startDate,
        endDate,
        exchangeList,
        symbolList);

    // Step 3: Fetch historical price klines
    Console.WriteLine("Step 3: Fetching historical price klines...");
    var priceHistory = await fetcher.FetchAllPriceKlines(
        startDate,
        endDate,
        exchangeList,
        symbolList,
        interval);

    // Step 4: Reconstruct snapshots with enrichment
    Console.WriteLine("Step 4: Reconstructing market snapshots...");
    var intervalTimeSpan = ParseInterval(interval);
    var snapshots = await reconstructor.BackfillHistoricalSnapshots(
        startDate,
        endDate,
        intervalTimeSpan,
        fundingHistory,
        priceHistory,
        liquidityMetrics);

    // Step 5: Save snapshots
    Console.WriteLine("Step 5: Saving snapshots to file...");
    await reconstructor.SaveSnapshotsToFile(snapshots, output);

    Console.WriteLine();
    Console.WriteLine($"✅ Backfill complete!");
    Console.WriteLine($"   Total snapshots: {snapshots.Count}");
    Console.WriteLine($"   Total opportunities: {snapshots.Sum(s => s.OpportunitiesCount)}");
    Console.WriteLine($"   Output file: {output}");

}, startDateOption, endDateOption, intervalOption, exchangesOption, symbolsOption, outputOption);

// === COMMAND: COLLECT (Phase 1 - Data Collection Only) ===
var collectCommand = new Command("collect", "Collect and persist raw market data to data/raw/ folder");

collectCommand.AddOption(startDateOption);
collectCommand.AddOption(endDateOption);
collectCommand.AddOption(exchangesOption);
collectCommand.AddOption(symbolsOption);

collectCommand.SetHandler(async (startDate, endDate, exchanges, symbols) =>
{
    Console.WriteLine("=== PHASE 1: DATA COLLECTION (DAY-BY-DAY) ===");
    Console.WriteLine($"Start Date: {startDate:yyyy-MM-dd}");
    Console.WriteLine($"End Date: {endDate:yyyy-MM-dd}");
    Console.WriteLine($"Exchanges: {exchanges}");
    Console.WriteLine();

    var exchangeList = exchanges.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList();
    var symbolList = await GetSymbolsAsync(symbols, exchangeList, host);

    if (symbolList.Count == 0)
    {
        Console.WriteLine("ERROR: No symbols found!");
        return;
    }

    Console.WriteLine($"Processing {symbolList.Count} symbols");
    Console.WriteLine();

    var fetcher = host.Services.GetRequiredService<HistoricalDataFetcher>();
    var liquidityFetcher = host.Services.GetRequiredService<LiquidityDataFetcher>();
    var persister = host.Services.GetRequiredService<RawDataPersister>();

    // === DAY-BY-DAY COLLECTION LOOP ===
    var currentDay = startDate.Date;
    var endDay = endDate.Date;
    var totalDays = (endDay - currentDay).Days + 1;
    var dayCount = 0;
    var skippedDays = 0;
    var collectedDays = 0;

    Console.WriteLine($"📅 Collecting data for {totalDays} day(s)...");
    Console.WriteLine();

    while (currentDay <= endDay)
    {
        dayCount++;
        var dayString = currentDay.ToString("yyyy-MM-dd");
        var dayDirectory = Path.Combine("data/raw", dayString);
        var manifestPath = Path.Combine(dayDirectory, "manifest.json");

        Console.WriteLine($"[Day {dayCount}/{totalDays}] {dayString}");

        // Check if this day already has data
        if (File.Exists(manifestPath))
        {
            Console.WriteLine($"  ⏭️  Skipping - data already exists at {dayDirectory}");
            skippedDays++;
            currentDay = currentDay.AddDays(1);
            continue;
        }

        try
        {
            Console.WriteLine($"  📥 Collecting...");

            // Fetch liquidity only once (current snapshot applies to all days)
            Dictionary<string, Dictionary<string, LiquidityMetricsDto>>? liquidityMetrics = null;
            if (dayCount == 1)
            {
                Console.WriteLine($"  Step 1/4: Fetching liquidity metrics...");
                liquidityMetrics = await liquidityFetcher.FetchCurrentLiquidityMetrics(exchangeList, symbolList);
            }

            // Fetch data for this specific day only
            var dayEnd = currentDay.AddDays(1).AddTicks(-1); // End of day

            Console.WriteLine($"  Step 2/4: Fetching funding rates...");
            var fundingHistory = await fetcher.FetchAllFundingRates(currentDay, dayEnd, exchangeList, symbolList);

            Console.WriteLine($"  Step 3/4: Fetching price klines...");
            var priceHistory = await fetcher.FetchAllPriceKlines(currentDay, dayEnd, exchangeList, symbolList, "1m");

            Console.WriteLine($"  Step 4/4: Saving data...");
            var manifest = await persister.SaveRawDataAsync(
                currentDay, dayEnd, exchangeList, symbolList,
                fundingHistory, priceHistory, liquidityMetrics);

            Console.WriteLine($"  ✅ Saved to {manifest.DataPath}");
            Console.WriteLine($"     Funding: {manifest.TotalFundingRates:N0}, Klines: {manifest.TotalPriceKlines:N0}");
            collectedDays++;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ❌ ERROR: {ex.Message}");
            Console.WriteLine($"  Skipping {dayString} and continuing...");
        }

        Console.WriteLine();
        currentDay = currentDay.AddDays(1);
    }

    Console.WriteLine("========================================");
    Console.WriteLine("✅ DATA COLLECTION COMPLETE!");
    Console.WriteLine("========================================");
    Console.WriteLine($"Total days: {totalDays}");
    Console.WriteLine($"Collected: {collectedDays} days");
    Console.WriteLine($"Skipped: {skippedDays} days (already had data)");
    Console.WriteLine($"Symbols: {symbolList.Count}");
    Console.WriteLine($"Exchanges: {string.Join(", ", exchangeList)}");
    Console.WriteLine($"Data saved to: data/raw/");
    Console.WriteLine();
    Console.WriteLine("Next step: Run 'detect' command to find opportunities from this data");
    Console.WriteLine();

}, startDateOption, endDateOption, exchangesOption, symbolsOption);

// === COMMAND: DETECT (Phase 2 - Opportunity Detection) ===
var detectCommand = new Command("detect", "Detect opportunities from persisted raw data in data/raw/");

detectCommand.AddOption(startDateOption);
detectCommand.AddOption(endDateOption);
detectCommand.AddOption(intervalOption);

var detectOutputOption = new Option<string?>(
    "--output",
    description: "Output file for opportunities (optional, auto-generated if not specified)",
    getDefaultValue: () => null);

var singleTimestampOption = new Option<string?>(
    "--single-timestamp",
    description: "Generate opportunity for a single specific timestamp (format: yyyy-MM-ddTHH:mm:ss). Data will be loaded from start/end date range.",
    getDefaultValue: () => null);

detectCommand.AddOption(detectOutputOption);
detectCommand.AddOption(singleTimestampOption);

detectCommand.SetHandler(async (startDate, endDate, interval, output, singleTimestamp) =>
{
    Console.WriteLine("=== PHASE 2: OPPORTUNITY DETECTION ===");
    Console.WriteLine($"Start Date: {startDate:yyyy-MM-dd}");
    Console.WriteLine($"End Date: {endDate:yyyy-MM-dd}");
    Console.WriteLine($"Interval: {interval}");

    // Parse single timestamp if provided
    DateTime? specificTimestamp = null;
    if (!string.IsNullOrEmpty(singleTimestamp))
    {
        if (DateTime.TryParse(singleTimestamp, out var parsedTime))
        {
            specificTimestamp = parsedTime;
            Console.WriteLine($"Single Timestamp Mode: {specificTimestamp:yyyy-MM-dd HH:mm:ss}");
        }
        else
        {
            Console.WriteLine($"❌ Invalid timestamp format: {singleTimestamp}");
            Console.WriteLine("Expected format: yyyy-MM-ddTHH:mm:ss (e.g., 2025-10-24T16:35:00)");
            return;
        }
    }
    Console.WriteLine();

    var loader = host.Services.GetRequiredService<RawDataLoader>();
    var reconstructor = host.Services.GetRequiredService<SnapshotReconstructor>();
    var opportunityPersister = host.Services.GetRequiredService<OpportunityPersister>();

    // Step 1: Load manifest to get exchanges
    Console.WriteLine("Step 1: Loading collection manifest...");
    var manifest = await loader.LoadManifestAsync(startDate, endDate);
    var exchanges = manifest.Exchanges;

    var intervalTimeSpan = ParseInterval(interval);
    var allSnapshots = new List<HistoricalMarketSnapshot>();

    // === SINGLE TIMESTAMP MODE ===
    if (specificTimestamp.HasValue)
    {
        Console.WriteLine($"Generating opportunity for single timestamp: {specificTimestamp:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine();

        // Load ALL available data from raw folder (for historical averages)
        var (fundingRates, priceKlines, liquidityMetrics) = await loader.LoadAllAvailableDataAsync(exchanges);

        // Check if we have data
        var hasFundingData = fundingRates.Any(kvp => kvp.Value.Any());
        var hasPriceData = priceKlines.Any(kvp => kvp.Value.Any());

        if (!hasFundingData && !hasPriceData)
        {
            Console.WriteLine("❌ No data found in specified date range");
            return;
        }

        Console.WriteLine($"Loaded data: {fundingRates.Sum(kvp => kvp.Value.Count)} funding rates, {priceKlines.Sum(kvp => kvp.Value.Sum(s => s.Value.Count))} klines");
        Console.WriteLine();

        // Generate opportunity for the single timestamp only
        Console.WriteLine("Generating opportunity...");
        var snapshot = await reconstructor.ReconstructSnapshotAtTimestamp(
            specificTimestamp.Value,
            fundingRates,
            priceKlines,
            liquidityMetrics ?? new Dictionary<string, Dictionary<string, LiquidityMetricsDto>>());

        allSnapshots.Add(snapshot);

        // Save the single snapshot
        var outputFile = output ?? $"data/opportunities/single_{specificTimestamp:yyyy-MM-dd_HHmm}.json";
        Console.WriteLine($"Saving to {outputFile}...");
        await opportunityPersister.SaveOpportunitiesAsync(allSnapshots, specificTimestamp.Value, specificTimestamp.Value, outputFile);

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("✅ SINGLE OPPORTUNITY DETECTION COMPLETE!");
        Console.WriteLine("========================================");
        Console.WriteLine($"Timestamp: {specificTimestamp:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"Opportunities detected: {snapshot.Opportunities.Count}");
        if (snapshot.Opportunities.Any())
        {
            var cfffCount = snapshot.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures);
            var cfpsCount = snapshot.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesPriceSpread);
            Console.WriteLine($"  CFFF (Funding): {cfffCount}");
            Console.WriteLine($"  CFPS (Price Spread): {cfpsCount}");
        }
        Console.WriteLine($"Saved to: {outputFile}");
        Console.WriteLine();
        return;
    }

    // === STANDARD DAY-BY-DAY MODE ===
    var totalDays = (int)(endDate.Date - startDate.Date).TotalDays + 1;
    Console.WriteLine($"Generating opportunities for {totalDays} day(s): {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");
    Console.WriteLine();

    // Step 2: Load ALL available data from raw folder (for historical averages)
    var (allFundingRates, allPriceKlines, allLiquidityMetrics) = await loader.LoadAllAvailableDataAsync(exchanges);

    // Check if we have any data for the range
    var hasAnyFundingData = allFundingRates.Any(kvp => kvp.Value.Any());
    var hasAnyPriceData = allPriceKlines.Any(kvp => kvp.Value.Any());

    if (!hasAnyFundingData && !hasAnyPriceData)
    {
        Console.WriteLine($"⚠️  No data found for date range - ensure 'collect' command has been run for these dates");
        return;
    }

    Console.WriteLine($"✓ Data loaded successfully");
    Console.WriteLine();

    // Generate list of all days to process
    var daysToProcess = new List<DateTime>();
    var currentDay = DateTime.SpecifyKind(startDate.Date, DateTimeKind.Utc);
    while (currentDay <= DateTime.SpecifyKind(endDate.Date, DateTimeKind.Utc))
    {
        daysToProcess.Add(currentDay);
        currentDay = currentDay.AddDays(1);
    }

    Console.WriteLine($"Processing {totalDays} day(s) in parallel...");
    Console.WriteLine();

    // Process all days in parallel
    var snapshotsLock = new object();
    var processedCount = 0;

    var dayTasks = daysToProcess.Select(async day =>
    {
        try
        {
            // Step 3: Detect opportunities for this day
            var dayEnd = day.AddDays(1).AddTicks(-1); // End of day
            var daySnapshots = await reconstructor.BackfillHistoricalSnapshots(
                day, dayEnd, intervalTimeSpan,
                allFundingRates, allPriceKlines, allLiquidityMetrics ?? new Dictionary<string, Dictionary<string, LiquidityMetricsDto>>());

            var dayOpportunities = daySnapshots.Sum(s => s.Opportunities.Count);

            // Thread-safe logging and collection update
            lock (snapshotsLock)
            {
                processedCount++;
                Console.WriteLine($"[{processedCount}/{totalDays}] {day:yyyy-MM-dd}: {daySnapshots.Count} snapshots, {dayOpportunities} opportunities");
                allSnapshots.AddRange(daySnapshots);
            }

            // Step 4: Save this day's opportunities immediately
            if (daySnapshots.Any())
            {
                await opportunityPersister.SaveOpportunitiesAsync(daySnapshots, day, day, output);
            }

            return true;
        }
        catch (Exception ex)
        {
            lock (snapshotsLock)
            {
                Console.WriteLine($"[ERROR] Failed to process {day:yyyy-MM-dd}: {ex.Message}");
            }
            return false;
        }
    });

    await Task.WhenAll(dayTasks);

    // Final summary
    var totalOpportunities = allSnapshots.Sum(s => s.Opportunities.Count);

    Console.WriteLine();
    Console.WriteLine("========================================");
    Console.WriteLine("✅ OPPORTUNITY DETECTION COMPLETE!");
    Console.WriteLine("========================================");
    Console.WriteLine($"Days processed: {totalDays:N0}");
    Console.WriteLine($"Snapshots created: {allSnapshots.Count:N0}");
    Console.WriteLine($"Opportunities detected: {totalOpportunities:N0}");
    if (totalOpportunities > 0)
    {
        var cfffCount = allSnapshots.Sum(s => s.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures));
        var cfpsCount = allSnapshots.Sum(s => s.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesPriceSpread));
        Console.WriteLine($"  CFFF (Funding): {cfffCount:N0}");
        Console.WriteLine($"  CFPS (Price Spread): {cfpsCount:N0}");
    }
    Console.WriteLine($"Saved to: data/opportunities/");
    Console.WriteLine();
    Console.WriteLine("Next step: Run 'simulate' command to generate position simulations");
    Console.WriteLine();

}, startDateOption, endDateOption, intervalOption, detectOutputOption, singleTimestampOption);

// === COMMAND: SIMULATE (Phase 3 - from saved opportunities) ===
var simulateCommand = new Command("simulate", "Simulate positions from persisted opportunities in data/opportunities/");

var opportunitiesFileOption = new Option<string?>(
    "--opportunities-file",
    description: "Path to specific JSON file (optional - if not provided, loads all from data/opportunities/)") { IsRequired = false };

var simulateOutputOption = new Option<string>(
    "--output",
    description: "Output CSV file for training data (saved in data/ directory)",
    getDefaultValue: () => "data/training_data.csv");

var holdDurationsOption = new Option<string>(
    "--durations",
    description: "Comma-separated hold durations (e.g., '1h,4h,1d,3d,7d' or '0.5,1,2,4,6,8,12,24,32,48,72' in hours)",
    getDefaultValue: () => "0.5,1,2,4,6,8,12,24,32,48,72");

simulateCommand.AddOption(opportunitiesFileOption);
simulateCommand.AddOption(simulateOutputOption);
simulateCommand.AddOption(holdDurationsOption);

simulateCommand.SetHandler(async (opportunitiesFile, outputFile, durations) =>
{
    Console.WriteLine("=== PHASE 3: SIMULATE POSITIONS FROM OPPORTUNITIES ===");
    Console.WriteLine($"Output file: {outputFile}");
    Console.WriteLine($"Hold durations: {durations}");
    Console.WriteLine();

    var opportunityPersister = host.Services.GetRequiredService<OpportunityPersister>();
    var simulator = host.Services.GetRequiredService<PositionSimulator>();
    var exporter = host.Services.GetRequiredService<CsvExporter>();

    // Step 1: Load opportunities
    Console.WriteLine("Step 1: Loading opportunities...");
    List<HistoricalMarketSnapshot> snapshots;

    if (string.IsNullOrEmpty(opportunitiesFile))
    {
        Console.WriteLine("  Loading ALL opportunities from data/opportunities/...");
        snapshots = await opportunityPersister.LoadAllOpportunitiesAsync();
    }
    else
    {
        Console.WriteLine($"  Loading from file: {opportunitiesFile}");
        snapshots = await opportunityPersister.LoadOpportunitiesAsync(opportunitiesFile);
    }
    Console.WriteLine();

    // Step 2: Simulate positions with recommended exit strategies
    Console.WriteLine("Step 2: Simulating positions with exit strategies...");
    var simulations = await simulator.SimulateAllPositions(snapshots);

    // Step 3: Export to CSV
    Console.WriteLine("Step 3: Exporting to CSV...");
    await exporter.ExportToCsv(simulations, outputFile);

    Console.WriteLine();
    Console.WriteLine($"✅ Simulation complete!");
    Console.WriteLine($"   Total simulations: {simulations.Count:N0}");
    if (simulations.Count > 0)
    {
        Console.WriteLine($"   Profitable: {simulations.Count(s => s.WasProfitable):N0} ({(simulations.Count(s => s.WasProfitable) * 100.0 / simulations.Count):F1}%)");
        Console.WriteLine($"   Average profit: {simulations.Average(s => s.ActualProfitPercent):F2}%");
    }
    Console.WriteLine($"   Output file: {outputFile}");

}, opportunitiesFileOption, simulateOutputOption, holdDurationsOption);

// === COMMAND: FULL (all 3 phases: collect + detect + simulate) ===
var fullCommand = new Command("full", "Run complete 3-phase pipeline: collect → detect → simulate");

fullCommand.AddOption(startDateOption);
fullCommand.AddOption(endDateOption);
fullCommand.AddOption(intervalOption);
fullCommand.AddOption(exchangesOption);
fullCommand.AddOption(symbolsOption);
fullCommand.AddOption(holdDurationsOption);

var fullOutputOption = new Option<string>(
    "--output",
    description: "Output CSV file for training data",
    getDefaultValue: () => "training_data.csv");

fullCommand.AddOption(fullOutputOption);

fullCommand.SetHandler(async (startDate, endDate, interval, exchanges, symbols, durations, output) =>
{
    Console.WriteLine("=== FULL 3-PHASE PIPELINE ===");
    Console.WriteLine($"Date range: {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");
    Console.WriteLine($"Interval: {interval}");
    Console.WriteLine($"Hold durations: {durations}");
    Console.WriteLine();

    var exchangeList = exchanges.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList();
    var symbolList = await GetSymbolsAsync(symbols, exchangeList, host);

    if (symbolList.Count == 0)
    {
        Console.WriteLine("ERROR: No symbols found!");
        return;
    }

    Console.WriteLine($"Processing {symbolList.Count} symbols across {exchangeList.Count} exchanges");
    Console.WriteLine();

    var fetcher = host.Services.GetRequiredService<HistoricalDataFetcher>();
    var liquidityFetcher = host.Services.GetRequiredService<LiquidityDataFetcher>();
    var reconstructor = host.Services.GetRequiredService<SnapshotReconstructor>();
    var persister = host.Services.GetRequiredService<RawDataPersister>();
    var opportunityPersister = host.Services.GetRequiredService<OpportunityPersister>();
    var simulator = host.Services.GetRequiredService<PositionSimulator>();
    var exporter = host.Services.GetRequiredService<CsvExporter>();

    // === PHASE 1: COLLECT RAW DATA ===
    Console.WriteLine("╔════════════════════════════════════════╗");
    Console.WriteLine("║ PHASE 1: COLLECT & PERSIST RAW DATA   ║");
    Console.WriteLine("╚════════════════════════════════════════╝");

    Console.WriteLine("Step 1/3: Fetching current liquidity metrics...");
    var liquidityMetrics = await liquidityFetcher.FetchCurrentLiquidityMetrics(exchangeList, symbolList);

    Console.WriteLine("Step 2/3: Fetching historical funding rates...");
    var fundingHistory = await fetcher.FetchAllFundingRates(startDate, endDate, exchangeList, symbolList);

    Console.WriteLine("Step 3/3: Fetching historical price klines...");
    var priceHistory = await fetcher.FetchAllPriceKlines(startDate, endDate, exchangeList, symbolList, interval);

    Console.WriteLine("Saving raw data to disk...");
    var manifest = await persister.SaveRawDataAsync(
        startDate, endDate, exchangeList, symbolList,
        fundingHistory, priceHistory, liquidityMetrics);

    // === PHASE 2: DETECT OPPORTUNITIES ===
    Console.WriteLine();
    Console.WriteLine("╔════════════════════════════════════════╗");
    Console.WriteLine("║ PHASE 2: DETECT OPPORTUNITIES         ║");
    Console.WriteLine("╚════════════════════════════════════════╝");

    Console.WriteLine("Reconstructing market snapshots...");
    var intervalTimeSpan = ParseInterval(interval);
    var snapshots = await reconstructor.BackfillHistoricalSnapshots(
        startDate, endDate, intervalTimeSpan, fundingHistory, priceHistory, liquidityMetrics);

    Console.WriteLine("Saving detected opportunities to disk...");
    var dateString = startDate.ToString("yyyy-MM-dd");
    var opportunitiesPath = $"{dateString}_opportunities.json";
    await opportunityPersister.SaveOpportunitiesAsync(snapshots, startDate, endDate, opportunitiesPath);

    var totalOpportunities = snapshots.Sum(s => s.Opportunities.Count);
    var cfffCount = snapshots.Sum(s => s.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesFutures));
    var cfpsCount = snapshots.Sum(s => s.Opportunities.Count(o => o.SubType == StrategySubType.CrossExchangeFuturesPriceSpread));

    Console.WriteLine($"   CFFF opportunities: {cfffCount:N0}");
    Console.WriteLine($"   CFPS opportunities: {cfpsCount:N0}");
    Console.WriteLine($"   Total: {totalOpportunities:N0}");

    // === PHASE 3: SIMULATE POSITIONS ===
    Console.WriteLine();
    Console.WriteLine("╔════════════════════════════════════════╗");
    Console.WriteLine("║ PHASE 3: SIMULATE POSITIONS           ║");
    Console.WriteLine("╚════════════════════════════════════════╝");

    Console.WriteLine("Simulating positions with exit strategies across all opportunities...");
    var simulations = await simulator.SimulateAllPositions(snapshots);

    Console.WriteLine("Exporting training data to CSV...");
    await exporter.ExportToCsv(simulations, output);

    // === SUMMARY ===
    Console.WriteLine();
    Console.WriteLine("╔════════════════════════════════════════╗");
    Console.WriteLine("║ ✅ PIPELINE COMPLETE!                  ║");
    Console.WriteLine("╚════════════════════════════════════════╝");
    Console.WriteLine($"📊 Snapshots: {snapshots.Count:N0}");
    Console.WriteLine($"🎯 Opportunities: {totalOpportunities:N0} (CFFF: {cfffCount:N0}, CFPS: {cfpsCount:N0})");
    Console.WriteLine($"🔬 Simulations: {simulations.Count:N0}");
    if (simulations.Count > 0)
    {
        Console.WriteLine($"💰 Profitable: {simulations.Count(s => s.WasProfitable):N0} ({(simulations.Count(s => s.WasProfitable) * 100.0 / simulations.Count):F1}%)");
        Console.WriteLine($"📈 Avg profit: {simulations.Average(s => s.ActualProfitPercent):F2}%");
    }
    Console.WriteLine();
    Console.WriteLine($"📁 Raw data: data/raw/");
    Console.WriteLine($"📁 Opportunities: data/opportunities/{opportunitiesPath}");
    Console.WriteLine($"📁 Training data: {output}");
    Console.WriteLine();

}, startDateOption, endDateOption, intervalOption, exchangesOption, symbolsOption, holdDurationsOption, fullOutputOption);

// === COMMAND: SNAPSHOT (real-time current market) ===
var snapshotCommand = new Command("snapshot", "Fetch current market data and detect opportunities in real-time");

var snapshotExchangesOption = new Option<string>(
    "--exchanges",
    description: "Comma-separated list of exchanges",
    getDefaultValue: () => "Binance,Bybit");

var snapshotSymbolsOption = new Option<string?>(
    "--symbols",
    description: "Comma-separated list of symbols (leave empty for auto-discovery)",
    getDefaultValue: () => null);

snapshotCommand.AddOption(snapshotExchangesOption);
snapshotCommand.AddOption(snapshotSymbolsOption);

snapshotCommand.SetHandler(async (exchanges, symbols) =>
{
    Console.WriteLine("=== Real-Time Market Snapshot ===" );
    Console.WriteLine($"Time: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
    Console.WriteLine($"Exchanges: {exchanges}");
    Console.WriteLine();

    var exchangeList = exchanges.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList();
    var symbolList = await GetSymbolsAsync(symbols, exchangeList, host);

    if (symbolList.Count == 0)
    {
        Console.WriteLine("ERROR: No symbols found!");
        return;
    }

    Console.WriteLine($"Processing {symbolList.Count} symbols");
    Console.WriteLine();

    var fetcher = host.Services.GetRequiredService<HistoricalDataFetcher>();
    var liquidityFetcher = host.Services.GetRequiredService<LiquidityDataFetcher>();
    var reconstructor = host.Services.GetRequiredService<SnapshotReconstructor>();

    // Fetch current market data using bulk endpoints (avoids rate limiting)
    Console.WriteLine("Fetching current liquidity metrics...");
    var liquidityMetrics = await liquidityFetcher.FetchCurrentLiquidityMetrics(exchangeList, symbolList);

    Console.WriteLine("Fetching current funding rates...");
    var fundingHistory = await fetcher.FetchCurrentFundingRates(exchangeList, symbolList);

    Console.WriteLine("Fetching current prices...");
    var priceHistory = await fetcher.FetchCurrentPrices(exchangeList, symbolList);

    // Reconstruct a single snapshot for current time
    Console.WriteLine();
    Console.WriteLine("Detecting opportunities from current market data...");
    var now = DateTime.UtcNow;
    var snapshots = await reconstructor.BackfillHistoricalSnapshots(
        now.AddMinutes(-1),
        now,
        TimeSpan.FromMinutes(1),
        fundingHistory,
        priceHistory,
        liquidityMetrics);

    if (snapshots.Count == 0)
    {
        Console.WriteLine("ERROR: Could not create snapshot!");
        return;
    }

    var snapshot = snapshots.Last();

    // Display results
    Console.WriteLine();
    Console.WriteLine("========================================");
    Console.WriteLine("DETECTED OPPORTUNITIES");
    Console.WriteLine("========================================");
    Console.WriteLine($"Snapshot Time: {snapshot.Timestamp:yyyy-MM-dd HH:mm:ss} UTC");
    Console.WriteLine($"BTC Price: ${snapshot.BtcPrice:N2}");
    Console.WriteLine($"Market Regime: {snapshot.MarketRegime}");
    Console.WriteLine($"Total Opportunities: {snapshot.Opportunities.Count}");
    Console.WriteLine();

    if (snapshot.Opportunities.Count == 0)
    {
        Console.WriteLine("No opportunities detected at this time.");
    }
    else
    {
        foreach (var opp in snapshot.Opportunities)
        {
            Console.WriteLine($"[{opp.SubType}] {opp.Symbol}");
            Console.WriteLine($"  Long: {opp.LongExchange} | Short: {opp.ShortExchange}");
            Console.WriteLine($"  Funding Profit (8h): {opp.FundProfit8h:F4}%");
            Console.WriteLine($"  Funding APR: {opp.FundApr:F2}%");
            Console.WriteLine($"  Break Even: {opp.BreakEvenTimeHours:F1}h");
            Console.WriteLine($"  Volume 24h: ${opp.Volume24h:N0}");
            Console.WriteLine($"  Liquidity: {opp.LiquidityStatus}");
            Console.WriteLine();
        }
    }

    Console.WriteLine("========================================");
    Console.WriteLine($"Detection used backend thresholds:");
    Console.WriteLine($"  MinSpreadPercentage: 0.1%");
    Console.WriteLine($"  MinPriceSpreadPercentage: 0.3%");
    Console.WriteLine("========================================");

}, snapshotExchangesOption, snapshotSymbolsOption);

// === ADD COMMANDS TO ROOT ===
rootCommand.AddCommand(collectCommand);  // Phase 1: Data collection
rootCommand.AddCommand(detectCommand);   // Phase 2: Opportunity detection
rootCommand.AddCommand(simulateCommand); // Phase 3: Position simulation
rootCommand.AddCommand(fullCommand);     // All 3 phases combined
rootCommand.AddCommand(backfillCommand); // Legacy (deprecated)
rootCommand.AddCommand(snapshotCommand); // Real-time snapshot

// === RUN CLI ===
return await rootCommand.InvokeAsync(args);

// === HELPER FUNCTIONS ===
static TimeSpan ParseInterval(string interval)
{
    var number = int.Parse(interval[..^1]);
    var unit = interval[^1];

    return unit switch
    {
        'm' => TimeSpan.FromMinutes(number),
        'h' => TimeSpan.FromHours(number),
        'd' => TimeSpan.FromDays(number),
        _ => throw new ArgumentException($"Invalid interval: {interval}")
    };
}

static decimal ParseDurationToHours(string duration)
{
    duration = duration.Trim();

    // If it's just a number (no unit suffix), treat it as hours
    if (decimal.TryParse(duration, NumberStyles.Any, CultureInfo.InvariantCulture, out var numericValue))
    {
        return numericValue;
    }

    // Otherwise, parse with unit suffix
    var numberPart = duration[..^1];
    var unit = duration[^1];
    var number = decimal.Parse(numberPart, CultureInfo.InvariantCulture);

    return unit switch
    {
        'h' => number,
        'd' => number * 24m,
        'm' => number / 60m,
        _ => throw new ArgumentException($"Invalid duration: {duration}. Expected format: number with optional unit (h/d/m), e.g., '1h', '2d', '30m', or just '1.5' for hours")
    };
}

static async Task<List<string>> GetSymbolsAsync(string? symbolsInput, List<string> exchanges, IHost host)
{
    if (!string.IsNullOrEmpty(symbolsInput))
    {
        // Use provided symbols
        var symbols = symbolsInput.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList();
        Console.WriteLine($"Using {symbols.Count} provided symbols");
        return symbols;
    }

    // Auto-discover symbols
    Console.WriteLine("Auto-discovering symbols from exchanges...");
    var symbolDiscovery = host.Services.GetRequiredService<SymbolDiscoveryService>();
    var discoveredSymbols = await symbolDiscovery.GetActiveSymbolsAsync();

    // Filter to only symbols that exist on BOTH exchanges
    var binanceSymbols = new HashSet<string>();
    var bybitSymbols = new HashSet<string>();

    using var scope = host.Services.CreateScope();

    if (exchanges.Contains("Binance"))
    {
        var binanceConnector = scope.ServiceProvider.GetService<BinanceConnector>();
        if (binanceConnector != null)
        {
            await binanceConnector.ConnectAsync(string.Empty, string.Empty);
            var symbols = await binanceConnector.GetActiveSymbolsAsync(0, 1000, 0);
            foreach (var symbol in symbols) binanceSymbols.Add(symbol);
        }
    }

    if (exchanges.Contains("Bybit"))
    {
        var bybitConnector = scope.ServiceProvider.GetService<BybitConnector>();
        if (bybitConnector != null)
        {
            await bybitConnector.ConnectAsync(string.Empty, string.Empty);
            var symbols = await bybitConnector.GetActiveSymbolsAsync(0, 1000, 0);
            foreach (var symbol in symbols) bybitSymbols.Add(symbol);
        }
    }

    // Get intersection - only symbols on BOTH exchanges
    var commonSymbols = discoveredSymbols
        .Where(s => binanceSymbols.Contains(s) && bybitSymbols.Contains(s))
        .ToList();

    Console.WriteLine($"Discovered {discoveredSymbols.Count} total symbols");
    Console.WriteLine($"Found {commonSymbols.Count} symbols available on BOTH Binance and Bybit");
    Console.WriteLine($"All symbols: {string.Join(", ", commonSymbols)}");

    return commonSymbols;
}
