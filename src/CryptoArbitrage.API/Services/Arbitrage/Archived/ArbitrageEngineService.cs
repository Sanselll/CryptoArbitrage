using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Arbitrage.Detection;
using CryptoArbitrage.API.Services.Data;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.Streaming;

namespace CryptoArbitrage.API.Services.Arbitrage.Archived;

/// <summary>
/// Orchestrator service that coordinates opportunity detection and real-time streaming.
/// Consumes data from the new data collection system instead of fetching directly.
/// Implements two independent loops:
/// 1. Slow loop: Read market data and detect opportunities (every 60s by default)
/// 2. Fast loop: Broadcast cached data to UI via SignalR (every 1s by default)
/// </summary>
public class ArbitrageEngineService : BackgroundService
{
    private readonly ILogger<ArbitrageEngineService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly ArbitrageConfig _config;
    private readonly IOpportunityDetectionService _opportunityDetectionService;
    private readonly ISignalRStreamingService _signalRStreamingService;
    private readonly IMarketDataService _marketDataService;
    private readonly IFundingRateService _fundingRateService;
    private readonly IDataRepository<UserDataSnapshot> _userDataRepository;

    // Cached data for broadcasting
    private List<FundingRateDto> _cachedFundingRates = new();
    private List<ArbitrageOpportunityDto> _cachedOpportunities = new();
    private readonly object _cacheLock = new();

    public ArbitrageEngineService(
        ILogger<ArbitrageEngineService> logger,
        IServiceProvider serviceProvider,
        ArbitrageConfig config,
        IOpportunityDetectionService opportunityDetectionService,
        ISignalRStreamingService signalRStreamingService,
        IMarketDataService marketDataService,
        IFundingRateService fundingRateService,
        IDataRepository<UserDataSnapshot> userDataRepository)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
        _opportunityDetectionService = opportunityDetectionService;
        _signalRStreamingService = signalRStreamingService;
        _marketDataService = marketDataService;
        _fundingRateService = fundingRateService;
        _userDataRepository = userDataRepository;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Arbitrage Engine Service starting...");
        _logger.LogInformation("Configuration: OpportunityCollection={OpportunityInterval}s, SignalRBroadcast={BroadcastInterval}s",
            _config.OpportunityCollectionIntervalSeconds,
            _config.SignalRBroadcastIntervalSeconds);

        try
        {
            // Start two independent loops in parallel
            var tasks = new[]
            {
                OpportunityCollectionLoopAsync(stoppingToken),    // Slow: Read data & detect opportunities
                SignalRBroadcastLoopAsync(stoppingToken)          // Fast: Broadcast cached data to UI
            };

            await Task.WhenAll(tasks);
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "Fatal error in Arbitrage Engine Service. Service will attempt to restart.");
            throw;
        }
    }

    /// <summary>
    /// Loop 1: Read market data from new services and detect opportunities (slow, configurable interval)
    /// This reads from the cached data collected by the background services
    /// </summary>
    private async Task OpportunityCollectionLoopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting opportunity collection loop (interval: {Interval}s)",
            _config.OpportunityCollectionIntervalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();

                // PHASE 1: Read cached market data from new data collection system
                var snapshot = await _marketDataService.GetMarketDataSnapshotAsync();

                if (snapshot == null)
                {
                    _logger.LogWarning("No market data snapshot available yet. Waiting for data collectors...");
                    await Task.Delay(TimeSpan.FromSeconds(_config.OpportunityCollectionIntervalSeconds), stoppingToken);
                    continue;
                }

                // Convert to old format for opportunity detection
                // The snapshot has FundingRates, SpotPrices, PerpPrices but they're in different structure
                // We need to build a MarketDataSnapshot compatible with OpportunityDetectionService
                var fundingRatesDict = await _marketDataService.GetFundingRatesAsync();
                var spotPricesDict = await _marketDataService.GetSpotPricesAsync();
                var perpPricesDict = await _marketDataService.GetPerpetualPricesAsync();

                var opportunitySnapshot = new MarketDataSnapshot
                {
                    FundingRates = fundingRatesDict,
                    SpotPrices = spotPricesDict,
                    PerpPrices = perpPricesDict,
                    FetchedAt = snapshot.FetchedAt
                };
                

                // PHASE 2: Detect opportunities from market data
                var opportunities = await _opportunityDetectionService.DetectOpportunitiesAsync(opportunitySnapshot);

                // PHASE 3: Update cached data (thread-safe)
                lock (_cacheLock)
                {
                    _cachedFundingRates = fundingRatesDict.Values.SelectMany(r => r).ToList();
                    _cachedOpportunities = opportunities;
                }

                sw.Stop();
                _logger.LogInformation(
                    "Opportunity collection completed in {Elapsed}ms: {Rates} rates, {Opps} opportunities",
                    sw.ElapsedMilliseconds, _cachedFundingRates.Count, _cachedOpportunities.Count);

                await Task.Delay(TimeSpan.FromSeconds(_config.OpportunityCollectionIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in opportunity collection loop");
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
        }
    }

    /// <summary>
    /// Loop 2: Broadcast cached data to UI (fast, configurable interval)
    /// This is cheap - just sends already-collected data to SignalR clients
    /// </summary>
    private async Task SignalRBroadcastLoopAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Starting SignalR broadcast loop (interval: {Interval}s)",
            _config.SignalRBroadcastIntervalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                List<FundingRateDto> ratesToBroadcast;
                List<ArbitrageOpportunityDto> oppsToBroadcast;

                // Get snapshot of cached data (thread-safe)
                lock (_cacheLock)
                {
                    ratesToBroadcast = _cachedFundingRates.ToList();
                    oppsToBroadcast = _cachedOpportunities.ToList();
                }

                // Broadcast to all clients
                if (ratesToBroadcast.Any())
                {
                    await _signalRStreamingService.BroadcastFundingRatesAsync(ratesToBroadcast, stoppingToken);
                }

                if (oppsToBroadcast.Any())
                {
                    await _signalRStreamingService.BroadcastOpportunitiesAsync(oppsToBroadcast, stoppingToken);
                }

                // Broadcast user-specific data (balances and positions)
                try
                {
                    var allUserData = await _userDataRepository.GetByPatternAsync("userdata:*", stoppingToken);

                    if (allUserData.Any())
                    {
                        // Group by userId
                        var userGroups = allUserData.GroupBy(kv => kv.Value.UserId);

                        foreach (var userGroup in userGroups)
                        {
                            var userId = userGroup.Key.ToString();
                            var userSnapshots = userGroup.Select(kv => kv.Value).ToList();

                            // Aggregate balances across all exchanges for this user
                            var balances = userSnapshots
                                .Select(s => new AccountBalanceDto
                                {
                                    Exchange = s.Exchange,
                                    TotalBalance = s.Balance?.TotalBalance ?? 0,
                                    AvailableBalance = s.Balance?.AvailableBalance ?? 0,
                                    MarginUsed = s.Balance?.MarginUsed ?? 0
                                })
                                .ToList();

                            // Aggregate positions across all exchanges for this user
                            var positions = userSnapshots
                                .SelectMany(s => s.Positions ?? new List<PositionDto>())
                                .ToList();

                            // Broadcast to user
                            if (balances.Any())
                            {
                                await _signalRStreamingService.BroadcastBalancesToUserAsync(userId, balances, stoppingToken);
                            }

                            if (positions.Any())
                            {
                                await _signalRStreamingService.BroadcastPositionsToUserAsync(userId, positions, stoppingToken);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error broadcasting user data");
                }

                await Task.Delay(TimeSpan.FromSeconds(_config.SignalRBroadcastIntervalSeconds), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in SignalR broadcast loop");
                await Task.Delay(TimeSpan.FromSeconds(1), stoppingToken);
            }
        }
    }

    public override Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Arbitrage Engine Service stopping...");
        return base.StopAsync(cancellationToken);
    }
}
