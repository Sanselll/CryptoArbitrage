using System.Text.Json;
using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Events;

namespace CryptoArbitrage.API.Services.Arbitrage.Detection;

/// <summary>
/// Dumps enriched opportunity snapshots to disk in the same format as HistoricalCollector
/// Subscribes to OpportunitiesEnriched event (after enrichment completes)
/// Output format matches HistoricalMarketSnapshot structure for consistent comparison
/// </summary>
public class OpportunitySnapshotDumper : IHostedService
{
    private readonly OpportunityDumpConfig _config;
    private readonly ILogger<OpportunitySnapshotDumper> _logger;
    private readonly IDataCollectionEventBus _eventBus;

    public OpportunitySnapshotDumper(
        OpportunityDumpConfig config,
        ILogger<OpportunitySnapshotDumper> logger,
        IDataCollectionEventBus eventBus)
    {
        _config = config;
        _logger = logger;
        _eventBus = eventBus;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (!_config.Enabled)
        {
            _logger.LogInformation("OpportunitySnapshotDumper is disabled");
            return Task.CompletedTask;
        }

        // Subscribe to OpportunitiesEnriched event (AFTER enrichment)
        _eventBus.Subscribe<List<ArbitrageOpportunityDto>>(
            DataCollectionConstants.EventTypes.OpportunitiesEnriched,
            OnOpportunitiesEnrichedAsync);

        _logger.LogInformation("OpportunitySnapshotDumper started and subscribed to OpportunitiesEnriched events");
        _logger.LogInformation("Dumps will be saved to: {BasePath}", _config.BasePath);
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("OpportunitySnapshotDumper stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Called when enriched opportunities are published
    /// </summary>
    private async Task OnOpportunitiesEnrichedAsync(DataCollectionEvent<List<ArbitrageOpportunityDto>> @event)
    {
        if (@event.Data == null || @event.Data.Count < _config.MinOpportunitiesThreshold)
        {
            _logger.LogDebug("Skipping dump: {Count} opportunities (threshold: {Threshold})",
                @event.Data?.Count ?? 0, _config.MinOpportunitiesThreshold);
            return;
        }

        try
        {
            await DumpSnapshotAsync(@event.Data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error dumping opportunity snapshot");
        }
    }

    /// <summary>
    /// Dumps opportunities to JSON file in HistoricalMarketSnapshot format
    /// </summary>
    private async Task DumpSnapshotAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        try
        {
            var timestamp = DateTime.UtcNow;
            var dateFolder = timestamp.ToString("yyyy-MM-dd");
            var timeString = timestamp.ToString("HHmm");

            // Create directory structure: data/backend_dumps/2025-10-25/
            var dumpDir = Path.Combine(_config.BasePath, dateFolder);
            Directory.CreateDirectory(dumpDir);

            // Filename: opportunities_1307.json
            var filename = $"opportunities_{timeString}.json";
            var filePath = Path.Combine(dumpDir, filename);

            // Build snapshot in HistoricalMarketSnapshot format (simplified)
            var snapshot = new OpportunitySnapshot
            {
                Timestamp = timestamp,
                Opportunities = opportunities
            };

            // Wrap in array to match HistoricalCollector format (array of snapshots)
            var snapshots = new[] { snapshot };

            // Serialize with camelCase naming (matching HistoricalCollector)
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };

            var json = JsonSerializer.Serialize(snapshots, options);

            // Write to file
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation(
                "Dumped {Count} opportunities to {FilePath} ({Size} KB)",
                opportunities.Count,
                filePath,
                new FileInfo(filePath).Length / 1024);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing opportunity snapshot to disk");
            throw;
        }
    }

    /// <summary>
    /// Simplified snapshot structure matching HistoricalCollector output
    /// Only includes timestamp and opportunities (no raw market data)
    /// </summary>
    private class OpportunitySnapshot
    {
        public DateTime Timestamp { get; set; }
        public List<ArbitrageOpportunityDto> Opportunities { get; set; } = new();
    }
}
