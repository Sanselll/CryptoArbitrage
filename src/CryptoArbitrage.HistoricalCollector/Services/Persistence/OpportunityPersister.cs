using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace CryptoArbitrage.HistoricalCollector.Services.Persistence;

/// <summary>
/// Persists detected opportunities and snapshots to disk
/// Saves to data/opportunities/ folder
/// </summary>
public class OpportunityPersister
{
    private readonly ILogger<OpportunityPersister> _logger;
    private const string BaseDataPath = "data/opportunities";

    public OpportunityPersister(ILogger<OpportunityPersister> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Save detected opportunities/snapshots to JSON files organized by day
    /// Creates day folders like: data/opportunities/2025-09-02/opportunities.json
    /// </summary>
    public async Task SaveOpportunitiesAsync(
        List<HistoricalMarketSnapshot> snapshots,
        DateTime? startDate = null,
        DateTime? endDate = null,
        string? outputPath = null)
    {
        // Group snapshots by day
        var snapshotsByDay = snapshots
            .GroupBy(s => s.Timestamp.Date)
            .OrderBy(g => g.Key)
            .ToList();

        _logger.LogInformation("Saving {TotalSnapshots} snapshots grouped into {Days} day(s)",
            snapshots.Count, snapshotsByDay.Count);

        int totalOpportunities = 0;

        foreach (var dayGroup in snapshotsByDay)
        {
            var date = dayGroup.Key;
            var daySnapshots = dayGroup.ToList();
            var dayOpportunities = daySnapshots.Sum(s => s.Opportunities.Count);
            totalOpportunities += dayOpportunities;

            // Create day folder
            var dayFolder = Path.Combine(BaseDataPath, date.ToString("yyyy-MM-dd"));
            Directory.CreateDirectory(dayFolder);

            // Save to opportunities.json in the day folder
            var filePath = Path.Combine(dayFolder, "opportunities.json");

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            var json = JsonSerializer.Serialize(daySnapshots, options);
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation("  Saved {Date}: {SnapshotCount} snapshots, {OpportunityCount} opportunities",
                date.ToString("yyyy-MM-dd"), daySnapshots.Count, dayOpportunities);
        }

        _logger.LogInformation("Total saved: {SnapshotCount} snapshots with {OpportunityCount} opportunities to {Path}",
            snapshots.Count, totalOpportunities, BaseDataPath);
    }

    /// <summary>
    /// Load opportunities/snapshots from JSON file
    /// </summary>
    public async Task<List<HistoricalMarketSnapshot>> LoadOpportunitiesAsync(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Opportunities file not found: {filePath}. Run 'detect' command first.");
        }

        var json = await File.ReadAllTextAsync(filePath);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        var snapshots = JsonSerializer.Deserialize<List<HistoricalMarketSnapshot>>(json, options)
            ?? throw new InvalidOperationException($"Failed to deserialize {filePath}");

        var totalOpportunities = snapshots.Sum(s => s.Opportunities.Count);
        _logger.LogInformation("Loaded {SnapshotCount} snapshots with {OpportunityCount} opportunities from {Path}",
            snapshots.Count, totalOpportunities, filePath);

        return snapshots;
    }
}
