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
    /// Only saves timestamp and opportunities (excludes prices, funding rates, etc. to reduce file size)
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

            // Convert to lightweight snapshots (only timestamp + opportunities)
            var lightweightSnapshots = daySnapshots.Select(s => new LightweightOpportunitySnapshot
            {
                Timestamp = s.Timestamp,
                Opportunities = s.Opportunities
            }).ToList();

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            var json = JsonSerializer.Serialize(lightweightSnapshots, options);
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation("  Saved {Date}: {SnapshotCount} snapshots, {OpportunityCount} opportunities",
                date.ToString("yyyy-MM-dd"), daySnapshots.Count, dayOpportunities);
        }

        _logger.LogInformation("Total saved: {SnapshotCount} snapshots with {OpportunityCount} opportunities to {Path}",
            snapshots.Count, totalOpportunities, BaseDataPath);
    }

    /// <summary>
    /// Load opportunities/snapshots from JSON file
    /// Handles both lightweight (new) and full (legacy) formats
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

        // Try to load as lightweight format first (new format)
        try
        {
            var lightweightSnapshots = JsonSerializer.Deserialize<List<LightweightOpportunitySnapshot>>(json, options);
            if (lightweightSnapshots != null)
            {
                // Convert to full HistoricalMarketSnapshot (for simulation compatibility)
                var snapshots = lightweightSnapshots.Select(ls => new HistoricalMarketSnapshot
                {
                    Timestamp = ls.Timestamp,
                    Opportunities = ls.Opportunities
                }).ToList();

                var totalOpportunities = snapshots.Sum(s => s.Opportunities.Count);
                _logger.LogInformation("Loaded {SnapshotCount} snapshots with {OpportunityCount} opportunities from {Path} (lightweight format)",
                    snapshots.Count, totalOpportunities, filePath);

                return snapshots;
            }
        }
        catch
        {
            // Fall through to try legacy format
        }

        // Fall back to legacy full format
        var legacySnapshots = JsonSerializer.Deserialize<List<HistoricalMarketSnapshot>>(json, options)
            ?? throw new InvalidOperationException($"Failed to deserialize {filePath}");

        var totalOpps = legacySnapshots.Sum(s => s.Opportunities.Count);
        _logger.LogInformation("Loaded {SnapshotCount} snapshots with {OpportunityCount} opportunities from {Path} (legacy format)",
            legacySnapshots.Count, totalOpps, filePath);

        return legacySnapshots;
    }

    /// <summary>
    /// Load ALL opportunities from all day folders in data/opportunities/
    /// Scans for all opportunities.json files in subfolders and merges them
    /// </summary>
    public async Task<List<HistoricalMarketSnapshot>> LoadAllOpportunitiesAsync()
    {
        if (!Directory.Exists(BaseDataPath))
        {
            throw new DirectoryNotFoundException($"Opportunities directory not found: {BaseDataPath}. Run 'detect' command first.");
        }

        _logger.LogInformation("Loading all opportunities from {Path}...", BaseDataPath);

        // Find all opportunities.json files in subfolders
        var opportunityFiles = Directory.GetFiles(BaseDataPath, "opportunities.json", SearchOption.AllDirectories)
            .OrderBy(f => f)
            .ToList();

        if (!opportunityFiles.Any())
        {
            throw new FileNotFoundException($"No opportunity files found in {BaseDataPath}. Run 'detect' command first.");
        }

        _logger.LogInformation("Found {Count} opportunity file(s) to load", opportunityFiles.Count);

        var allSnapshots = new List<HistoricalMarketSnapshot>();
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        // Load each file and merge snapshots
        foreach (var filePath in opportunityFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(filePath);
                List<HistoricalMarketSnapshot>? snapshots = null;

                // Try lightweight format first
                try
                {
                    var lightweightSnapshots = JsonSerializer.Deserialize<List<LightweightOpportunitySnapshot>>(json, options);
                    if (lightweightSnapshots != null)
                    {
                        snapshots = lightweightSnapshots.Select(ls => new HistoricalMarketSnapshot
                        {
                            Timestamp = ls.Timestamp,
                            Opportunities = ls.Opportunities
                        }).ToList();
                    }
                }
                catch
                {
                    // Fall back to legacy format
                    snapshots = JsonSerializer.Deserialize<List<HistoricalMarketSnapshot>>(json, options);
                }

                if (snapshots != null && snapshots.Any())
                {
                    var dayFolder = Path.GetFileName(Path.GetDirectoryName(filePath));
                    var opportunityCount = snapshots.Sum(s => s.Opportunities.Count);

                    _logger.LogInformation("  Loaded {Day}: {SnapshotCount} snapshots, {OpportunityCount} opportunities",
                        dayFolder, snapshots.Count, opportunityCount);

                    allSnapshots.AddRange(snapshots);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load opportunities from {Path}, skipping", filePath);
            }
        }

        // Sort by timestamp to ensure chronological order
        allSnapshots = allSnapshots.OrderBy(s => s.Timestamp).ToList();

        var totalOpportunities = allSnapshots.Sum(s => s.Opportunities.Count);
        _logger.LogInformation("Loaded total: {SnapshotCount} snapshots with {OpportunityCount} opportunities from {FileCount} file(s)",
            allSnapshots.Count, totalOpportunities, opportunityFiles.Count);

        return allSnapshots;
    }
}
