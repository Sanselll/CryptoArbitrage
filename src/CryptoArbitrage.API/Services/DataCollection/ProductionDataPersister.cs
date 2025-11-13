using System.Text.Json;
using CryptoArbitrage.API.Models.DataCollection;

namespace CryptoArbitrage.API.Services.DataCollection;

/// <summary>
/// Service for persisting production data snapshots to disk for ML training
/// </summary>
public class ProductionDataPersister
{
    private readonly ILogger<ProductionDataPersister> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _basePath;
    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public ProductionDataPersister(
        ILogger<ProductionDataPersister> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;

        // Default to data/production in the project root
        _basePath = _configuration["ProductionDataCollection:BasePath"]
            ?? Path.Combine(Directory.GetCurrentDirectory(), "data", "production");

        EnsureBasePathExists();
    }

    /// <summary>
    /// Save a snapshot to disk
    /// </summary>
    public async Task SaveSnapshotAsync(ProductionSnapshot snapshot, CancellationToken cancellationToken = default)
    {
        try
        {
            var date = snapshot.Timestamp.ToString("yyyy-MM-dd");
            var time = snapshot.Timestamp.ToString("HH-mm");

            var dayPath = Path.Combine(_basePath, date);
            Directory.CreateDirectory(dayPath);

            var filePath = Path.Combine(dayPath, $"{time}.json");

            // Serialize and save
            var json = JsonSerializer.Serialize(snapshot, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json, cancellationToken);

            _logger.LogInformation(
                "Saved production snapshot to {FilePath}. Symbols: {SymbolCount}, Opportunities: {OpportunityCount}",
                filePath, snapshot.Metadata.TotalSymbols, snapshot.Metadata.TotalOpportunities);

            // Update daily manifest
            await UpdateManifestAsync(snapshot, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save production snapshot for {Timestamp}", snapshot.Timestamp);
            throw;
        }
    }

    /// <summary>
    /// Update the daily manifest file
    /// </summary>
    private async Task UpdateManifestAsync(ProductionSnapshot snapshot, CancellationToken cancellationToken)
    {
        try
        {
            var date = snapshot.Timestamp.ToString("yyyy-MM-dd");
            var dayPath = Path.Combine(_basePath, date);
            var manifestPath = Path.Combine(dayPath, "manifest.json");

            DailyManifest manifest;

            // Load existing manifest or create new one
            if (File.Exists(manifestPath))
            {
                var existingJson = await File.ReadAllTextAsync(manifestPath, cancellationToken);
                manifest = JsonSerializer.Deserialize<DailyManifest>(existingJson, _jsonOptions) ?? new DailyManifest();
            }
            else
            {
                manifest = new DailyManifest
                {
                    Date = date,
                    SymbolsTracked = snapshot.Metadata.TotalSymbols
                };
            }

            // Add or update snapshot info
            var time = snapshot.Timestamp.ToString("HH:mm");
            var fileName = snapshot.Timestamp.ToString("HH-mm") + ".json";

            var existingSnapshot = manifest.Snapshots.FirstOrDefault(s => s.Time == time);
            if (existingSnapshot != null)
            {
                existingSnapshot.Opportunities = snapshot.Metadata.TotalOpportunities;
            }
            else
            {
                manifest.Snapshots.Add(new SnapshotInfo
                {
                    Time = time,
                    File = fileName,
                    Opportunities = snapshot.Metadata.TotalOpportunities
                });
            }

            // Sort by time
            manifest.Snapshots = manifest.Snapshots.OrderBy(s => s.Time).ToList();
            manifest.TotalSnapshots = manifest.Snapshots.Count;

            // Save manifest
            var manifestJson = JsonSerializer.Serialize(manifest, _jsonOptions);
            await File.WriteAllTextAsync(manifestPath, manifestJson, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to update manifest for {Date}", snapshot.Timestamp.ToString("yyyy-MM-dd"));
            // Don't throw - manifest update failure shouldn't stop snapshot saving
        }
    }

    /// <summary>
    /// Get the storage path for a specific date
    /// </summary>
    public string GetDayPath(DateTime date)
    {
        return Path.Combine(_basePath, date.ToString("yyyy-MM-dd"));
    }

    /// <summary>
    /// Get all available dates with snapshots
    /// </summary>
    public IEnumerable<string> GetAvailableDates()
    {
        if (!Directory.Exists(_basePath))
            return Enumerable.Empty<string>();

        return Directory.GetDirectories(_basePath)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrEmpty(name))
            .OrderBy(name => name)!;
    }

    /// <summary>
    /// Load a specific snapshot
    /// </summary>
    public async Task<ProductionSnapshot?> LoadSnapshotAsync(DateTime timestamp, CancellationToken cancellationToken = default)
    {
        try
        {
            var date = timestamp.ToString("yyyy-MM-dd");
            var time = timestamp.ToString("HH-mm");
            var filePath = Path.Combine(_basePath, date, $"{time}.json");

            if (!File.Exists(filePath))
                return null;

            var json = await File.ReadAllTextAsync(filePath, cancellationToken);
            return JsonSerializer.Deserialize<ProductionSnapshot>(json, _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load snapshot for {Timestamp}", timestamp);
            return null;
        }
    }

    /// <summary>
    /// Load daily manifest
    /// </summary>
    public async Task<DailyManifest?> LoadManifestAsync(string date, CancellationToken cancellationToken = default)
    {
        try
        {
            var manifestPath = Path.Combine(_basePath, date, "manifest.json");

            if (!File.Exists(manifestPath))
                return null;

            var json = await File.ReadAllTextAsync(manifestPath, cancellationToken);
            return JsonSerializer.Deserialize<DailyManifest>(json, _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load manifest for {Date}", date);
            return null;
        }
    }

    private void EnsureBasePathExists()
    {
        try
        {
            if (!Directory.Exists(_basePath))
            {
                Directory.CreateDirectory(_basePath);
                _logger.LogInformation("Created production data directory at {BasePath}", _basePath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create production data directory at {BasePath}", _basePath);
            throw;
        }
    }
}
