namespace CryptoArbitrage.API.Services;

/// <summary>
/// Result of API key validation including permissions and IP restrictions
/// </summary>
public class ApiKeyValidationResult
{
    public bool IsValid { get; set; }
    public List<string> MissingPermissions { get; set; } = new();
    public bool IsIpRestricted { get; set; }
    public List<string> AllowedIps { get; set; } = new();
    public string? ServerIp { get; set; }
    public string DetailedMessage { get; set; } = string.Empty;

    /// <summary>
    /// Creates a successful validation result
    /// </summary>
    public static ApiKeyValidationResult Success()
    {
        return new ApiKeyValidationResult
        {
            IsValid = true,
            DetailedMessage = "API key validation successful. All required permissions are enabled."
        };
    }

    /// <summary>
    /// Creates a failed validation result with detailed error message
    /// </summary>
    public static ApiKeyValidationResult Failure(
        List<string> missingPermissions,
        bool isIpRestricted = false,
        List<string>? allowedIps = null,
        string? serverIp = null)
    {
        var result = new ApiKeyValidationResult
        {
            IsValid = false,
            MissingPermissions = missingPermissions,
            IsIpRestricted = isIpRestricted,
            AllowedIps = allowedIps ?? new(),
            ServerIp = serverIp
        };

        // Build detailed message
        var messageParts = new List<string>();

        if (missingPermissions.Any())
        {
            messageParts.Add($"Missing required permissions: {string.Join(", ", missingPermissions)}");
        }

        if (isIpRestricted)
        {
            if (!string.IsNullOrEmpty(serverIp))
            {
                messageParts.Add($"API key is IP-restricted. Please add this server's IP address to the whitelist: {serverIp}");
            }
            else
            {
                messageParts.Add("API key is IP-restricted. Please check your IP whitelist settings.");
            }
        }

        result.DetailedMessage = string.Join(" | ", messageParts);
        return result;
    }
}
