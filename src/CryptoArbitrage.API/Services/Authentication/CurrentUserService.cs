using System.Security.Claims;

namespace CryptoArbitrage.API.Services.Authentication;

/// <summary>
/// Provides the authenticated user's identity information from JWT token claims.
/// CRITICAL SECURITY: This service extracts user identity from the JWT token claims,
/// ensuring that user identity is ALWAYS server-validated and never trusted from client input.
/// </summary>
public class CurrentUserService : ICurrentUserService
{
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly ILogger<CurrentUserService> _logger;
    private static readonly AsyncLocal<string?> _backgroundUserId = new AsyncLocal<string?>();

    public CurrentUserService(IHttpContextAccessor httpContextAccessor, ILogger<CurrentUserService> logger)
    {
        _httpContextAccessor = httpContextAccessor;
        _logger = logger;
    }

    /// <summary>
    /// Gets the authenticated user's ID from the NameIdentifier claim in the JWT token.
    /// For background operations, returns the ID set via SetBackgroundUserContext.
    /// </summary>
    public string? UserId
    {
        get
        {
            // First check if we're in a background context
            if (_backgroundUserId.Value != null)
            {
                return _backgroundUserId.Value;
            }

            // Otherwise, get from HTTP context (JWT token)
            return _httpContextAccessor.HttpContext?.User
                ?.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        }
    }

    /// <summary>
    /// Gets the authenticated user's email from the Email claim in the JWT token.
    /// </summary>
    public string? Email => _httpContextAccessor.HttpContext?.User
        ?.FindFirst(ClaimTypes.Email)?.Value;

    /// <summary>
    /// Gets a value indicating whether the user is authenticated (has a valid UserId claim).
    /// </summary>
    public bool IsAuthenticated => !string.IsNullOrEmpty(UserId);

    /// <summary>
    /// Validates that the current authenticated user owns the specified resource.
    /// CRITICAL: This method MUST be called before every update/delete operation to prevent
    /// unauthorized access to other users' resources.
    /// </summary>
    /// <exception cref="UnauthorizedAccessException">If user is not authenticated or does not own the resource</exception>
    public void ValidateUserOwnsResource(string resourceUserId)
    {
        if (string.IsNullOrEmpty(UserId))
        {
            _logger.LogWarning("Attempted access to resource without authentication");
            throw new UnauthorizedAccessException("User not authenticated");
        }

        if (UserId != resourceUserId)
        {
            _logger.LogWarning(
                "User {UserId} attempted unauthorized access to resource owned by {ResourceUserId}",
                UserId, resourceUserId);
            throw new UnauthorizedAccessException("Access denied to resource");
        }

        _logger.LogDebug("User {UserId} validated ownership of resource", UserId);
    }

    /// <summary>
    /// Sets the user context for background operations (e.g., agent trading).
    /// Returns an IDisposable that restores the previous context when disposed.
    /// IMPORTANT: Only use for trusted background operations like agent trading.
    /// </summary>
    public IDisposable SetBackgroundUserContext(string userId)
    {
        if (string.IsNullOrEmpty(userId))
        {
            throw new ArgumentException("User ID cannot be null or empty", nameof(userId));
        }

        _logger.LogDebug("Setting background user context for user {UserId}", userId);

        var previousUserId = _backgroundUserId.Value;
        _backgroundUserId.Value = userId;

        return new BackgroundUserContextScope(previousUserId, _logger);
    }

    private class BackgroundUserContextScope : IDisposable
    {
        private readonly string? _previousUserId;
        private readonly ILogger _logger;
        private bool _disposed;

        public BackgroundUserContextScope(string? previousUserId, ILogger logger)
        {
            _previousUserId = previousUserId;
            _logger = logger;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _backgroundUserId.Value = _previousUserId;
                _logger.LogDebug("Restored previous background user context");
                _disposed = true;
            }
        }
    }
}
