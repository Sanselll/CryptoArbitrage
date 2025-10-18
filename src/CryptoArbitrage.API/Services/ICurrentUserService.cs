namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service for retrieving authenticated user information from the current HTTP context.
/// CRITICAL: This is the single source of truth for user identity - NEVER trust client-provided user IDs.
/// All user identity information is extracted from the JWT token claims.
/// </summary>
public interface ICurrentUserService
{
    /// <summary>
    /// Gets the authenticated user's unique identifier from the JWT token.
    /// Returns null if user is not authenticated.
    /// </summary>
    string? UserId { get; }

    /// <summary>
    /// Gets the authenticated user's email from the JWT token.
    /// Returns null if user is not authenticated.
    /// </summary>
    string? Email { get; }

    /// <summary>
    /// Gets a value indicating whether the user is authenticated.
    /// </summary>
    bool IsAuthenticated { get; }

    /// <summary>
    /// Validates that the current authenticated user owns the specified resource.
    /// Throws UnauthorizedAccessException if user does not own the resource.
    /// CRITICAL: Must be called before every update/delete operation.
    /// </summary>
    /// <param name="resourceUserId">The UserId of the resource to validate ownership</param>
    /// <exception cref="UnauthorizedAccessException">Thrown if user is not authenticated or does not own the resource</exception>
    void ValidateUserOwnsResource(string resourceUserId);
}
