using Microsoft.AspNetCore.Mvc;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Base controller providing common error handling and response patterns.
/// Eliminates repetitive try-catch-log blocks across all controllers.
/// </summary>
public abstract class BaseController : ControllerBase
{
    protected readonly ILogger Logger;

    protected BaseController(ILogger logger)
    {
        Logger = logger;
    }

    /// <summary>
    /// Executes an action with automatic error handling and logging.
    /// Returns Ok(result) on success, or StatusCode(500) on error.
    /// </summary>
    protected async Task<ActionResult<T>> ExecuteAsync<T>(
        Func<Task<T>> action,
        string? operationName = null)
    {
        try
        {
            var result = await action();
            return Ok(result);
        }
        catch (Exception ex)
        {
            var operation = operationName ?? "operation";
            Logger.LogError(ex, "Error during {Operation}", operation);
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Executes an action with automatic error handling, logging, and custom error messages.
    /// Allows specifying both the operation name and a user-friendly error message.
    /// </summary>
    protected async Task<ActionResult<T>> ExecuteAsync<T>(
        Func<Task<T>> action,
        string operationName,
        string userErrorMessage)
    {
        try
        {
            var result = await action();
            return Ok(result);
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error during {Operation}", operationName);
            return StatusCode(500, new { errorMessage = userErrorMessage });
        }
    }

    /// <summary>
    /// Executes an action returning ActionResult with automatic error handling.
    /// Use when you need more control over the response (e.g., NotFound, Created, etc.).
    /// </summary>
    protected async Task<ActionResult> ExecuteActionAsync(
        Func<Task<ActionResult>> action,
        string? operationName = null)
    {
        try
        {
            return await action();
        }
        catch (Exception ex)
        {
            var operation = operationName ?? "operation";
            Logger.LogError(ex, "Error during {Operation}", operation);
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Executes an action that returns a specific ActionResult type with automatic error handling.
    /// Use when you need typed results (e.g., ActionResult&lt;T&gt;).
    /// </summary>
    protected async Task<ActionResult<T>> ExecuteActionAsync<T>(
        Func<Task<ActionResult<T>>> action,
        string? operationName = null)
    {
        try
        {
            return await action();
        }
        catch (Exception ex)
        {
            var operation = operationName ?? "operation";
            Logger.LogError(ex, "Error during {Operation}", operation);
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Validates that user is authenticated. Returns Unauthorized ActionResult if not.
    /// </summary>
    protected ActionResult? ValidateAuthentication(string? userId)
    {
        if (string.IsNullOrEmpty(userId))
            return Unauthorized(new { error = "User not authenticated" });
        return null;
    }

    /// <summary>
    /// Executes an action with user authentication check.
    /// Automatically returns Unauthorized if userId is null/empty.
    /// </summary>
    protected async Task<ActionResult<T>> ExecuteAuthenticatedAsync<T>(
        string? userId,
        Func<Task<T>> action,
        string? operationName = null)
    {
        var authResult = ValidateAuthentication(userId);
        if (authResult != null)
            return authResult;

        return await ExecuteAsync(action, operationName);
    }
}
