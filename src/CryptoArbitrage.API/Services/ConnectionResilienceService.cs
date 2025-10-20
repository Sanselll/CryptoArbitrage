using System.Net.Sockets;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Provides retry logic with exponential backoff and circuit breaker pattern
/// for handling transient network failures
/// </summary>
public class ConnectionResilienceService
{
    private readonly ILogger<ConnectionResilienceService> _logger;
    private readonly int _maxRetryAttempts;
    private readonly int _initialRetryDelayMs;
    private readonly int _maxRetryDelayMs;
    private readonly int _circuitBreakerThreshold;
    private readonly TimeSpan _circuitBreakerTimeout;

    // Circuit breaker state per connector
    private readonly Dictionary<string, CircuitBreakerState> _circuitBreakers = new();
    private readonly object _circuitBreakerLock = new();

    public ConnectionResilienceService(
        ILogger<ConnectionResilienceService> logger,
        IConfiguration configuration)
    {
        _logger = logger;

        // Load configuration with defaults
        var config = configuration.GetSection("ResilienceConfig");
        _maxRetryAttempts = config.GetValue("MaxRetryAttempts", 3);
        _initialRetryDelayMs = config.GetValue("InitialRetryDelayMs", 1000);
        _maxRetryDelayMs = config.GetValue("MaxRetryDelayMs", 30000);
        _circuitBreakerThreshold = config.GetValue("CircuitBreakerThreshold", 5);
        _circuitBreakerTimeout = TimeSpan.FromSeconds(config.GetValue("CircuitBreakerTimeoutSeconds", 60));
    }

    /// <summary>
    /// Execute an operation with exponential backoff retry logic
    /// </summary>
    public async Task<T> ExecuteWithRetryAsync<T>(
        string operationName,
        string connectorKey,
        Func<Task<T>> operation,
        CancellationToken cancellationToken = default)
    {
        // Check circuit breaker first
        if (IsCircuitOpen(connectorKey))
        {
            throw new InvalidOperationException(
                $"Circuit breaker is open for {connectorKey}. Will retry after timeout.");
        }

        int attempt = 0;
        int delay = _initialRetryDelayMs;

        while (attempt < _maxRetryAttempts)
        {
            try
            {
                attempt++;
                var result = await operation();

                // Success - reset circuit breaker
                ResetCircuitBreaker(connectorKey);

                if (attempt > 1)
                {
                    _logger.LogInformation(
                        "Operation {Operation} succeeded on attempt {Attempt}/{MaxAttempts}",
                        operationName, attempt, _maxRetryAttempts);
                }

                return result;
            }
            catch (Exception ex) when (IsTransientError(ex) && attempt < _maxRetryAttempts)
            {
                _logger.LogWarning(ex,
                    "Transient error in {Operation} (attempt {Attempt}/{MaxAttempts}). " +
                    "Retrying in {Delay}ms...",
                    operationName, attempt, _maxRetryAttempts, delay);

                await Task.Delay(delay, cancellationToken);

                // Exponential backoff with jitter
                delay = Math.Min(delay * 2 + Random.Shared.Next(0, 1000), _maxRetryDelayMs);
            }
            catch (Exception ex)
            {
                // Non-transient error or max retries reached
                RecordFailure(connectorKey);

                _logger.LogError(ex,
                    "Operation {Operation} failed after {Attempt} attempts",
                    operationName, attempt);

                throw;
            }
        }

        // Should never reach here, but just in case
        RecordFailure(connectorKey);
        throw new InvalidOperationException(
            $"Operation {operationName} failed after {_maxRetryAttempts} attempts");
    }

    /// <summary>
    /// Execute an operation with retry, returning a boolean success indicator instead of throwing
    /// </summary>
    public async Task<bool> ExecuteWithRetryBoolAsync(
        string operationName,
        string connectorKey,
        Func<Task<bool>> operation,
        CancellationToken cancellationToken = default)
    {
        try
        {
            return await ExecuteWithRetryAsync(operationName, connectorKey, operation, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute {Operation} for {Connector}",
                operationName, connectorKey);
            return false;
        }
    }

    /// <summary>
    /// Determine if an exception represents a transient error that should be retried
    /// </summary>
    private bool IsTransientError(Exception ex)
    {
        // Network-related errors that are typically transient
        return ex is HttpRequestException ||
               ex is SocketException ||
               ex is TimeoutException ||
               ex is TaskCanceledException ||
               ex is OperationCanceledException ||
               (ex.InnerException != null && IsTransientError(ex.InnerException)) ||
               (ex.Message?.Contains("nodename nor servname provided") == true) ||
               (ex.Message?.Contains("NetworkError") == true) ||
               (ex.Message?.Contains("Connection refused") == true) ||
               (ex.Message?.Contains("No such host") == true) ||
               (ex.Message?.Contains("ChannelClosed") == true) ||
               (ex.Message?.Contains("Connection reset") == true);
    }

    /// <summary>
    /// Check if circuit breaker is open for a connector
    /// </summary>
    private bool IsCircuitOpen(string connectorKey)
    {
        lock (_circuitBreakerLock)
        {
            if (_circuitBreakers.TryGetValue(connectorKey, out var state))
            {
                if (state.IsOpen)
                {
                    // Check if timeout has elapsed
                    if (DateTime.UtcNow - state.OpenedAt >= _circuitBreakerTimeout)
                    {
                        _logger.LogInformation(
                            "Circuit breaker timeout elapsed for {Connector}. Attempting to close circuit.",
                            connectorKey);
                        state.IsOpen = false;
                        state.FailureCount = 0;
                        return false;
                    }
                    return true;
                }
            }
            return false;
        }
    }

    /// <summary>
    /// Record a failure and potentially open the circuit breaker
    /// </summary>
    private void RecordFailure(string connectorKey)
    {
        lock (_circuitBreakerLock)
        {
            if (!_circuitBreakers.ContainsKey(connectorKey))
            {
                _circuitBreakers[connectorKey] = new CircuitBreakerState();
            }

            var state = _circuitBreakers[connectorKey];
            state.FailureCount++;
            state.LastFailureAt = DateTime.UtcNow;

            if (state.FailureCount >= _circuitBreakerThreshold && !state.IsOpen)
            {
                state.IsOpen = true;
                state.OpenedAt = DateTime.UtcNow;

                _logger.LogWarning(
                    "Circuit breaker opened for {Connector} after {Count} failures. " +
                    "Will retry after {Timeout} seconds.",
                    connectorKey, state.FailureCount, _circuitBreakerTimeout.TotalSeconds);
            }
        }
    }

    /// <summary>
    /// Reset circuit breaker after successful operation
    /// </summary>
    private void ResetCircuitBreaker(string connectorKey)
    {
        lock (_circuitBreakerLock)
        {
            if (_circuitBreakers.TryGetValue(connectorKey, out var state))
            {
                if (state.FailureCount > 0 || state.IsOpen)
                {
                    _logger.LogInformation(
                        "Circuit breaker reset for {Connector} after successful operation",
                        connectorKey);
                }
                state.FailureCount = 0;
                state.IsOpen = false;
            }
        }
    }

    /// <summary>
    /// Get health status for a connector
    /// </summary>
    public ConnectionHealth GetConnectionHealth(string connectorKey)
    {
        lock (_circuitBreakerLock)
        {
            if (_circuitBreakers.TryGetValue(connectorKey, out var state))
            {
                if (state.IsOpen)
                {
                    return new ConnectionHealth
                    {
                        IsHealthy = false,
                        Status = "CircuitOpen",
                        Message = $"Circuit breaker open since {state.OpenedAt:yyyy-MM-dd HH:mm:ss} UTC. " +
                                 $"Will retry after {_circuitBreakerTimeout.TotalSeconds}s timeout.",
                        FailureCount = state.FailureCount,
                        LastFailureAt = state.LastFailureAt
                    };
                }
                else if (state.FailureCount > 0)
                {
                    return new ConnectionHealth
                    {
                        IsHealthy = true,
                        Status = "Degraded",
                        Message = $"{state.FailureCount} recent failures. Connection working but unstable.",
                        FailureCount = state.FailureCount,
                        LastFailureAt = state.LastFailureAt
                    };
                }
            }

            return new ConnectionHealth
            {
                IsHealthy = true,
                Status = "Healthy",
                Message = "No recent failures",
                FailureCount = 0,
                LastFailureAt = null
            };
        }
    }

    private class CircuitBreakerState
    {
        public int FailureCount { get; set; }
        public bool IsOpen { get; set; }
        public DateTime OpenedAt { get; set; }
        public DateTime? LastFailureAt { get; set; }
    }
}

public class ConnectionHealth
{
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public int FailureCount { get; set; }
    public DateTime? LastFailureAt { get; set; }
}
