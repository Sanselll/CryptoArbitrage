using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.Reconciliation.Configuration;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.Reconciliation;

/// <summary>
/// Core service for reconciling positions with transaction history
/// Links commissions and funding fees to positions and calculates accurate trading fees
/// </summary>
public class PositionReconciliationService
{
    private readonly ILogger<PositionReconciliationService> _logger;
    private readonly PositionReconciliationConfiguration _config;
    private readonly IDataRepository<List<TransactionDto>> _transactionRepository;

    public PositionReconciliationService(
        ILogger<PositionReconciliationService> logger,
        PositionReconciliationConfiguration config,
        IDataRepository<List<TransactionDto>> transactionRepository)
    {
        _logger = logger;
        _config = config;
        _transactionRepository = transactionRepository;
    }

    /// <summary>
    /// Reconcile a single position with available transactions
    /// </summary>
    public async Task<ReconciliationResult> ReconcilePositionAsync(
        Position position,
        ArbitrageDbContext dbContext,
        CancellationToken cancellationToken = default)
    {
        var result = new ReconciliationResult
        {
            PositionId = position.Id,
            StartStatus = position.ReconciliationStatus
        };

        try
        {
            _logger.LogDebug("Reconciling Position {PositionId} ({Symbol} {Exchange} {Type} {Side})",
                position.Id, position.Symbol, position.Exchange, position.Type, position.Side);

            // Get user's transactions for this exchange
            var transactionKey = $"transactionhistory:{position.UserId}:{position.Exchange}";
            var transactions = await _transactionRepository.GetAsync(transactionKey, cancellationToken);

            if (transactions == null || transactions.Count == 0)
            {
                _logger.LogDebug("No transactions available for user {UserId} on {Exchange}",
                    position.UserId, position.Exchange);
                result.Success = false;
                result.ErrorMessage = "No transactions available";
                return result;
            }

            _logger.LogDebug("Found {Count} transactions for reconciliation", transactions.Count);

            // Step 1: Find and link trading commissions using OrderId
            var commissionResults = await ReconcileTradingCommissionsAsync(
                position, transactions, dbContext, cancellationToken);

            result.CommissionsLinked = commissionResults.LinkedCount;
            result.TradingFeesPaid = commissionResults.TotalFees;

            // Step 2: Find and link funding fees using time-window matching (perpetual only)
            if (position.Type == PositionType.Perpetual)
            {
                var fundingResults = await ReconcileFundingFeesAsync(
                    position, transactions, dbContext, cancellationToken);

                result.FundingFeesLinked = fundingResults.LinkedCount;
                result.FundingFeePaid = fundingResults.FeesPaid;
                result.FundingFeeReceived = fundingResults.FeesReceived;
            }

            // Step 3: Fees are now calculated on-the-fly from PositionTransaction table (no storage needed)

            // Step 4: Calculate and persist P&L breakdown for closed positions
            if (position.Status == PositionStatus.Closed && position.ClosedAt.HasValue && position.ExitPrice.HasValue)
            {
                CalculateAndPersistPnL(position, dbContext);
            }

            // Step 5: Determine new reconciliation status
            var newStatus = DetermineReconciliationStatus(position, dbContext);
            position.ReconciliationStatus = newStatus;

            if (newStatus == ReconciliationStatus.FullyReconciled)
            {
                position.ReconciliationCompletedAt = DateTime.UtcNow;
            }

            await dbContext.SaveChangesAsync(cancellationToken);

            result.EndStatus = newStatus;
            result.Success = true;

            _logger.LogInformation(
                "Reconciled Position {PositionId}: Status {OldStatus}→{NewStatus}, " +
                "Commissions: {Commissions}, Trading Fees: ${TradingFees:F4}, " +
                "Funding Paid: ${FundingPaid:F4}, Funding Received: ${FundingReceived:F4}",
                position.Id, result.StartStatus, result.EndStatus,
                result.CommissionsLinked, result.TradingFeesPaid,
                result.FundingFeePaid, result.FundingFeeReceived);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reconciling Position {PositionId}", position.Id);
            result.Success = false;
            result.ErrorMessage = ex.Message;
            return result;
        }
    }

    /// <summary>
    /// Find and link trading commissions using OrderId matching
    /// </summary>
    private async Task<CommissionReconciliationResult> ReconcileTradingCommissionsAsync(
        Position position,
        List<TransactionDto> transactions,
        ArbitrageDbContext dbContext,
        CancellationToken cancellationToken)
    {
        var result = new CommissionReconciliationResult();

        // Define time window for matching
        var startTime = position.OpenedAt.AddMinutes(-_config.CommissionPreMatchWindowMinutes);
        var endTime = (position.ClosedAt ?? DateTime.UtcNow)
            .AddMinutes(_config.CommissionPostMatchWindowMinutes);

        // Find commission transactions matching OrderId
        var orderIds = new List<string>();
        if (!string.IsNullOrEmpty(position.OrderId))
            orderIds.Add(position.OrderId);
        if (!string.IsNullOrEmpty(position.CloseOrderId))
            orderIds.Add(position.CloseOrderId);

        if (orderIds.Count == 0)
        {
            _logger.LogWarning("Position {PositionId} has no OrderId or CloseOrderId", position.Id);
            return result;
        }

        var commissionTransactions = transactions
            .Where(t => (t.Type == TransactionType.Commission || t.Type == TransactionType.Trade)
                && t.Symbol == position.Symbol
                && t.Exchange == position.Exchange
                && !string.IsNullOrEmpty(t.OrderId)
                && orderIds.Contains(t.OrderId)
                && t.CreatedAt >= startTime
                && t.CreatedAt <= endTime
                && t.Fee > 0) // Only include if there's actually a fee
            .ToList();

        _logger.LogDebug("Found {Count} commission transactions for Position {PositionId} (OrderIds: {OrderIds})",
            commissionTransactions.Count, position.Id, string.Join(", ", orderIds));

        // Check if already linked (idempotency)
        var existingLinks = await dbContext.PositionTransactions
            .Where(pt => pt.PositionId == position.Id
                && pt.TransactionType == TransactionType.Commission)
            .Select(pt => pt.TransactionId)
            .ToListAsync(cancellationToken);

        // Link new commissions
        foreach (var tx in commissionTransactions)
        {
            if (existingLinks.Contains(tx.TransactionId))
            {
                _logger.LogDebug("Transaction {TxId} already linked to Position {PositionId}",
                    tx.TransactionId, position.Id);
                continue;
            }

            var positionTransaction = new PositionTransaction
            {
                PositionId = position.Id,
                TransactionId = tx.TransactionId,
                Exchange = tx.Exchange,
                Symbol = tx.Symbol,
                TransactionType = TransactionType.Commission,
                Amount = tx.Amount,
                Fee = tx.Fee,
                SignedFee = tx.SignedFee,
                OrderId = tx.OrderId,
                AllocationPercentage = 1.0m, // 100% - commissions are position-specific
                TransactionCreatedAt = tx.CreatedAt,
                LinkedAt = DateTime.UtcNow,
                Asset = tx.FeeAsset
            };

            dbContext.PositionTransactions.Add(positionTransaction);
            result.LinkedCount++;
            result.TotalFees += tx.Fee;

            _logger.LogDebug("Linked commission Transaction {TxId} to Position {PositionId}: Fee ${Fee:F6}",
                tx.TransactionId, position.Id, tx.Fee);
        }

        return result;
    }

    /// <summary>
    /// Find and link funding fees using time-window matching with quantity-weighted allocation
    /// </summary>
    private async Task<FundingFeeReconciliationResult> ReconcileFundingFeesAsync(
        Position position,
        List<TransactionDto> transactions,
        ArbitrageDbContext dbContext,
        CancellationToken cancellationToken)
    {
        var result = new FundingFeeReconciliationResult();

        // Define time window for matching
        var startTime = position.OpenedAt.AddMinutes(-_config.FundingFeeTimeWindowMinutes);
        var endTime = (position.ClosedAt ?? DateTime.UtcNow)
            .AddMinutes(_config.FundingFeeTimeWindowMinutes);

        // Find funding fee transactions in time window
        var fundingTransactions = transactions
            .Where(t => t.Type == TransactionType.FundingFee
                && t.Symbol == position.Symbol
                && t.Exchange == position.Exchange
                && t.CreatedAt >= startTime
                && t.CreatedAt <= endTime)
            .ToList();

        _logger.LogInformation("FUNDING_FEE_DIAGNOSTIC: Found {Count} funding fee transactions for Position {PositionId} " +
            "(Symbol={Symbol}, Exchange={Exchange}, Side={Side}, OpenedAt={OpenedAt}, ClosedAt={ClosedAt})",
            fundingTransactions.Count, position.Id, position.Symbol, position.Exchange, position.Side,
            position.OpenedAt, position.ClosedAt);

        // Log each transaction found
        foreach (var txLog in fundingTransactions)
        {
            _logger.LogInformation("FUNDING_FEE_DIAGNOSTIC: Transaction {TxId} details: " +
                "Symbol={Symbol}, Exchange={Exchange}, CreatedAt={CreatedAt}, SignedFee={SignedFee}",
                txLog.TransactionId, txLog.Symbol, txLog.Exchange, txLog.CreatedAt, txLog.SignedFee);
        }

        // Check if already linked (idempotency)
        var existingLinks = await dbContext.PositionTransactions
            .Where(pt => pt.PositionId == position.Id
                && pt.TransactionType == TransactionType.FundingFee)
            .Select(pt => pt.TransactionId)
            .ToListAsync(cancellationToken);

        // Process each funding fee transaction
        foreach (var tx in fundingTransactions)
        {
            if (existingLinks.Contains(tx.TransactionId))
            {
                _logger.LogDebug("Funding fee {TxId} already linked to Position {PositionId}",
                    tx.TransactionId, position.Id);
                continue;
            }

            // Find overlapping positions for quantity-weighted allocation
            _logger.LogInformation("FUNDING_FEE_DIAGNOSTIC: Searching for overlapping positions for Transaction {TxId} " +
                "(Looking for: Side={Side}, Symbol={Symbol}, Exchange={Exchange}, TxTime={TxTime})",
                tx.TransactionId, position.Side, position.Symbol, position.Exchange, tx.CreatedAt);

            var overlappingPositions = await dbContext.Positions
                .Where(p => p.UserId == position.UserId
                    && p.Symbol == position.Symbol
                    && p.Exchange == position.Exchange
                    && p.Type == PositionType.Perpetual
                    && p.Side == position.Side // Same side (long/short)
                    && p.OpenedAt <= tx.CreatedAt
                    && (p.ClosedAt == null || p.ClosedAt >= tx.CreatedAt))
                .ToListAsync(cancellationToken);

            _logger.LogInformation("FUNDING_FEE_DIAGNOSTIC: Found {Count} overlapping positions for Transaction {TxId} at {Time}. " +
                "Positions: [{PositionIds}]",
                overlappingPositions.Count, tx.TransactionId, tx.CreatedAt,
                string.Join(", ", overlappingPositions.Select(p => $"P{p.Id}")));

            // Calculate quantity-weighted allocation
            var totalQuantity = overlappingPositions.Sum(p => p.Quantity);
            var allocationPercentage = totalQuantity > 0
                ? position.Quantity / totalQuantity
                : 0m;

            if (allocationPercentage == 0)
            {
                _logger.LogWarning("Zero allocation percentage for Position {PositionId} funding fee {TxId}",
                    position.Id, tx.TransactionId);
                continue;
            }

            // Calculate allocated fee (use SignedFee for proper sign)
            var signedFee = tx.SignedFee ?? (tx.Fee * (tx.Amount >= 0 ? 1 : -1));
            var allocatedFee = signedFee * allocationPercentage;

            var positionTransaction = new PositionTransaction
            {
                PositionId = position.Id,
                TransactionId = tx.TransactionId,
                Exchange = tx.Exchange,
                Symbol = tx.Symbol,
                TransactionType = TransactionType.FundingFee,
                Amount = tx.Amount * allocationPercentage,
                Fee = Math.Abs(allocatedFee),
                SignedFee = allocatedFee,
                OrderId = null, // Funding fees don't have OrderId
                AllocationPercentage = allocationPercentage,
                TransactionCreatedAt = tx.CreatedAt,
                LinkedAt = DateTime.UtcNow,
                Asset = tx.Asset,
                Notes = overlappingPositions.Count > 1
                    ? $"Split across {overlappingPositions.Count} positions"
                    : null
            };

            dbContext.PositionTransactions.Add(positionTransaction);
            result.LinkedCount++;

            // Track paid vs received
            if (allocatedFee < 0)
            {
                result.FeesPaid += Math.Abs(allocatedFee);
                _logger.LogDebug("Linked funding PAID Transaction {TxId} to Position {PositionId}: " +
                    "${Fee:F6} ({Pct:P2} allocation)",
                    tx.TransactionId, position.Id, Math.Abs(allocatedFee), allocationPercentage);
            }
            else
            {
                result.FeesReceived += allocatedFee;
                _logger.LogDebug("Linked funding RECEIVED Transaction {TxId} to Position {PositionId}: " +
                    "${Fee:F6} ({Pct:P2} allocation)",
                    tx.TransactionId, position.Id, allocatedFee, allocationPercentage);
            }
        }

        return result;
    }

    /// <summary>
    /// Calculate and persist P&L breakdown on the Position entity
    /// Formula: RealizedPnL = FundingEarned + PricePnL - TradingFees
    /// </summary>
    private void CalculateAndPersistPnL(Position position, ArbitrageDbContext dbContext)
    {
        // Get all linked transactions for this position
        var linkedTransactions = dbContext.PositionTransactions
            .Where(pt => pt.PositionId == position.Id)
            .ToList();

        // Calculate Funding Earned (sum of all funding fees)
        // Positive SignedFee = received funding (longs receive when funding is positive)
        // Negative SignedFee = paid funding (shorts pay when funding is positive)
        var fundingEarnedUsd = linkedTransactions
            .Where(t => t.TransactionType == TransactionType.FundingFee)
            .Sum(t => t.SignedFee ?? 0);

        // Calculate Trading Fees (sum of all commissions/trading fees)
        var tradingFeesUsd = linkedTransactions
            .Where(t => t.TransactionType == TransactionType.Commission || t.TransactionType == TransactionType.Trade)
            .Sum(t => t.Fee);

        // Calculate Price P&L
        // For LONG: (ExitPrice - EntryPrice) * Quantity
        // For SHORT: (EntryPrice - ExitPrice) * Quantity
        decimal pricePnLUsd;
        if (position.Side == PositionSide.Long)
        {
            pricePnLUsd = (position.ExitPrice!.Value - position.EntryPrice) * position.Quantity;
        }
        else // Short
        {
            pricePnLUsd = (position.EntryPrice - position.ExitPrice!.Value) * position.Quantity;
        }

        // Calculate Realized P&L
        // Formula: RealizedPnL = FundingEarned + PricePnL - TradingFees
        var realizedPnLUsd = fundingEarnedUsd + pricePnLUsd - tradingFeesUsd;

        // Calculate Realized P&L Percentage (relative to initial margin)
        var realizedPnLPct = position.InitialMargin > 0
            ? (realizedPnLUsd / position.InitialMargin) * 100
            : 0m;

        // Persist to Position entity
        position.FundingEarnedUsd = fundingEarnedUsd;
        position.TradingFeesUsd = tradingFeesUsd;
        position.PricePnLUsd = pricePnLUsd;
        position.RealizedPnLUsd = realizedPnLUsd;
        position.RealizedPnLPct = realizedPnLPct;

        _logger.LogInformation(
            "Calculated P&L for Position {PositionId}: " +
            "FundingEarned=${Funding:F4}, TradingFees=${Fees:F4}, PricePnL=${PricePnL:F4}, " +
            "RealizedPnL=${RealizedPnL:F4} ({RealizedPct:F2}%)",
            position.Id, fundingEarnedUsd, tradingFeesUsd, pricePnLUsd, realizedPnLUsd, realizedPnLPct);
    }

    /// <summary>
    /// Determine reconciliation status based on expected vs actual transactions
    /// </summary>
    private ReconciliationStatus DetermineReconciliationStatus(
        Position position,
        ArbitrageDbContext dbContext)
    {
        // Get linked transactions
        var linkedTransactions = dbContext.PositionTransactions
            .Where(pt => pt.PositionId == position.Id)
            .ToList();

        if (linkedTransactions.Count == 0)
        {
            // No transactions linked - check if stale
            if (position.ClosedAt.HasValue &&
                (DateTime.UtcNow - position.ClosedAt.Value).TotalHours > _config.StaleThresholdHours)
            {
                return ReconciliationStatus.StaleUnreconciled;
            }
            return ReconciliationStatus.Preliminary;
        }

        // Check for expected transactions
        var hasOpeningCommission = linkedTransactions.Any(t =>
            t.TransactionType == TransactionType.Commission
            && t.OrderId == position.OrderId);

        var hasClosingCommission = true;
        if (position.Status == PositionStatus.Closed && !string.IsNullOrEmpty(position.CloseOrderId))
        {
            hasClosingCommission = linkedTransactions.Any(t =>
                t.TransactionType == TransactionType.Commission
                && t.OrderId == position.CloseOrderId);
        }

        // Check funding fees for perpetual positions
        var hasExpectedFunding = true;
        if (position.Type == PositionType.Perpetual)
        {
            var durationHours = ((position.ClosedAt ?? DateTime.UtcNow) - position.OpenedAt).TotalHours;

            if (durationHours >= (double)_config.MinHoursForFundingFees)
            {
                var expectedFundingCount = (int)(durationHours / (double)_config.FundingIntervalHours);
                var actualFundingCount = linkedTransactions.Count(t =>
                    t.TransactionType == TransactionType.FundingFee);

                // Allow tolerance of ±1 funding fee
                hasExpectedFunding = actualFundingCount >= (expectedFundingCount - 1);

                _logger.LogDebug("Position {PositionId} funding check: Expected ~{Expected}, Got {Actual}",
                    position.Id, expectedFundingCount, actualFundingCount);
            }
        }

        // Determine status
        // IMPORTANT: OPEN positions can NEVER be FullyReconciled because new funding fees keep coming every 8h
        if (position.Status == PositionStatus.Open || position.ClosedAt == null)
        {
            // OPEN position: return PartiallyReconciled (will be re-evaluated every cycle for new funding fees)
            return linkedTransactions.Count > 0
                ? ReconciliationStatus.PartiallyReconciled
                : ReconciliationStatus.Preliminary;
        }

        // CLOSED position: can be marked as FullyReconciled
        if (hasOpeningCommission && hasClosingCommission && hasExpectedFunding)
        {
            return ReconciliationStatus.FullyReconciled;
        }
        else if (linkedTransactions.Count > 0)
        {
            // At least some transactions linked

            // Check if stale
            if (position.ClosedAt.HasValue &&
                (DateTime.UtcNow - position.ClosedAt.Value).TotalHours > _config.StaleThresholdHours)
            {
                return ReconciliationStatus.StaleUnreconciled;
            }

            return ReconciliationStatus.PartiallyReconciled;
        }
        else
        {
            return ReconciliationStatus.Preliminary;
        }
    }
}

// Helper classes for results
public class ReconciliationResult
{
    public int PositionId { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public ReconciliationStatus StartStatus { get; set; }
    public ReconciliationStatus EndStatus { get; set; }
    public int CommissionsLinked { get; set; }
    public int FundingFeesLinked { get; set; }
    public decimal TradingFeesPaid { get; set; }
    public decimal FundingFeePaid { get; set; }
    public decimal FundingFeeReceived { get; set; }
}

public class CommissionReconciliationResult
{
    public int LinkedCount { get; set; }
    public decimal TotalFees { get; set; }
}

public class FundingFeeReconciliationResult
{
    public int LinkedCount { get; set; }
    public decimal FeesPaid { get; set; }
    public decimal FeesReceived { get; set; }
}
