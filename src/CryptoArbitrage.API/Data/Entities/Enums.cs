namespace CryptoArbitrage.API.Data.Entities;

public enum PositionSide
{
    Long = 0,
    Short = 1
}

public enum PositionStatus
{
    Open = 0,
    Closed = 1,
    Liquidated = 2,
    PartiallyFilled = 3
}

public enum OpportunityStatus
{
    Detected = 0,
    Executed = 1,
    Expired = 2,
    Ignored = 3
}

public enum ExecutionState
{
    Running = 0,
    Stopped = 1,
    Failed = 2
}

public enum PositionType
{
    Perpetual = 0,  // Futures/perpetual position
    Spot = 1        // Spot order/position
}

public enum FundingDirection
{
    LongPaysShort = 0,   // Positive funding rate - longs pay shorts
    ShortPaysLong = 1    // Negative funding rate - shorts pay longs
}

public enum ReconciliationStatus
{
    Preliminary = 0,         // Just closed, waiting for transactions
    PartiallyReconciled = 1, // Some transactions found, but not all
    FullyReconciled = 2,     // All expected transactions found
    StaleUnreconciled = 3    // >24 hours old, still incomplete (flag for manual review)
}
