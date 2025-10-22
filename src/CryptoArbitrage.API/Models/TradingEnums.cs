namespace CryptoArbitrage.API.Models;

public enum OrderType
{
    Limit = 0,
    Market = 1,
    StopLoss = 2,
    StopLossLimit = 3,
    TakeProfit = 4,
    TakeProfitLimit = 5,
    StopMarket = 6,
    TakeProfitMarket = 7
}

public enum OrderSide
{
    Buy = 0,
    Sell = 1
}

public enum OrderStatus
{
    New = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Canceled = 3,
    PendingCancel = 4,
    Rejected = 5,
    Expired = 6,
    PendingNew = 7,
    Insurance = 8,
    Adl = 9,
    ExpiredInMatch = 10
}

public enum TimeInForce
{
    GTC = 0,  // Good Till Cancel
    IOC = 1,  // Immediate Or Cancel
    FOK = 2,  // Fill Or Kill
    GTX = 3   // Good Till Crossing (Post Only)
}

public enum TradeRole
{
    Maker = 0,
    Taker = 1
}

public enum TransactionType
{
    Deposit = 0,
    Withdrawal = 1,
    Transfer = 2,
    Commission = 3,
    Funding = 4,
    Rebate = 5,
    Airdrop = 6,
    Other = 7,
    RealizedPnL = 8,
    Trade = 9,
    Liquidation = 10,
    Bonus = 11,
    WelcomeBonus = 12,
    FundingFee = 13,
    InsuranceClear = 14,
    ReferralKickback = 15,
    CommissionRebate = 16,
    ContestReward = 17,
    InternalTransfer = 18,
    Settlement = 19,
    Delivery = 20,
    Adl = 21
}

public enum TransactionStatus
{
    Pending = 0,
    Completed = 1,
    Failed = 2,
    Canceled = 3,
    Confirmed = 4
}
