export interface FundingRate {
  exchange: string;
  symbol: string;
  rate: number;
  annualizedRate: number;
  average3DayRate?: number;
  average24hRate?: number;
  fundingIntervalHours?: number;
  volume24h?: number;
  fundingTime: string;
  nextFundingTime: string;
  recordedAt: string;
}

export enum PositionSide {
  Long = 0,
  Short = 1
}

export enum PositionStatus {
  Open = 0,
  Closed = 1,
  Liquidated = 2,
  PartiallyFilled = 3
}

export enum PositionType {
  Perpetual = 0,
  Spot = 1
}

export interface Position {
  id: number;
  executionId?: number;
  exchange: string;
  symbol: string;
  type: PositionType;
  side: PositionSide;
  status: PositionStatus;
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  leverage: number;
  initialMargin: number;
  realizedPnL: number;
  unrealizedPnL: number;
  totalFundingFeePaid: number;
  totalFundingFeeReceived: number;
  netFundingFee: number;
  tradingFeePaid: number;
  reconciliationStatus: number;
  reconciliationCompletedAt?: string;
  openedAt: string;
  closedAt?: string;
  activeOpportunityId?: number;

  // RL Predictions (Reinforcement Learning)
  rlExitProbability?: number;             // RL EXIT action probability (0-1)
  rlHoldProbability?: number;             // RL HOLD action probability (0-1)
  rlConfidence?: string;                  // RL confidence level: HIGH/MEDIUM/LOW
  rlStateValue?: number;                  // RL state value estimate
  rlModelVersion?: string;                // RL model version
}

export enum OpportunityStatus {
  Detected = 0,
  Executed = 1,
  Expired = 2,
  Ignored = 3
}

export enum ExecutionState {
  Running = 0,
  Stopped = 1,
  Failed = 2
}

export enum StrategySubType {
  SpotPerpetualSameExchange = 0,
  CrossExchangeFuturesFutures = 1,
  CrossExchangeSpotFutures = 2,
  CrossExchangeFuturesPriceSpread = 3
}

export enum LiquidityStatus {
  Good = 0,
  Medium = 1,
  Low = 2
}

export interface ArbitrageOpportunity {
  id: number;
  symbol: string;
  strategy?: number; // 0 = CrossExchange, 1 = SpotPerpetual
  subType?: StrategySubType;

  // Cross-exchange fields
  longExchange: string;
  shortExchange: string;
  longFundingRate: number;
  shortFundingRate: number;
  longFundingIntervalHours?: number;   // Funding interval for long exchange (1h, 4h, 8h, etc.)
  shortFundingIntervalHours?: number;  // Funding interval for short exchange
  longNextFundingTime?: string;        // When long funding rate expires/renews
  shortNextFundingTime?: string;       // When short funding rate expires/renews

  // Cross-exchange price fields - NEW properly-named fields
  longExchangePrice?: number;          // Price on long exchange (perpetual)
  shortExchangePrice?: number;         // Price on short exchange (perpetual)
  currentPriceSpreadPercent?: number;  // Current instant price spread: (Short - Long) / Long * 100

  // Spot-perpetual fields
  exchange?: string;
  // DEPRECATED: spotPrice/perpetualPrice are confusing for cross-exchange
  // For cross-exchange: spotPrice = longExchangePrice, perpetualPrice = shortExchangePrice
  // Use longExchangePrice/shortExchangePrice for cross-exchange instead
  spotPrice?: number;
  perpetualPrice?: number;
  fundingRate?: number;
  annualizedFundingRate?: number;
  pricePremium?: number;

  // Common fields
  spreadRate: number;
  annualizedSpread: number;
  estimatedProfitPercentage: number;
  positionCostPercent: number;
  breakEvenTimeHours?: number;
  volume24h?: number;

  // Calculated metrics (current funding rate)
  fundProfit8h: number;           // 8-hour profit percentage using current funding rate
  fundApr: number;                // Annualized percentage rate using current funding rate

  // Projected metrics (24-hour average)
  fund_profit_8h_24h_proj?: number;   // 8-hour profit % using 24h average funding rate
  fund_apr_24h_proj?: number;         // APR % using 24h average funding rate
  fundBreakEvenTime24hProj?: number;  // Break-even hours using 24h average funding rate

  // Projected metrics (3-day average)
  fund_profit_8h_3d_proj?: number;    // 8-hour profit % using 3D average funding rate
  fund_apr_3d_proj?: number;          // APR % using 3D average funding rate
  fundBreakEvenTime3dProj?: number;   // Break-even hours using 3D average funding rate

  // Price spread projection metrics (for CFPS - CrossExchangeFuturesPriceSpread)
  price_spread_24h_avg?: number;      // 24-hour average price spread %
  price_spread_3d_avg?: number;       // 3-day average price spread %

  // Spread history metrics (30 samples) - for Cross-Exchange only
  spread_30_sample_avg?: number;      // Average spread based on last 30 price samples
  spreadVolatilityStdDev?: number;    // Standard deviation of spread samples
  spreadVolatilityCv?: number;        // Coefficient of variation (StdDev/Mean)

  // Per-exchange volumes for cross-exchange arbitrage
  longVolume24h?: number;
  shortVolume24h?: number;

  // Liquidity metrics
  bidAskSpreadPercent?: number;
  orderbookDepthUsd?: number;
  liquidityStatus?: LiquidityStatus;
  liquidityWarning?: string;

  // RL Predictions (Reinforcement Learning)
  rlEnterProbability?: number;            // RL ENTER action probability (0-1)
  rlHoldProbability?: number;             // RL HOLD action probability (0-1)
  rlConfidence?: string;                  // RL confidence level: HIGH/MEDIUM/LOW
  rlStateValue?: number;                  // RL state value estimate
  rlModelVersion?: string;                // RL model version

  status: OpportunityStatus;
  detectedAt: string;
  executedAt?: string;

  // Execution fields (merged from Execution table)
  executionId?: number;
  executionState?: ExecutionState;
  executionStartedAt?: string;
  executionFundingEarned?: number;

  // Computed unique key for frontend tracking
  uniqueKey?: string;
}

export interface ActiveOpportunity {
  id: number;
  symbol: string;
  exchange: string;
  strategy: number;

  // Funding rate information (captured at execution time)
  fundingRate: number;
  annualizedFundingRate: number;
  spreadRate: number;
  annualizedSpread: number;

  // Execution details
  executedAt: string;
  closedAt?: string;
  isActive: boolean;

  // Position management
  positionSizeUsd: number;
  leverage: number;
  stopLossPercentage?: number;
  takeProfitPercentage?: number;

  // Spot position (bought asset)
  spotQuantity: number;
  spotEntryPrice: number;
  spotExitPrice?: number;
  spotBuyOrderId?: string;
  spotSellOrderId?: string;

  // Perpetual position (short futures)
  perpQuantity: number;
  perpEntryPrice: number;
  perpExitPrice?: number;
  perpOpenOrderId?: string;
  perpCloseOrderId?: string;

  // P&L tracking
  realizedPnL: number;
  totalFundingFeesEarned: number;
  totalFundingFeesPaid: number;
  netFundingFees: number;

  // Optional notes
  notes?: string;
}

export interface AccountBalance {
  exchange: string;
  // Combined totals (Spot + Futures)
  totalBalance: number;
  availableBalance: number;
  // Operational balance (USDT + coins in positions + futures balance)
  operationalBalanceUsd: number;
  // Spot balances
  spotBalanceUsd: number;
  spotAvailableUsd: number;
  spotAssets: Record<string, number>;
  // Futures balances
  futuresBalanceUsd: number;
  futuresAvailableUsd: number;
  marginUsed: number;
  unrealizedPnL: number;
  updatedAt: string;
}

// Notification types
export enum NotificationType {
  NegativeFunding = 0,
  ExecutionStateChange = 1,
  ExchangeConnectivity = 2,
  OpportunityDetected = 3,
  LiquidationRisk = 4,
  General = 5
}

export enum NotificationSeverity {
  Info = 0,
  Success = 1,
  Warning = 2,
  Error = 3
}

export interface Notification {
  id: string;
  type: NotificationType;
  severity: NotificationSeverity;
  title: string;
  message: string;
  data?: any;
  autoClose: boolean;
  autoCloseDelay?: number;
  timestamp: string;
}

export interface DashboardData {
  fundingRates: FundingRate[];
  openPositions: Position[];
  opportunities: ArbitrageOpportunity[];
  balances: AccountBalance[];
  totalPnL: number;
  todayPnL: number;
  totalMarginUsed: number;
  activeOpportunities: number;
  updatedAt: string;
}

// Trading data types
export enum OrderType {
  Limit = 0,
  Market = 1,
  StopLoss = 2,
  StopLossLimit = 3,
  TakeProfit = 4,
  TakeProfitLimit = 5
}

export enum OrderSide {
  Buy = 0,
  Sell = 1
}

export enum OrderStatus {
  New = 0,
  PartiallyFilled = 1,
  Filled = 2,
  Canceled = 3,
  PendingCancel = 4,
  Rejected = 5,
  Expired = 6
}

export enum TimeInForce {
  GTC = 0,  // Good Till Cancel
  IOC = 1,  // Immediate Or Cancel
  FOK = 2,  // Fill Or Kill
  GTX = 3   // Good Till Crossing (Post Only)
}

export interface Order {
  exchange: string;
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  type: OrderType;
  side: OrderSide;
  status: OrderStatus;
  price?: number;
  stopPrice?: number;
  quantity: number;
  filledQuantity: number;
  remainingQuantity: number;
  quoteQuantity?: number;
  timeInForce?: TimeInForce;
  createdAt: string;
  updatedAt?: string;
}

export enum TradeRole {
  Maker = 0,
  Taker = 1
}

export interface Trade {
  exchange: string;
  tradeId: string;
  orderId: string;
  symbol: string;
  side: OrderSide;
  price: number;
  quantity: number;
  quoteQuantity: number;
  commission: number;
  commissionAsset: string;
  role: TradeRole;
  executedAt: string;
}

export enum TransactionType {
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

export enum TransactionStatus {
  Pending = 0,
  Completed = 1,
  Failed = 2,
  Canceled = 3,
  Confirmed = 4
}

export interface Transaction {
  exchange: string;
  transactionId: string;
  type: TransactionType;
  status: TransactionStatus;
  asset: string;
  amount: number;
  fee?: number;
  feeAsset?: string;
  symbol?: string;
  signedFee?: number;
  orderId?: string;
  clientOrderId?: string;
  side?: string;
  tradePrice?: number;
  fromAddress?: string;
  toAddress?: string;
  txHash?: string;
  createdAt: string;
  updatedAt?: string;
}
