export interface FundingRate {
  exchange: string;
  symbol: string;
  rate: number;
  annualizedRate: number;
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
  openedAt: string;
  closedAt?: string;
  activeOpportunityId?: number;
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
  CrossExchangeSpotFutures = 2
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

  // Spot-perpetual fields
  exchange?: string;
  spotPrice?: number;
  perpetualPrice?: number;
  fundingRate?: number;
  annualizedFundingRate?: number;
  pricePremium?: number;

  // Common fields
  spreadRate: number;
  annualizedSpread: number;
  estimatedProfitPercentage: number;
  volume24h?: number;

  // Liquidity metrics
  bidAskSpreadPercent?: number;
  orderbookDepthUsd?: number;
  liquidityStatus?: LiquidityStatus;
  liquidityWarning?: string;

  status: OpportunityStatus;
  detectedAt: string;
  executedAt?: string;
  activeOpportunityExecutedAt?: string;

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
