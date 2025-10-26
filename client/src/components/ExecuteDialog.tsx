import { useState } from 'react';
import { X, DollarSign, TrendingUp, Wallet, AlertCircle, Sparkles } from 'lucide-react';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { ExchangeBadge } from './ui/ExchangeBadge';
import { MLScoreBadge } from './ui/MLScoreBadge';
import { useArbitrageStore } from '../stores/arbitrageStore';

interface ExecuteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onExecute: (params: ExecutionParams) => void;
  isExecuting?: boolean;
  opportunity: {
    symbol: string;
    strategy: number; // 0 = SpotPerpetual, 1 = CrossExchange
    exchange?: string;
    longExchange?: string;
    shortExchange?: string;
    spotPrice?: number;
    perpetualPrice?: number;
    fundingRate?: number;
    longFundingRate?: number;
    shortFundingRate?: number;
    annualizedSpread?: number;
    estimatedProfitPercentage?: number;
    // ML predictions
    mlCompositeScore?: number;
    mlPredictedProfitPercent?: number;
    mlSuccessProbability?: number;
    mlPredictedDurationHours?: number;
    mlPredictedHoldHours?: number; // Alias for backward compatibility
    mlModelVersion?: string;
  };
}

export interface ExecutionParams {
  positionSizeUsd: number;
  leverage: number;
  stopLossPercentage?: number;
  takeProfitPercentage?: number;
}

export const ExecuteDialog = ({ isOpen, onClose, onExecute, opportunity, isExecuting = false }: ExecuteDialogProps) => {
  const [positionSize, setPositionSize] = useState<number>(10);
  const [positionSizeInput, setPositionSizeInput] = useState<string>('10');
  const [leverage, setLeverage] = useState<number>(1);
  const { balances } = useArbitrageStore();

  if (!isOpen) return null;

  const isSpotPerp = opportunity.strategy === 1; // 1 = SpotPerpetual
  const isCrossFutures = opportunity.strategy === 0; // 0 = CrossExchange

  // Calculate the effective funding rate based on strategy
  const effectiveFundingRate = isSpotPerp
    ? (opportunity.fundingRate || 0)
    : ((opportunity.longFundingRate || 0) - (opportunity.shortFundingRate || 0));

  // Calculate estimated 8h earnings: position size * funding rate
  const estimated8hEarnings = positionSize * Math.abs(effectiveFundingRate);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    onExecute({
      positionSizeUsd: positionSize,
      leverage,
      stopLossPercentage: undefined,
      takeProfitPercentage: undefined,
    });
  };

  // Get balance for the relevant exchange(s)
  const getExchangeBalance = (exchangeName: string) => {
    return balances.find((b) => b.exchange === exchangeName);
  };

  const primaryExchange = opportunity.exchange || opportunity.longExchange || '';
  const primaryBalance = getExchangeBalance(primaryExchange);

  // For cross-futures, also get short exchange balance
  const secondaryExchange = isCrossFutures ? (opportunity.shortExchange || '') : '';
  const secondaryBalance = isCrossFutures ? getExchangeBalance(secondaryExchange) : null;

  // Calculate requirements based on strategy
  const requiredSpot = isSpotPerp ? positionSize : 0;
  const requiredMargin = positionSize / leverage;

  // For spot-perp: need both spot and futures margin
  // For cross-futures: only need futures margin on both exchanges
  let canExecute = false;
  let balanceMessage = '';

  if (isSpotPerp && primaryBalance) {
    const spotSufficient = primaryBalance.spotAvailableUsd >= requiredSpot;
    const marginSufficient = primaryBalance.futuresAvailableUsd >= requiredMargin;
    canExecute = spotSufficient && marginSufficient;
    if (!canExecute) {
      balanceMessage = !spotSufficient
        ? `Insufficient spot USDT (need $${requiredSpot.toFixed(2)})`
        : `Insufficient futures margin (need $${requiredMargin.toFixed(2)})`;
    }
  } else if (isCrossFutures && primaryBalance && secondaryBalance) {
    const longSufficient = primaryBalance.futuresAvailableUsd >= requiredMargin;
    const shortSufficient = secondaryBalance.futuresAvailableUsd >= requiredMargin;
    canExecute = longSufficient && shortSufficient;
    if (!canExecute) {
      balanceMessage = !longSufficient
        ? `Insufficient ${primaryExchange} futures (need $${requiredMargin.toFixed(2)})`
        : `Insufficient ${secondaryExchange} futures (need $${requiredMargin.toFixed(2)})`;
    }
  }

  // Validate input is not empty and is a valid number
  const isValidInput = positionSizeInput !== '' && !isNaN(parseFloat(positionSizeInput)) && parseFloat(positionSizeInput) > 0;

  canExecute = canExecute && isValidInput && positionSize >= 10 && positionSize <= 10000;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <LoadingOverlay isLoading={isExecuting} message="Executing trade..." />
      <div className="bg-binance-bg-secondary border border-binance-border rounded-lg shadow-2xl w-full max-w-md m-4">
        {/* Compact Header */}
        <div className="flex items-center justify-between p-3 border-b border-binance-border">
          <div>
            <h2 className="text-base font-bold text-binance-text">Execute {opportunity.symbol}</h2>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-[10px] text-binance-text-secondary">
                {isSpotPerp ? 'Spot-Perpetual' : 'Cross-Exchange'}
              </span>
              <span className="text-binance-text-secondary">•</span>
              {isSpotPerp ? (
                <ExchangeBadge exchange={opportunity.exchange || ''} />
              ) : (
                <div className="flex items-center gap-1">
                  <ExchangeBadge exchange={opportunity.longExchange || ''} />
                  <span className="text-[10px] text-binance-text-secondary">/</span>
                  <ExchangeBadge exchange={opportunity.shortExchange || ''} />
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-binance-text-secondary hover:text-binance-text transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Compact Opportunity Info */}
        <div className="p-3 bg-binance-bg border-b border-binance-border">
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            <div>
              <p className="text-binance-text-secondary">Funding (8h)</p>
              <p className={`font-mono font-bold ${
                effectiveFundingRate >= 0 ? 'text-binance-green' : 'text-binance-red'
              }`}>
                {(effectiveFundingRate * 100).toFixed(4)}%
              </p>
            </div>
            <div>
              <p className="text-binance-text-secondary">Est. 8h Earn</p>
              <p className="font-mono font-bold text-binance-green">
                ${estimated8hEarnings.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-binance-text-secondary">Est. APR</p>
              <p className="font-mono font-bold text-binance-green">
                {opportunity.estimatedProfitPercentage?.toFixed(2) || '-'}%
              </p>
            </div>
          </div>
        </div>

        {/* ML Predictions */}
        {opportunity.mlCompositeScore !== undefined && (
          <div className="p-3 bg-binance-bg-secondary/50 border-b border-binance-border">
            <div className="flex items-center gap-1 mb-2">
              <Sparkles className="w-3 h-3 text-purple-400" />
              <h3 className="text-[10px] font-semibold text-binance-text">AI Predictions</h3>
              {opportunity.mlModelVersion && (
                <Badge variant="info" size="sm" className="ml-auto text-[8px] px-1 py-0">
                  {opportunity.mlModelVersion}
                </Badge>
              )}
            </div>
            <div className="flex justify-center">
              <MLScoreBadge
                score={opportunity.mlCompositeScore}
                profitPrediction={opportunity.mlPredictedProfitPercent}
                successProbability={opportunity.mlSuccessProbability}
                holdDuration={opportunity.mlPredictedDurationHours || opportunity.mlPredictedHoldHours}
                modelVersion={opportunity.mlModelVersion}
                size="md"
                showDetails
              />
            </div>
          </div>
        )}

        {/* Balance Display */}
        {balances.length > 0 && (
          <div className="p-3 bg-binance-bg-secondary border-b border-binance-border">
            <div className="flex items-center gap-1 mb-2">
              <Wallet className="w-3 h-3 text-binance-yellow" />
              <h3 className="text-[10px] font-semibold text-binance-text">Available Balance</h3>
            </div>

            {/* Spot-Perpetual: Show Spot USDT and Futures Margin */}
            {isSpotPerp && primaryBalance && (
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <ExchangeBadge exchange={primaryExchange} size="small" />
                    <span className="text-[8px] text-binance-text-secondary">Spot</span>
                  </div>
                  <p className={`font-mono font-bold ${primaryBalance.spotAvailableUsd >= requiredSpot ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${primaryBalance.spotAvailableUsd.toFixed(2)}
                  </p>
                </div>
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <ExchangeBadge exchange={primaryExchange} size="small" />
                    <span className="text-[8px] text-binance-text-secondary">Futures</span>
                  </div>
                  <p className={`font-mono font-bold ${primaryBalance.futuresAvailableUsd >= requiredMargin ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${primaryBalance.futuresAvailableUsd.toFixed(2)}
                  </p>
                </div>
              </div>
            )}

            {/* Cross-Futures: Show both exchanges' futures balances */}
            {isCrossFutures && primaryBalance && secondaryBalance && (
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <ExchangeBadge exchange={primaryExchange} size="small" />
                    <span className="text-[8px] text-binance-text-secondary">Futures</span>
                  </div>
                  <p className={`font-mono font-bold ${primaryBalance.futuresAvailableUsd >= requiredMargin ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${primaryBalance.futuresAvailableUsd.toFixed(2)}
                  </p>
                </div>
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <ExchangeBadge exchange={secondaryExchange} size="small" />
                    <span className="text-[8px] text-binance-text-secondary">Futures</span>
                  </div>
                  <p className={`font-mono font-bold ${secondaryBalance.futuresAvailableUsd >= requiredMargin ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${secondaryBalance.futuresAvailableUsd.toFixed(2)}
                  </p>
                </div>
              </div>
            )}

            {balanceMessage && (
              <div className="mt-2 flex items-start gap-1 text-binance-red">
                <AlertCircle className="w-3 h-3 mt-0.5" />
                <p className="text-[10px]">{balanceMessage}</p>
              </div>
            )}
          </div>
        )}

        {/* Compact Execution Form */}
        <form onSubmit={handleSubmit} className="p-3 space-y-3">
          {/* Compact Position Size */}
          <div>
            <label className="flex items-center gap-1 text-[11px] font-semibold text-binance-text mb-1">
              <DollarSign className="w-3 h-3 text-binance-yellow" />
              Position Size (USD)
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={positionSizeInput}
              onChange={(e) => {
                const value = e.target.value;
                // Allow only numbers and decimal point
                if (value === '' || /^\d*\.?\d*$/.test(value)) {
                  setPositionSizeInput(value);
                  const parsed = parseFloat(value);
                  if (!isNaN(parsed) && parsed > 0) {
                    setPositionSize(parsed);
                  } else if (value === '') {
                    setPositionSize(10); // Default when empty
                  }
                }
              }}
              onBlur={() => {
                // Ensure valid number on blur
                const parsed = parseFloat(positionSizeInput);
                if (isNaN(parsed) || parsed < 10) {
                  setPositionSize(10);
                  setPositionSizeInput('10');
                } else if (parsed > 10000) {
                  setPositionSize(10000);
                  setPositionSizeInput('10000');
                } else {
                  setPositionSize(parsed);
                  setPositionSizeInput(parsed.toString());
                }
              }}
              className="w-full px-2 py-1.5 bg-binance-bg border border-binance-border rounded text-binance-text font-mono text-sm focus:outline-none focus:ring-1 focus:ring-binance-yellow disabled:opacity-50"
              placeholder="10"
              required
              disabled={isExecuting}
            />
            <p className="text-[10px] text-binance-text-secondary mt-0.5">
              Min: $10 • Max: $10,000
            </p>
          </div>

          {/* Compact Leverage */}
          <div>
            <label className="flex items-center justify-between text-[11px] font-semibold text-binance-text mb-1">
              <div className="flex items-center gap-1">
                <TrendingUp className="w-3 h-3 text-binance-yellow" />
                Leverage
              </div>
              <span className="text-base font-bold text-binance-yellow">
                {leverage}x
              </span>
            </label>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={leverage}
              onChange={(e) => setLeverage(parseInt(e.target.value))}
              className="w-full h-1"
              disabled={isExecuting}
            />
            <div className="flex justify-between text-[10px] text-binance-text-secondary mt-0.5">
              <span>1x</span>
              <span>Margin: ${requiredMargin.toFixed(2)}</span>
              <span>5x</span>
            </div>
          </div>

          {/* Compact Summary */}
          <div className="bg-binance-bg border border-binance-border rounded p-2 space-y-1 text-[10px]">
            {isSpotPerp ? (
              <>
                <div className="flex justify-between">
                  <span className="text-binance-text-secondary">Spot Purchase:</span>
                  <span className="font-mono font-bold text-binance-text">${requiredSpot.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-binance-text-secondary">Perp Margin:</span>
                  <span className="font-mono font-bold text-binance-text">${requiredMargin.toFixed(2)}</span>
                </div>
                <div className="flex justify-between border-t border-binance-border pt-1">
                  <span className="text-binance-text">Total Required:</span>
                  <span className="font-mono font-bold text-binance-text">${(requiredSpot + requiredMargin).toFixed(2)}</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex justify-between">
                  <span className="text-binance-text-secondary">{primaryExchange} Margin:</span>
                  <span className="font-mono font-bold text-binance-text">${requiredMargin.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-binance-text-secondary">{secondaryExchange} Margin:</span>
                  <span className="font-mono font-bold text-binance-text">${requiredMargin.toFixed(2)}</span>
                </div>
                <div className="flex justify-between border-t border-binance-border pt-1">
                  <span className="text-binance-text">Total Margin:</span>
                  <span className="font-mono font-bold text-binance-text">${(requiredMargin * 2).toFixed(2)}</span>
                </div>
              </>
            )}
          </div>

          {/* Compact Actions */}
          <div className="flex gap-2 pt-1">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={onClose}
              className="flex-1 h-8 text-[11px]"
              disabled={isExecuting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="sm"
              className="flex-1 h-8 text-[11px]"
              disabled={isExecuting || !canExecute}
              title={!canExecute && !isExecuting ? balanceMessage || 'Invalid parameters' : ''}
            >
              {isExecuting ? 'Executing...' : 'Execute'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
