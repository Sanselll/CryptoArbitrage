import { useState, useEffect } from 'react';
import { X, DollarSign, TrendingUp, Wallet, AlertCircle } from 'lucide-react';
import { Button } from './ui/Button';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { apiService, type ExecutionBalances } from '../services/apiService';

interface ExecuteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onExecute: (params: ExecutionParams) => void;
  isExecuting?: boolean;
  opportunity: {
    symbol: string;
    strategy: number; // 1 = SpotPerpetual, 2 = CrossExchange
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
  const [balances, setBalances] = useState<ExecutionBalances | null>(null);
  const [loadingBalances, setLoadingBalances] = useState(false);
  const [balanceError, setBalanceError] = useState<string | null>(null);

  // Fetch balances when dialog opens
  useEffect(() => {
    // Determine the exchange to use (SpotPerp uses 'exchange', CrossExchange uses 'longExchange')
    const exchangeName = opportunity.exchange || opportunity.longExchange;

    if (isOpen && exchangeName) {
      const fetchBalances = async () => {
        setLoadingBalances(true);
        setBalanceError(null);
        try {
          const data = await apiService.getExecutionBalances(exchangeName, leverage);
          setBalances(data);
        } catch (error: any) {
          setBalanceError(error.message || 'Failed to fetch balances');
        } finally {
          setLoadingBalances(false);
        }
      };
      fetchBalances();
    }
  }, [isOpen, opportunity.exchange, opportunity.longExchange, leverage]);

  if (!isOpen) return null;

  const isSpotPerp = opportunity.strategy === 1;

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

  // Calculate requirements
  const requiredSpot = positionSize;
  const requiredMargin = positionSize / leverage;
  const totalRequired = balances?.isUnifiedAccount
    ? requiredSpot + requiredMargin
    : requiredSpot; // For Binance, spot and margin are separate

  // Check if we have sufficient balance
  const spotSufficient = balances ? balances.spotUsdtAvailable >= requiredSpot : false;
  const marginSufficient = balances ? balances.futuresAvailable >= requiredMargin : false;
  const totalSufficient = balances?.isUnifiedAccount
    ? balances.totalAvailable >= totalRequired
    : spotSufficient && marginSufficient;

  // Validate input is not empty and is a valid number
  const isValidInput = positionSizeInput !== '' && !isNaN(parseFloat(positionSizeInput)) && parseFloat(positionSizeInput) > 0;

  const canExecute = !loadingBalances && !balanceError && totalSufficient && isValidInput && positionSize >= 10 && positionSize <= 10000;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <LoadingOverlay isLoading={isExecuting} message="Executing trade..." />
      <div className="bg-binance-bg-secondary border border-binance-border rounded-lg shadow-2xl w-full max-w-md m-4">
        {/* Compact Header */}
        <div className="flex items-center justify-between p-3 border-b border-binance-border">
          <div>
            <h2 className="text-base font-bold text-binance-text">Execute {opportunity.symbol}</h2>
            <p className="text-[10px] text-binance-text-secondary">
              {isSpotPerp ? 'Spot-Perpetual' : 'Cross-Exchange'} •
              {isSpotPerp ? opportunity.exchange : `${opportunity.longExchange} / ${opportunity.shortExchange}`}
            </p>
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

        {/* Balance Display */}
        {loadingBalances ? (
          <div className="p-3 bg-binance-bg-secondary border-b border-binance-border">
            <p className="text-[10px] text-binance-text-secondary">Loading balances...</p>
          </div>
        ) : balanceError ? (
          <div className="p-3 bg-binance-bg-secondary border-b border-binance-border">
            <div className="flex items-center gap-1 text-binance-red">
              <AlertCircle className="w-3 h-3" />
              <p className="text-[10px]">{balanceError}</p>
            </div>
          </div>
        ) : balances ? (
          <div className="p-3 bg-binance-bg-secondary border-b border-binance-border">
            <div className="flex items-center gap-1 mb-2">
              <Wallet className="w-3 h-3 text-binance-yellow" />
              <h3 className="text-[10px] font-semibold text-binance-text">Available Balance</h3>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              {balances.isUnifiedAccount ? (
                <>
                  <div>
                    <p className="text-binance-text-secondary">Total Available</p>
                    <p className={`font-mono font-bold ${totalSufficient ? 'text-binance-green' : 'text-binance-red'}`}>
                      ${balances.totalAvailable.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-binance-text-secondary">Required</p>
                    <p className="font-mono font-bold text-binance-text">
                      ${totalRequired.toFixed(2)}
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <p className="text-binance-text-secondary">Spot USDT</p>
                    <p className={`font-mono font-bold ${spotSufficient ? 'text-binance-green' : 'text-binance-red'}`}>
                      ${balances.spotUsdtAvailable.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-binance-text-secondary">Futures Margin</p>
                    <p className={`font-mono font-bold ${marginSufficient ? 'text-binance-green' : 'text-binance-red'}`}>
                      ${balances.futuresAvailable.toFixed(2)}
                    </p>
                  </div>
                </>
              )}
            </div>
            {!totalSufficient && (
              <div className="mt-2 flex items-start gap-1 text-binance-red">
                <AlertCircle className="w-3 h-3 mt-0.5" />
                <p className="text-[10px]">Insufficient balance for this position size</p>
              </div>
            )}
          </div>
        ) : null}

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
              Min: $10 • Max: ${balances?.maxPositionSize.toFixed(0) || '10,000'}
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
              <span className="font-mono font-bold text-binance-text">${totalRequired.toFixed(2)}</span>
            </div>
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
              title={!canExecute && !isExecuting ? 'Insufficient balance or invalid parameters' : ''}
            >
              {isExecuting ? 'Executing...' : 'Execute'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
