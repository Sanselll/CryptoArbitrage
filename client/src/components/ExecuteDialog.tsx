import { useState } from 'react';
import { X, DollarSign, TrendingUp, Shield, Target } from 'lucide-react';
import { Button } from './ui/Button';
import { LoadingOverlay } from './ui/LoadingOverlay';

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
  const [positionSize, setPositionSize] = useState<number>(1000);
  const [leverage, setLeverage] = useState<number>(1);
  const [stopLoss, setStopLoss] = useState<string>('');
  const [takeProfit, setTakeProfit] = useState<string>('');

  if (!isOpen) return null;

  const isSpotPerp = opportunity.strategy === 1;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    onExecute({
      positionSizeUsd: positionSize,
      leverage,
      stopLossPercentage: stopLoss ? parseFloat(stopLoss) : undefined,
      takeProfitPercentage: takeProfit ? parseFloat(takeProfit) : undefined,
    });
  };

  const estimatedMargin = positionSize / leverage;
  const totalPosition = positionSize * (isSpotPerp ? 2 : 2); // Both legs

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <LoadingOverlay isLoading={isExecuting} message="Executing trade..." />
      <div className="bg-binance-bg-secondary border border-binance-border rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto m-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-binance-border">
          <div>
            <h2 className="text-2xl font-bold text-binance-text">Execute Arbitrage</h2>
            <p className="text-sm text-binance-text-secondary mt-1">
              {opportunity.symbol} - {isSpotPerp ? 'Spot-Perpetual' : 'Cross-Exchange'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-binance-text-secondary hover:text-binance-text transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Opportunity Summary */}
        <div className="p-6 bg-binance-bg border-b border-binance-border">
          <h3 className="text-sm font-semibold text-binance-text-secondary mb-3">Opportunity Details</h3>
          <div className="grid grid-cols-2 gap-4">
            {isSpotPerp ? (
              <>
                <div>
                  <p className="text-xs text-binance-text-secondary">Spot Price</p>
                  <p className="text-lg font-mono font-bold text-binance-text">
                    ${opportunity.spotPrice?.toFixed(6) || '-'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-binance-text-secondary">Perp Price</p>
                  <p className="text-lg font-mono font-bold text-binance-text">
                    ${opportunity.perpetualPrice?.toFixed(6) || '-'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-binance-text-secondary">Funding Rate (8h)</p>
                  <p className={`text-lg font-mono font-bold ${
                    (opportunity.fundingRate || 0) >= 0 ? 'text-binance-green' : 'text-binance-red'
                  }`}>
                    {((opportunity.fundingRate || 0) * 100).toFixed(4)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-binance-text-secondary">Estimated APR</p>
                  <p className="text-lg font-mono font-bold text-binance-green">
                    {opportunity.estimatedProfitPercentage?.toFixed(2) || '-'}%
                  </p>
                </div>
              </>
            ) : (
              <>
                <div>
                  <p className="text-xs text-binance-text-secondary">Long Exchange</p>
                  <p className="text-lg font-semibold text-binance-text">{opportunity.longExchange}</p>
                </div>
                <div>
                  <p className="text-xs text-binance-text-secondary">Short Exchange</p>
                  <p className="text-lg font-semibold text-binance-text">{opportunity.shortExchange}</p>
                </div>
                <div className="col-span-2">
                  <p className="text-xs text-binance-text-secondary">Estimated APR</p>
                  <p className="text-lg font-mono font-bold text-binance-green">
                    {((opportunity.annualizedSpread || 0) * 100).toFixed(2)}%
                  </p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Execution Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Position Size */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-binance-text mb-2">
              <DollarSign className="w-4 h-4 text-binance-yellow" />
              Position Size (USD per side)
            </label>
            <input
              type="number"
              min="100"
              max="10000"
              step="100"
              value={positionSize}
              onChange={(e) => setPositionSize(parseFloat(e.target.value))}
              className="w-full px-4 py-3 bg-binance-bg border border-binance-border rounded-md text-binance-text font-mono text-lg focus:outline-none focus:ring-2 focus:ring-binance-yellow disabled:opacity-50 disabled:cursor-not-allowed"
              required
              disabled={isExecuting}
            />
            <p className="text-xs text-binance-text-secondary mt-1">
              Min: $100 | Max: $10,000
            </p>
          </div>

          {/* Leverage */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-binance-text mb-2">
              <TrendingUp className="w-4 h-4 text-binance-yellow" />
              Leverage
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={leverage}
                onChange={(e) => setLeverage(parseInt(e.target.value))}
                className="flex-1"
                disabled={isExecuting}
              />
              <span className="text-2xl font-bold text-binance-yellow w-16 text-right">
                {leverage}x
              </span>
            </div>
            <p className="text-xs text-binance-text-secondary mt-1">
              Higher leverage increases risk and potential liquidation
            </p>
          </div>

          {/* Stop Loss (Optional) */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-binance-text mb-2">
              <Shield className="w-4 h-4 text-binance-red" />
              Stop Loss (%)
              <span className="text-binance-text-secondary font-normal">(Optional)</span>
            </label>
            <input
              type="number"
              min="0.1"
              max="50"
              step="0.1"
              value={stopLoss}
              onChange={(e) => setStopLoss(e.target.value)}
              placeholder="e.g., 2.0"
              className="w-full px-4 py-3 bg-binance-bg border border-binance-border rounded-md text-binance-text font-mono focus:outline-none focus:ring-2 focus:ring-binance-yellow disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isExecuting}
            />
            <p className="text-xs text-binance-text-secondary mt-1">
              Close position if loss exceeds this percentage
            </p>
          </div>

          {/* Take Profit (Optional) */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-binance-text mb-2">
              <Target className="w-4 h-4 text-binance-green" />
              Take Profit (%)
              <span className="text-binance-text-secondary font-normal">(Optional)</span>
            </label>
            <input
              type="number"
              min="0.1"
              max="100"
              step="0.1"
              value={takeProfit}
              onChange={(e) => setTakeProfit(e.target.value)}
              placeholder="e.g., 5.0"
              className="w-full px-4 py-3 bg-binance-bg border border-binance-border rounded-md text-binance-text font-mono focus:outline-none focus:ring-2 focus:ring-binance-yellow disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isExecuting}
            />
            <p className="text-xs text-binance-text-secondary mt-1">
              Close position when profit reaches this percentage
            </p>
          </div>

          {/* Summary */}
          <div className="bg-binance-bg border border-binance-border rounded-lg p-4 space-y-2">
            <h4 className="font-semibold text-binance-text mb-2">Execution Summary</h4>
            <div className="flex justify-between text-sm">
              <span className="text-binance-text-secondary">Required Margin:</span>
              <span className="font-mono font-bold text-binance-text">${estimatedMargin.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-binance-text-secondary">Total Position (both sides):</span>
              <span className="font-mono font-bold text-binance-text">${totalPosition.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-binance-text-secondary">Number of Positions:</span>
              <span className="font-mono font-bold text-binance-text">2</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <Button
              type="button"
              variant="secondary"
              size="lg"
              onClick={onClose}
              className="flex-1"
              disabled={isExecuting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="flex-1"
              disabled={isExecuting}
            >
              {isExecuting ? 'Executing...' : 'Execute Trade'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
