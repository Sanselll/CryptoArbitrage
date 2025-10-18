import { TrendingUp, Wallet, Lock } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { ExchangeBadge } from './ui/ExchangeBadge';

export const BalanceWidget = () => {
  const { balances } = useArbitrageStore();

  const getMarginRiskLevel = (utilization: number) => {
    if (utilization > 75) return { label: 'High Risk', variant: 'danger' as const };
    if (utilization > 50) return { label: 'Medium', variant: 'warning' as const };
    return { label: 'Low Risk', variant: 'success' as const };
  };

  return (
    <div className="flex gap-2">
      {balances.map((balance) => {
        // Total capital = all assets (spot + futures)
        const totalCapital = balance.totalBalance || 0;
        const totalUnrealizedPnL = balance.unrealizedPnL;

        const spotBalanceUsd = balance.spotBalanceUsd || 0;
        const spotAvailableUsd = balance.spotAvailableUsd || 0;
        const futuresBalance = balance.futuresBalanceUsd || 0;
        const futuresAvailable = balance.futuresAvailableUsd || 0;
        const totalMarginUsed = balance.marginUsed;

        // Available Capital = USDT available to deploy in new trades
        // Binance: Spot Available USDT + Futures Available USDT
        // Bybit: Available USDT (AvailableToWithdraw from unified account)
        const availableCapital = balance.exchange === 'Binance'
          ? spotAvailableUsd + futuresAvailable
          : futuresAvailable;

        // In Positions = coins locked in spot positions + margin used in futures
        // spotBalanceUsd - spotAvailableUsd = coins locked in spot
        const coinsInSpotPositions = spotBalanceUsd - spotAvailableUsd;
        const inPositions = coinsInSpotPositions + totalMarginUsed;

        // Margin % (only when positions exist)
        const marginUtilization = futuresBalance > 0 ? (totalMarginUsed / futuresBalance) * 100 : 0;
        const hasPositions = totalMarginUsed > 0 || coinsInSpotPositions > 0;
        const riskLevel = getMarginRiskLevel(marginUtilization);

        const pnlPercent = totalCapital > 0 ? (totalUnrealizedPnL / totalCapital) * 100 : 0;

        return (
          <Card key={balance.exchange} className="card-hover w-[240px]">
            <CardContent className="p-2">
              {/* Exchange Header with P&L */}
              <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-binance-border/30">
                <ExchangeBadge exchange={balance.exchange} />
                <div className="flex items-center gap-1.5">
                  <TrendingUp
                    className={`w-2.5 h-2.5 ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  />
                  <div
                    className={`text-xs font-bold font-mono ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  >
                    {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
                  </div>
                  <div
                    className={`text-[9px] font-medium ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  >
                    ({totalUnrealizedPnL >= 0 ? '+' : ''}{pnlPercent.toFixed(1)}%)
                  </div>
                </div>
              </div>

              {/* Capital Total */}
              <div className="mb-1.5 pb-1 border-b border-binance-border/30">
                <div className="text-[9px] text-binance-text-secondary font-bold mb-0.5">Capital</div>
                <div className="text-base font-bold text-binance-text font-mono">
                  ${totalCapital.toFixed(0)}
                </div>
              </div>

              {/* Column Headers */}
              <div className="grid grid-cols-2 gap-2 mb-1">
                <div className="text-[9px] text-binance-text-secondary font-bold">Available Capital</div>
                <div className="text-[9px] text-binance-text-secondary font-bold">In Positions</div>
              </div>

              {/* Grid Layout with 2 columns */}
              <div className="grid grid-cols-2 gap-2">
                {/* Column 1: Available Capital */}
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Wallet className="w-2.5 h-2.5 text-binance-text-secondary" />
                    <span className="text-sm font-bold text-binance-green font-mono">
                      ${availableCapital.toFixed(0)}
                    </span>
                  </div>
                  <div className="text-[9px] text-binance-text-muted">
                    Free to trade
                  </div>
                </div>

                {/* Column 2: In Positions */}
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Lock className="w-2.5 h-2.5 text-binance-text-secondary" />
                    <span className="text-sm font-bold text-binance-text font-mono">
                      ${inPositions.toFixed(0)}
                    </span>
                  </div>
                  {hasPositions ? (
                    <div className="flex items-center gap-1">
                      <span className="text-[9px] text-binance-text-secondary">Margin</span>
                      <Badge variant={riskLevel.variant} size="sm" className="text-[8px] px-1 py-0">
                        {marginUtilization.toFixed(0)}%
                      </Badge>
                    </div>
                  ) : (
                    <div className="text-[9px] text-binance-text-muted">No positions</div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};
