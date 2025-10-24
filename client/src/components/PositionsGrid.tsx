import { Layers, StopCircle, TrendingUp, TrendingDown, ArrowUpDown } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { PositionSide, PositionStatus, PositionType, StrategySubType, OpportunitySuggestion, RecommendedStrategyType } from '../types/index';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';
import { EmptyState } from './ui/EmptyState';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { ExchangeBadge } from './ui/ExchangeBadge';
import { SuggestionBadge } from './ui/SuggestionBadge';
import { AlertDialog, ConfirmDialog } from './ui/Dialog';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';
import { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { useDialog } from '../hooks/useDialog';

// Helper function to get strategy type label from position types
const getStrategyLabelFromPositions = (
  perp: any | null,
  spot: any | null,
  longPerp: any | null,
  shortPerp: any | null
): { text: string; color: string } => {
  // If we have two perp positions (Cross-Fut)
  if (longPerp && shortPerp) {
    return { text: 'Cross-Fut', color: 'bg-purple-500/20 text-purple-400' };
  }
  // If we have both spot and perp on same exchange, it's Spot-Perp
  if (spot && perp && spot.exchange === perp.exchange) {
    return { text: 'Spot-Perp', color: 'bg-blue-500/20 text-blue-400' };
  }
  // If we have spot on one exchange and perp on another, it's Cross-Spot
  if (spot && perp && spot.exchange !== perp.exchange) {
    return { text: 'Cross-Spot', color: 'bg-green-500/20 text-green-400' };
  }
  return { text: 'Unknown', color: 'bg-gray-500/20 text-gray-400' };
};

interface PositionPair {
  executionId: number | undefined;
  symbol: string;
  exchange: string;
  perp: any | null;
  spot: any | null;
  // For Cross-Fut strategies (two perpetual positions on different exchanges)
  longPerp: any | null;
  shortPerp: any | null;
}

// Helper function to format execution time (time elapsed since position opened)
const formatExecutionTime = (openedAt: string) => {
  const now = new Date();
  const dateString = openedAt.endsWith('Z') ? openedAt : `${openedAt}Z`;
  const opened = new Date(dateString);
  const diff = now.getTime() - opened.getTime();

  if (diff < 0) return '0h 0m 0s';

  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);
  return `${hours}h ${minutes}m ${seconds}s`;
};

type SortField = 'ai' | 'symbol' | 'pnl' | 'time';
type SortDirection = 'asc' | 'desc';

export const PositionsGrid = () => {
  const { positions, opportunities, fundingRates } = useArbitrageStore();
  const [isClosing, setIsClosing] = useState(false);
  const [executionTimes, setExecutionTimes] = useState<{ [key: number]: string }>({});
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('time');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const { alertState, showSuccess, showError, closeAlert, confirmState, showConfirm, closeConfirm } = useDialog();

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const newExecutionTimes: { [key: number]: string } = {};

      openPositions.forEach((position) => {
        if (position.executionId && position.openedAt) {
          newExecutionTimes[position.executionId] = formatExecutionTime(position.openedAt);
        }
      });
      setExecutionTimes(newExecutionTimes);
    }, 1000);

    return () => clearInterval(interval);
  }, [positions]);

  const openPositions = positions.filter((p) => p.status === PositionStatus.Open);

  // Group positions by executionId (two positions per execution: perp + spot OR longPerp + shortPerp)
  const positionPairs: PositionPair[] = [];
  const grouped = new Map<number | undefined, {
    perp: any | null;
    spot: any | null;
    longPerp: any | null;
    shortPerp: any | null;
    symbol: string;
    exchange: string;
  }>();

  openPositions.forEach((position) => {
    const key = position.executionId;
    if (!grouped.has(key)) {
      grouped.set(key, {
        perp: null,
        spot: null,
        longPerp: null,
        shortPerp: null,
        symbol: position.symbol,
        exchange: position.exchange
      });
    }
    const pair = grouped.get(key)!;

    if (position.type === PositionType.Perpetual) {
      // Check if we already have a perp position - if so, this is Cross-Fut
      if (pair.perp !== null) {
        // We have two perpetual positions - determine which is long and which is short
        if (position.side === PositionSide.Long) {
          pair.longPerp = position;
          pair.shortPerp = pair.perp; // The first one must be short
        } else {
          pair.shortPerp = position;
          pair.longPerp = pair.perp; // The first one must be long
        }
        pair.perp = null; // Clear the single perp since we're using longPerp/shortPerp
      } else if (pair.longPerp === null && pair.shortPerp === null) {
        // First perp position we've seen for this execution
        pair.perp = position;
      }
    } else if (position.type === PositionType.Spot) {
      pair.spot = position;
    }
  });

  grouped.forEach((pair, executionId) => {
    positionPairs.push({
      executionId,
      symbol: pair.symbol,
      exchange: pair.exchange,
      perp: pair.perp,
      spot: pair.spot,
      longPerp: pair.longPerp,
      shortPerp: pair.shortPerp
    });
  });

  // Sort position pairs
  positionPairs.sort((a, b) => {
    let aValue: number, bValue: number;

    switch (sortField) {
      case 'ai':
        aValue = a.perp?.entryConfidenceScore || a.spot?.entryConfidenceScore || -Infinity;
        bValue = b.perp?.entryConfidenceScore || b.spot?.entryConfidenceScore || -Infinity;
        break;
      case 'symbol':
        return sortDirection === 'asc'
          ? a.symbol.localeCompare(b.symbol)
          : b.symbol.localeCompare(a.symbol);
      case 'pnl':
        const aTotalPnL = (a.perp?.unrealizedPnL || 0) + (a.spot?.unrealizedPnL || 0) +
                          (a.longPerp?.unrealizedPnL || 0) + (a.shortPerp?.unrealizedPnL || 0);
        const bTotalPnL = (b.perp?.unrealizedPnL || 0) + (b.spot?.unrealizedPnL || 0) +
                          (b.longPerp?.unrealizedPnL || 0) + (b.shortPerp?.unrealizedPnL || 0);
        aValue = aTotalPnL;
        bValue = bTotalPnL;
        break;
      case 'time':
        const aTime = a.perp?.openedAt || a.spot?.openedAt || a.longPerp?.openedAt || a.shortPerp?.openedAt || '';
        const bTime = b.perp?.openedAt || b.spot?.openedAt || b.longPerp?.openedAt || b.shortPerp?.openedAt || '';
        return sortDirection === 'desc'
          ? new Date(bTime).getTime() - new Date(aTime).getTime()
          : new Date(aTime).getTime() - new Date(bTime).getTime();
      default:
        aValue = 0;
        bValue = 0;
    }

    return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
  });

  const totalPnL = openPositions.reduce((sum, p) => sum + p.unrealizedPnL, 0);

  const handleStop = async (executionId: number) => {
    showConfirm(
      `Are you sure you want to stop this execution?\n\nThis will close both spot and perpetual positions and realize the P&L.`,
      async () => {
        setIsClosing(true);
        try {
          const response = await apiService.stopExecution(executionId);

          if (response.success) {
            // Manually refresh positions immediately after successful stop
            try {
              const freshPositions = await apiService.getPositions();
              useArbitrageStore.getState().setPositions(freshPositions);
            } catch (refreshError) {
              console.error('Failed to refresh positions after stop:', refreshError);
              // Don't fail the whole operation if refresh fails
            }

            showSuccess(
              `${response.message}\n\nExecution stopped.`,
              'Success'
            );
          } else {
            showError(
              `${response.errorMessage}`,
              'Failed to Stop Execution'
            );
          }
        } catch (error: any) {
          console.error('Error stopping execution:', error);
          showError(
            `${error.message || 'Unknown error'}`,
            'Failed to Stop Execution'
          );
        } finally {
          setIsClosing(false);
        }
      },
      {
        title: 'Stop Execution',
        confirmText: 'Stop',
        cancelText: 'Cancel',
        variant: 'danger',
      }
    );
  };

  return (
    <>
      <LoadingOverlay
        isLoading={isClosing}
        message="Closing position..."
      />

      <div className="h-full flex flex-col overflow-x-auto overflow-y-auto">
        {openPositions.length === 0 ? (
          <EmptyState
            icon={<Layers className="w-12 h-12" />}
            title="No active executions"
            description="Execute arbitrage opportunities to start trading"
          />
        ) : (
          <Table>
            <TableHeader className="sticky top-0 z-30">
              <TableRow hover={false}>
                <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border">Symbol</TableHead>
                <TableHead className="py-1">Strategy</TableHead>
                <TableHead className="py-1">Exchange</TableHead>
                <TableHead className="py-1">Type</TableHead>
                <TableHead className="py-1">Side</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead className="text-right">Lev</TableHead>
                <TableHead className="text-right">Position P&L</TableHead>
                <TableHead className="text-right">Est. Fund</TableHead>
                <TableHead className="text-right">P&L</TableHead>
                <TableHead className="text-right">Fund P&L</TableHead>
                <TableHead className="text-right">Total P&L</TableHead>
                <TableHead className="text-right">Time</TableHead>
                <TableHead
                  className="sticky right-[180px] z-40 bg-binance-bg-secondary border-l border-binance-border text-center cursor-pointer hover:bg-binance-bg-tertiary"
                  title="AI confidence score at entry"
                  onClick={() => handleSort('ai')}
                >
                  <div className="flex items-center justify-center gap-1">
                    Entry AI
                    {sortField === 'ai' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="sticky right-[100px] z-40 bg-binance-bg-secondary border-l border-binance-border text-center"
                  title="Exit signals and recommendations"
                >
                  Exit AI
                </TableHead>
                <TableHead className="sticky right-0 z-40 bg-binance-bg-secondary border-l border-binance-border text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positionPairs.map((pair, pairIndex) => {
                // Render two rows for each pair
                const perpPosition = pair.perp;
                const spotPosition = pair.spot;
                const longPerpPosition = pair.longPerp;
                const shortPerpPosition = pair.shortPerp;
                const strategyLabel = getStrategyLabelFromPositions(perpPosition, spotPosition, longPerpPosition, shortPerpPosition);
                const rows = [];

                // For Cross-Fut, use longPerp and shortPerp; otherwise use perp and spot
                const isCrossFut = longPerpPosition && shortPerpPosition;

                // Calculate estimated funding for next settlement FIRST (needed for total P&L)
                // For Cross-Fut, we need to calculate funding for both positions
                let estimatedFunding = 0;
                if (isCrossFut) {
                  // For Cross-Fut, get funding rates from both exchanges
                  const longFundingRate = fundingRates.find(fr =>
                    fr.symbol === pair.symbol && fr.exchange === longPerpPosition.exchange
                  );
                  const shortFundingRate = fundingRates.find(fr =>
                    fr.symbol === pair.symbol && fr.exchange === shortPerpPosition.exchange
                  );
                  const longRate = longFundingRate ? longFundingRate.rate : 0;
                  const shortRate = shortFundingRate ? shortFundingRate.rate : 0;
                  const longValue = longPerpPosition.quantity * longPerpPosition.entryPrice;
                  const shortValue = shortPerpPosition.quantity * shortPerpPosition.entryPrice;
                  // Long position: negative rate = receive (positive), positive rate = pay (negative)
                  // Short position: negative rate = pay (negative), positive rate = receive (positive)
                  estimatedFunding = -longRate * longValue + shortRate * shortValue;
                } else {
                  // For Spot-Perp or Cross-Spot
                  const currentFundingRate = fundingRates.find(fr =>
                    fr.symbol === pair.symbol && fr.exchange === pair.exchange
                  );
                  const fundingRate = currentFundingRate ? currentFundingRate.rate : 0;
                  const perpPositionValue = perpPosition ? perpPosition.quantity * perpPosition.entryPrice : 0;
                  const perpSide = perpPosition ? perpPosition.side : PositionSide.Long;
                  // Long: negative rate = receive (positive), positive rate = pay (negative)
                  // Short: negative rate = pay (negative), positive rate = receive (positive)
                  estimatedFunding = fundingRate * perpPositionValue * (perpSide === PositionSide.Long ? -1 : 1);
                }

                // Calculate combined P&L for the pair
                let combinedUnrealizedPnL = 0;
                let combinedFunding = 0;

                if (isCrossFut) {
                  combinedUnrealizedPnL = (longPerpPosition?.unrealizedPnL || 0) + (shortPerpPosition?.unrealizedPnL || 0);
                  combinedFunding = (longPerpPosition?.netFundingFee || 0) + (shortPerpPosition?.netFundingFee || 0);
                } else {
                  combinedUnrealizedPnL = (perpPosition?.unrealizedPnL || 0) + (spotPosition?.unrealizedPnL || 0);
                  combinedFunding = (perpPosition?.netFundingFee || 0) + (spotPosition?.netFundingFee || 0);
                }

                // Total P&L = unrealized P&L from price changes + historical funding already paid/received + estimated funding for next settlement
                const totalPairPnL = combinedUnrealizedPnL + combinedFunding + estimatedFunding;

                // For Cross-Fut: render both long and short perp rows
                if (isCrossFut && longPerpPosition && shortPerpPosition) {
                  // Long Perp row
                  const longPnlPercent =
                    longPerpPosition.entryPrice > 0
                      ? (longPerpPosition.unrealizedPnL / (longPerpPosition.entryPrice * longPerpPosition.quantity)) * 100
                      : 0;

                  const longFundingRate = fundingRates.find(fr =>
                    fr.symbol === pair.symbol && fr.exchange === longPerpPosition.exchange
                  );
                  const longRate = longFundingRate ? longFundingRate.rate : 0;
                  const longValue = longPerpPosition.quantity * longPerpPosition.entryPrice;
                  // Long: negative rate = receive (positive), positive rate = pay (negative)
                  const longEstimatedFunding = -longRate * longValue;

                  const uniqueKey = `cross-fut-${pair.executionId}-${pairIndex}`;
                  const isHovered = hoveredRow === uniqueKey;

                  rows.push(
                    <TableRow
                      key={`long-perp-${longPerpPosition.id}-${pairIndex}`}
                      className={`border-b-0 ${isHovered ? 'bg-[#2b3139]' : ''}`}
                      hover={false}
                      onMouseEnter={() => setHoveredRow(uniqueKey)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      <TableCell className={`sticky left-0 z-20 border-r border-binance-border font-bold text-xs ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>{pair.symbol}</TableCell>
                      <TableCell rowSpan={2} className="py-1">
                        <Badge
                          size="sm"
                          className={`text-[10px] ${strategyLabel.color}`}
                        >
                          {strategyLabel.text}
                        </Badge>
                      </TableCell>
                      <TableCell className="py-1">
                        <ExchangeBadge exchange={longPerpPosition.exchange} />
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge variant="info" size="sm" className="text-[10px]">Perpetual</Badge>
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge
                          variant="success"
                          size="sm"
                          className="gap-0.5 text-[10px]"
                        >
                          <TrendingUp className="w-2.5 h-2.5" />
                          Long
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${longPerpPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">{longPerpPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${(longPerpPosition.quantity * longPerpPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm" className="text-[10px]">
                          <span className="font-mono">{longPerpPosition.leverage}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              longPerpPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {longPerpPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {longPerpPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-0.5 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-300 ${
                                longPerpPosition.unrealizedPnL >= 0 ? 'bg-binance-green' : 'bg-binance-red'
                              }`}
                              style={{
                                width: `${Math.min(Math.abs(longPnlPercent), 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <span
                          className={`font-mono text-[11px] ${
                            longEstimatedFunding >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {longEstimatedFunding >= 0 ? '+' : ''}$
                          {longEstimatedFunding.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            combinedUnrealizedPnL >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {combinedUnrealizedPnL >= 0 ? '+' : ''}$
                          {combinedUnrealizedPnL.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            (combinedFunding + estimatedFunding) >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {(combinedFunding + estimatedFunding) >= 0 ? '+' : ''}$
                          {(combinedFunding + estimatedFunding).toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            totalPairPnL >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {totalPairPnL >= 0 ? '+' : ''}$
                          {totalPairPnL.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span className="font-mono text-[11px] text-binance-text-secondary">
                          {pair.executionId ? executionTimes[pair.executionId] || '--' : '--'}
                        </span>
                      </TableCell>
                      <TableCell className={`sticky right-[180px] z-20 border-l border-binance-border text-center ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {(() => {
                          // Get entry confidence from either position
                          const entryScore = pair.perp?.entryConfidenceScore || pair.spot?.entryConfidenceScore || pair.longPerp?.entryConfidenceScore;
                          const recommendedStrategy = pair.perp?.recommendedStrategy || pair.spot?.recommendedStrategy || pair.longPerp?.recommendedStrategy;

                          if (!entryScore) {
                            return <span className="text-xs text-binance-text-secondary">--</span>;
                          }

                          // Create a minimal suggestion object for the badge
                          const mockSuggestion: OpportunitySuggestion = {
                            confidenceScore: entryScore,
                            recommendedStrategy: RecommendedStrategyType.Hybrid,
                            entryRecommendation: entryScore >= 80 ? 3 : entryScore >= 60 ? 2 : entryScore >= 40 ? 1 : 0,
                            suggestedPositionSizeUsd: 0,
                            suggestedLeverage: 0,
                            suggestedHoldingPeriodHours: 0,
                            profitTargetPercent: pair.perp?.profitTargetPercent || pair.spot?.profitTargetPercent || 0,
                            maxHoldingHours: pair.perp?.maxHoldingHours || pair.spot?.maxHoldingHours || 0,
                            scoreBreakdown: {
                              fundingQuality: 0,
                              profitPotential: 0,
                              spreadEfficiency: 0,
                              marketQuality: 0
                            },
                            reasoning: "Entry snapshot from position execution",
                            scoringFactors: [],
                            warnings: []
                          };

                          return <SuggestionBadge suggestion={mockSuggestion} entryScore={entryScore} recommendedStrategy={recommendedStrategy} />;
                        })()}
                      </TableCell>
                      <TableCell className={`sticky right-[100px] z-20 border-l border-binance-border text-center ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {(() => {
                          // Get exit signals from either position
                          const exitSignals = pair.perp?.exitSignals || pair.spot?.exitSignals || pair.longPerp?.exitSignals || pair.shortPerp?.exitSignals;

                          if (!exitSignals || exitSignals.length === 0) {
                            return <span className="text-xs text-binance-text-secondary">--</span>;
                          }

                          // Filter for triggered signals
                          const triggeredSignals = exitSignals.filter((s: any) => s.isTriggered);

                          if (triggeredSignals.length === 0) {
                            return <span className="text-xs text-green-400">✓ Hold</span>;
                          }

                          // Get highest urgency signal
                          const criticalSignal = triggeredSignals.find((s: any) => s.urgency === 3); // Critical
                          const highSignal = triggeredSignals.find((s: any) => s.urgency === 2); // High
                          const mediumSignal = triggeredSignals.find((s: any) => s.urgency === 1); // Medium
                          const topSignal = criticalSignal || highSignal || mediumSignal || triggeredSignals[0];

                          const urgencyColor = topSignal.urgency === 3 ? 'text-red-400' :
                                               topSignal.urgency === 2 ? 'text-orange-400' :
                                               topSignal.urgency === 1 ? 'text-yellow-400' : 'text-blue-400';

                          const urgencyLabel = topSignal.urgency === 3 ? 'EXIT NOW' :
                                               topSignal.urgency === 2 ? 'Exit Soon' :
                                               topSignal.urgency === 1 ? 'Consider Exit' : 'Monitor';

                          return (
                            <div className="flex flex-col items-center gap-0.5">
                              <span className={`text-[10px] font-bold ${urgencyColor}`}>
                                {urgencyLabel}
                              </span>
                              <span className="text-[9px] text-binance-text-secondary">
                                {triggeredSignals.length} signal{triggeredSignals.length > 1 ? 's' : ''}
                              </span>
                            </div>
                          );
                        })()}
                      </TableCell>
                      <TableCell className={`sticky right-0 z-20 border-l border-binance-border text-right ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {pair.executionId ? (
                          <Button
                            variant="danger"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStop(pair.executionId!);
                            }}
                            className="gap-0.5 h-6 px-2 text-[10px]"
                          >
                            <StopCircle className="w-2.5 h-2.5" />
                            Stop
                          </Button>
                        ) : (
                          <span className="text-[10px] text-binance-text-secondary">--</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );

                  // Short Perp row
                  const shortPnlPercent =
                    shortPerpPosition.entryPrice > 0
                      ? (shortPerpPosition.unrealizedPnL / (shortPerpPosition.entryPrice * shortPerpPosition.quantity)) * 100
                      : 0;

                  const shortFundingRate = fundingRates.find(fr =>
                    fr.symbol === pair.symbol && fr.exchange === shortPerpPosition.exchange
                  );
                  const shortRate = shortFundingRate ? shortFundingRate.rate : 0;
                  const shortValue = shortPerpPosition.quantity * shortPerpPosition.entryPrice;
                  // Short: negative rate = pay (negative), positive rate = receive (positive)
                  const shortEstimatedFunding = shortRate * shortValue;

                  rows.push(
                    <TableRow
                      key={`short-perp-${shortPerpPosition.id}-${pairIndex}`}
                      className={`border-t border-binance-border/30 ${isHovered ? 'bg-[#2b3139]' : ''}`}
                      hover={false}
                      onMouseEnter={() => setHoveredRow(uniqueKey)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      <TableCell className="py-1">
                        <ExchangeBadge exchange={shortPerpPosition.exchange} />
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge variant="info" size="sm" className="text-[10px]">Perpetual</Badge>
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge
                          variant="danger"
                          size="sm"
                          className="gap-0.5 text-[10px]"
                        >
                          <TrendingDown className="w-2.5 h-2.5" />
                          Short
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${shortPerpPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">{shortPerpPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${(shortPerpPosition.quantity * shortPerpPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm" className="text-[10px]">
                          <span className="font-mono">{shortPerpPosition.leverage || 1}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              shortPerpPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {shortPerpPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {shortPerpPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-0.5 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-300 ${
                                shortPerpPosition.unrealizedPnL >= 0 ? 'bg-binance-green' : 'bg-binance-red'
                              }`}
                              style={{
                                width: `${Math.min(Math.abs(shortPnlPercent), 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <span
                          className={`font-mono text-[11px] ${
                            shortEstimatedFunding >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {shortEstimatedFunding >= 0 ? '+' : ''}$
                          {shortEstimatedFunding.toFixed(2)}
                        </span>
                      </TableCell>
                    </TableRow>
                  );
                }
                // For Spot-Perp or Cross-Spot: render perp and spot rows
                else if (perpPosition) {
                  const pnlPercent =
                    perpPosition.entryPrice > 0
                      ? (perpPosition.unrealizedPnL / (perpPosition.entryPrice * perpPosition.quantity)) * 100
                      : 0;

                  const uniqueKey = `perp-spot-${pair.executionId}-${pairIndex}`;
                  const isHovered = hoveredRow === uniqueKey;

                  rows.push(
                    <TableRow
                      key={`perp-${perpPosition.id}-${pairIndex}`}
                      className={`border-b-0 ${isHovered ? 'bg-[#2b3139]' : ''}`}
                      hover={false}
                      onMouseEnter={() => setHoveredRow(uniqueKey)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      <TableCell className={`sticky left-0 z-20 border-r border-binance-border font-bold text-xs ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>{pair.symbol}</TableCell>
                      <TableCell rowSpan={2} className="py-1">
                        <Badge
                          size="sm"
                          className={`text-[10px] ${strategyLabel.color}`}
                        >
                          {strategyLabel.text}
                        </Badge>
                      </TableCell>
                      <TableCell rowSpan={2} className="py-1">
                        <ExchangeBadge exchange={pair.exchange} />
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge variant="info" size="sm" className="text-[10px]">Perpetual</Badge>
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge
                          variant={perpPosition.side === PositionSide.Long ? 'success' : 'danger'}
                          size="sm"
                          className="gap-0.5 text-[10px]"
                        >
                          {perpPosition.side === PositionSide.Long ? (
                            <TrendingUp className="w-2.5 h-2.5" />
                          ) : (
                            <TrendingDown className="w-2.5 h-2.5" />
                          )}
                          {perpPosition.side === PositionSide.Long ? 'Long' : 'Short'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${perpPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">{perpPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${(perpPosition.quantity * perpPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm" className="text-[10px]">
                          <span className="font-mono">{perpPosition.leverage || 1}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              perpPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {perpPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {perpPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-0.5 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-300 ${
                                perpPosition.unrealizedPnL >= 0 ? 'bg-binance-green' : 'bg-binance-red'
                              }`}
                              style={{
                                width: `${Math.min(Math.abs(pnlPercent), 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <span
                          className={`font-mono text-[11px] ${
                            estimatedFunding >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {estimatedFunding >= 0 ? '+' : ''}$
                          {estimatedFunding.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            combinedUnrealizedPnL >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {combinedUnrealizedPnL >= 0 ? '+' : ''}$
                          {combinedUnrealizedPnL.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            (combinedFunding + estimatedFunding) >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {(combinedFunding + estimatedFunding) >= 0 ? '+' : ''}$
                          {(combinedFunding + estimatedFunding).toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-[11px] font-bold ${
                            totalPairPnL >= 0
                              ? 'text-binance-green'
                              : 'text-binance-red'
                          }`}
                        >
                          {totalPairPnL >= 0 ? '+' : ''}$
                          {totalPairPnL.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span className="font-mono text-[11px] text-binance-text-secondary">
                          {pair.executionId ? executionTimes[pair.executionId] || '--' : '--'}
                        </span>
                      </TableCell>
                      <TableCell className={`sticky right-[180px] z-20 border-l border-binance-border text-center ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {(() => {
                          // Get entry confidence from either position
                          const entryScore = pair.perp?.entryConfidenceScore || pair.spot?.entryConfidenceScore || pair.longPerp?.entryConfidenceScore;
                          const recommendedStrategy = pair.perp?.recommendedStrategy || pair.spot?.recommendedStrategy || pair.longPerp?.recommendedStrategy;

                          if (!entryScore) {
                            return <span className="text-xs text-binance-text-secondary">--</span>;
                          }

                          // Create a minimal suggestion object for the badge
                          const mockSuggestion: OpportunitySuggestion = {
                            confidenceScore: entryScore,
                            recommendedStrategy: RecommendedStrategyType.Hybrid,
                            entryRecommendation: entryScore >= 80 ? 3 : entryScore >= 60 ? 2 : entryScore >= 40 ? 1 : 0,
                            suggestedPositionSizeUsd: 0,
                            suggestedLeverage: 0,
                            suggestedHoldingPeriodHours: 0,
                            profitTargetPercent: pair.perp?.profitTargetPercent || pair.spot?.profitTargetPercent || 0,
                            maxHoldingHours: pair.perp?.maxHoldingHours || pair.spot?.maxHoldingHours || 0,
                            scoreBreakdown: {
                              fundingQuality: 0,
                              profitPotential: 0,
                              spreadEfficiency: 0,
                              marketQuality: 0
                            },
                            reasoning: "Entry snapshot from position execution",
                            scoringFactors: [],
                            warnings: []
                          };

                          return <SuggestionBadge suggestion={mockSuggestion} entryScore={entryScore} recommendedStrategy={recommendedStrategy} />;
                        })()}
                      </TableCell>
                      <TableCell className={`sticky right-[100px] z-20 border-l border-binance-border text-center ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {(() => {
                          // Get exit signals from either position
                          const exitSignals = pair.perp?.exitSignals || pair.spot?.exitSignals || pair.longPerp?.exitSignals || pair.shortPerp?.exitSignals;

                          if (!exitSignals || exitSignals.length === 0) {
                            return <span className="text-xs text-binance-text-secondary">--</span>;
                          }

                          // Filter for triggered signals
                          const triggeredSignals = exitSignals.filter((s: any) => s.isTriggered);

                          if (triggeredSignals.length === 0) {
                            return <span className="text-xs text-green-400">✓ Hold</span>;
                          }

                          // Get highest urgency signal
                          const criticalSignal = triggeredSignals.find((s: any) => s.urgency === 3); // Critical
                          const highSignal = triggeredSignals.find((s: any) => s.urgency === 2); // High
                          const mediumSignal = triggeredSignals.find((s: any) => s.urgency === 1); // Medium
                          const topSignal = criticalSignal || highSignal || mediumSignal || triggeredSignals[0];

                          const urgencyColor = topSignal.urgency === 3 ? 'text-red-400' :
                                               topSignal.urgency === 2 ? 'text-orange-400' :
                                               topSignal.urgency === 1 ? 'text-yellow-400' : 'text-blue-400';

                          const urgencyLabel = topSignal.urgency === 3 ? 'EXIT NOW' :
                                               topSignal.urgency === 2 ? 'Exit Soon' :
                                               topSignal.urgency === 1 ? 'Consider Exit' : 'Monitor';

                          return (
                            <div className="flex flex-col items-center gap-0.5">
                              <span className={`text-[10px] font-bold ${urgencyColor}`}>
                                {urgencyLabel}
                              </span>
                              <span className="text-[9px] text-binance-text-secondary">
                                {triggeredSignals.length} signal{triggeredSignals.length > 1 ? 's' : ''}
                              </span>
                            </div>
                          );
                        })()}
                      </TableCell>
                      <TableCell className={`sticky right-0 z-20 border-l border-binance-border text-right ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                        {pair.executionId ? (
                          <Button
                            variant="danger"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStop(pair.executionId!);
                            }}
                            className="gap-0.5 h-6 px-2 text-[10px]"
                          >
                            <StopCircle className="w-2.5 h-2.5" />
                            Stop
                          </Button>
                        ) : (
                          <span className="text-[10px] text-binance-text-secondary">--</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                }

                // Spot row (for Spot-Perp and Cross-Spot only, not for Cross-Fut)
                if (spotPosition && !isCrossFut) {
                  const pnlPercent =
                    spotPosition.entryPrice > 0
                      ? (spotPosition.unrealizedPnL / (spotPosition.entryPrice * spotPosition.quantity)) * 100
                      : 0;

                  const uniqueKey = `perp-spot-${pair.executionId}-${pairIndex}`;
                  const isHovered = hoveredRow === uniqueKey;

                  rows.push(
                    <TableRow
                      key={`spot-${spotPosition.id}-${pairIndex}`}
                      className={`border-t border-binance-border/30 ${isHovered ? 'bg-[#2b3139]' : ''}`}
                      hover={false}
                      onMouseEnter={() => setHoveredRow(uniqueKey)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      <TableCell className="py-1">
                        <Badge variant="secondary" size="sm" className="text-[10px]">Spot</Badge>
                      </TableCell>
                      <TableCell className="py-1">
                        <Badge
                          variant={spotPosition.side === PositionSide.Long ? 'success' : 'danger'}
                          size="sm"
                          className="gap-0.5 text-[10px]"
                        >
                          {spotPosition.side === PositionSide.Long ? (
                            <TrendingUp className="w-2.5 h-2.5" />
                          ) : (
                            <TrendingDown className="w-2.5 h-2.5" />
                          )}
                          {spotPosition.side === PositionSide.Long ? 'Long' : 'Short'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${spotPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">{spotPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${(spotPosition.quantity * spotPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm" className="text-[10px]">
                          <span className="font-mono">{spotPosition.leverage || 1}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              spotPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {spotPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {spotPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-0.5 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-300 ${
                                spotPosition.unrealizedPnL >= 0 ? 'bg-binance-green' : 'bg-binance-red'
                              }`}
                              style={{
                                width: `${Math.min(Math.abs(pnlPercent), 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      </TableCell>
                    </TableRow>
                  );
                }

                return rows;
              })}
            </TableBody>
          </Table>
        )}
      </div>

    <AlertDialog
      isOpen={alertState.isOpen}
      onClose={closeAlert}
      title={alertState.title}
      message={alertState.message}
      variant={alertState.variant}
    />

    <ConfirmDialog
      isOpen={confirmState.isOpen}
      onClose={closeConfirm}
      onConfirm={confirmState.onConfirm}
      title={confirmState.title}
      message={confirmState.message}
      confirmText={confirmState.confirmText}
      cancelText={confirmState.cancelText}
      variant={confirmState.variant}
    />
    </>
  );
};
