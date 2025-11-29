import { History, TrendingUp, TrendingDown, Calendar } from 'lucide-react';
import { useArbitrageStore, ExecutionHistoryFilter } from '../stores/arbitrageStore';
import { PositionSide, PositionType } from '../types/index';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';
import { EmptyState } from './ui/EmptyState';
import { ExchangeBadge } from './ui/ExchangeBadge';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';
import { useMemo, useState } from 'react';

// Helper function to format duration
const formatDuration = (durationSeconds: number): string => {
  if (durationSeconds < 0) return '0m';

  const days = Math.floor(durationSeconds / (24 * 60 * 60));
  const hours = Math.floor((durationSeconds % (24 * 60 * 60)) / (60 * 60));
  const minutes = Math.floor((durationSeconds % (60 * 60)) / 60);

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  } else if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else {
    return `${minutes}m`;
  }
};

// Helper function to format date/time
const formatDateTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Filter preset configuration
const filterPresets: { key: ExecutionHistoryFilter; label: string }[] = [
  { key: 'today', label: 'Today' },
  { key: '7days', label: '7 Days' },
  { key: '30days', label: '30 Days' },
  { key: 'all', label: 'All Time' },
];

export const ExecutionHistoryGrid = () => {
  const { executionHistory, executionHistoryFilter, setExecutionHistoryFilter } = useArbitrageStore();
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  // Filter executions based on selected filter
  const filteredHistory = useMemo(() => {
    if (executionHistoryFilter === 'all') {
      return executionHistory;
    }

    const now = new Date();
    let cutoffDate: Date;

    switch (executionHistoryFilter) {
      case 'today':
        cutoffDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        break;
      case '7days':
        cutoffDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case '30days':
        cutoffDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      default:
        return executionHistory;
    }

    return executionHistory.filter(execution => {
      const closedAt = new Date(execution.closedAt);
      return closedAt >= cutoffDate;
    });
  }, [executionHistory, executionHistoryFilter]);

  // Calculate summary stats
  const summaryStats = useMemo(() => {
    const totalPnL = filteredHistory.reduce((sum, e) => sum + e.totalPnL, 0);
    const winCount = filteredHistory.filter(e => e.totalPnL > 0).length;
    const winRate = filteredHistory.length > 0
      ? (winCount / filteredHistory.length * 100).toFixed(1)
      : '0.0';

    return { totalPnL, winCount, winRate };
  }, [filteredHistory]);

  if (executionHistory.length === 0) {
    return (
      <EmptyState
        icon={<History className="w-12 h-12" />}
        title="No execution history"
        description="Completed executions will appear here"
      />
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Filter buttons and summary */}
      <div className="flex items-center justify-between px-2 py-2 border-b border-binance-border">
        <div className="flex items-center gap-2">
          {filterPresets.map((preset) => (
            <Button
              key={preset.key}
              variant={executionHistoryFilter === preset.key ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setExecutionHistoryFilter(preset.key)}
              className="h-6 px-2 text-[10px]"
            >
              {preset.label}
            </Button>
          ))}
        </div>

        {/* Summary stats */}
        <div className="flex items-center gap-4 text-[11px]">
          <span className="text-binance-text-secondary">
            {filteredHistory.length} executions
          </span>
          <span className={summaryStats.totalPnL >= 0 ? 'text-binance-green' : 'text-binance-red'}>
            Total: {summaryStats.totalPnL >= 0 ? '+' : ''}${summaryStats.totalPnL.toFixed(2)}
          </span>
          <span className="text-binance-text-secondary">
            Win Rate: {summaryStats.winRate}%
          </span>
        </div>
      </div>

      {filteredHistory.length === 0 ? (
        <EmptyState
          icon={<Calendar className="w-12 h-12" />}
          title="No executions in this period"
          description="Try selecting a different time range"
        />
      ) : (
        <div className="flex-1 overflow-x-auto overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 z-30">
              <TableRow hover={false}>
                <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border">Symbol</TableHead>
                <TableHead className="py-1">Exchange</TableHead>
                <TableHead className="py-1">Side</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Exit</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead className="text-right">Lev</TableHead>
                <TableHead className="text-right">Price P&L</TableHead>
                <TableHead className="text-right">Funding</TableHead>
                <TableHead className="text-right">Fees</TableHead>
                <TableHead className="text-right">Total P&L</TableHead>
                <TableHead className="text-right">Duration</TableHead>
                <TableHead className="text-right">Closed</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredHistory.map((execution) => {
                const isHovered = hoveredRow === execution.id;
                const positions = execution.positions || [];
                const numPositions = positions.length;

                // Sort positions: Long first, then Short; or Perp first, then Spot
                const sortedPositions = [...positions].sort((a, b) => {
                  // For cross-exchange (both perp), sort by side: Long first
                  if (a.type === PositionType.Perpetual && b.type === PositionType.Perpetual) {
                    return a.side === PositionSide.Long ? -1 : 1;
                  }
                  // For spot-perp, sort by type: Perp first
                  return a.type === PositionType.Perpetual ? -1 : 1;
                });

                return sortedPositions.map((position, positionIndex) => {
                  const isFirstRow = positionIndex === 0;
                  const isLastRow = positionIndex === numPositions - 1;
                  const positionValue = position.quantity * position.entryPrice;
                  const pnlPercent = positionValue > 0
                    ? (position.pricePnL / positionValue) * 100
                    : 0;

                  return (
                    <TableRow
                      key={`${execution.id}-${position.id}`}
                      className={`${!isLastRow ? 'border-b-0' : ''} ${!isFirstRow ? 'border-t border-binance-border/30' : ''} ${isHovered ? 'bg-[#2b3139]' : ''}`}
                      hover={false}
                      onMouseEnter={() => setHoveredRow(execution.id)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      {/* Symbol - only on first row */}
                      {isFirstRow && (
                        <TableCell
                          className={`sticky left-0 z-20 border-r border-binance-border font-bold text-xs ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}
                          rowSpan={numPositions}
                        >
                          {execution.symbol}
                        </TableCell>
                      )}

                      {/* Exchange */}
                      <TableCell className="py-1">
                        <ExchangeBadge exchange={position.exchange} />
                      </TableCell>

                      {/* Side */}
                      <TableCell className="py-1">
                        {position.type === PositionType.Perpetual ? (
                          <Badge
                            variant={position.side === PositionSide.Long ? 'success' : 'danger'}
                            size="sm"
                            className="gap-0.5 text-[10px]"
                          >
                            {position.side === PositionSide.Long ? (
                              <>
                                <TrendingUp className="w-2.5 h-2.5" />
                                Long
                              </>
                            ) : (
                              <>
                                <TrendingDown className="w-2.5 h-2.5" />
                                Short
                              </>
                            )}
                          </Badge>
                        ) : (
                          <Badge variant="info" size="sm" className="text-[10px]">
                            Spot
                          </Badge>
                        )}
                      </TableCell>

                      {/* Entry Price */}
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${position.entryPrice.toFixed(2)}</span>
                      </TableCell>

                      {/* Exit Price */}
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">
                          {position.exitPrice > 0 ? `$${position.exitPrice.toFixed(2)}` : '-'}
                        </span>
                      </TableCell>

                      {/* Size (Quantity) */}
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">{position.quantity.toFixed(4)}</span>
                      </TableCell>

                      {/* Value */}
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px]">${positionValue.toFixed(2)}</span>
                      </TableCell>

                      {/* Leverage */}
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm" className="text-[10px]">
                          <span className="font-mono">{position.leverage}x</span>
                        </Badge>
                      </TableCell>

                      {/* Price P&L */}
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-0.5">
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              position.pricePnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                            }`}
                          >
                            {position.pricePnL >= 0 ? '+' : ''}${position.pricePnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-0.5 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-300 ${
                                position.pricePnL >= 0 ? 'bg-binance-green' : 'bg-binance-red'
                              }`}
                              style={{
                                width: `${Math.min(Math.abs(pnlPercent), 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </TableCell>

                      {/* Funding */}
                      <TableCell className="text-right">
                        <span
                          className={`font-mono text-[11px] ${
                            position.fundingEarned >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {position.fundingEarned !== 0
                            ? `${position.fundingEarned >= 0 ? '+' : ''}$${position.fundingEarned.toFixed(2)}`
                            : '-'}
                        </span>
                      </TableCell>

                      {/* Fees */}
                      <TableCell className="text-right">
                        <span className="font-mono text-[11px] text-red-400">
                          {position.tradingFees > 0 ? `-$${position.tradingFees.toFixed(2)}` : '-'}
                        </span>
                      </TableCell>

                      {/* Total P&L - only on first row, rowSpan */}
                      {isFirstRow && (
                        <TableCell className="text-right" rowSpan={numPositions}>
                          <span
                            className={`font-mono text-[11px] font-bold ${
                              execution.totalPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                            }`}
                          >
                            {execution.totalPnL >= 0 ? '+' : ''}${execution.totalPnL.toFixed(2)}
                          </span>
                        </TableCell>
                      )}

                      {/* Duration - only on first row, rowSpan */}
                      {isFirstRow && (
                        <TableCell className="text-right" rowSpan={numPositions}>
                          <span className="font-mono text-[11px] text-binance-text-secondary">
                            {formatDuration(execution.durationSeconds)}
                          </span>
                        </TableCell>
                      )}

                      {/* Closed - only on first row, rowSpan */}
                      {isFirstRow && (
                        <TableCell className="text-right" rowSpan={numPositions}>
                          <span className="font-mono text-[11px] text-binance-text-secondary">
                            {formatDateTime(execution.closedAt)}
                          </span>
                        </TableCell>
                      )}
                    </TableRow>
                  );
                });
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};
