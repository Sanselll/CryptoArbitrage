import { Layers, X, Edit, TrendingUp, TrendingDown } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { PositionSide, PositionStatus, PositionType } from '../types/index';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';
import { EmptyState } from './ui/EmptyState';
import { LoadingOverlay } from './ui/LoadingOverlay';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';
import { useState } from 'react';
import { apiService } from '../services/apiService';

interface PositionPair {
  executionId: number | undefined;
  symbol: string;
  exchange: string;
  perp: any | null;
  spot: any | null;
}

export const PositionsGrid = () => {
  const { positions } = useArbitrageStore();
  const [isClosing, setIsClosing] = useState(false);

  const openPositions = positions.filter((p) => p.status === PositionStatus.Open);

  // Group positions by executionId (two positions per execution: perp + spot)
  const positionPairs: PositionPair[] = [];
  const grouped = new Map<number | undefined, { perp: any | null; spot: any | null; symbol: string; exchange: string }>();

  openPositions.forEach((position) => {
    const key = position.executionId;
    if (!grouped.has(key)) {
      grouped.set(key, { perp: null, spot: null, symbol: position.symbol, exchange: position.exchange });
    }
    const pair = grouped.get(key)!;
    if (position.type === PositionType.Perpetual) {
      pair.perp = position;
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
      spot: pair.spot
    });
  });

  const totalPnL = openPositions.reduce((sum, p) => sum + p.unrealizedPnL, 0);

  const handleClose = async (executionId: number) => {
    const confirmed = window.confirm(
      `Are you sure you want to close this position?\n\n` +
      `This will close both spot and perpetual positions and realize the P&L.`
    );

    if (!confirmed) return;

    setIsClosing(true);
    try {
      const response = await apiService.stopExecution(executionId);

      if (response.success) {
        // Manually refresh positions immediately after successful close
        try {
          const freshPositions = await apiService.getPositions();
          useArbitrageStore.getState().setPositions(freshPositions);
        } catch (refreshError) {
          console.error('Failed to refresh positions after close:', refreshError);
          // Don't fail the whole operation if refresh fails
        }

        alert(
          `Success!\n\n` +
          `${response.message}\n` +
          `Position closed.`
        );
      } else {
        alert(`Failed to close position:\n\n${response.errorMessage}`);
      }
    } catch (error: any) {
      console.error('Error closing position:', error);
      alert(`Failed to close position:\n\n${error.message || 'Unknown error'}`);
    } finally {
      setIsClosing(false);
    }
  };

  const handleEdit = (positionId: number) => {
    console.log('Edit position:', positionId);
    // TODO: Implement edit logic
  };

  return (
    <>
      <LoadingOverlay
        isLoading={isClosing}
        message="Closing position..."
      />

      <Card className="h-full flex flex-col">
        <CardHeader className="p-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-binance-yellow" />
              Open Positions
            </CardTitle>
          <div className="flex items-center gap-3">
            <span className="text-xs text-binance-text-secondary">Total P&L:</span>
            <span
              className={`text-sm font-bold font-mono ${
                totalPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
              }`}
            >
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </span>
            <Badge variant="info" size="sm">
              {openPositions.length} Active
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-0">
        {openPositions.length === 0 ? (
          <EmptyState
            icon={<Layers className="w-12 h-12" />}
            title="No open positions"
            description="Execute arbitrage opportunities to create hedged positions"
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow hover={false}>
                <TableHead>Symbol</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Size (USDT)</TableHead>
                <TableHead className="text-right">Lev</TableHead>
                <TableHead className="text-right">Unrealized P&L</TableHead>
                <TableHead className="text-right">Funding</TableHead>
                <TableHead className="text-right">Est. Total P&L</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positionPairs.map((pair, pairIndex) => {
                // Render two rows for each pair (perpetual and spot)
                const perpPosition = pair.perp;
                const spotPosition = pair.spot;
                const rows = [];

                // Calculate combined P&L for the pair
                const combinedUnrealizedPnL = (perpPosition?.unrealizedPnL || 0) + (spotPosition?.unrealizedPnL || 0);
                const combinedFunding = (perpPosition?.netFundingFee || 0) + (spotPosition?.netFundingFee || 0);
                const totalPairPnL = combinedUnrealizedPnL + combinedFunding;

                // Perpetual row
                if (perpPosition) {
                  const pnlPercent =
                    perpPosition.entryPrice > 0
                      ? (perpPosition.unrealizedPnL / (perpPosition.entryPrice * perpPosition.quantity)) * 100
                      : 0;

                  rows.push(
                    <TableRow key={`perp-${perpPosition.id}-${pairIndex}`} className="border-b-0">
                      <TableCell className="font-bold" rowSpan={2}>{pair.symbol}</TableCell>
                      <TableCell>
                        <Badge variant="info" size="sm">Perpetual</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={perpPosition.side === PositionSide.Long ? 'success' : 'danger'}
                          size="sm"
                          className="gap-1"
                        >
                          {perpPosition.side === PositionSide.Long ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : (
                            <TrendingDown className="w-3 h-3" />
                          )}
                          {perpPosition.side === PositionSide.Long ? 'Long' : 'Short'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">${perpPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">{perpPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">${(perpPosition.quantity * perpPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm">
                          <span className="font-mono">{perpPosition.leverage}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-1">
                          <span
                            className={`font-mono text-sm font-bold ${
                              perpPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {perpPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {perpPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-1 overflow-hidden">
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
                          className={`font-mono text-sm ${
                            perpPosition.netFundingFee >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {perpPosition.netFundingFee >= 0 ? '+' : ''}$
                          {perpPosition.netFundingFee.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right" rowSpan={2}>
                        <span
                          className={`font-mono text-sm font-bold ${
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
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => pair.executionId && handleEdit(pair.executionId)}
                            className="h-8 w-8 p-0"
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="danger"
                            size="sm"
                            onClick={() => pair.executionId && handleClose(pair.executionId)}
                            className="gap-1"
                          >
                            <X className="w-3 h-3" />
                            Close
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                }

                // Spot row
                if (spotPosition) {
                  const pnlPercent =
                    spotPosition.entryPrice > 0
                      ? (spotPosition.unrealizedPnL / (spotPosition.entryPrice * spotPosition.quantity)) * 100
                      : 0;

                  rows.push(
                    <TableRow key={`spot-${spotPosition.id}-${pairIndex}`} className="border-t border-binance-border/30">
                      <TableCell>
                        <Badge variant="secondary" size="sm">Spot</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={spotPosition.side === PositionSide.Long ? 'success' : 'danger'}
                          size="sm"
                          className="gap-1"
                        >
                          {spotPosition.side === PositionSide.Long ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : (
                            <TrendingDown className="w-3 h-3" />
                          )}
                          {spotPosition.side === PositionSide.Long ? 'Long' : 'Short'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">${spotPosition.entryPrice.toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">{spotPosition.quantity.toFixed(4)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="font-mono text-sm">${(spotPosition.quantity * spotPosition.entryPrice).toFixed(2)}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="warning" size="sm">
                          <span className="font-mono">{spotPosition.leverage}x</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex flex-col items-end gap-1">
                          <span
                            className={`font-mono text-sm font-bold ${
                              spotPosition.unrealizedPnL >= 0
                                ? 'text-binance-green'
                                : 'text-binance-red'
                            }`}
                          >
                            {spotPosition.unrealizedPnL >= 0 ? '+' : ''}$
                            {spotPosition.unrealizedPnL.toFixed(2)}
                          </span>
                          <div className="w-full bg-binance-bg-tertiary rounded-full h-1 overflow-hidden">
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
                        <span className="font-mono text-sm text-binance-text-secondary">--</span>
                      </TableCell>
                    </TableRow>
                  );
                }

                return rows;
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
    </>
  );
};
