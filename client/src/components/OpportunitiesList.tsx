import { Target, ArrowUpCircle, ArrowDownCircle, Play, Clock, StopCircle } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';
import { EmptyState } from './ui/EmptyState';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';
import { useState, useEffect } from 'react';
import { ExecuteDialog, ExecutionParams } from './ExecuteDialog';
import { apiService } from '../services/apiService';
import { PositionStatus } from '../types/index';

// Helper function to calculate time until next funding (every 8 hours at 00:00, 08:00, 16:00 UTC)
const getNextFundingTime = () => {
  const now = new Date();
  const currentHour = now.getUTCHours();
  const nextFundingHour = Math.ceil((currentHour + 1) / 8) * 8;
  const nextFunding = new Date(now);
  nextFunding.setUTCHours(nextFundingHour === 24 ? 0 : nextFundingHour, 0, 0, 0);
  if (nextFundingHour === 24) {
    nextFunding.setUTCDate(nextFunding.getUTCDate() + 1);
  }
  return nextFunding;
};

const formatTimeUntil = (targetDate: Date) => {
  const now = new Date();
  const diff = targetDate.getTime() - now.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);
  return `${hours}h ${minutes}m ${seconds}s`;
};

// Helper function to format execution time (time elapsed since position opened)
const formatExecutionTime = (openedAt: string) => {
  const now = new Date();
  // Ensure the date is parsed as UTC by appending 'Z' if not present
  const dateString = openedAt.endsWith('Z') ? openedAt : `${openedAt}Z`;
  const opened = new Date(dateString);
  const diff = now.getTime() - opened.getTime();

  // Prevent negative time display
  if (diff < 0) return '0h 0m 0s';

  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);
  return `${hours}h ${minutes}m ${seconds}s`;
};

export const OpportunitiesList = () => {
  const { opportunities, positions } = useArbitrageStore();
  const [timeUntilFunding, setTimeUntilFunding] = useState('');
  const [executionTimes, setExecutionTimes] = useState<{ [key: string]: string }>({});
  const [selectedOpportunity, setSelectedOpportunity] = useState<any>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      const nextFunding = getNextFundingTime();
      setTimeUntilFunding(formatTimeUntil(nextFunding));

      // Update execution times based on opportunity's activeOpportunityExecutedAt
      const newExecutionTimes: { [key: string]: string } = {};

      // Get current opportunities from the store
      const currentOpportunities = useArbitrageStore.getState().opportunities;
      currentOpportunities
        .filter((opp) => (opp as any).activeOpportunityExecutedAt)
        .forEach((opp) => {
          const isSpotPerp = (opp as any).strategy === 1;
          const key = isSpotPerp
            ? `${opp.symbol}-${(opp as any).exchange}`
            : `${opp.symbol}-${(opp as any).longExchange}`;
          newExecutionTimes[key] = formatExecutionTime((opp as any).activeOpportunityExecutedAt);
        });
      setExecutionTimes(newExecutionTimes);
    }, 1000);

    return () => clearInterval(interval);
  }, []); // Empty dependency array - interval runs once on mount

  const activeOpportunities = opportunities
    .filter((opp) => {
      // Only show Detected opportunities
      if (opp.status !== 0) return false;

      return true; // Show all detected opportunities, including those being executed
    })
    .sort((a, b) => b.annualizedSpread - a.annualizedSpread)
    .slice(0, 20); // Show top 20 opportunities

  const handleExecute = (opp: any) => {
    setSelectedOpportunity(opp);
    setIsDialogOpen(true);
  };

  const handleStop = async (opp: any) => {
    // Check if this opportunity has an execution running
    if (!opp.executionId) {
      alert('No execution found for this opportunity');
      return;
    }

    const confirmed = window.confirm(
      `Are you sure you want to stop this execution?\n\n` +
      `Symbol: ${opp.symbol}\n` +
      `Exchange: ${opp.exchange}\n` +
      `This will close all positions and sell the spot asset.`
    );

    if (!confirmed) return;

    try {
      const response = await apiService.stopExecution(opp.executionId);

      if (response.success) {
        alert(
          `Success!\n\n` +
          `${response.message}\n` +
          `Execution stopped and positions closed.`
        );
      } else {
        alert(`Failed to stop execution:\n\n${response.errorMessage}`);
      }
    } catch (error: any) {
      console.error('Error stopping execution:', error);
      alert(`Failed to stop execution:\n\n${error.message || 'Unknown error'}`);
    }
  };

  const handleExecuteConfirm = async (params: ExecutionParams) => {
    if (!selectedOpportunity) return;

    setIsExecuting(true);
    try {
      const isSpotPerp = selectedOpportunity.strategy === 1;

      const request = {
        symbol: selectedOpportunity.symbol,
        strategy: selectedOpportunity.strategy,
        exchange: isSpotPerp ? selectedOpportunity.exchange : '',
        longExchange: !isSpotPerp ? selectedOpportunity.longExchange : undefined,
        shortExchange: !isSpotPerp ? selectedOpportunity.shortExchange : undefined,
        positionSizeUsd: params.positionSizeUsd,
        leverage: params.leverage,
        stopLossPercentage: params.stopLossPercentage,
        takeProfitPercentage: params.takeProfitPercentage,
        // Add funding rate information for ArbitrageOpportunity record
        fundingRate: isSpotPerp ? (selectedOpportunity.fundingRate || 0) : 0,
        longFundingRate: !isSpotPerp ? selectedOpportunity.longFundingRate : undefined,
        shortFundingRate: !isSpotPerp ? selectedOpportunity.shortFundingRate : undefined,
        spreadRate: isSpotPerp
          ? Math.abs(selectedOpportunity.fundingRate || 0)
          : Math.abs((selectedOpportunity.longFundingRate || 0) - (selectedOpportunity.shortFundingRate || 0)),
        annualizedSpread: isSpotPerp
          ? (selectedOpportunity.fundingRate || 0) * 3 * 365
          : selectedOpportunity.annualizedSpread || 0,
        estimatedProfitPercentage: isSpotPerp
          ? selectedOpportunity.estimatedProfitPercentage || 0
          : (selectedOpportunity.annualizedSpread || 0) * 100,
      };

      const response = await apiService.executeOpportunity(request);

      if (response.success) {
        alert(`Success!\n\n${response.message}\n\nPositions created: ${response.positionIds.length}\nOrders: ${response.orderIds.join(', ')}`);
        setIsDialogOpen(false);
      } else {
        alert(`Execution failed:\n\n${response.errorMessage}`);
      }
    } catch (error: any) {
      console.error('Error executing opportunity:', error);
      alert(`Failed to execute opportunity:\n\n${error.message || 'Unknown error'}`);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <>
      <Card className="h-full flex flex-col">
        <CardHeader className="p-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Target className="w-4 h-4 text-binance-yellow" />
              Arbitrage Opportunities
            </CardTitle>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-binance-text-secondary text-sm">
                <Clock className="w-4 h-4" />
                <span className="font-mono">{timeUntilFunding}</span>
              </div>
              <Badge variant="info" size="sm">
                {activeOpportunities.length} Active
              </Badge>
            </div>
          </div>
        </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-0">
        {activeOpportunities.length === 0 ? (
          <EmptyState
            icon={<Target className="w-12 h-12" />}
            title="No opportunities found"
            description="Scanning markets for profitable arbitrage opportunities"
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow hover={false}>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Spot Price</TableHead>
                <TableHead className="text-right">Perp Price</TableHead>
                <TableHead className="text-right">8h Rate</TableHead>
                <TableHead className="text-right">APR</TableHead>
                <TableHead className="text-right">Status</TableHead>
                <TableHead className="text-right">Time</TableHead>
                <TableHead className="text-right">Est. Funding Fee</TableHead>
                <TableHead className="text-right">P&L</TableHead>
                <TableHead className="text-right">Est. Total P&L</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {activeOpportunities.map((opp, index) => {
                // Determine if this is spot-perpetual or cross-exchange
                const isSpotPerp = opp.strategy === 1; // SpotPerpetual = 1
                const uniqueKey = `${opp.symbol}-${opp.exchange || opp.longExchange}-${index}`;

                // Find matching positions for this opportunity
                const matchingPositions = positions.filter(
                  (p) =>
                    p.status === PositionStatus.Open &&
                    p.symbol === opp.symbol &&
                    (isSpotPerp ? p.exchange === opp.exchange : true)
                );

                const isExecuting = matchingPositions.length > 0;
                const totalPnL = matchingPositions.reduce((sum, p) => sum + p.unrealizedPnL, 0);
                const executionKey = isSpotPerp ? `${opp.symbol}-${opp.exchange}` : `${opp.symbol}-${opp.longExchange}`;

                // Calculate estimated funding fee for next settlement
                // Note: Binance pays the FULL funding fee if you hold at settlement time (00:00, 08:00, 16:00 UTC)
                // regardless of when you opened the position during the 8-hour period
                let estimatedEarnings = 0;
                if (isExecuting && matchingPositions.length > 0) {
                  // Funding rate for the period (funding happens every 8 hours)
                  const fundingRate = isSpotPerp ? opp.fundingRate : (opp.longFundingRate + opp.shortFundingRate) / 2;

                  // ONLY perpetual positions pay/receive funding fees (spot positions don't)
                  // Import PositionType to filter correctly
                  const perpPositions = matchingPositions.filter((p) => p.type === 0); // 0 = Perpetual
                  const perpPositionValue = perpPositions.reduce((sum, p) => sum + (p.quantity * p.entryPrice), 0);

                  // Calculate the FULL funding fee you'll receive at next settlement
                  // Negative funding rate = you RECEIVE funding (short pays long)
                  // Positive funding rate = you PAY funding (long pays short)
                  // Since we're SHORT perpetual, negative rate means we receive (positive earnings)
                  estimatedEarnings = -fundingRate * perpPositionValue;
                }

                return (
                  <TableRow key={uniqueKey}>
                    <TableCell className="font-bold">{opp.symbol}</TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-sm font-semibold text-binance-text">
                        {isSpotPerp ? `$${opp.spotPrice?.toFixed(6) || '-'}` : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-sm font-semibold text-binance-text">
                        {isSpotPerp ? `$${opp.perpetualPrice?.toFixed(6) || '-'}` : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={`font-mono text-sm font-bold ${
                        isSpotPerp
                          ? (opp.fundingRate >= 0 ? 'text-binance-green' : 'text-binance-red')
                          : 'text-binance-text-secondary'
                      }`}>
                        {isSpotPerp
                          ? `${(opp.fundingRate * 100).toFixed(4)}%`
                          : `${((opp.longFundingRate + opp.shortFundingRate) / 2 * 100).toFixed(4)}%`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant="success" size="sm">
                        <span className="font-mono font-bold">
                          {isSpotPerp
                            ? (opp.estimatedProfitPercentage).toFixed(2)
                            : (opp.annualizedSpread * 100).toFixed(2)}%
                        </span>
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {isExecuting ? (
                        <Badge variant="info" size="sm">
                          Executing
                        </Badge>
                      ) : (
                        <Badge variant="secondary" size="sm">
                          Ready
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-sm text-binance-text-secondary">
                        {isExecuting ? executionTimes[executionKey] || '-' : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      {isExecuting ? (
                        <span
                          className={`font-mono text-sm font-bold ${
                            estimatedEarnings >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {estimatedEarnings >= 0 ? '+' : ''}${estimatedEarnings.toFixed(2)}
                        </span>
                      ) : (
                        <span className="font-mono text-sm text-binance-text-secondary">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {isExecuting ? (
                        <span
                          className={`font-mono text-sm font-bold ${
                            totalPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                        </span>
                      ) : (
                        <span className="font-mono text-sm text-binance-text-secondary">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {isExecuting ? (
                        <span
                          className={`font-mono text-sm font-bold ${
                            (totalPnL + estimatedEarnings) >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {(totalPnL + estimatedEarnings) >= 0 ? '+' : ''}${(totalPnL + estimatedEarnings).toFixed(2)}
                        </span>
                      ) : (
                        <span className="font-mono text-sm text-binance-text-secondary">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {opp.executionId ? (
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => handleStop(opp)}
                          className="gap-1"
                        >
                          <StopCircle className="w-3 h-3" />
                          Stop
                        </Button>
                      ) : (
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => handleExecute(opp)}
                          className="gap-1"
                        >
                          <Play className="w-3 h-3" />
                          Execute
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>

    {selectedOpportunity && (
      <ExecuteDialog
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        onExecute={handleExecuteConfirm}
        opportunity={selectedOpportunity}
      />
    )}
    </>
  );
};
