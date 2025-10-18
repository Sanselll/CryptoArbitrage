import { Target, ArrowUpCircle, ArrowDownCircle, Play, Clock, StopCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';
import { EmptyState } from './ui/EmptyState';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { ExchangeBadge } from './ui/ExchangeBadge';
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
import { ExecuteDialog, ExecutionParams } from './ExecuteDialog';
import { apiService } from '../services/apiService';
import { PositionStatus, StrategySubType } from '../types/index';
import { useDialog } from '../hooks/useDialog';

// Helper function to get strategy type label
const getStrategyLabel = (subType?: number): { text: string; color: string } => {
  switch (subType) {
    case StrategySubType.SpotPerpetualSameExchange:
    case 0:
      return { text: 'Spot-Perp', color: 'bg-blue-500/20 text-blue-400' };
    case StrategySubType.CrossExchangeFuturesFutures:
    case 1:
      return { text: 'Cross-Fut', color: 'bg-purple-500/20 text-purple-400' };
    case StrategySubType.CrossExchangeSpotFutures:
    case 2:
      return { text: 'Cross-Spot', color: 'bg-green-500/20 text-green-400' };
    default:
      return { text: 'Unknown', color: 'bg-gray-500/20 text-gray-400' };
  }
};

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
  const navigate = useNavigate();
  const { opportunities, positions } = useArbitrageStore();
  const [timeUntilFunding, setTimeUntilFunding] = useState('');
  const [executionTimes, setExecutionTimes] = useState<{ [key: string]: string }>({});
  const [selectedOpportunity, setSelectedOpportunity] = useState<any>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const { alertState, showSuccess, showError, showInfo, closeAlert, confirmState, showConfirm, closeConfirm } = useDialog();

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

  const handleExecute = async (opp: any) => {
    try {
      // Fetch user's connected exchanges
      const userApiKeys = await apiService.getUserApiKeys();
      const connectedExchanges = userApiKeys
        .filter((key) => key.isEnabled)
        .map((key) => key.exchangeName);

      // Determine which exchanges are needed
      const isSpotPerp = opp.strategy === 1;
      const requiredExchanges = isSpotPerp
        ? [opp.exchange]
        : [opp.longExchange, opp.shortExchange];

      // Check if user has all required exchanges connected
      const missingExchanges = requiredExchanges.filter(
        (exchange) => !connectedExchanges.includes(exchange)
      );

      if (missingExchanges.length > 0) {
        showError(
          `To execute this opportunity, you need to connect the following exchange(s):\n\n${missingExchanges.join(', ')}\n\nPlease add your API keys in Profile Settings to continue.`,
          'Exchange Connection Required',
          'Go to Profile Settings',
          () => {
            // Navigate to Profile Settings
            navigate('/profile');
          }
        );
        return;
      }

      // All required exchanges are connected - open dialog
      setSelectedOpportunity(opp);
      setIsDialogOpen(true);
    } catch (error: any) {
      console.error('Error validating exchanges:', error);
      showError(
        `Failed to validate exchange connections: ${error.message}`,
        'Validation Error'
      );
    }
  };

  const handleStop = async (opp: any) => {
    // Check if this opportunity has an execution running
    if (!opp.executionId) {
      showInfo('No execution found for this opportunity', 'No Execution');
      return;
    }

    showConfirm(
      `Symbol: ${opp.symbol}\nExchange: ${opp.exchange}\n\nThis will close all positions and sell the spot asset.`,
      async () => {
        setIsStopping(true);
        try {
          const response = await apiService.stopExecution(opp.executionId);

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
              `${response.message}\n\nExecution stopped and positions closed.`,
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
          setIsStopping(false);
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

  const handleExecuteConfirm = async (params: ExecutionParams) => {
    if (!selectedOpportunity) return;

    setIsExecuting(true);
    try {
      const isSpotPerp = selectedOpportunity.strategy === 1;

      const request = {
        symbol: selectedOpportunity.symbol,
        strategy: selectedOpportunity.strategy,
        subType: selectedOpportunity.subType || 0,  // Include the subType
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
        showSuccess(
          `${response.message}\n\nPositions created: ${response.positionIds.length}\nOrders: ${response.orderIds.join(', ')}`,
          'Success'
        );
        setIsDialogOpen(false);
      } else {
        showError(
          `${response.errorMessage}`,
          'Execution Failed'
        );
      }
    } catch (error: any) {
      console.error('Error executing opportunity:', error);
      showError(
        `${error.message || 'Unknown error'}`,
        'Failed to Execute'
      );
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <>
      <LoadingOverlay
        isLoading={isExecuting}
        message="Executing strategy..."
      />
      <LoadingOverlay
        isLoading={isStopping}
        message="Stopping execution..."
      />

      <Card className="h-full flex flex-col">
        <CardHeader className="p-2">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-1.5 text-sm">
              <Target className="w-3 h-3 text-binance-yellow" />
              Arbitrage Opportunities
            </CardTitle>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 text-binance-text-secondary text-[10px]">
                <Clock className="w-3 h-3" />
                <span className="font-mono">{timeUntilFunding}</span>
              </div>
              <Badge variant="info" size="sm" className="text-[10px]">
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
                <TableHead>Type</TableHead>
                <TableHead>Exchange</TableHead>
                <TableHead className="text-right">Spread</TableHead>
                <TableHead className="text-right">8h Profit</TableHead>
                <TableHead className="text-right">APR</TableHead>
                <TableHead className="text-right">24h Vol</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {activeOpportunities.map((opp, index) => {
                // Determine if this is spot-perpetual or cross-exchange
                const isSpotPerp = opp.strategy === 1; // SpotPerpetual = 1
                const uniqueKey = `${opp.symbol}-${opp.exchange || opp.longExchange}-${index}`;
                const strategyLabel = getStrategyLabel(opp.subType);

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
                    <TableCell className="font-bold text-xs py-1">{opp.symbol}</TableCell>
                    <TableCell className="py-1">
                      <Badge
                        size="sm"
                        className={`text-[10px] ${strategyLabel.color}`}
                      >
                        {strategyLabel.text}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-1">
                      {isSpotPerp ? (
                        <ExchangeBadge exchange={opp.exchange} />
                      ) : (
                        <div className="flex items-center gap-1">
                          <ExchangeBadge exchange={opp.longExchange} />
                          <span className="text-binance-text-muted">/</span>
                          <ExchangeBadge exchange={opp.shortExchange} />
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] font-bold text-binance-text-primary">
                        {(() => {
                          // Calculate spread (price difference)
                          // Based on strategy subtype:
                          // - Spot-Perp: (perpPrice - spotPrice) / spotPrice
                          // - Cross-Fut: (shortExchangePrice - longExchangePrice) / longExchangePrice
                          // - Cross-Spot: (shortExchangeFutPrice - longExchangeSpotPrice) / longExchangeSpotPrice

                          const spotPrice = opp.spotPrice || 0;
                          const perpPrice = opp.perpetualPrice || 0;

                          if (spotPrice > 0 && perpPrice > 0) {
                            // For all types: spotPrice represents the long side, perpPrice represents the short side
                            const spread = ((perpPrice - spotPrice) / spotPrice) * 100;
                            return `${spread >= 0 ? '+' : ''}${spread.toFixed(4)}%`;
                          }
                          return '-';
                        })()}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] font-bold text-binance-green">
                        {(() => {
                          // Calculate 8-hour profit: APR / 365 days / 3 funding periods per day
                          const apr = isSpotPerp
                            ? opp.estimatedProfitPercentage
                            : opp.annualizedSpread * 100;
                          const profit8h = apr / 365 / 3;
                          return `${profit8h >= 0 ? '+' : ''}${profit8h.toFixed(4)}%`;
                        })()}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <Badge variant="success" size="sm" className="text-[10px]">
                        <span className="font-mono font-bold">
                          {isSpotPerp
                            ? (opp.estimatedProfitPercentage).toFixed(2)
                            : (opp.annualizedSpread * 100).toFixed(2)}%
                        </span>
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.volume24h
                          ? `$${(opp.volume24h / 1000000).toFixed(2)}M`
                          : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      {opp.executionId ? (
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => handleStop(opp)}
                          className="gap-0.5 h-6 px-2 text-[10px]"
                        >
                          <StopCircle className="w-2.5 h-2.5" />
                          Stop
                        </Button>
                      ) : isExecuting ? (
                        <span className="text-[11px] text-binance-text-secondary">
                          Executing...
                        </span>
                      ) : (
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => handleExecute(opp)}
                          className="gap-0.5 h-6 px-2 text-[10px]"
                        >
                          <Play className="w-2.5 h-2.5" />
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
        isExecuting={isExecuting}
      />
    )}

    <AlertDialog
      isOpen={alertState.isOpen}
      onClose={closeAlert}
      title={alertState.title}
      message={alertState.message}
      variant={alertState.variant}
      actionText={alertState.actionText}
      onAction={alertState.onAction}
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
