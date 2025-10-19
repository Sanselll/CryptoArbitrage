import { Target, ArrowUpCircle, ArrowDownCircle, Play, Clock, StopCircle, TrendingUp, TrendingDown } from 'lucide-react';
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

// Helper function to get exchange-specific time until next funding
const getExchangeFundingTime = (nextFundingTimeStr?: string): string => {
  if (!nextFundingTimeStr) {
    // Fallback to default calculation
    return formatTimeUntil(getNextFundingTime());
  }
  const nextFundingDate = new Date(nextFundingTimeStr);
  return formatTimeUntil(nextFundingDate);
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
  const { opportunities, positions, fundingRates } = useArbitrageStore();
  const [timeUntilFunding, setTimeUntilFunding] = useState('');
  const [executionTimes, setExecutionTimes] = useState<{ [key: string]: string }>({});
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
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
    .slice(0, 50); // Show top 50 opportunities

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
            <Badge variant="info" size="sm" className="text-[10px]">
              {activeOpportunities.length} Active
            </Badge>
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
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Fee Rate</TableHead>
                <TableHead className="text-right">Fee Interval</TableHead>
                <TableHead className="text-right">Next Funding</TableHead>
                <TableHead className="text-right">24h Volume</TableHead>
                <TableHead className="text-right">Spread</TableHead>
                <TableHead className="text-right">8h Profit</TableHead>
                <TableHead className="text-right">APR</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {activeOpportunities.map((opp, index) => {
                // Determine if this is spot-perpetual or cross-exchange
                const isSpotPerp = opp.strategy === 1; // SpotPerpetual = 1
                const isCrossFut = opp.subType === StrategySubType.CrossExchangeFuturesFutures;
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

                // Calculate 8-hour profit and spread for merged cells
                const apr = isSpotPerp
                  ? opp.estimatedProfitPercentage
                  : opp.annualizedSpread * 100;
                const profit8h = apr / 365 / 3;

                const spotPrice = opp.spotPrice || 0;
                const perpPrice = opp.perpetualPrice || 0;
                const spread = (spotPrice > 0 && perpPrice > 0)
                  ? ((perpPrice - spotPrice) / spotPrice) * 100
                  : 0;

                // Get funding rates for each exchange
                const longFundingData = fundingRates.find(fr =>
                  fr.symbol === opp.symbol &&
                  fr.exchange === (isSpotPerp ? opp.exchange : opp.longExchange)
                );
                const shortFundingData = isCrossFut ? fundingRates.find(fr =>
                  fr.symbol === opp.symbol &&
                  fr.exchange === opp.shortExchange
                ) : null;

                const rows = [];

                const isHovered = hoveredRow === uniqueKey;

                // First row (perp position for spot-perp, or long exchange for cross-exchange)
                rows.push(
                  <TableRow
                    key={`${uniqueKey}-1`}
                    className={`border-b-0 ${isHovered ? 'bg-[rgba(43,49,57,0.4)]' : ''}`}
                    hover={false}
                    onMouseEnter={() => setHoveredRow(uniqueKey)}
                    onMouseLeave={() => setHoveredRow(null)}
                  >
                    <TableCell className="font-bold text-xs py-1" rowSpan={2}>{opp.symbol}</TableCell>
                    <TableCell rowSpan={2} className="py-1">
                      <Badge
                        size="sm"
                        className={`text-[10px] ${strategyLabel.color}`}
                      >
                        {strategyLabel.text}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-1">
                      <ExchangeBadge exchange={isSpotPerp ? opp.exchange : opp.longExchange} />
                    </TableCell>
                    <TableCell className="py-1">
                      <Badge
                        variant={isSpotPerp ? "danger" : (isCrossFut ? "success" : "danger")}
                        size="sm"
                        className="gap-0.5 text-[10px]"
                      >
                        {isSpotPerp ? (
                          <>
                            <TrendingDown className="w-2.5 h-2.5" />
                            Short
                          </>
                        ) : (isCrossFut ? (
                          <>
                            <TrendingUp className="w-2.5 h-2.5" />
                            Long
                          </>
                        ) : (
                          <>
                            <TrendingDown className="w-2.5 h-2.5" />
                            Short
                          </>
                        ))}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px]">
                        {longFundingData
                          ? `${(longFundingData.rate * 100).toFixed(4)}%`
                          : (isSpotPerp
                            ? `${(opp.fundingRate * 100).toFixed(4)}%`
                            : `${(opp.longFundingRate * 100).toFixed(4)}%`)
                        }
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {longFundingData?.fundingIntervalHours || 8}h
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <div className="flex items-center justify-end gap-0.5">
                        <Clock className="w-2.5 h-2.5 text-binance-text-secondary" />
                        <span className="font-mono text-[11px] text-binance-text-secondary">
                          {getExchangeFundingTime(longFundingData?.nextFundingTime)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {longFundingData?.volume24h ? `$${(longFundingData.volume24h / 1000000).toFixed(2)}M` : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <span className="font-mono text-[11px] font-bold text-binance-text-primary">
                        {spread !== 0 ? `${spread >= 0 ? '+' : ''}${spread.toFixed(4)}%` : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <span className="font-mono text-[11px] font-bold text-binance-green">
                        {`${profit8h >= 0 ? '+' : ''}${profit8h.toFixed(4)}%`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <Badge variant="success" size="sm" className="text-[10px]">
                        <span className="font-mono font-bold">
                          {apr.toFixed(2)}%
                        </span>
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
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

                // Second row (spot position for spot-perp, or short exchange for cross-exchange)
                rows.push(
                  <TableRow
                    key={`${uniqueKey}-2`}
                    className={`border-t border-binance-border/30 ${isHovered ? 'bg-[rgba(43,49,57,0.4)]' : ''}`}
                    hover={false}
                    onMouseEnter={() => setHoveredRow(uniqueKey)}
                    onMouseLeave={() => setHoveredRow(null)}
                  >
                    <TableCell className="py-1">
                      <ExchangeBadge exchange={isSpotPerp ? opp.exchange : (isCrossFut ? opp.shortExchange : opp.shortExchange)} />
                    </TableCell>
                    <TableCell className="py-1">
                      <Badge
                        variant={isSpotPerp ? "success" : (isCrossFut ? "danger" : "success")}
                        size="sm"
                        className="gap-0.5 text-[10px]"
                      >
                        {isSpotPerp ? (
                          <>
                            <TrendingUp className="w-2.5 h-2.5" />
                            Long
                          </>
                        ) : (isCrossFut ? (
                          <>
                            <TrendingDown className="w-2.5 h-2.5" />
                            Short
                          </>
                        ) : (
                          <>
                            <TrendingUp className="w-2.5 h-2.5" />
                            Long
                          </>
                        ))}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : (
                          shortFundingData
                            ? `${(shortFundingData.rate * 100).toFixed(4)}%`
                            : `${(opp.shortFundingRate * 100).toFixed(4)}%`
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : `${shortFundingData?.fundingIntervalHours || 8}h`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1">
                      {isSpotPerp ? (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      ) : (
                        <div className="flex items-center justify-end gap-0.5">
                          <Clock className="w-2.5 h-2.5 text-binance-text-secondary" />
                          <span className="font-mono text-[11px] text-binance-text-secondary">
                            {getExchangeFundingTime(shortFundingData?.nextFundingTime)}
                          </span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right py-1">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : (shortFundingData?.volume24h ? `$${(shortFundingData.volume24h / 1000000).toFixed(2)}M` : '--')}
                      </span>
                    </TableCell>
                  </TableRow>
                );

                return rows;
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
