import { Target, ArrowUpCircle, ArrowDownCircle, Play, Clock, TrendingUp, TrendingDown, ArrowUpDown } from 'lucide-react';
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
import { PositionStatus, StrategySubType, LiquidityStatus } from '../types/index';
import { useDialog } from '../hooks/useDialog';

// Helper function to get strategy type label (abbreviated)
const getStrategyLabel = (subType?: number): { text: string; fullText: string; color: string } => {
  switch (subType) {
    case StrategySubType.SpotPerpetualSameExchange:
    case 0:
      return { text: 'SP', fullText: 'Spot-Perp', color: 'bg-blue-500/20 text-blue-400' };
    case StrategySubType.CrossExchangeFuturesFutures:
    case 1:
      return { text: 'CFFF', fullText: 'Cross-Fut Funding', color: 'bg-purple-500/20 text-purple-400' };
    case StrategySubType.CrossExchangeSpotFutures:
    case 2:
      return { text: 'CFSF', fullText: 'Cross-Fut Spot', color: 'bg-green-500/20 text-green-400' };
    case StrategySubType.CrossExchangeFuturesPriceSpread:
    case 3:
      return { text: 'CFPS', fullText: 'Cross-Fut Price Spread', color: 'bg-orange-500/20 text-orange-400' };
    default:
      return { text: '?', fullText: 'Unknown', color: 'bg-gray-500/20 text-gray-400' };
  }
};

// Helper function to get liquidity badge details
const getLiquidityBadge = (status?: LiquidityStatus): { text: string; variant: 'success' | 'warning' | 'danger' } => {
  switch (status) {
    case LiquidityStatus.Good:
      return { text: 'Good', variant: 'success' };
    case LiquidityStatus.Medium:
      return { text: 'Medium', variant: 'warning' };
    case LiquidityStatus.Low:
      return { text: 'Low', variant: 'danger' };
    default:
      return { text: 'Unknown', variant: 'warning' };
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

type SortField = 'spread' | '8hProfit' | '8hProfit3d' | 'apr' | 'apr3d';
type SortDirection = 'asc' | 'desc';

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
  const [sortField, setSortField] = useState<SortField>('apr');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null); // null means show all
  const { alertState, showSuccess, showError, showInfo, closeAlert, confirmState, showConfirm, closeConfirm } = useDialog();

  // Helper function to find matching positions for an opportunity
  const findMatchingPositions = (opp: any) => {
    const isSpotPerp = opp.strategy === 1; // SpotPerpetual
    const matched = positions.filter((pos) => {
      // IMPORTANT: Only match OPEN positions (same logic as PositionsGrid)
      if (pos.status !== PositionStatus.Open) {
        return false;
      }

      if (isSpotPerp) {
        // For spot-perp: match by symbol and exchange
        return pos.symbol === opp.symbol && pos.exchange === opp.exchange;
      } else {
        // For cross-exchange: match by symbol and either longExchange or shortExchange
        return (
          pos.symbol === opp.symbol &&
          (pos.exchange === opp.longExchange || pos.exchange === opp.shortExchange)
        );
      }
    });

    return matched;
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const nextFunding = getNextFundingTime();
      setTimeUntilFunding(formatTimeUntil(nextFunding));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  // Pre-calculate values for all opportunities for sorting
  const opportunitiesWithCalculations = opportunities.map(opp => {
    const isSpotPerp = opp.strategy === 1;
    const isCrossFut = opp.subType === StrategySubType.CrossExchangeFuturesFutures;

    const longFundingData = fundingRates.find(fr =>
      fr.symbol === opp.symbol &&
      fr.exchange === (isSpotPerp ? opp.exchange : opp.longExchange)
    );
    const shortFundingData = isCrossFut ? fundingRates.find(fr =>
      fr.symbol === opp.symbol &&
      fr.exchange === opp.shortExchange
    ) : null;

    // Calculate spread
    const spotPrice = opp.spotPrice || 0;
    const perpPrice = opp.perpetualPrice || 0;
    const spread = (spotPrice > 0 && perpPrice > 0)
      ? ((perpPrice - spotPrice) / spotPrice) * 100
      : 0;

    // Calculate APR
    const apr = isSpotPerp
      ? opp.estimatedProfitPercentage
      : opp.annualizedSpread * 100;

    // Calculate 8H profit
    let profit8h: number;
    if (isSpotPerp) {
      const fundingIntervalHours = longFundingData?.fundingIntervalHours || 8;
      const periodsIn8Hours = 8 / fundingIntervalHours;
      profit8h = (apr / 365) * periodsIn8Hours;
    } else if (isCrossFut) {
      const longInterval = opp.longFundingIntervalHours || longFundingData?.fundingIntervalHours || 8;
      const shortInterval = opp.shortFundingIntervalHours || shortFundingData?.fundingIntervalHours || 8;
      const longDailyRate = (opp.longFundingRate * 100) * (24 / longInterval);
      const shortDailyRate = (opp.shortFundingRate * 100) * (24 / shortInterval);
      const netDailyRate = shortDailyRate - longDailyRate;
      profit8h = netDailyRate / 3;
    } else {
      const shortInterval = opp.shortFundingIntervalHours || shortFundingData?.fundingIntervalHours || 8;
      const periodsIn8Hours = 8 / shortInterval;
      profit8h = (apr / 365) * periodsIn8Hours;
    }

    // Calculate 3D average metrics
    let apr3d: number | null = null;
    let profit8h3d: number | null = null;

    if (longFundingData?.average3DayRate !== undefined && longFundingData?.average3DayRate !== null) {
      if (isSpotPerp) {
        const fundingIntervalHours = longFundingData?.fundingIntervalHours || 8;
        const periodsPerYear = (365 * 24) / fundingIntervalHours;
        apr3d = (longFundingData.average3DayRate * 100) * periodsPerYear;
        const periodsIn8Hours = 8 / fundingIntervalHours;
        profit8h3d = (apr3d / 365) * periodsIn8Hours;
      } else if (isCrossFut && shortFundingData?.average3DayRate !== undefined && shortFundingData?.average3DayRate !== null) {
        const longInterval = longFundingData?.fundingIntervalHours || 8;
        const shortInterval = shortFundingData?.fundingIntervalHours || 8;
        const longDailyRate = (longFundingData.average3DayRate * 100) * (24 / longInterval);
        const shortDailyRate = (shortFundingData.average3DayRate * 100) * (24 / shortInterval);
        const netDailyRate = shortDailyRate - longDailyRate;
        apr3d = netDailyRate * 365;
        profit8h3d = netDailyRate / 3;
      } else if (shortFundingData?.average3DayRate !== undefined && shortFundingData?.average3DayRate !== null) {
        const shortInterval = shortFundingData?.fundingIntervalHours || 8;
        const periodsPerYear = (365 * 24) / shortInterval;
        apr3d = (shortFundingData.average3DayRate * 100) * periodsPerYear;
        const periodsIn8Hours = 8 / shortInterval;
        profit8h3d = (apr3d / 365) * periodsIn8Hours;
      }
    }

    return {
      ...opp,
      _calculated: {
        spread,
        apr,
        profit8h,
        apr3d,
        profit8h3d
      }
    };
  });

  // Get unique strategy types that have opportunities
  const availableStrategies = new Set(
    opportunitiesWithCalculations
      .filter(opp => opp.status === 0)
      .map(opp => opp.subType ?? 0)
  );

  const activeOpportunities = opportunitiesWithCalculations
    .filter((opp) => {
      // Only show Detected opportunities
      if (opp.status !== 0) return false;

      // Filter by selected strategy (null means show all)
      if (selectedStrategy !== null && opp.subType !== selectedStrategy) return false;

      return true; // Show all detected opportunities, including those being executed
    })
    .sort((a, b) => {
      let aValue: number, bValue: number;

      switch (sortField) {
        case 'spread':
          aValue = a._calculated.spread;
          bValue = b._calculated.spread;
          break;
        case '8hProfit':
          aValue = a._calculated.profit8h;
          bValue = b._calculated.profit8h;
          break;
        case '8hProfit3d':
          aValue = a._calculated.profit8h3d ?? -Infinity;
          bValue = b._calculated.profit8h3d ?? -Infinity;
          break;
        case 'apr':
          aValue = a._calculated.apr;
          bValue = b._calculated.apr;
          break;
        case 'apr3d':
          aValue = a._calculated.apr3d ?? -Infinity;
          bValue = b._calculated.apr3d ?? -Infinity;
          break;
        default:
          aValue = a._calculated.apr;
          bValue = b._calculated.apr;
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    })
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

      // Check if opportunity has low liquidity
      if (opp.liquidityStatus === LiquidityStatus.Low) {
        showConfirm(
          `Symbol: ${opp.symbol}\n\n⚠️ WARNING: Low Liquidity Detected\n\n${opp.liquidityWarning || 'This asset has low liquidity which may result in execution failures or unfavorable prices.'}\n\nMarket orders may fail. Consider using limit orders instead.\n\nDo you still want to proceed?`,
          () => {
            // User confirmed - open dialog
            setSelectedOpportunity(opp);
            setIsDialogOpen(true);
          },
          {
            title: 'Low Liquidity Warning',
            confirmText: 'Proceed Anyway',
            cancelText: 'Cancel',
            variant: 'danger',
          }
        );
        return;
      }

      // All required exchanges are connected and liquidity is acceptable - open dialog
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
    // Find matching positions for this opportunity
    const matchingPositions = findMatchingPositions(opp);
    const executionId = matchingPositions[0]?.executionId;

    // Check if this opportunity has an execution running
    if (!executionId) {
      showInfo('No execution found for this opportunity', 'No Execution');
      return;
    }

    showConfirm(
      `Symbol: ${opp.symbol}\nExchange: ${opp.exchange}\n\nThis will close all positions and sell the spot asset.`,
      async () => {
        setIsStopping(true);
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
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-1.5 text-sm">
                <Target className="w-3 h-3 text-binance-yellow" />
                Arbitrage Opportunities
              </CardTitle>
              <Badge variant="info" size="sm" className="text-[10px]">
                {activeOpportunities.length} Active
              </Badge>
            </div>
            {availableStrategies.size > 0 && (
              <div className="flex items-center gap-1 flex-wrap">
                {availableStrategies.size > 1 && (
                  <button
                    onClick={() => setSelectedStrategy(null)}
                    title="Show all strategies"
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-all ${
                      selectedStrategy === null
                        ? 'bg-binance-yellow/20 text-binance-yellow border-binance-yellow'
                        : 'bg-gray-700/20 text-gray-600 opacity-50 border-gray-700'
                    } hover:opacity-100 border`}
                  >
                    ALL
                  </button>
                )}
                {[
                  StrategySubType.SpotPerpetualSameExchange,
                  StrategySubType.CrossExchangeFuturesFutures,
                  StrategySubType.CrossExchangeSpotFutures,
                  StrategySubType.CrossExchangeFuturesPriceSpread
                ]
                  .filter(strategyType => availableStrategies.has(strategyType))
                  .map((strategyType) => {
                    const label = getStrategyLabel(strategyType);
                    const isSelected = selectedStrategy === strategyType;
                    return (
                      <button
                        key={strategyType}
                        onClick={() => setSelectedStrategy(strategyType)}
                        title={label.fullText}
                        className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-all ${
                          isSelected
                            ? `${label.color} border-transparent`
                            : 'bg-gray-700/20 text-gray-600 opacity-50 border-gray-700'
                        } hover:opacity-100 border`}
                      >
                        {label.text}
                      </button>
                    );
                  })}
              </div>
            )}
          </div>
        </CardHeader>

      <CardContent className="flex-1 overflow-x-auto overflow-y-auto p-0">
        {activeOpportunities.length === 0 ? (
          <EmptyState
            icon={<Target className="w-12 h-12" />}
            title="No opportunities found"
            description="Scanning markets for profitable arbitrage opportunities"
          />
        ) : (
          <Table>
            <TableHeader className="sticky top-0 z-30">
              <TableRow hover={false}>
                <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border">Symbol</TableHead>
                <TableHead className="sticky left-[80px] z-40 bg-binance-bg-secondary border-r border-binance-border">Type</TableHead>
                <TableHead className="sticky left-[145px] z-40 bg-binance-bg-secondary border-r border-binance-border">Exchange</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Fee Rate</TableHead>
                <TableHead className="text-right">3D Avg</TableHead>
                <TableHead className="text-right">Fee Interval</TableHead>
                <TableHead className="text-right">Next Funding</TableHead>
                <TableHead className="text-right">24h Volume</TableHead>
                <TableHead className="text-right">Liquidity</TableHead>
                <TableHead
                  className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('spread')}
                >
                  <div className="flex items-center justify-end gap-1">
                    Spread
                    {sortField === 'spread' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('8hProfit')}
                >
                  <div className="flex items-center justify-end gap-1">
                    8h Profit
                    {sortField === '8hProfit' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('8hProfit3d')}
                >
                  <div className="flex items-center justify-end gap-1">
                    8h Profit (3D)
                    {sortField === '8hProfit3d' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('apr')}
                >
                  <div className="flex items-center justify-end gap-1">
                    APR
                    {sortField === 'apr' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('apr3d')}
                >
                  <div className="flex items-center justify-end gap-1">
                    APR (3D)
                    {sortField === 'apr3d' && (
                      <ArrowUpDown className="w-3 h-3" />
                    )}
                  </div>
                </TableHead>
                <TableHead className="sticky right-0 z-40 bg-binance-bg-secondary border-l border-binance-border text-right">Action</TableHead>
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

                // Get funding rates for each exchange
                const longFundingData = fundingRates.find(fr =>
                  fr.symbol === opp.symbol &&
                  fr.exchange === (isSpotPerp ? opp.exchange : opp.longExchange)
                );
                const shortFundingData = isCrossFut ? fundingRates.find(fr =>
                  fr.symbol === opp.symbol &&
                  fr.exchange === opp.shortExchange
                ) : null;

                // Calculate 8-hour profit and spread for merged cells
                const apr = isSpotPerp
                  ? opp.estimatedProfitPercentage
                  : opp.annualizedSpread * 100;

                // Calculate 8H profit based on funding intervals
                let profit8h: number;
                if (isSpotPerp) {
                  // For spot-perp, use the long exchange funding interval
                  const fundingIntervalHours = longFundingData?.fundingIntervalHours || 8;
                  const periodsIn8Hours = 8 / fundingIntervalHours;
                  profit8h = (apr / 365) * periodsIn8Hours;
                } else if (isCrossFut) {
                  // For cross-futures, calculate weighted average based on both intervals
                  const longInterval = opp.longFundingIntervalHours || longFundingData?.fundingIntervalHours || 8;
                  const shortInterval = opp.shortFundingIntervalHours || shortFundingData?.fundingIntervalHours || 8;

                  // Calculate daily earnings from each position
                  const longDailyRate = (opp.longFundingRate * 100) * (24 / longInterval);
                  const shortDailyRate = (opp.shortFundingRate * 100) * (24 / shortInterval);

                  // Net daily rate = short rate - long rate (we earn on short, pay on long)
                  const netDailyRate = shortDailyRate - longDailyRate;

                  // 8H profit = (daily rate / 3)
                  profit8h = netDailyRate / 3;
                } else {
                  // For cross-exchange spot-futures, only short position has funding
                  const shortInterval = opp.shortFundingIntervalHours || shortFundingData?.fundingIntervalHours || 8;
                  const periodsIn8Hours = 8 / shortInterval;
                  profit8h = (apr / 365) * periodsIn8Hours;
                }

                const spotPrice = opp.spotPrice || 0;
                const perpPrice = opp.perpetualPrice || 0;
                const spread = (spotPrice > 0 && perpPrice > 0)
                  ? ((perpPrice - spotPrice) / spotPrice) * 100
                  : 0;

                // Calculate APR and 8H profit based on 3D average rates
                let apr3d: number | null = null;
                let profit8h3d: number | null = null;

                if (longFundingData?.average3DayRate !== undefined && longFundingData?.average3DayRate !== null) {
                  if (isSpotPerp) {
                    // For spot-perp, use the long exchange 3D average
                    const fundingIntervalHours = longFundingData?.fundingIntervalHours || 8;
                    const periodsPerYear = (365 * 24) / fundingIntervalHours;
                    apr3d = (longFundingData.average3DayRate * 100) * periodsPerYear;
                    const periodsIn8Hours = 8 / fundingIntervalHours;
                    profit8h3d = (apr3d / 365) * periodsIn8Hours;
                  } else if (isCrossFut && shortFundingData?.average3DayRate !== undefined && shortFundingData?.average3DayRate !== null) {
                    // For cross-futures, calculate based on both 3D averages
                    const longInterval = longFundingData?.fundingIntervalHours || 8;
                    const shortInterval = shortFundingData?.fundingIntervalHours || 8;

                    // Calculate daily earnings from each position using 3D averages
                    const longDailyRate = (longFundingData.average3DayRate * 100) * (24 / longInterval);
                    const shortDailyRate = (shortFundingData.average3DayRate * 100) * (24 / shortInterval);

                    // Net daily rate = short rate - long rate
                    const netDailyRate = shortDailyRate - longDailyRate;
                    apr3d = netDailyRate * 365;

                    // 8H profit = (daily rate / 3)
                    profit8h3d = netDailyRate / 3;
                  } else if (shortFundingData?.average3DayRate !== undefined && shortFundingData?.average3DayRate !== null) {
                    // For cross-exchange spot-futures, only short position has funding
                    const shortInterval = shortFundingData?.fundingIntervalHours || 8;
                    const periodsPerYear = (365 * 24) / shortInterval;
                    apr3d = (shortFundingData.average3DayRate * 100) * periodsPerYear;
                    const periodsIn8Hours = 8 / shortInterval;
                    profit8h3d = (apr3d / 365) * periodsIn8Hours;
                  }
                }

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
                    <TableCell className={`sticky left-0 z-20 border-r border-binance-border font-bold text-xs py-1 ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>{opp.symbol}</TableCell>
                    <TableCell rowSpan={2} className={`sticky left-[80px] z-20 border-r border-binance-border py-1 ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
                      <Badge
                        size="sm"
                        className={`text-[10px] font-mono ${strategyLabel.color}`}
                        title={strategyLabel.fullText}
                      >
                        {strategyLabel.text}
                      </Badge>
                    </TableCell>
                    <TableCell className={`sticky left-[145px] z-20 border-r border-binance-border py-1 ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
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
                        {longFundingData?.average3DayRate
                          ? `${(longFundingData.average3DayRate * 100).toFixed(4)}%`
                          : '--'}
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
                        {opp.longVolume24h && opp.longVolume24h > 0
                          ? `$${(opp.longVolume24h / 1000000).toFixed(2)}M`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      {opp.liquidityStatus !== undefined ? (
                        <Badge
                          variant={getLiquidityBadge(opp.liquidityStatus).variant}
                          size="sm"
                          className="text-[10px]"
                          title={opp.liquidityWarning || undefined}
                        >
                          {getLiquidityBadge(opp.liquidityStatus).text}
                        </Badge>
                      ) : (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <span className={`font-mono text-[11px] font-bold ${
                        spread >= 0 ? 'text-binance-green' : 'text-binance-red'
                      }`}>
                        {spread !== 0 ? `${spread >= 0 ? '+' : ''}${spread.toFixed(4)}%` : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <span className="font-mono text-[11px] font-bold text-binance-green">
                        {`${profit8h >= 0 ? '+' : ''}${profit8h.toFixed(4)}%`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right py-1" rowSpan={2}>
                      <span className="font-mono text-[11px] font-bold text-binance-green">
                        {profit8h3d !== null
                          ? `${profit8h3d >= 0 ? '+' : ''}${profit8h3d.toFixed(4)}%`
                          : '--'}
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
                      {apr3d !== null ? (
                        <Badge variant="success" size="sm" className="text-[10px]">
                          <span className="font-mono font-bold">
                            {apr3d.toFixed(2)}%
                          </span>
                        </Badge>
                      ) : (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      )}
                    </TableCell>
                    <TableCell className={`sticky right-0 z-20 border-l border-binance-border text-right py-1 ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
                      {findMatchingPositions(opp).length > 0 ? (
                        <Badge
                          variant="success"
                          size="sm"
                          className="text-[10px]"
                        >
                          Executed
                        </Badge>
                      ) : isExecuting ? (
                        <Badge
                          variant="warning"
                          size="sm"
                          className="text-[10px]"
                        >
                          Executing
                        </Badge>
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
                    <TableCell className={`sticky left-[145px] z-20 border-r border-binance-border py-1 ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
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
                        {isSpotPerp ? '--' : (
                          shortFundingData?.average3DayRate
                            ? `${(shortFundingData.average3DayRate * 100).toFixed(4)}%`
                            : '--'
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
                        {isSpotPerp
                          ? '--'
                          : (opp.shortVolume24h && opp.shortVolume24h > 0
                            ? `$${(opp.shortVolume24h / 1000000).toFixed(2)}M`
                            : '--')}
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
