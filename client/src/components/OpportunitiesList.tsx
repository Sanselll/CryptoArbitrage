import { Target, ArrowUpCircle, ArrowDownCircle, Play, Clock, TrendingUp, TrendingDown, ArrowUpDown, Search } from 'lucide-react';
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

type SortField = 'spread' | 'priceSpread24h' | 'priceSpread3d' | 'fundProfit8h' | 'fundProfit8h3d' | 'fundProfit8h24h' | 'fundApr' | 'fundApr3d' | 'fundApr24h'
  | 'volume' | 'liquidity' | 'posCost' | 'breakEven' | 'fundBreakEven24h' | 'fundBreakEven3d';
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
  const [sortField, setSortField] = useState<SortField>('fundApr');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null); // null means show all
  const [symbolFilter, setSymbolFilter] = useState<string>(''); // Symbol search filter
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

  // No need to calculate - all metrics come from backend now!
  const opportunitiesWithCalculations = opportunities.map(opp => {
    // Calculate spread for display only (not used in backend)
    const spotPrice = opp.spotPrice || 0;
    const perpPrice = opp.perpetualPrice || 0;
    const spread = (spotPrice > 0 && perpPrice > 0)
      ? ((perpPrice - spotPrice) / spotPrice) * 100
      : 0;

    return {
      ...opp,
      _calculated: { spread }  // Only keep spread for UI display
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

      // Filter by symbol (case-insensitive partial match)
      if (symbolFilter && !opp.symbol.toLowerCase().includes(symbolFilter.toLowerCase())) return false;

      return true; // Show all detected opportunities, including those being executed
    })
    .sort((a, b) => {
      let aValue: number, bValue: number;

      switch (sortField) {
        case 'spread':
          aValue = a._calculated.spread;
          bValue = b._calculated.spread;
          break;
        case 'priceSpread24h':
          aValue = a.priceSpread24hAvg ?? -Infinity;
          bValue = b.priceSpread24hAvg ?? -Infinity;
          break;
        case 'priceSpread3d':
          aValue = a.priceSpread3dAvg ?? -Infinity;
          bValue = b.priceSpread3dAvg ?? -Infinity;
          break;
        case 'fundProfit8h':
          aValue = a.fundProfit8h;
          bValue = b.fundProfit8h;
          break;
        case 'fundProfit8h3d':
          aValue = a.fundProfit8h3dProj ?? -Infinity;
          bValue = b.fundProfit8h3dProj ?? -Infinity;
          break;
        case 'fundProfit8h24h':
          aValue = a.fundProfit8h24hProj ?? -Infinity;
          bValue = b.fundProfit8h24hProj ?? -Infinity;
          break;
        case 'fundApr':
          aValue = a.fundApr;
          bValue = b.fundApr;
          break;
        case 'fundApr3d':
          aValue = a.fundApr3dProj ?? -Infinity;
          bValue = b.fundApr3dProj ?? -Infinity;
          break;
        case 'fundApr24h':
          aValue = a.fundApr24hProj ?? -Infinity;
          bValue = b.fundApr24hProj ?? -Infinity;
          break;
        case 'volume':
          aValue = a.volume24h ?? 0;
          bValue = b.volume24h ?? 0;
          break;
        case 'liquidity':
          aValue = a.orderbookDepthUsd ?? 0;
          bValue = b.orderbookDepthUsd ?? 0;
          break;
        case 'posCost':
          aValue = a.positionCostPercent;
          bValue = b.positionCostPercent;
          break;
        case 'breakEven':
          aValue = a.breakEvenTimeHours ?? Infinity;
          bValue = b.breakEvenTimeHours ?? Infinity;
          break;
        case 'fundBreakEven24h':
          aValue = a.fundBreakEvenTime24hProj ?? Infinity;
          bValue = b.fundBreakEvenTime24hProj ?? Infinity;
          break;
        case 'fundBreakEven3d':
          aValue = a.fundBreakEvenTime3dProj ?? Infinity;
          bValue = b.fundBreakEvenTime3dProj ?? Infinity;
          break;
        default:
          aValue = a.fundApr;
          bValue = b.fundApr;
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    })
    .slice(0, 200); // Show top 200 opportunities

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
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <CardTitle className="flex items-center gap-1.5 text-sm">
                <Target className="w-3 h-3 text-binance-yellow" />
                Arbitrage Opportunities
              </CardTitle>
              {/* Symbol Filter */}
              <div className="relative">
                <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-binance-text-secondary pointer-events-none" />
                <input
                  type="text"
                  placeholder="Filter symbol..."
                  value={symbolFilter}
                  onChange={(e) => setSymbolFilter(e.target.value)}
                  className="w-40 pl-7 pr-2 py-1 text-xs bg-binance-bg-tertiary border border-binance-border rounded text-binance-text placeholder:text-binance-text-secondary/50 focus:outline-none focus:border-binance-yellow/50 focus:ring-1 focus:ring-binance-yellow/20 transition-all"
                />
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
            <Badge variant="info" size="sm" className="text-[10px]">
              {activeOpportunities.length} Active
            </Badge>
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
                  <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border" title="trading pair (e.g., BTCUSDT)">Symbol</TableHead>
                  <TableHead className="sticky left-[80px] z-40 bg-binance-bg-secondary border-r border-binance-border" title="arbitrage strategy type">Type</TableHead>
                  <TableHead className="sticky left-[145px] z-40 bg-binance-bg-secondary border-r border-binance-border" title="exchange where position is held">Exchange</TableHead>
                  <TableHead className="py-1" title="long (buy) or short (sell) position">Side</TableHead>
                  <TableHead className="text-right" title="current funding rate (% per interval)">Fund Rate</TableHead>
                  <TableHead className="text-right" title="24-hour time-weighted average funding rate">Fund Rate (24h)</TableHead>
                  <TableHead className="text-right" title="3-day time-weighted average funding rate">Fund Rate (3D)</TableHead>
                  <TableHead className="text-right w-[40px]" title="funding interval in hours (1h, 4h, 8h)">Fund Int</TableHead>
                  <TableHead className="text-right min-w-[100px]" title="countdown to next funding payment">Next Funding</TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('volume')} title="24-hour trading volume in USDT"><div className="flex items-center justify-end gap-1">
                        24h Volume
                        {sortField === 'volume' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('liquidity')} title="orderbook depth and bid-ask spread quality"><div className="flex items-center justify-end gap-1">
                        Liquidity
                        {sortField === 'liquidity' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('spread')} title="price difference between spot and perpetual (%)"><div className="flex items-center justify-end gap-1">
                        Spread
                        {sortField === 'spread' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('priceSpread24h')} title="24-hour average price spread (%)"><div className="flex items-center justify-end gap-1">
                        Spread (24h AVG)
                        {sortField === 'priceSpread24h' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('priceSpread3d')} title="3-day average price spread (%)"><div className="flex items-center justify-end gap-1">
                        Spread (3D AVG)
                        {sortField === 'priceSpread3d' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('posCost')} title="cost to open + close position (trading fees %)"><div className="flex items-center justify-end gap-1">
                        Pos Cost
                        {sortField === 'posCost' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('breakEven')} title="hours to recover position cost at current rate"><div className="flex items-center justify-end gap-1">
                        Break Even
                        {sortField === 'breakEven' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundBreakEven24h')} title="break-even time projected from 24h average rate"><div className="flex items-center justify-end gap-1">
                        Break Even (24h PROJ)
                        {sortField === 'fundBreakEven24h' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundBreakEven3d')} title="break-even time projected from 3D average rate"><div className="flex items-center justify-end gap-1">
                        Break Even (3D PROJ)
                        {sortField === 'fundBreakEven3d' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundProfit8h')} title="estimated profit % per 8-hour period at current rate"><div className="flex items-center justify-end gap-1">
                        Fund 8h
                        {sortField === 'fundProfit8h' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundProfit8h24h')} title="projected 8h profit based on 24h average rate"><div className="flex items-center justify-end gap-1">
                        Fund 8h (24h PROJ)
                        {sortField === 'fundProfit8h24h' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundProfit8h3d')} title="projected 8h profit based on 3D average rate"><div className="flex items-center justify-end gap-1">
                        Fund 8h (3D PROJ)
                        {sortField === 'fundProfit8h3d' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundApr')} title="annualized percentage rate at current funding rate"><div className="flex items-center justify-end gap-1">
                        Fund APR
                        {sortField === 'fundApr' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundApr24h')} title="projected APR based on 24h average rate"><div className="flex items-center justify-end gap-1">
                        Fund APR (24h PROJ)
                        {sortField === 'fundApr24h' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
                  onClick={() => handleSort('fundApr3d')} title="projected APR based on 3D average rate"><div className="flex items-center justify-end gap-1">
                        Fund APR (3D PROJ)
                        {sortField === 'fundApr3d' && (
                          <ArrowUpDown className="w-3 h-3" />
                        )}
                      </div></TableHead>
                <TableHead className="sticky right-0 z-40 bg-binance-bg-secondary border-l border-binance-border text-right" title="execute or view opportunity details">Action</TableHead>
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

                // All metrics now come from backend - no calculations needed!
                const profit8h = opp.fundProfit8h;
                const apr = opp.fundApr;
                const profit8h24h = opp.fundProfit8h24hProj;
                const apr24h = opp.fundApr24hProj;
                const profit8h3d = opp.fundProfit8h3dProj;
                const apr3d = opp.fundApr3dProj;

                // Calculate spread for display only
                const spotPrice = opp.spotPrice || 0;
                const perpPrice = opp.perpetualPrice || 0;
                const spread = (spotPrice > 0 && perpPrice > 0)
                  ? ((perpPrice - spotPrice) / spotPrice) * 100
                  : 0;

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
                    <TableCell className={`sticky left-0 z-20 border-r border-binance-border font-bold text-xs ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>{opp.symbol}</TableCell>
                    <TableCell rowSpan={2} className={`sticky left-[80px] z-20 border-r border-binance-border ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
                      <Badge
                        size="sm"
                        className={`text-[10px] font-mono ${strategyLabel.color}`}
                        title={strategyLabel.fullText}
                      >
                        {strategyLabel.text}
                      </Badge>
                    </TableCell>
                    <TableCell className={`sticky left-[145px] z-20 border-r border-binance-border ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
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
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px]">
                        {longFundingData
                          ? `${(longFundingData.rate * 100).toFixed(4)}%`
                          : (isSpotPerp
                            ? `${(opp.fundingRate * 100).toFixed(4)}%`
                            : `${(opp.longFundingRate * 100).toFixed(4)}%`)
                        }
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {longFundingData?.average24hRate
                          ? `${(longFundingData.average24hRate * 100).toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {longFundingData?.average3DayRate
                          ? `${(longFundingData.average3DayRate * 100).toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right w-[40px]">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {longFundingData?.fundingIntervalHours || 8}h
                      </span>
                    </TableCell>
                    <TableCell className="text-right min-w-[100px]">
                      <div className="flex items-center justify-end gap-0.5 whitespace-nowrap">
                        <Clock className="w-2.5 h-2.5 text-binance-text-secondary" />
                        <span className="font-mono text-[11px] text-binance-text-secondary">
                          {getExchangeFundingTime(longFundingData?.nextFundingTime)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.longVolume24h && opp.longVolume24h > 0
                          ? `$${(opp.longVolume24h / 1000000).toFixed(2)}M`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
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
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] font-bold ${
                        spread >= 0 ? 'text-binance-green' : 'text-binance-red'
                      }`}>
                        {spread !== 0 ? `${spread >= 0 ? '+' : ''}${spread.toFixed(4)}%` : '-'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] ${
                        opp.priceSpread24hAvg !== null && opp.priceSpread24hAvg !== undefined
                          ? opp.priceSpread24hAvg >= 0 ? 'text-binance-green' : 'text-binance-red'
                          : 'text-binance-text-secondary'
                      }`}>
                        {opp.priceSpread24hAvg !== null && opp.priceSpread24hAvg !== undefined
                          ? `${opp.priceSpread24hAvg >= 0 ? '+' : ''}${opp.priceSpread24hAvg.toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] ${
                        opp.priceSpread3dAvg !== null && opp.priceSpread3dAvg !== undefined
                          ? opp.priceSpread3dAvg >= 0 ? 'text-binance-green' : 'text-binance-red'
                          : 'text-binance-text-secondary'
                      }`}>
                        {opp.priceSpread3dAvg !== null && opp.priceSpread3dAvg !== undefined
                          ? `${opp.priceSpread3dAvg >= 0 ? '+' : ''}${opp.priceSpread3dAvg.toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.positionCostPercent.toFixed(2)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.breakEvenTimeHours !== null && opp.breakEvenTimeHours !== undefined
                          ? `${Math.round(opp.breakEvenTimeHours)}h`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.fundBreakEvenTime24hProj !== null && opp.fundBreakEvenTime24hProj !== undefined
                          ? `${Math.round(opp.fundBreakEvenTime24hProj)}h`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {opp.fundBreakEvenTime3dProj !== null && opp.fundBreakEvenTime3dProj !== undefined
                          ? `${Math.round(opp.fundBreakEvenTime3dProj)}h`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] font-bold ${
                        profit8h >= 0 ? 'text-binance-green' : 'text-binance-red'
                      }`}>
                        {`${profit8h >= 0 ? '+' : ''}${profit8h.toFixed(4)}%`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] font-bold ${
                        profit8h24h !== null
                          ? profit8h24h >= 0 ? 'text-binance-green' : 'text-binance-red'
                          : 'text-binance-text-secondary'
                      }`}>
                        {profit8h24h !== null
                          ? `${profit8h24h >= 0 ? '+' : ''}${profit8h24h.toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <span className={`font-mono text-[11px] font-bold ${
                        profit8h3d !== null
                          ? profit8h3d >= 0 ? 'text-binance-green' : 'text-binance-red'
                          : 'text-binance-text-secondary'
                      }`}>
                        {profit8h3d !== null
                          ? `${profit8h3d >= 0 ? '+' : ''}${profit8h3d.toFixed(4)}%`
                          : '--'}
                      </span>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      <Badge variant={apr >= 0 ? "success" : "danger"} size="sm" className="text-[10px]">
                        <span className="font-mono font-bold">
                          {apr.toFixed(2)}%
                        </span>
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      {apr24h !== null ? (
                        <Badge variant={apr24h >= 0 ? "success" : "danger"} size="sm" className="text-[10px]">
                          <span className="font-mono font-bold">
                            {apr24h.toFixed(2)}%
                          </span>
                        </Badge>
                      ) : (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right" rowSpan={2}>
                      {apr3d !== null ? (
                        <Badge variant={apr3d >= 0 ? "success" : "danger"} size="sm" className="text-[10px]">
                          <span className="font-mono font-bold">
                            {apr3d.toFixed(2)}%
                          </span>
                        </Badge>
                      ) : (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      )}
                    </TableCell>
                    <TableCell className={`sticky right-0 z-20 border-l border-binance-border text-right ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`} rowSpan={2}>
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
                    <TableCell className={`sticky left-[145px] z-20 border-r border-binance-border ${isHovered ? 'bg-[#2b3139]' : 'bg-binance-bg-secondary'}`}>
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
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : (
                          shortFundingData
                            ? `${(shortFundingData.rate * 100).toFixed(4)}%`
                            : `${(opp.shortFundingRate * 100).toFixed(4)}%`
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : (
                          shortFundingData?.average24hRate
                            ? `${(shortFundingData.average24hRate * 100).toFixed(4)}%`
                            : '--'
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : (
                          shortFundingData?.average3DayRate
                            ? `${(shortFundingData.average3DayRate * 100).toFixed(4)}%`
                            : '--'
                        )}
                      </span>
                    </TableCell>
                    <TableCell className="text-right w-[40px]">
                      <span className="font-mono text-[11px] text-binance-text-secondary">
                        {isSpotPerp ? '--' : `${shortFundingData?.fundingIntervalHours || 8}h`}
                      </span>
                    </TableCell>
                    <TableCell className="text-right min-w-[100px]">
                      {isSpotPerp ? (
                        <span className="font-mono text-[11px] text-binance-text-secondary">--</span>
                      ) : (
                        <div className="flex items-center justify-end gap-0.5 whitespace-nowrap">
                          <Clock className="w-2.5 h-2.5 text-binance-text-secondary" />
                          <span className="font-mono text-[11px] text-binance-text-secondary">
                            {getExchangeFundingTime(shortFundingData?.nextFundingTime)}
                          </span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
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
