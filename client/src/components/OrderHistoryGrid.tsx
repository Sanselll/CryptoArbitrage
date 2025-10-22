import { FileText, ArrowUpDown } from 'lucide-react';
import { useState, useMemo } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { OrderType, OrderSide, OrderStatus } from '../types/index';
import { EmptyState } from './ui/EmptyState';
import { ExchangeBadge } from './ui/ExchangeBadge';
import { Badge } from './ui/Badge';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';

const getOrderTypeLabel = (type: OrderType): string => {
  switch (type) {
    case OrderType.Limit: return 'Limit';
    case OrderType.Market: return 'Market';
    case OrderType.StopLoss: return 'Stop Loss';
    case OrderType.StopLossLimit: return 'Stop Loss Limit';
    case OrderType.TakeProfit: return 'Take Profit';
    case OrderType.TakeProfitLimit: return 'Take Profit Limit';
    default: return 'Unknown';
  }
};

const getOrderStatusBadge = (status: OrderStatus) => {
  switch (status) {
    case OrderStatus.New:
      return <Badge variant="blue" size="sm" className="text-[10px]">New</Badge>;
    case OrderStatus.PartiallyFilled:
      return <Badge variant="yellow" size="sm" className="text-[10px]">Partially Filled</Badge>;
    case OrderStatus.Filled:
      return <Badge variant="green" size="sm" className="text-[10px]">Filled</Badge>;
    case OrderStatus.Canceled:
      return <Badge variant="gray" size="sm" className="text-[10px]">Canceled</Badge>;
    case OrderStatus.PendingCancel:
      return <Badge variant="yellow" size="sm" className="text-[10px]">Pending Cancel</Badge>;
    case OrderStatus.Rejected:
      return <Badge variant="red" size="sm" className="text-[10px]">Rejected</Badge>;
    case OrderStatus.Expired:
      return <Badge variant="gray" size="sm" className="text-[10px]">Expired</Badge>;
    default:
      return <Badge variant="gray" size="sm" className="text-[10px]">Unknown</Badge>;
  }
};

type SortField = 'updatedAt' | 'price' | 'quantity';
type SortDirection = 'asc' | 'desc';

export const OrderHistoryGrid = () => {
  const { orderHistory } = useArbitrageStore();
  const [sortField, setSortField] = useState<SortField>('updatedAt');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedOrders = useMemo(() => {
    if (!orderHistory || orderHistory.length === 0) return [];

    const sorted = [...orderHistory].sort((a, b) => {
      let aValue: number, bValue: number;

      switch (sortField) {
        case 'updatedAt':
          aValue = new Date(a.updatedAt || a.createdAt).getTime();
          bValue = new Date(b.updatedAt || b.createdAt).getTime();
          break;
        case 'price':
          aValue = a.price || 0;
          bValue = b.price || 0;
          break;
        case 'quantity':
          aValue = a.quantity || 0;
          bValue = b.quantity || 0;
          break;
        default:
          aValue = new Date(a.updatedAt || a.createdAt).getTime();
          bValue = new Date(b.updatedAt || b.createdAt).getTime();
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    });

    return sorted.slice(0, 100); // Limit to 100 items
  }, [orderHistory, sortField, sortDirection]);

  if (!orderHistory || orderHistory.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="h-12 w-12" />}
        title="No Order History"
        description="You don't have any order history at the moment."
      />
    );
  }

  return (
    <div className="h-full overflow-x-auto">
      <Table>
        <TableHeader className="sticky top-0 z-30">
          <TableRow hover={false}>
            <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border">Exchange</TableHead>
            <TableHead className="sticky left-[120px] z-40 bg-binance-bg-secondary border-r border-binance-border">Symbol</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Side</TableHead>
            <TableHead>Status</TableHead>
            <TableHead
              className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('price')}
            >
              <div className="flex items-center justify-end gap-1">
                Price
                {sortField === 'price' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
            <TableHead className="text-right">Stop Price</TableHead>
            <TableHead
              className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('quantity')}
            >
              <div className="flex items-center justify-end gap-1">
                Quantity
                {sortField === 'quantity' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
            <TableHead className="text-right">Filled</TableHead>
            <TableHead className="text-right">Remaining</TableHead>
            <TableHead>Created At</TableHead>
            <TableHead
              className="cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('updatedAt')}
            >
              <div className="flex items-center gap-1">
                Updated At
                {sortField === 'updatedAt' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedOrders.map((order, index) => {
            const fillPercentage = order.quantity > 0 ? (order.filledQuantity / order.quantity) * 100 : 0;

            return (
              <TableRow key={`${order.exchange}-${order.orderId}-${index}`}>
                <TableCell className="sticky left-0 z-20 bg-binance-bg-secondary border-r border-binance-border">
                  <ExchangeBadge exchange={order.exchange} />
                </TableCell>
                <TableCell className="sticky left-[120px] z-20 bg-binance-bg-secondary border-r border-binance-border font-medium text-xs">
                  {order.symbol}
                </TableCell>
                <TableCell>
                  <span className="text-[11px] text-gray-300">{getOrderTypeLabel(order.type)}</span>
                </TableCell>
                <TableCell>
                  {order.side === OrderSide.Buy ? (
                    <span className="text-green-400 font-medium text-[11px]">Buy</span>
                  ) : (
                    <span className="text-red-400 font-medium text-[11px]">Sell</span>
                  )}
                </TableCell>
                <TableCell>{getOrderStatusBadge(order.status)}</TableCell>
                <TableCell className="text-right font-mono">
                  <span className="text-[11px]">
                    {order.price != null ? order.price.toFixed(8) : '-'}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono text-sm">
                  <span className="text-[11px]">
                    {order.stopPrice != null ? order.stopPrice.toFixed(8) : '-'}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <span className="text-[11px]">
                    {order.quantity != null ? order.quantity.toFixed(8) : '-'}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono">
                  <div className="flex flex-col items-end">
                    <span className="text-[11px]">
                      {order.filledQuantity != null ? order.filledQuantity.toFixed(8) : '-'}
                    </span>
                    {fillPercentage > 0 && (
                      <span className="text-[10px] text-gray-400">({fillPercentage.toFixed(1)}%)</span>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-right font-mono text-sm">
                  <span className="text-[11px]">
                    {order.remainingQuantity != null ? order.remainingQuantity.toFixed(8) : '-'}
                  </span>
                </TableCell>
                <TableCell className="text-sm text-gray-400">
                  <span className="text-[11px]">
                    {new Date(order.createdAt).toLocaleString()}
                  </span>
                </TableCell>
                <TableCell className="text-sm text-gray-400">
                  <span className="text-[11px]">
                    {order.updatedAt ? new Date(order.updatedAt).toLocaleString() : '-'}
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
};
