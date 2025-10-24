import { FileText, ArrowUpDown } from 'lucide-react';
import { useState, useMemo } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { OrderSide, TradeRole } from '../types/index';
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

type SortField = 'executedAt' | 'price' | 'quantity';
type SortDirection = 'asc' | 'desc';

export const TradeHistoryGrid = () => {
  const { tradeHistory } = useArbitrageStore();
  const [sortField, setSortField] = useState<SortField>('executedAt');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedTrades = useMemo(() => {
    if (!tradeHistory || tradeHistory.length === 0) return [];

    const sorted = [...tradeHistory].sort((a, b) => {
      let aValue: number, bValue: number;

      switch (sortField) {
        case 'executedAt':
          aValue = new Date(a.executedAt).getTime();
          bValue = new Date(b.executedAt).getTime();
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
          aValue = new Date(a.executedAt).getTime();
          bValue = new Date(b.executedAt).getTime();
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    });

    return sorted.slice(0, 100); // Limit to 100 items
  }, [tradeHistory, sortField, sortDirection]);

  if (!tradeHistory || tradeHistory.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="h-12 w-12" />}
        title="No Trade History"
        description="You don't have any trade history at the moment."
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
            <TableHead>Side</TableHead>
            <TableHead>Role</TableHead>
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
            <TableHead className="text-right">Quote Qty</TableHead>
            <TableHead className="text-right">Commission</TableHead>
            <TableHead>Commission Asset</TableHead>
            <TableHead
              className="cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('executedAt')}
            >
              <div className="flex items-center gap-1">
                Executed At
                {sortField === 'executedAt' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedTrades.map((trade, index) => (
            <TableRow key={`${trade.exchange}-${trade.tradeId}-${index}`}>
              <TableCell className="sticky left-0 z-20 bg-binance-bg-secondary border-r border-binance-border">
                <ExchangeBadge exchange={trade.exchange} />
              </TableCell>
              <TableCell className="sticky left-[120px] z-20 bg-binance-bg-secondary border-r border-binance-border font-medium text-xs">
                {trade.symbol}
              </TableCell>
              <TableCell>
                {trade.side === OrderSide.Buy ? (
                  <span className="text-green-400 font-medium text-[11px]">Buy</span>
                ) : (
                  <span className="text-red-400 font-medium text-[11px]">Sell</span>
                )}
              </TableCell>
              <TableCell>
                {trade.role === TradeRole.Maker ? (
                  <Badge variant="info" size="sm" className="text-[10px]">Maker</Badge>
                ) : (
                  <Badge variant="secondary" size="sm" className="text-[10px]">Taker</Badge>
                )}
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className="text-[11px]">
                  {trade.price != null ? trade.price.toFixed(8) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className="text-[11px]">
                  {trade.quantity != null ? trade.quantity.toFixed(8) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className="text-[11px]">
                  {trade.quoteQuantity != null ? trade.quoteQuantity.toFixed(2) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className="text-[11px]">
                  {trade.commission != null ? trade.commission.toFixed(8) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-sm text-gray-300">
                <span className="text-[11px]">
                  {trade.commissionAsset || '-'}
                </span>
              </TableCell>
              <TableCell className="text-sm text-gray-400">
                <span className="text-[11px]">
                  {new Date(trade.executedAt).toLocaleString()}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};
