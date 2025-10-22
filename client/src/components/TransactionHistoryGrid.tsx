import { FileText, ArrowUpDown } from 'lucide-react';
import { useState, useMemo } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { TransactionType, TransactionStatus } from '../types/index';
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

const getTransactionTypeLabel = (type: TransactionType): string => {
  switch (type) {
    case TransactionType.Deposit: return 'Deposit';
    case TransactionType.Withdrawal: return 'Withdrawal';
    case TransactionType.Transfer: return 'Transfer';
    case TransactionType.Commission: return 'Commission';
    case TransactionType.Funding: return 'Funding';
    case TransactionType.Rebate: return 'Rebate';
    case TransactionType.Airdrop: return 'Airdrop';
    case TransactionType.Other: return 'Other';
    case TransactionType.RealizedPnL: return 'Realized PnL';
    case TransactionType.Trade: return 'Trade';
    case TransactionType.Liquidation: return 'Liquidation';
    case TransactionType.Bonus: return 'Bonus';
    case TransactionType.WelcomeBonus: return 'Welcome Bonus';
    case TransactionType.FundingFee: return 'Funding Fee';
    case TransactionType.InsuranceClear: return 'Insurance Clear';
    case TransactionType.ReferralKickback: return 'Referral Kickback';
    case TransactionType.CommissionRebate: return 'Commission Rebate';
    case TransactionType.ContestReward: return 'Contest Reward';
    case TransactionType.InternalTransfer: return 'Internal Transfer';
    case TransactionType.Settlement: return 'Settlement';
    case TransactionType.Delivery: return 'Delivery';
    case TransactionType.Adl: return 'ADL';
    default: return 'Unknown';
  }
};

const getTransactionStatusBadge = (status: TransactionStatus) => {
  switch (status) {
    case TransactionStatus.Pending:
      return <Badge variant="yellow" size="sm" className="text-[10px]">Pending</Badge>;
    case TransactionStatus.Completed:
      return <Badge variant="green" size="sm" className="text-[10px]">Completed</Badge>;
    case TransactionStatus.Failed:
      return <Badge variant="red" size="sm" className="text-[10px]">Failed</Badge>;
    case TransactionStatus.Canceled:
      return <Badge variant="gray" size="sm" className="text-[10px]">Canceled</Badge>;
    case TransactionStatus.Confirmed:
      return <Badge variant="green" size="sm" className="text-[10px]">Confirmed</Badge>;
    default:
      return <Badge variant="gray" size="sm" className="text-[10px]">Unknown</Badge>;
  }
};

const getTransactionTypeBadge = (type: TransactionType) => {
  switch (type) {
    case TransactionType.Deposit:
      return <Badge variant="green" size="sm" className="text-[10px]">Deposit</Badge>;
    case TransactionType.Withdrawal:
      return <Badge variant="red" size="sm" className="text-[10px]">Withdrawal</Badge>;
    case TransactionType.Transfer:
      return <Badge variant="blue" size="sm" className="text-[10px]">Transfer</Badge>;
    case TransactionType.Commission:
      return <Badge variant="gray" size="sm" className="text-[10px]">Commission</Badge>;
    case TransactionType.Funding:
      return <Badge variant="blue" size="sm" className="text-[10px]">Funding</Badge>;
    case TransactionType.Rebate:
      return <Badge variant="green" size="sm" className="text-[10px]">Rebate</Badge>;
    case TransactionType.Airdrop:
      return <Badge variant="green" size="sm" className="text-[10px]">Airdrop</Badge>;
    case TransactionType.RealizedPnL:
      return <Badge variant="blue" size="sm" className="text-[10px]">Realized PnL</Badge>;
    case TransactionType.Trade:
      return <Badge variant="blue" size="sm" className="text-[10px]">Trade</Badge>;
    case TransactionType.Liquidation:
      return <Badge variant="red" size="sm" className="text-[10px]">Liquidation</Badge>;
    case TransactionType.Bonus:
      return <Badge variant="green" size="sm" className="text-[10px]">Bonus</Badge>;
    case TransactionType.WelcomeBonus:
      return <Badge variant="green" size="sm" className="text-[10px]">Welcome Bonus</Badge>;
    case TransactionType.FundingFee:
      return <Badge variant="yellow" size="sm" className="text-[10px]">Funding Fee</Badge>;
    case TransactionType.InsuranceClear:
      return <Badge variant="gray" size="sm" className="text-[10px]">Insurance Clear</Badge>;
    case TransactionType.ReferralKickback:
      return <Badge variant="green" size="sm" className="text-[10px]">Referral</Badge>;
    case TransactionType.CommissionRebate:
      return <Badge variant="green" size="sm" className="text-[10px]">Comm Rebate</Badge>;
    case TransactionType.ContestReward:
      return <Badge variant="green" size="sm" className="text-[10px]">Contest</Badge>;
    case TransactionType.InternalTransfer:
      return <Badge variant="blue" size="sm" className="text-[10px]">Internal</Badge>;
    case TransactionType.Settlement:
      return <Badge variant="gray" size="sm" className="text-[10px]">Settlement</Badge>;
    case TransactionType.Delivery:
      return <Badge variant="gray" size="sm" className="text-[10px]">Delivery</Badge>;
    case TransactionType.Adl:
      return <Badge variant="red" size="sm" className="text-[10px]">ADL</Badge>;
    default:
      return <Badge variant="gray" size="sm" className="text-[10px]">{getTransactionTypeLabel(type)}</Badge>;
  }
};

type SortField = 'createdAt' | 'amount';
type SortDirection = 'asc' | 'desc';

export const TransactionHistoryGrid = () => {
  const { transactionHistory } = useArbitrageStore();
  const [sortField, setSortField] = useState<SortField>('createdAt');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedTransactions = useMemo(() => {
    if (!transactionHistory || transactionHistory.length === 0) return [];

    const sorted = [...transactionHistory].sort((a, b) => {
      let aValue: number, bValue: number;

      switch (sortField) {
        case 'createdAt':
          aValue = new Date(a.createdAt).getTime();
          bValue = new Date(b.createdAt).getTime();
          break;
        case 'amount':
          aValue = a.amount || 0;
          bValue = b.amount || 0;
          break;
        default:
          aValue = new Date(a.createdAt).getTime();
          bValue = new Date(b.createdAt).getTime();
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    });

    return sorted.slice(0, 100); // Limit to 100 items
  }, [transactionHistory, sortField, sortDirection]);

  if (!transactionHistory || transactionHistory.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="h-12 w-12" />}
        title="No Transaction History"
        description="You don't have any transaction history at the moment."
      />
    );
  }

  return (
    <div className="h-full overflow-x-auto">
      <Table>
        <TableHeader className="sticky top-0 z-30">
          <TableRow hover={false}>
            <TableHead className="sticky left-0 z-40 bg-binance-bg-secondary border-r border-binance-border">Exchange</TableHead>
            <TableHead className="sticky left-[120px] z-40 bg-binance-bg-secondary border-r border-binance-border">Type</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Asset</TableHead>
            <TableHead
              className="text-right cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('amount')}
            >
              <div className="flex items-center justify-end gap-1">
                Amount
                {sortField === 'amount' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
            <TableHead className="text-right">Fee</TableHead>
            <TableHead>Fee Asset</TableHead>
            <TableHead
              className="cursor-pointer hover:bg-binance-bg-hover transition-colors"
              onClick={() => handleSort('createdAt')}
            >
              <div className="flex items-center gap-1">
                Created At
                {sortField === 'createdAt' && (
                  <ArrowUpDown className="w-3 h-3" />
                )}
              </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedTransactions.map((transaction, index) => (
            <TableRow key={`${transaction.exchange}-${transaction.transactionId}-${index}`}>
              <TableCell className="sticky left-0 z-20 bg-binance-bg-secondary border-r border-binance-border">
                <ExchangeBadge exchange={transaction.exchange} />
              </TableCell>
              <TableCell className="sticky left-[120px] z-20 bg-binance-bg-secondary border-r border-binance-border">
                {getTransactionTypeBadge(transaction.type)}
              </TableCell>
              <TableCell>{getTransactionStatusBadge(transaction.status)}</TableCell>
              <TableCell className="font-medium text-gray-300">
                <span className="text-[11px]">
                  {transaction.asset}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span className={`text-[11px] ${
                  transaction.type === TransactionType.Deposit ||
                  transaction.type === TransactionType.Rebate ||
                  transaction.type === TransactionType.Airdrop
                    ? 'text-green-400'
                    : transaction.type === TransactionType.Withdrawal ||
                      transaction.type === TransactionType.Commission
                    ? 'text-red-400'
                    : 'text-gray-300'
                }`}>
                  {transaction.amount != null ? transaction.amount.toFixed(8) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-right font-mono text-sm">
                <span className="text-[11px]">
                  {transaction.fee != null ? transaction.fee.toFixed(8) : '-'}
                </span>
              </TableCell>
              <TableCell className="text-sm text-gray-300">
                <span className="text-[11px]">
                  {transaction.feeAsset || '-'}
                </span>
              </TableCell>
              <TableCell className="text-sm text-gray-400">
                <span className="text-[11px]">
                  {new Date(transaction.createdAt).toLocaleString()}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};
