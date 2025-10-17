import { Clock, TrendingUp, TrendingDown } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { EmptyState } from './ui/EmptyState';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from './ui/Table';

export const FundingRateMonitor = () => {
  const { fundingRates } = useArbitrageStore();

  // Group funding rates by symbol
  const groupedRates = fundingRates.reduce((acc, rate) => {
    if (!acc[rate.symbol]) {
      acc[rate.symbol] = [];
    }
    acc[rate.symbol].push(rate);
    return acc;
  }, {} as Record<string, typeof fundingRates>);

  const getTimeUntilFunding = (nextFundingTime: string) => {
    const now = new Date().getTime();
    const funding = new Date(nextFundingTime).getTime();
    const diff = funding - now;

    if (diff < 0) return 'In progress';

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

    return `${hours}h ${minutes}m`;
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="p-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-binance-yellow" />
            Funding Rates
          </CardTitle>
          <Badge variant="success" size="sm">
            Live Data
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-0">
        {fundingRates.length === 0 ? (
          <EmptyState
            icon={<TrendingUp className="w-12 h-12" />}
            title="No funding rate data"
            description="Waiting for live funding rate updates from exchanges"
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow hover={false}>
                <TableHead>Symbol</TableHead>
                <TableHead>Exchange</TableHead>
                <TableHead className="text-right">Funding Rate</TableHead>
                <TableHead className="text-right">APR</TableHead>
                <TableHead className="text-right">Next Funding</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(groupedRates).map(([symbol, rates]) =>
                rates.map((rate, idx) => (
                  <TableRow key={`${symbol}-${rate.exchange}-${idx}`}>
                    <TableCell className="font-medium">{symbol}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" size="sm">
                        {rate.exchange}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        {rate.rate >= 0 ? (
                          <TrendingUp className="w-3 h-3 text-binance-green" />
                        ) : (
                          <TrendingDown className="w-3 h-3 text-binance-red" />
                        )}
                        <span
                          className={`font-mono text-sm font-semibold ${
                            rate.rate >= 0 ? 'text-binance-green' : 'text-binance-red'
                          }`}
                        >
                          {(rate.rate * 100).toFixed(4)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <span
                        className={`font-mono text-sm font-semibold ${
                          rate.annualizedRate >= 0
                            ? 'text-binance-green'
                            : 'text-binance-red'
                        }`}
                      >
                        {rate.annualizedRate >= 0 ? '+' : ''}
                        {(rate.annualizedRate * 100).toFixed(2)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1 text-binance-text-muted">
                        <Clock className="w-3 h-3" />
                        <span className="text-xs">
                          {getTimeUntilFunding(rate.nextFundingTime)}
                        </span>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
};
