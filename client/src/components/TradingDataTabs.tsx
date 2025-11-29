import { Card } from './ui/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/Tabs';
import { PositionsGrid } from './PositionsGrid';
import { ExecutionHistoryGrid } from './ExecutionHistoryGrid';
import { OpenOrdersGrid } from './OpenOrdersGrid';
import { OrderHistoryGrid } from './OrderHistoryGrid';
import { TradeHistoryGrid } from './TradeHistoryGrid';
import { TransactionHistoryGrid } from './TransactionHistoryGrid';

export const TradingDataTabs = () => {
  return (
    <Card className="h-full flex flex-col p-0">
      <Tabs defaultValue="positions" className="h-full flex flex-col">
        <TabsList className="flex-shrink-0 px-2 border-b border-gray-700">
          <TabsTrigger value="positions">Active Executions</TabsTrigger>
          <TabsTrigger value="execution-history">Execution History</TabsTrigger>
          <TabsTrigger value="open-orders">Open Orders</TabsTrigger>
          <TabsTrigger value="order-history">Order History</TabsTrigger>
          <TabsTrigger value="trades">Trades</TabsTrigger>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-hidden min-h-0">
          <TabsContent value="positions" className="h-full overflow-auto">
            <PositionsGrid />
          </TabsContent>

          <TabsContent value="execution-history" className="h-full overflow-auto">
            <ExecutionHistoryGrid />
          </TabsContent>

          <TabsContent value="open-orders" className="h-full overflow-auto">
            <OpenOrdersGrid />
          </TabsContent>

          <TabsContent value="order-history" className="h-full overflow-auto">
            <OrderHistoryGrid />
          </TabsContent>

          <TabsContent value="trades" className="h-full overflow-auto">
            <TradeHistoryGrid />
          </TabsContent>

          <TabsContent value="transactions" className="h-full overflow-auto">
            <TransactionHistoryGrid />
          </TabsContent>
        </div>
      </Tabs>
    </Card>
  );
};
