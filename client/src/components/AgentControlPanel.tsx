import { useState, useEffect } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { apiService } from '../services/apiService';
import { signalRService } from '../services/signalRService';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/Tabs';

export function AgentControlPanel() {
  const agent = useArbitrageStore((state) => state.agent);
  const setAgentStatus = useArbitrageStore((state) => state.setAgentStatus);
  const setAgentStats = useArbitrageStore((state) => state.setAgentStats);
  const setAgentConfig = useArbitrageStore((state) => state.setAgentConfig);
  const addAgentDecision = useArbitrageStore((state) => state.addAgentDecision);
  const [isLoading, setIsLoading] = useState(false);
  const [realtimeDuration, setRealtimeDuration] = useState(0);
  const [config, setConfig] = useState({
    maxLeverage: 1.0,
    targetUtilization: 0.9,
    maxPositions: 3,
    predictionIntervalSeconds: 3600, // 1 hour to match training environment
  });

  // Fetch agent config and status on mount
  useEffect(() => {
    const fetchAgentData = async () => {
      try {
        const [configData, statusData] = await Promise.all([
          apiService.getAgentConfig(),
          apiService.getAgentStatus(),
        ]);
        setConfig({
          maxLeverage: configData.maxLeverage,
          targetUtilization: configData.targetUtilization,
          maxPositions: configData.maxPositions,
          predictionIntervalSeconds: configData.predictionIntervalSeconds,
        });

        // Update Zustand store with fetched status
        setAgentStatus(statusData.status, statusData.durationSeconds, statusData.errorMessage);
        if (statusData.stats) {
          setAgentStats(statusData.stats);
        }
        if (statusData.config) {
          setAgentConfig(statusData.config);
        }
      } catch (error) {
        console.error('Error fetching agent data:', error);
      }
    };

    fetchAgentData();
  }, [setAgentStatus, setAgentStats, setAgentConfig]);

  // Subscribe to agent decision broadcasts
  useEffect(() => {
    console.log('[AgentControlPanel] Setting up onAgentDecision listener');
    const unsubscribe = signalRService.onAgentDecision((decision) => {
      console.log('[AgentControlPanel] Received agent decision:', decision);
      addAgentDecision(decision);
    });

    return () => {
      console.log('[AgentControlPanel] Cleaning up onAgentDecision listener');
      unsubscribe();
    };
  }, [addAgentDecision]);

  // Sync realtime duration with backend updates
  useEffect(() => {
    if (agent.durationSeconds !== undefined) {
      setRealtimeDuration(agent.durationSeconds);
    }
  }, [agent.durationSeconds]);

  // Realtime timer - increment duration every second when agent is running
  useEffect(() => {
    if (agent.status !== 'running') {
      return;
    }

    const interval = setInterval(() => {
      setRealtimeDuration((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [agent.status]);

  const handleStart = async () => {
    setIsLoading(true);
    try {
      const statusData = await apiService.startAgent(config);
      // Update store with returned status
      setAgentStatus(statusData.status, statusData.durationSeconds, statusData.errorMessage);
      if (statusData.stats) {
        setAgentStats(statusData.stats);
      }
      if (statusData.config) {
        setAgentConfig(statusData.config);
      }
    } catch (error: any) {
      console.error('Error starting agent:', error);
      alert(error.message || 'Failed to start agent');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      await apiService.stopAgent();
      // Update status to stopped
      setAgentStatus('stopped', 0);
    } catch (error: any) {
      console.error('Error stopping agent:', error);
      alert(error.message || 'Failed to stop agent');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePause = async () => {
    setIsLoading(true);
    try {
      await apiService.pauseAgent();
      // Update status to paused
      setAgentStatus('paused', agent.durationSeconds);
    } catch (error: any) {
      console.error('Error pausing agent:', error);
      alert(error.message || 'Failed to pause agent');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResume = async () => {
    setIsLoading(true);
    try {
      await apiService.resumeAgent();
      // Update status to running
      setAgentStatus('running', agent.durationSeconds);
    } catch (error: any) {
      console.error('Error resuming agent:', error);
      alert(error.message || 'Failed to resume agent');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateConfig = async () => {
    if (agent.status === 'running') {
      alert('Cannot update config while agent is running. Please stop the agent first.');
      return;
    }

    setIsLoading(true);
    try {
      const updatedConfig = await apiService.updateAgentConfig(config);
      setConfig({
        maxLeverage: updatedConfig.maxLeverage,
        targetUtilization: updatedConfig.targetUtilization,
        maxPositions: updatedConfig.maxPositions,
        predictionIntervalSeconds: updatedConfig.predictionIntervalSeconds,
      });
    } catch (error: any) {
      console.error('Error updating config:', error);
      alert(error.message || 'Failed to update configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-binance-green';
      case 'paused':
        return 'text-binance-yellow';
      case 'error':
        return 'text-binance-red';
      default:
        return 'text-binance-text-secondary';
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle>AI Trading Agent</CardTitle>
      </CardHeader>

      <Tabs defaultValue="control" className="flex-1 flex flex-col min-h-0">
        <TabsList className="px-4">
          <TabsTrigger value="control">Control</TabsTrigger>
          <TabsTrigger value="decisions">Decisions</TabsTrigger>
        </TabsList>

        {/* Control Tab */}
        <TabsContent value="control" className="flex-1 overflow-y-auto">
          <CardContent className="space-y-4">
            {/* Status Display */}
            <div className="bg-binance-bg rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-binance-text-secondary">Status:</span>
                <span className={`text-sm font-semibold uppercase ${getStatusColor(agent.status)}`}>
                  {agent.status}
                </span>
              </div>

              {realtimeDuration > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-binance-text-secondary">Running:</span>
                  <span className="text-sm text-binance-text">
                    {formatDuration(realtimeDuration)}
                  </span>
                </div>
              )}

              {agent.errorMessage && (
                <div className="text-xs text-binance-red bg-binance-red bg-opacity-10 rounded p-2">
                  {agent.errorMessage}
                </div>
              )}
            </div>

            {/* Stats Display */}
            {agent.stats && (
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-binance-bg rounded-lg p-3">
                  <div className="text-xs text-binance-text-secondary">Session P&L</div>
                  <div className={`text-lg font-bold ${(agent.stats.sessionPnLUsd ?? 0) >= 0 ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${(agent.stats.sessionPnLUsd ?? 0).toFixed(2)}
                  </div>
                  <div className={`text-xs ${(agent.stats.sessionPnLPct ?? 0) >= 0 ? 'text-binance-green' : 'text-binance-red'}`}>
                    {(agent.stats.sessionPnLPct ?? 0) >= 0 ? '+' : ''}{(agent.stats.sessionPnLPct ?? 0).toFixed(2)}%
                  </div>
                </div>

                <div className="bg-binance-bg rounded-lg p-3">
                  <div className="text-xs text-binance-text-secondary">Win Rate</div>
                  <div className="text-lg font-bold text-binance-text">
                    {(agent.stats.winRate ?? 0).toFixed(1)}%
                  </div>
                  <div className="text-xs text-binance-text-secondary">
                    {agent.stats.winningTrades ?? 0}W / {agent.stats.losingTrades ?? 0}L
                  </div>
                </div>

                <div className="bg-binance-bg rounded-lg p-3">
                  <div className="text-xs text-binance-text-secondary">Trades</div>
                  <div className="text-lg font-bold text-binance-text">
                    {agent.stats.totalTrades ?? 0}
                  </div>
                  <div className="text-xs text-binance-text-secondary">
                    {agent.stats.activePositions ?? 0} active
                  </div>
                </div>

                <div className="col-span-2 bg-binance-bg rounded-lg p-3">
                  <div className="text-xs text-binance-text-secondary mb-1">Decisions</div>
                  <div className="flex justify-between text-xs">
                    <span className="text-binance-text-secondary">
                      Hold: {agent.stats.holdDecisions ?? 0}
                    </span>
                    <span className="text-binance-green">
                      Enter: {agent.stats.enterDecisions ?? 0}
                    </span>
                    <span className="text-binance-red">
                      Exit: {agent.stats.exitDecisions ?? 0}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Configuration Form */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-binance-text">Configuration</h4>

              <div>
                <label className="block text-xs text-binance-text-secondary mb-1">
                  Max Leverage: {config.maxLeverage.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.5"
                  value={config.maxLeverage}
                  onChange={(e) => setConfig({ ...config, maxLeverage: parseFloat(e.target.value) })}
                  disabled={agent.status === 'running'}
                  className="w-full h-2 bg-binance-bg rounded-lg appearance-none cursor-pointer slider"
                />
              </div>

              <div>
                <label className="block text-xs text-binance-text-secondary mb-1">
                  Target Utilization: {(config.targetUtilization * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  value={config.targetUtilization}
                  onChange={(e) => setConfig({ ...config, targetUtilization: parseFloat(e.target.value) })}
                  disabled={agent.status === 'running'}
                  className="w-full h-2 bg-binance-bg rounded-lg appearance-none cursor-pointer slider"
                />
              </div>

              <div>
                <label className="block text-xs text-binance-text-secondary mb-1">
                  Max Positions: {config.maxPositions}
                </label>
                <input
                  type="range"
                  min="1"
                  max="3"
                  step="1"
                  value={config.maxPositions}
                  onChange={(e) => setConfig({ ...config, maxPositions: parseInt(e.target.value) })}
                  disabled={agent.status === 'running'}
                  className="w-full h-2 bg-binance-bg rounded-lg appearance-none cursor-pointer slider"
                />
              </div>

              {agent.status !== 'running' && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleUpdateConfig}
                  isLoading={isLoading}
                  className="w-full"
                >
                  Update Configuration
                </Button>
              )}
            </div>

            {/* Control Buttons */}
            <div className="space-y-2">
              {agent.status === 'stopped' || agent.status === 'error' ? (
                <Button
                  variant="success"
                  onClick={handleStart}
                  isLoading={isLoading}
                  className="w-full"
                >
                  Start Agent
                </Button>
              ) : agent.status === 'running' ? (
                <>
                  <Button
                    variant="secondary"
                    onClick={handlePause}
                    isLoading={isLoading}
                    className="w-full"
                  >
                    Pause Agent
                  </Button>
                  <Button
                    variant="danger"
                    onClick={handleStop}
                    isLoading={isLoading}
                    className="w-full"
                  >
                    Stop Agent
                  </Button>
                </>
              ) : agent.status === 'paused' ? (
                <>
                  <Button
                    variant="success"
                    onClick={handleResume}
                    isLoading={isLoading}
                    className="w-full"
                  >
                    Resume Agent
                  </Button>
                  <Button
                    variant="danger"
                    onClick={handleStop}
                    isLoading={isLoading}
                    className="w-full"
                  >
                    Stop Agent
                  </Button>
                </>
              ) : null}
            </div>
          </CardContent>
        </TabsContent>

        {/* Decisions Tab */}
        <TabsContent value="decisions" className="flex-1 overflow-y-auto">
          <CardContent className="p-0">
            {agent.decisions.length === 0 ? (
              <div className="p-4 text-center text-sm text-binance-text-secondary">
                No decisions yet. Start the agent to see decision history.
              </div>
            ) : (
              <div className="divide-y divide-binance-border">
                {agent.decisions.map((decision, index) => {
                  const symbol = decision.symbol || decision.opportunitySymbol;
                  const isSuccess = decision.executionStatus === 'success';
                  const isFailed = decision.executionStatus === 'failed';

                  return (
                    <div key={index} className="p-3 hover:bg-binance-bg-hover">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-sm font-semibold uppercase ${
                              decision.action === 'ENTER'
                                ? 'text-binance-green'
                                : decision.action === 'EXIT'
                                ? 'text-binance-red'
                                : 'text-binance-text-secondary'
                            }`}
                          >
                            {decision.action}
                          </span>
                          {symbol && (
                            <span className="text-sm font-medium text-binance-text">
                              {symbol}
                            </span>
                          )}
                        </div>
                        <span className="text-xs text-binance-text-secondary">
                          {new Date(decision.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {/* ENTER specific fields */}
                      {decision.action === 'ENTER' && (
                        <>
                          {decision.amountUsd !== undefined && (
                            <div className="text-sm text-binance-text mb-1">
                              Amount: <span className="font-medium">${decision.amountUsd.toFixed(2)}</span>
                            </div>
                          )}
                          {decision.confidence && (
                            <div className="text-xs text-binance-text-secondary">
                              Confidence: {decision.confidence}
                            </div>
                          )}
                        </>
                      )}

                      {/* EXIT specific fields */}
                      {decision.action === 'EXIT' && (
                        <>
                          {decision.profitUsd !== undefined && (
                            <div className={`text-sm mb-1 ${decision.profitUsd >= 0 ? 'text-binance-green' : 'text-binance-red'}`}>
                              Profit: <span className="font-medium">
                                {decision.profitUsd >= 0 ? '+' : ''}${decision.profitUsd.toFixed(2)}
                              </span>
                              {decision.profitPct !== undefined && (
                                <span className="ml-2">
                                  ({decision.profitPct >= 0 ? '+' : ''}{decision.profitPct.toFixed(2)}%)
                                </span>
                              )}
                            </div>
                          )}
                          {decision.durationHours !== undefined && (
                            <div className="text-xs text-binance-text-secondary">
                              Duration: {decision.durationHours.toFixed(1)}h
                            </div>
                          )}
                        </>
                      )}

                      {/* Execution status indicator */}
                      <div className="flex items-center gap-2 mt-2">
                        {isSuccess && (
                          <span className="text-xs px-2 py-0.5 rounded bg-binance-green bg-opacity-20 text-binance-green">
                            ✓ Executed
                          </span>
                        )}
                        {isFailed && (
                          <span className="text-xs px-2 py-0.5 rounded bg-binance-red bg-opacity-20 text-binance-red">
                            ✗ Failed
                          </span>
                        )}
                        {decision.errorMessage && (
                          <span className="text-xs text-binance-red">
                            {decision.errorMessage}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </TabsContent>
      </Tabs>
    </Card>
  );
}
