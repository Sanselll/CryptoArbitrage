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
  const setAgentDecisions = useArbitrageStore((state) => state.setAgentDecisions);
  const [isLoading, setIsLoading] = useState(false);
  const [realtimeDuration, setRealtimeDuration] = useState(0);
  const [config, setConfig] = useState({
    maxLeverage: 1.0,
    targetUtilization: 0.9,
    maxPositions: 3,
    predictionIntervalSeconds: 3600, // 1 hour to match training environment
  });

  // Fetch agent config, status, and decisions on mount
  useEffect(() => {
    const fetchAgentData = async () => {
      try {
        const [configData, statusData, decisionsData] = await Promise.all([
          apiService.getAgentConfig(),
          apiService.getAgentStatus(),
          apiService.getAgentDecisions(100),
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

        // Load existing decisions from repository
        setAgentDecisions(decisionsData);
      } catch (error) {
        console.error('Error fetching agent data:', error);
      }
    };

    fetchAgentData();
  }, [setAgentStatus, setAgentStats, setAgentConfig, setAgentDecisions]);

  // Note: Agent decisions are already subscribed in arbitrageStore.ts
  // No need for duplicate subscription here

  // Calculate duration based on actual time difference (works even when tab is in background)
  useEffect(() => {
    if (agent.status !== 'running' || agent.durationSeconds === undefined) {
      setRealtimeDuration(agent.durationSeconds || 0);
      return;
    }

    // Calculate start time based on backend duration
    const startTime = Date.now() - (agent.durationSeconds * 1000);

    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setRealtimeDuration(elapsed);
    }, 1000);

    return () => clearInterval(interval);
  }, [agent.status, agent.durationSeconds]);

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
      <CardHeader className="p-2">
        <CardTitle className="text-base">AI Trading Agent</CardTitle>
      </CardHeader>

      <Tabs defaultValue="control" className="flex-1 flex flex-col min-h-0">
        <TabsList className="px-2">
          <TabsTrigger value="control" className="text-sm">Control</TabsTrigger>
          <TabsTrigger value="decisions" className="text-sm">Decisions</TabsTrigger>
        </TabsList>

        {/* Control Tab */}
        <TabsContent value="control" className="flex-1 overflow-y-auto">
          <CardContent className="p-2 space-y-2">
            {/* Status Display */}
            <div className="bg-binance-bg rounded-lg p-2 space-y-1.5">
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
                  <div className={`text-lg font-bold ${(agent.stats.totalPnLUsd ?? 0) >= 0 ? 'text-binance-green' : 'text-binance-red'}`}>
                    ${(agent.stats.totalPnLUsd ?? 0).toFixed(2)}
                  </div>
                  <div className={`text-xs ${(agent.stats.totalPnLPct ?? 0) >= 0 ? 'text-binance-green' : 'text-binance-red'}`}>
                    {(agent.stats.totalPnLPct ?? 0) >= 0 ? '+' : ''}{(agent.stats.totalPnLPct ?? 0).toFixed(2)}%
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

            {/* Configuration Display (read-only) */}
            <div className="bg-binance-bg rounded-lg p-2 space-y-1">
              <h4 className="text-xs font-semibold text-binance-text-secondary">Configuration</h4>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-binance-text-secondary">Leverage:</span>
                  <span className="text-binance-text ml-1">{config.maxLeverage.toFixed(1)}x</span>
                </div>
                <div>
                  <span className="text-binance-text-secondary">Util:</span>
                  <span className="text-binance-text ml-1">{(config.targetUtilization * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-binance-text-secondary">Pos:</span>
                  <span className="text-binance-text ml-1">{config.maxPositions}</span>
                </div>
              </div>
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
                    <div key={index} className="p-2 hover:bg-binance-bg-hover relative">
                      {/* Status badge in top-right corner */}
                      <div className="absolute top-1 right-1">
                        {isSuccess && (
                          <div className="w-1.5 h-1.5 rounded-full bg-binance-green" title="Executed" />
                        )}
                        {isFailed && (
                          <div className="w-1.5 h-1.5 rounded-full bg-binance-red" title="Failed" />
                        )}
                      </div>

                      {/* Compact header: action + symbol */}
                      <div className="flex items-baseline gap-1.5 mb-1">
                        <span
                          className={`text-[10px] font-bold uppercase leading-none ${
                            decision.action === 'ENTER'
                              ? 'text-binance-green'
                              : decision.action === 'EXIT'
                              ? 'text-binance-red'
                              : 'text-binance-text-secondary'
                          }`}
                        >
                          {decision.action}
                        </span>
                        <span className="text-[11px] font-medium text-binance-text leading-none">
                          {symbol || 'N/A'}
                        </span>
                        <span className="text-[9px] text-binance-text-secondary ml-auto leading-none">
                          {new Date(decision.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {/* Compact data row */}
                      <div className="flex items-center gap-2 text-[10px]">
                        {decision.action === 'ENTER' && decision.amountUsd != null && (
                          <span className="text-binance-text">
                            ${decision.amountUsd.toFixed(2)}
                          </span>
                        )}
                        {decision.action === 'EXIT' && decision.profitUsd != null && (
                          <>
                            <span className={decision.profitUsd >= 0 ? 'text-binance-green' : 'text-binance-red'}>
                              {decision.profitUsd >= 0 ? '+' : ''}${decision.profitUsd.toFixed(2)}
                            </span>
                            {decision.profitPct != null && (
                              <span className={decision.profitPct >= 0 ? 'text-binance-green' : 'text-binance-red'}>
                                ({decision.profitPct >= 0 ? '+' : ''}{decision.profitPct.toFixed(1)}%)
                              </span>
                            )}
                            {decision.durationHours != null && (
                              <span className="text-binance-text-secondary">
                                • {decision.durationHours.toFixed(1)}h
                              </span>
                            )}
                          </>
                        )}
                        {(decision.enterProbability != null || decision.exitProbability != null || decision.confidence) && (
                          <span className="text-binance-text-secondary ml-auto">
                            {decision.action === 'ENTER' && decision.enterProbability != null
                              ? `${(decision.enterProbability * 100).toFixed(0)}%`
                              : decision.action === 'EXIT' && decision.exitProbability != null
                              ? `${(decision.exitProbability * 100).toFixed(0)}%`
                              : decision.confidence || ''}
                          </span>
                        )}
                      </div>

                      {/* Error message if any */}
                      {decision.errorMessage && (
                        <div className="text-[9px] text-binance-red mt-0.5 truncate" title={decision.errorMessage}>
                          {decision.errorMessage}
                        </div>
                      )}
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
