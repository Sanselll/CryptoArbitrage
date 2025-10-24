import { TrendingUp, DollarSign, Clock, Target, AlertTriangle, BarChart3, ChevronDown, ChevronUp } from 'lucide-react';
import { OpportunitySuggestion, EntryRecommendation, RecommendedStrategyType } from '../../types/index';
import { useState } from 'react';

interface SuggestionTooltipProps {
  suggestion: OpportunitySuggestion;
  isPositionEntry?: boolean;
  recommendedStrategy?: string;
}

// Helper function to get recommendation label
const getRecommendationLabel = (recommendation: EntryRecommendation): string => {
  switch (recommendation) {
    case EntryRecommendation.StrongBuy: return 'Strong Buy';
    case EntryRecommendation.Buy: return 'Buy';
    case EntryRecommendation.Hold: return 'Hold';
    case EntryRecommendation.Skip: return 'Skip';
    default: return 'Unknown';
  }
};

// Helper function to get strategy label
const getStrategyLabel = (strategy: RecommendedStrategyType): { label: string; icon: string } => {
  switch (strategy) {
    case RecommendedStrategyType.FundingOnly:
      return { label: 'Funding Rate Arbitrage', icon: '💰' };
    case RecommendedStrategyType.SpreadOnly:
      return { label: 'Price Spread Arbitrage', icon: '📊' };
    case RecommendedStrategyType.Hybrid:
      return { label: 'Hybrid Strategy', icon: '🔄' };
    default:
      return { label: 'Unknown', icon: '❓' };
  }
};

// Helper to get recommendation color class
const getRecommendationColor = (recommendation: EntryRecommendation): string => {
  switch (recommendation) {
    case EntryRecommendation.StrongBuy: return 'text-green-400';
    case EntryRecommendation.Buy: return 'text-blue-400';
    case EntryRecommendation.Hold: return 'text-yellow-400';
    case EntryRecommendation.Skip: return 'text-red-400';
    default: return 'text-gray-400';
  }
};

export const SuggestionTooltip = ({ suggestion, isPositionEntry, recommendedStrategy }: SuggestionTooltipProps) => {
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);
  const strategyInfo = getStrategyLabel(suggestion.recommendedStrategy);
  const recommendationColor = getRecommendationColor(suggestion.entryRecommendation);
  const recommendationLabel = getRecommendationLabel(suggestion.entryRecommendation);

  return (
    <div className="w-[420px] pointer-events-auto">
      <div className="bg-binance-bg-tertiary border border-binance-border rounded shadow-xl p-3">
        {/* Header */}
        <div className="flex items-center justify-between mb-2.5 pb-2.5 border-b border-binance-border">
          <div className="flex items-center gap-2">
            <div className="text-xl font-bold text-binance-text">
              {Math.round(suggestion.confidenceScore)}
            </div>
            <div>
              <div className={`text-xs font-bold ${recommendationColor}`}>
                {isPositionEntry ? 'Entry Score' : recommendationLabel}
              </div>
              <div className="text-[10px] text-binance-text-secondary">
                {isPositionEntry ? `Strategy: ${recommendedStrategy || 'N/A'}` : 'AI Confidence'}
              </div>
            </div>
          </div>
          <div className="text-xl">{strategyInfo.icon}</div>
        </div>

        {/* Strategy */}
        <div className="mb-2">
          <div className="flex items-center gap-1 mb-1">
            <TrendingUp className="w-3 h-3 text-binance-yellow" />
            <span className="text-[10px] font-semibold text-binance-text">Strategy</span>
          </div>
          <div className="text-[11px] text-binance-text-secondary pl-3.5">
            {strategyInfo.label}
          </div>
        </div>

        {/* Score Breakdown - 2 columns */}
        <div className="mb-2">
          <div className="flex items-center gap-1 mb-1.5">
            <BarChart3 className="w-3 h-3 text-binance-yellow" />
            <span className="text-[10px] font-semibold text-binance-text">Score Breakdown</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5 pl-3.5">
            {Object.entries(suggestion.scoreBreakdown)
              .filter(([key]) => ['fundingQuality', 'profitPotential', 'spreadEfficiency', 'marketQuality', 'timeEfficiency', 'riskScore', 'executionSafety'].includes(key))
              .map(([key, value]) => {
                const label = key
                  .replace(/([A-Z])/g, ' $1')
                  .replace(/^./, (str) => str.toUpperCase())
                  .replace('Time Efficiency', 'Time Eff')
                  .replace('Execution Safety', 'Exec Safe');
                const percentage = Math.round(value as number);
                return (
                  <div key={key} className="flex items-center gap-1.5">
                    <div className="text-[10px] text-binance-text-secondary w-16 flex-shrink-0">{label}</div>
                    <div className="flex-1 h-1.5 bg-binance-bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          percentage >= 80 ? 'bg-green-500' :
                          percentage >= 60 ? 'bg-blue-500' :
                          percentage >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <div className="text-[10px] font-mono text-binance-text w-7 text-right flex-shrink-0">{percentage}</div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Execution Cost Info */}
        {suggestion.scoreBreakdown.executionCostPercent !== undefined && (
          <div className="mb-2 bg-binance-bg-secondary rounded p-1.5">
            <div className="text-[9px] text-binance-text-secondary mb-0.5">Execution Cost vs Profit</div>
            <div className="flex items-center justify-between">
              <div className="text-[10px] text-binance-text">
                Cost: <span className="text-red-400">{suggestion.scoreBreakdown.executionCostPercent.toFixed(3)}%</span>
              </div>
              <div className="text-[10px] text-binance-text">
                Net: <span className={suggestion.scoreBreakdown.profitAfterCosts > 0 ? 'text-green-400' : 'text-red-400'}>
                  {suggestion.scoreBreakdown.profitAfterCosts.toFixed(3)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Scoring Factors - Text Analysis */}
        {suggestion.scoreBreakdown.scoringFactors && suggestion.scoreBreakdown.scoringFactors.length > 0 && (
          <div className="mb-2">
            <div
              className="flex items-center justify-between mb-1 cursor-pointer hover:bg-binance-bg-secondary/50 rounded px-1 -mx-1 transition-colors"
              onClick={() => setIsAnalysisExpanded(!isAnalysisExpanded)}
            >
              <div className="flex items-center gap-1">
                <span className="text-[10px] font-semibold text-binance-text">Analysis</span>
                <span className="text-[9px] text-binance-text-secondary">
                  ({suggestion.scoreBreakdown.scoringFactors.length} factors)
                </span>
              </div>
              {isAnalysisExpanded ? (
                <ChevronUp className="w-3 h-3 text-binance-text-secondary" />
              ) : (
                <ChevronDown className="w-3 h-3 text-binance-text-secondary" />
              )}
            </div>
            <div
              className={`text-[9px] text-binance-text-secondary space-y-0.5 pl-3.5 leading-tight overflow-hidden transition-all duration-200 ${
                isAnalysisExpanded ? 'max-h-[200px] overflow-y-auto' : 'max-h-[36px]'
              }`}
            >
              {(isAnalysisExpanded
                ? suggestion.scoreBreakdown.scoringFactors
                : suggestion.scoreBreakdown.scoringFactors.slice(0, 2)
              ).map((factor, idx) => (
                <div key={idx} className="break-words">{factor}</div>
              ))}
              {!isAnalysisExpanded && suggestion.scoreBreakdown.scoringFactors.length > 2 && (
                <div className="text-binance-text-secondary/60 italic">
                  +{suggestion.scoreBreakdown.scoringFactors.length - 2} more...
                </div>
              )}
            </div>
          </div>
        )}

        {/* Suggested Parameters */}
        {!isPositionEntry && (
          <div className="mb-2 grid grid-cols-2 gap-1.5">
            <div className="bg-binance-bg-secondary rounded p-1.5">
              <div className="flex items-center gap-0.5 mb-0.5">
                <DollarSign className="w-2.5 h-2.5 text-binance-yellow" />
                <span className="text-[9px] text-binance-text-secondary">Position</span>
              </div>
              <div className="text-[11px] font-bold text-binance-text">
                ${suggestion.suggestedPositionSizeUsd.toLocaleString()}
              </div>
            </div>
            <div className="bg-binance-bg-secondary rounded p-1.5">
              <div className="flex items-center gap-0.5 mb-0.5">
                <TrendingUp className="w-2.5 h-2.5 text-binance-yellow" />
                <span className="text-[9px] text-binance-text-secondary">Leverage</span>
              </div>
              <div className="text-[11px] font-bold text-binance-text">
                {suggestion.suggestedLeverage}x
              </div>
            </div>
            <div className="bg-binance-bg-secondary rounded p-1.5">
              <div className="flex items-center gap-0.5 mb-0.5">
                <Clock className="w-2.5 h-2.5 text-binance-yellow" />
                <span className="text-[9px] text-binance-text-secondary">Hold</span>
              </div>
              <div className="text-[11px] font-bold text-binance-text">
                {Math.round(suggestion.suggestedHoldingPeriodHours)}h
              </div>
            </div>
            <div className="bg-binance-bg-secondary rounded p-1.5">
              <div className="flex items-center gap-0.5 mb-0.5">
                <Target className="w-2.5 h-2.5 text-binance-yellow" />
                <span className="text-[9px] text-binance-text-secondary">Target</span>
              </div>
              <div className="text-[11px] font-bold text-green-400">
                {suggestion.profitTargetPercent.toFixed(2)}%
              </div>
            </div>
          </div>
        )}

        {/* Warnings */}
        {suggestion.warnings && suggestion.warnings.length > 0 && (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded p-1.5">
            <div className="flex items-center gap-0.5 mb-0.5">
              <AlertTriangle className="w-2.5 h-2.5 text-yellow-400" />
              <span className="text-[9px] font-semibold text-yellow-400">Warnings</span>
            </div>
            <div className="text-[9px] text-yellow-400/90 pl-2.5 leading-tight">
              {suggestion.warnings.slice(0, 2).join('; ')}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
