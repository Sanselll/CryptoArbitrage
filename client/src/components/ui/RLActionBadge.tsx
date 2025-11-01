import { cn } from '../../lib/cn';
import { TrendingUp, TrendingDown, Bot, Zap, Pause } from 'lucide-react';

interface RLActionBadgeProps {
  actionType: 'ENTER' | 'EXIT';  // ENTER for opportunities, EXIT for positions
  probability?: number;           // Action probability (0-1)
  confidence?: string;            // HIGH, MEDIUM, LOW
  holdProbability?: number;       // HOLD action probability (0-1)
  stateValue?: number;            // RL state value estimate
  modelVersion?: string;          // Model version
  size?: 'sm' | 'md';
  showDetails?: boolean;
  className?: string;
}

const getConfidenceVariant = (confidence?: string): 'success' | 'warning' | 'danger' => {
  if (confidence === 'HIGH') return 'success';
  if (confidence === 'MEDIUM') return 'warning';
  return 'danger';
};

const getProbabilityColor = (probability?: number): string => {
  if (probability === undefined || probability === null) return 'text-binance-text-secondary';
  if (probability >= 0.7) return 'text-binance-green';
  if (probability >= 0.4) return 'text-binance-yellow';
  return 'text-binance-red';
};

const getConfidenceColor = (confidence?: string): string => {
  if (confidence === 'HIGH') return 'text-binance-green';
  if (confidence === 'MEDIUM') return 'text-binance-yellow';
  return 'text-binance-red';
};

const getActionIcon = (actionType: string, probability?: number) => {
  if (probability === undefined || probability === null) {
    return <Bot className="w-2.5 h-2.5" />;
  }

  if (actionType === 'ENTER') {
    return <TrendingUp className="w-2.5 h-2.5" />;
  } else if (actionType === 'EXIT') {
    return <TrendingDown className="w-2.5 h-2.5" />;
  } else if (actionType === 'HOLD') {
    return <Pause className="w-2.5 h-2.5" />;
  } else {
    return <Bot className="w-2.5 h-2.5" />;
  }
};

export const RLActionBadge = ({
  actionType,
  probability,
  confidence,
  holdProbability,
  stateValue,
  modelVersion,
  size = 'sm',
  showDetails = false,
  className
}: RLActionBadgeProps) => {
  // Don't render if no RL data
  if (probability === undefined && confidence === undefined) {
    return null;
  }

  // Determine which action has the highest probability
  const actionProb = probability ?? 0;
  const holdProb = holdProbability ?? 0;

  const actualAction = actionProb > holdProb ? actionType : 'HOLD';
  const actualProbability = Math.max(actionProb, holdProb);
  const secondaryProbability = Math.min(actionProb, holdProb);

  // Determine the secondary action label
  const secondaryActionLabel = actualAction === 'HOLD' ? actionType : 'HOLD';

  // Color based on action type: ENTER=green, HOLD=yellow, EXIT=red
  const getActionVariant = (action: string): 'success' | 'warning' | 'danger' => {
    if (action === 'ENTER') return 'success';
    if (action === 'HOLD') return 'warning';
    return 'danger'; // EXIT
  };

  const variant = getActionVariant(actualAction);

  // Color classes based on variant
  const variantClasses = {
    success: 'bg-binance-green/10 text-binance-green border-binance-green/20',
    warning: 'bg-binance-yellow/10 text-binance-yellow border-binance-yellow/20',
    danger: 'bg-binance-red/10 text-binance-red border-binance-red/20',
  };

  if (showDetails) {
    // Detailed view for dialogs
    return (
      <div className={cn('inline-flex flex-col gap-1 w-full', className)}>
        <div className={cn(
          'inline-flex items-center justify-center gap-1.5 font-bold text-xl font-mono rounded border transition-colors px-4 py-1',
          variantClasses[variant]
        )}>
          <Bot className="w-4 h-4" />
          <span className="text-xs uppercase tracking-wide">{actualAction}</span>
          <Zap className="w-3 h-3" />
          <span>{actualProbability !== undefined ? `${(actualProbability * 100).toFixed(0)}%` : '--'}</span>
        </div>
        <div className="flex items-center justify-center gap-3 text-[10px]">
          <div className="flex items-center gap-0.5">
            <span className="text-binance-text-secondary">Confidence:</span>
            <span className={cn('font-mono font-bold', getConfidenceColor(confidence))}>
              {confidence || '--'}
            </span>
          </div>
          <div className="flex items-center gap-0.5">
            <span className="text-binance-text-secondary">{secondaryActionLabel}:</span>
            <span className={cn('font-mono font-medium', getProbabilityColor(secondaryProbability))}>
              {secondaryProbability !== undefined ? `${(secondaryProbability * 100).toFixed(0)}%` : '--'}
            </span>
          </div>
          {stateValue !== undefined && (
            <div className="flex items-center gap-0.5">
              <span className="text-binance-text-secondary">Value:</span>
              <span className={cn('font-mono font-medium', stateValue > 0 ? 'text-binance-green' : 'text-binance-red')}>
                {stateValue.toFixed(1)}
              </span>
            </div>
          )}
        </div>
        {modelVersion && (
          <div className="text-[8px] text-binance-text-secondary text-center font-mono">
            {modelVersion}
          </div>
        )}
      </div>
    );
  }

  // Compact view for tables
  return (
    <div className={cn('inline-flex flex-col items-center justify-center w-full', className)}>
      <div className={cn(
        'inline-flex items-center justify-center gap-1 font-bold font-mono rounded border transition-colors px-2 py-0.5 mb-1 text-[10px]',
        variantClasses[variant]
      )}>
        {getActionIcon(actualAction as 'ENTER' | 'EXIT', actualProbability)}
        <span className="text-[8px] uppercase tracking-wide">{actualAction}</span>
        <span>{actualProbability !== undefined ? `${(actualProbability * 100).toFixed(0)}%` : '--'}</span>
      </div>
      <div className="flex items-center justify-center gap-1.5 text-[8px]">
        <div className="flex items-center gap-0.5">
          <span className={cn('font-mono font-bold', getConfidenceColor(confidence))}>
            {confidence ? confidence.charAt(0) : '--'}
          </span>
        </div>
        {secondaryProbability !== undefined && (
          <div className="flex items-center gap-0.5">
            <span className="text-binance-text-secondary/70">{secondaryActionLabel.charAt(0)}:</span>
            <span className={cn('font-mono font-medium', getProbabilityColor(secondaryProbability))}>
              {(secondaryProbability * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
