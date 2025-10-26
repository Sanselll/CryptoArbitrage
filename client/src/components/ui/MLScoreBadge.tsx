import { cn } from '../../lib/cn';
import { TrendingUp, TrendingDown, Target, Clock, Brain } from 'lucide-react';

interface MLScoreBadgeProps {
  score: number; // 0-100
  profitPrediction?: number; // Expected profit %
  successProbability?: number; // 0-1
  holdDuration?: number; // hours
  modelVersion?: string;
  size?: 'sm' | 'md';
  showDetails?: boolean;
  className?: string;
}

const getScoreVariant = (score: number): 'success' | 'warning' | 'danger' => {
  if (score >= 70) return 'success';
  if (score >= 40) return 'warning';
  return 'danger';
};

const getSuccessColor = (probability?: number): string => {
  if (probability === undefined || probability === null) return 'text-binance-text-secondary';
  if (probability >= 0.7) return 'text-binance-green';
  if (probability >= 0.4) return 'text-binance-yellow';
  return 'text-binance-red';
};

const getProfitColor = (profit?: number): string => {
  if (profit === undefined || profit === null) return 'text-binance-text-secondary';
  if (profit >= 1.0) return 'text-binance-green';
  if (profit >= 0) return 'text-binance-yellow';
  return 'text-binance-red';
};

const getDurationColor = (hours?: number): string => {
  if (hours === undefined || hours === null) return 'text-binance-text-secondary';
  if (hours <= 24) return 'text-binance-green';
  if (hours <= 72) return 'text-binance-yellow';
  return 'text-binance-red';
};

export const MLScoreBadge = ({
  score,
  profitPrediction,
  successProbability,
  holdDuration,
  size = 'sm',
  showDetails = false,
  className
}: MLScoreBadgeProps) => {
  const variant = getScoreVariant(score);

  // Color classes based on variant (matching Badge component)
  const variantClasses = {
    success: 'bg-binance-green/10 text-binance-green border-binance-green/20',
    warning: 'bg-binance-yellow/10 text-binance-yellow border-binance-yellow/20',
    danger: 'bg-binance-red/10 text-binance-red border-binance-red/20',
  };

  if (showDetails) {
    // Detailed view for Execute Dialog - badge for score, separate colored metrics
    return (
      <div className={cn('inline-flex flex-col gap-1 w-full', className)}>
        <div className={cn(
          'inline-flex items-center justify-center gap-1 font-bold text-xl font-mono rounded border transition-colors px-4 py-1',
          variantClasses[variant]
        )}>
          <Brain className="w-4 h-4" />
          {Math.round(score)}
        </div>
        <div className="flex items-center justify-center gap-3 text-[10px]">
          <div className="flex items-center gap-0.5">
            <Target className={cn('w-3 h-3', getSuccessColor(successProbability))} />
            <span className={cn('font-mono font-medium', getSuccessColor(successProbability))}>
              {successProbability !== undefined && successProbability !== null ? `${(successProbability * 100).toFixed(0)}%` : '--'}
            </span>
          </div>
          <div className="flex items-center gap-0.5">
            {profitPrediction !== undefined && profitPrediction !== null && profitPrediction < 0 ? (
              <TrendingDown className={cn('w-3 h-3', getProfitColor(profitPrediction))} />
            ) : (
              <TrendingUp className={cn('w-3 h-3', getProfitColor(profitPrediction))} />
            )}
            <span className={cn('font-mono font-medium', getProfitColor(profitPrediction))}>
              {profitPrediction !== undefined && profitPrediction !== null ? `${profitPrediction >= 0 ? '+' : ''}${profitPrediction.toFixed(2)}%` : '--'}
            </span>
          </div>
          <div className="flex items-center gap-0.5">
            <Clock className={cn('w-3 h-3', getDurationColor(holdDuration))} />
            <span className={cn('font-mono font-medium', getDurationColor(holdDuration))}>
              {holdDuration !== undefined && holdDuration !== null ? `${Math.round(holdDuration)}h` : '--'}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Compact view for table - badge for score matching APR badge size, separate colored metrics below
  return (
    <div className={cn('inline-flex flex-col items-center justify-center w-full', className)}>
      <div className={cn(
        'inline-flex items-center justify-center gap-1 font-bold font-mono rounded border transition-colors px-2 py-0.5 mb-1 text-[10px]',
        variantClasses[variant]
      )}>
        <Brain className="w-2.5 h-2.5" />
        {Math.round(score)}
      </div>
      <div className="flex items-center justify-center gap-1 text-[8px]">
        <div className="flex items-center gap-0.5">
          <Target className={cn('w-2.5 h-2.5', getSuccessColor(successProbability))} />
          <span className={cn('font-mono font-medium', getSuccessColor(successProbability))}>
            {successProbability !== undefined && successProbability !== null ? `${(successProbability * 100).toFixed(0)}%` : '--'}
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          {profitPrediction !== undefined && profitPrediction !== null && profitPrediction < 0 ? (
            <TrendingDown className={cn('w-2.5 h-2.5', getProfitColor(profitPrediction))} />
          ) : (
            <TrendingUp className={cn('w-2.5 h-2.5', getProfitColor(profitPrediction))} />
          )}
          <span className={cn('font-mono font-medium', getProfitColor(profitPrediction))}>
            {profitPrediction !== undefined && profitPrediction !== null ? `${profitPrediction >= 0 ? '+' : ''}${profitPrediction.toFixed(1)}%` : '--'}
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <Clock className={cn('w-2.5 h-2.5', getDurationColor(holdDuration))} />
          <span className={cn('font-mono font-medium', getDurationColor(holdDuration))}>
            {holdDuration !== undefined && holdDuration !== null ? `${Math.round(holdDuration)}h` : '--'}
          </span>
        </div>
      </div>
    </div>
  );
};
