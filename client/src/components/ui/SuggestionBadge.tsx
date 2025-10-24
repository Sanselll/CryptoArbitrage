import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Brain } from 'lucide-react';
import { EntryRecommendation, OpportunitySuggestion, RecommendedStrategyType } from '../../types/index';
import { SuggestionTooltip } from './SuggestionTooltip';

interface SuggestionBadgeProps {
  suggestion: OpportunitySuggestion;
  entryScore?: number; // For positions grid - shows entry confidence
  recommendedStrategy?: string; // For positions grid
}

// Helper function to get recommendation details
const getRecommendationDetails = (recommendation: EntryRecommendation): {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
} => {
  switch (recommendation) {
    case EntryRecommendation.StrongBuy:
      return {
        label: 'Strong Buy',
        color: 'text-green-400',
        bgColor: 'bg-green-500/20',
        borderColor: 'border-green-500/50'
      };
    case EntryRecommendation.Buy:
      return {
        label: 'Buy',
        color: 'text-blue-400',
        bgColor: 'bg-blue-500/20',
        borderColor: 'border-blue-500/50'
      };
    case EntryRecommendation.Hold:
      return {
        label: 'Hold',
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-500/20',
        borderColor: 'border-yellow-500/50'
      };
    case EntryRecommendation.Skip:
    default:
      return {
        label: 'Skip',
        color: 'text-red-400',
        bgColor: 'bg-red-500/20',
        borderColor: 'border-red-500/50'
      };
  }
};

// Helper to get color by confidence score
const getScoreColor = (score: number): {
  color: string;
  bgColor: string;
  borderColor: string;
} => {
  if (score >= 80) {
    return {
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
      borderColor: 'border-green-500/50'
    };
  } else if (score >= 60) {
    return {
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
      borderColor: 'border-blue-500/50'
    };
  } else if (score >= 40) {
    return {
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/20',
      borderColor: 'border-yellow-500/50'
    };
  } else {
    return {
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
      borderColor: 'border-red-500/50'
    };
  }
};

export const SuggestionBadge = ({ suggestion, entryScore, recommendedStrategy }: SuggestionBadgeProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const badgeRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // For positions grid, use entry score if provided
  const displayScore = entryScore ?? suggestion?.confidenceScore;
  const isPositionEntry = entryScore !== undefined;

  if (!displayScore) return null;

  const scoreDetails = getScoreColor(displayScore);
  const recDetails = suggestion ? getRecommendationDetails(suggestion.entryRecommendation) : scoreDetails;

  // Update tooltip position when shown
  useEffect(() => {
    if (showTooltip && badgeRef.current) {
      const rect = badgeRef.current.getBoundingClientRect();
      const tooltipWidth = 420; // Match the tooltip width
      const tooltipHeight = 300; // Approximate tooltip height
      const padding = 16; // Padding from viewport edges

      let top = rect.bottom + window.scrollY + 8;
      let left = rect.left + window.scrollX;

      // Adjust horizontal position if tooltip would overflow right edge
      if (left + tooltipWidth > window.innerWidth - padding) {
        left = window.innerWidth - tooltipWidth - padding + window.scrollX;
      }

      // Adjust horizontal position if tooltip would overflow left edge
      if (left < padding) {
        left = padding + window.scrollX;
      }

      // If tooltip would overflow bottom, show it above the badge instead
      if (rect.bottom + tooltipHeight > window.innerHeight) {
        top = rect.top + window.scrollY - tooltipHeight - 8;

        // If showing above would overflow top, just show below and let it scroll
        if (top < padding) {
          top = rect.bottom + window.scrollY + 8;
        }
      }

      setTooltipPosition({ top, left });
    }
  }, [showTooltip]);

  // Close tooltip when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        tooltipRef.current &&
        badgeRef.current &&
        !tooltipRef.current.contains(event.target as Node) &&
        !badgeRef.current.contains(event.target as Node)
      ) {
        setShowTooltip(false);
      }
    };

    if (showTooltip) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showTooltip]);

  return (
    <>
      <div
        ref={badgeRef}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={`
          inline-flex items-center gap-1 px-2 py-1 rounded border
          ${scoreDetails.bgColor} ${scoreDetails.borderColor}
          cursor-help transition-all hover:scale-105
        `}
        title={isPositionEntry ? "Entry AI Score" : "AI Confidence Score"}
      >
        <Brain className={`w-3 h-3 ${scoreDetails.color}`} />
        <span className={`font-mono text-xs font-bold ${scoreDetails.color}`}>
          {Math.round(displayScore)}
        </span>
      </div>

      {showTooltip && suggestion && createPortal(
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            zIndex: 9999
          }}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          <SuggestionTooltip
            suggestion={suggestion}
            isPositionEntry={isPositionEntry}
            recommendedStrategy={recommendedStrategy}
          />
        </div>,
        document.body
      )}
    </>
  );
};
