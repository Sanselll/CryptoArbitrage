import { forwardRef } from 'react'
import { cn } from '../../lib/cn'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info' | 'secondary'
  size?: 'sm' | 'md' | 'lg'
}

const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ children, className, variant = 'default', size = 'md', ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center font-medium rounded transition-colors',
          // Size variants
          {
            'px-2 py-0.5 text-xs': size === 'sm',
            'px-2.5 py-1 text-sm': size === 'md',
            'px-3 py-1.5 text-base': size === 'lg',
          },
          // Color variants
          {
            'bg-binance-bg-tertiary text-binance-text border border-binance-border':
              variant === 'default',
            'bg-binance-green/10 text-binance-green border border-binance-green/20':
              variant === 'success',
            'bg-binance-red/10 text-binance-red border border-binance-red/20':
              variant === 'danger',
            'bg-binance-yellow/10 text-binance-yellow border border-binance-yellow/20':
              variant === 'warning',
            'bg-binance-blue/10 text-binance-blue border border-binance-blue/20':
              variant === 'info',
            'bg-binance-bg-hover text-binance-text-secondary border border-binance-border':
              variant === 'secondary',
          },
          className
        )}
        {...props}
      >
        {children}
      </span>
    )
  }
)

Badge.displayName = 'Badge'

export { Badge }
