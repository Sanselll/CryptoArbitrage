import { forwardRef } from 'react'
import { cn } from '../../lib/cn'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      className,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      disabled,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          'inline-flex items-center justify-center font-medium rounded-md transition-all duration-200',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-binance-yellow focus-visible:ring-offset-2 focus-visible:ring-offset-binance-bg',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          // Size variants
          {
            'px-3 py-1.5 text-sm': size === 'sm',
            'px-4 py-2 text-sm': size === 'md',
            'px-6 py-3 text-base': size === 'lg',
          },
          // Color variants - TradingAgent style
          {
            'bg-binance-yellow text-gray-900 hover:bg-opacity-90 font-semibold':
              variant === 'primary',
            'bg-binance-bg-secondary border border-binance-border text-binance-text hover:bg-binance-bg-hover':
              variant === 'secondary',
            'bg-binance-green text-white hover:bg-opacity-90 font-semibold':
              variant === 'success',
            'bg-binance-red text-white hover:bg-opacity-80 font-semibold':
              variant === 'danger',
            'bg-transparent text-binance-text hover:bg-binance-bg-hover':
              variant === 'ghost',
          },
          className
        )}
        {...props}
      >
        {isLoading && (
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'

export { Button }
