import { cn } from '../../lib/cn'

interface EmptyStateProps {
  title: string
  description?: string
  icon?: React.ReactNode
  action?: React.ReactNode
  className?: string
}

export function EmptyState({
  title,
  description,
  icon,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center p-8 text-center',
        className
      )}
    >
      {icon && (
        <div className="mb-4 text-binance-text-muted opacity-50">{icon}</div>
      )}
      <h3 className="text-lg font-semibold text-binance-text mb-2">{title}</h3>
      {description && (
        <p className="text-sm text-binance-text-secondary max-w-sm">
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
