import { getExchangeFaviconUrl, getExchangeBorderColor, getExchangeBgColor, getExchangeColor } from '../../lib/exchangeUtils';

interface ExchangeBadgeProps {
  exchange: string;
  className?: string;
  size?: 'default' | 'small';
}

export const ExchangeBadge = ({ exchange, className = '', size = 'default' }: ExchangeBadgeProps) => {
  const faviconUrl = getExchangeFaviconUrl(exchange);
  const borderColor = getExchangeBorderColor(exchange);
  const bgColor = getExchangeBgColor(exchange);
  const textColor = getExchangeColor(exchange);

  const sizeClasses = size === 'small'
    ? 'gap-0.5 px-1 py-0'
    : 'gap-1.5 px-2 py-0.5';

  const iconSize = size === 'small' ? 'w-2 h-2' : 'w-3 h-3';
  const textSize = size === 'small' ? 'text-[8px]' : 'text-[10px]';

  return (
    <div className={`inline-flex items-center rounded border ${borderColor} ${bgColor} ${sizeClasses} ${className}`}>
      {faviconUrl && (
        <img
          src={faviconUrl}
          alt={`${exchange} icon`}
          className={iconSize}
          onError={(e) => {
            // Hide image if it fails to load
            e.currentTarget.style.display = 'none';
          }}
        />
      )}
      <span className={`font-semibold ${textColor} ${textSize}`}>
        {exchange}
      </span>
    </div>
  );
};
