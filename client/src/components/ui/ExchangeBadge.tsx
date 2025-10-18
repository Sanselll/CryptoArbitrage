import { getExchangeFaviconUrl, getExchangeBorderColor, getExchangeBgColor, getExchangeColor } from '../../lib/exchangeUtils';

interface ExchangeBadgeProps {
  exchange: string;
  className?: string;
}

export const ExchangeBadge = ({ exchange, className = '' }: ExchangeBadgeProps) => {
  const faviconUrl = getExchangeFaviconUrl(exchange);
  const borderColor = getExchangeBorderColor(exchange);
  const bgColor = getExchangeBgColor(exchange);
  const textColor = getExchangeColor(exchange);

  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border ${borderColor} ${bgColor} ${className}`}>
      {faviconUrl && (
        <img
          src={faviconUrl}
          alt={`${exchange} icon`}
          className="w-3 h-3"
          onError={(e) => {
            // Hide image if it fails to load
            e.currentTarget.style.display = 'none';
          }}
        />
      )}
      <span className={`text-[10px] font-semibold ${textColor}`}>
        {exchange}
      </span>
    </div>
  );
};
