import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utility function to merge Tailwind CSS classes with proper precedence
 * Combines clsx for conditional classes and tailwind-merge for conflict resolution
 *
 * @example
 * cn('px-2 py-1', condition && 'bg-red-500', 'bg-blue-500')
 * // Result: 'px-2 py-1 bg-blue-500' (bg-blue-500 takes precedence)
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
