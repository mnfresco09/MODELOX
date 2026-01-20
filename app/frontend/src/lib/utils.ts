import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return n.toFixed(decimals)
}

export function formatPercent(n: number): string {
  return n.toFixed(2) + '%'
}

export function getScoreColor(score: number): string {
  if (score >= 10) return 'text-emerald-400'
  if (score >= 5) return 'text-green-400'
  if (score >= 0) return 'text-yellow-400'
  return 'text-red-400'
}
