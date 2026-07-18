export type TradingViewWidget = {
  remove?: () => void;
  onChartReady?: (callback: () => void) => void;
  activeChart?: () => {
    createStudy?: (name: string, forceOverlay?: boolean, lock?: boolean, inputs?: Record<string, unknown>) => void;
    setResolution?: (resolution: string, callback?: () => void) => void;
  };
};

export type TradingViewWidgetOptions = Record<string, unknown>;

declare global {
  interface Window {
    TradingView?: {
      widget: new (options: TradingViewWidgetOptions) => TradingViewWidget;
    };
  }
}

let loadPromise: Promise<boolean> | null = null;

export function hasTradingViewLibrary(): boolean {
  return typeof window !== 'undefined' && typeof window.TradingView?.widget === 'function';
}

export function loadTradingViewLibrary(): Promise<boolean> {
  if (hasTradingViewLibrary()) return Promise.resolve(true);
  if (loadPromise) return loadPromise;
  loadPromise = new Promise((resolve) => {
    if (typeof document === 'undefined') {
      resolve(false);
      return;
    }
    const existing = document.querySelector<HTMLScriptElement>('script[data-tradingview-charting-library="true"]');
    if (existing) {
      existing.addEventListener('load', () => resolve(hasTradingViewLibrary()), { once: true });
      existing.addEventListener('error', () => resolve(false), { once: true });
      return;
    }
    const script = document.createElement('script');
    script.src = '/charting_library/charting_library.js';
    script.async = true;
    script.dataset.tradingviewChartingLibrary = 'true';
    script.onload = () => resolve(hasTradingViewLibrary());
    script.onerror = () => resolve(false);
    document.head.appendChild(script);
  });
  return loadPromise;
}
