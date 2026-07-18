import { fetchMarketBars } from './api';
import type { Candle, Market } from './types';

type DatafeedCallback = (...args: unknown[]) => void;

type SymbolInfo = {
  name: string;
  ticker: string;
  description: string;
  type: string;
  session: string;
  timezone: string;
  exchange: string;
  minmov: number;
  pricescale: number;
  has_intraday: boolean;
  has_daily: boolean;
  supported_resolutions: string[];
  volume_precision: number;
  data_status: string;
};

type PeriodParams = {
  from?: number;
  to?: number;
  countBack?: number;
  firstDataRequest?: boolean;
};

type RealtimeSubscription = {
  socket: WebSocket;
};

const SUPPORTED_RESOLUTIONS = ['1', '5', '15', '30', '60', '240', '1D'];

const RESOLUTION_TO_INTERVAL: Record<string, string> = {
  '1': '1m',
  '5': '5m',
  '15': '15m',
  '30': '30m',
  '60': '1h',
  '240': '4h',
  '1D': '1d'
};

function wsUrlForNetwork(network: string | undefined): string {
  return String(network || '').toLowerCase() === 'testnet'
    ? 'wss://api.hyperliquid-testnet.xyz/ws'
    : 'wss://api.hyperliquid.xyz/ws';
}

function normalizeRealtimeCandle(raw: Record<string, unknown>): Candle | null {
  const timestamp = Number(raw.time ?? raw.t ?? raw.T ?? 0);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return null;
  const open = Number(raw.open ?? raw.o ?? 0);
  const high = Number(raw.high ?? raw.h ?? 0);
  const low = Number(raw.low ?? raw.l ?? 0);
  const close = Number(raw.close ?? raw.c ?? 0);
  if ([open, high, low, close].some((item) => !Number.isFinite(item) || item <= 0)) return null;
  return {
    time: timestamp > 10_000_000_000 ? Math.floor(timestamp / 1000) : Math.floor(timestamp),
    open,
    high,
    low,
    close,
    volume: Number(raw.volume ?? raw.v ?? 0)
  };
}

function toTradingViewBar(candle: Candle) {
  return {
    time: Number(candle.time) * 1000,
    open: Number(candle.open),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
    volume: Number(candle.volume || 0)
  };
}

function marketLabel(market: Market): string {
  return market.display_name || `${market.symbol}-USDC`;
}

function symbolInfoForMarket(market: Market): SymbolInfo {
  return {
    name: marketLabel(market),
    ticker: market.symbol,
    description: marketLabel(market),
    type: 'crypto',
    session: '24x7',
    timezone: 'Etc/UTC',
    exchange: 'Hyperliquid',
    minmov: 1,
    pricescale: 100000,
    has_intraday: true,
    has_daily: true,
    supported_resolutions: SUPPORTED_RESOLUTIONS,
    volume_precision: 4,
    data_status: 'streaming'
  };
}

function findMarket(markets: Market[], symbolName: string): Market | undefined {
  return markets.find(
    (market) =>
      market.symbol === symbolName ||
      market.execution_symbol === symbolName ||
      market.display_name === symbolName ||
      marketLabel(market) === symbolName
  );
}

export function intervalToTradingViewResolution(interval: string): string {
  const match = Object.entries(RESOLUTION_TO_INTERVAL).find(([, value]) => value === interval);
  return match?.[0] || '1';
}

export function createHyperliquidDatafeed({
  token,
  markets,
  network
}: {
  token: string;
  markets: Market[];
  network: string | undefined;
}) {
  const subscriptions = new Map<string, RealtimeSubscription>();

  return {
    onReady(callback: DatafeedCallback) {
      setTimeout(
        () =>
          callback({
            supported_resolutions: SUPPORTED_RESOLUTIONS,
            exchanges: [{ value: 'Hyperliquid', name: 'Hyperliquid', desc: 'Hyperliquid' }],
            symbols_types: [{ name: 'crypto', value: 'crypto' }],
            supports_marks: false,
            supports_timescale_marks: false,
            supports_time: true
          }),
        0
      );
    },

    searchSymbols(userInput: string, _exchange: string, _symbolType: string, onResultReadyCallback: DatafeedCallback) {
      const needle = String(userInput || '').toLowerCase();
      const results = markets
        .filter((market) =>
          [market.symbol, market.execution_symbol, market.display_name, market.dex, market.market_name]
            .filter(Boolean)
            .join(' ')
            .toLowerCase()
            .includes(needle)
        )
        .slice(0, 50)
        .map((market) => ({
          symbol: market.symbol,
          full_name: marketLabel(market),
          description: marketLabel(market),
          exchange: 'Hyperliquid',
          ticker: market.symbol,
          type: 'crypto'
        }));
      onResultReadyCallback(results);
    },

    resolveSymbol(
      symbolName: string,
      onSymbolResolvedCallback: DatafeedCallback,
      onResolveErrorCallback: DatafeedCallback
    ) {
      const market = findMarket(markets, symbolName);
      if (!market) {
        onResolveErrorCallback('unknown_symbol');
        return;
      }
      onSymbolResolvedCallback(symbolInfoForMarket(market));
    },

    async getBars(
      symbolInfo: SymbolInfo,
      resolution: string,
      periodParams: PeriodParams,
      onHistoryCallback: DatafeedCallback,
      onErrorCallback: DatafeedCallback
    ) {
      try {
        const result = await fetchMarketBars(
          token,
          symbolInfo.ticker,
          resolution,
          periodParams.from,
          periodParams.to,
          periodParams.countBack
        );
        const bars = result.bars.map(toTradingViewBar);
        onHistoryCallback(bars, { noData: bars.length === 0 || result.no_data });
      } catch (exc) {
        onErrorCallback(exc instanceof Error ? exc.message : String(exc));
      }
    },

    subscribeBars(
      symbolInfo: SymbolInfo,
      resolution: string,
      onRealtimeCallback: DatafeedCallback,
      subscriberUID: string
    ) {
      const market = findMarket(markets, symbolInfo.ticker);
      if (!market) return;
      const socket = new WebSocket(wsUrlForNetwork(network));
      const interval = RESOLUTION_TO_INTERVAL[resolution] || '1m';
      const coin = market.ws_symbol || market.execution_symbol || market.symbol;
      socket.onopen = () => {
        socket.send(JSON.stringify({ method: 'subscribe', subscription: { type: 'candle', coin, interval } }));
      };
      socket.onmessage = (event) => {
        let payload: { channel?: string; data?: unknown };
        try {
          payload = JSON.parse(String(event.data));
        } catch {
          return;
        }
        if (payload.channel !== 'candle' || !payload.data) return;
        const rawItems = Array.isArray(payload.data) ? payload.data : [payload.data];
        for (const item of rawItems) {
          if (!item || typeof item !== 'object') continue;
          const candle = normalizeRealtimeCandle(item as Record<string, unknown>);
          if (candle) onRealtimeCallback(toTradingViewBar(candle));
        }
      };
      subscriptions.set(subscriberUID, { socket });
    },

    unsubscribeBars(subscriberUID: string) {
      subscriptions.get(subscriberUID)?.socket.close();
      subscriptions.delete(subscriberUID);
    }
  };
}
