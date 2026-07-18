import {
  type CSSProperties,
  type FormEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react';
import { createPortal } from 'react-dom';
import {
  Activity,
  BarChart3,
  BookOpen,
  Lock,
  LogOut,
  MinusCircle,
  PlusCircle,
  RefreshCw,
  SlidersHorizontal,
  Star
} from 'lucide-react';
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  HistogramSeries,
  LineSeries,
  LineStyle,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type LineData,
  type BarPrice,
  TickMarkType,
  type Time,
  type UTCTimestamp
} from 'lightweight-charts';
import {
  clearToken,
  fetchAccount,
  fetchAccountHistory,
  fetchFavoriteMarkets,
  fetchMarketBars,
  fetchMarketBook,
  fetchMarginLimits,
  fetchMarketSnapshot,
  fetchMarkets,
  fetchSession,
  getStoredToken,
  postLeverage,
  postMargin,
  postOrder,
  postPositionTpsl,
  putFavoriteMarkets,
  storeToken
} from './api';
import type {
  Account,
  AccountHistory,
  Candle,
  MarginLimits,
  Market,
  MarketSnapshot,
  MarketTrade,
  OrderBook,
  OrderBookLevel,
  Position,
  Session
} from './types';
import './styles.css';

const CHART_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
const WINDOW_SECONDS_BY_INTERVAL: Record<string, number> = {
  '1m': 86400,
  '5m': 259200,
  '15m': 604800,
  '30m': 1209600,
  '1h': 2592000,
  '4h': 7776000,
  '1d': 31536000
};
const CHART_INDICATORS_KEY = 'webTradeChartIndicators';
const CHART_TIMEZONE_KEY = 'webTradeChartTimezone';
const ORDER_BOOK_VISIBLE_LEVELS = 4;
const CHART_RIGHT_OFFSET_BARS = 8;
const CHART_RESET_VISIBLE_BARS = 120;
const CHART_PRICE_AXIS_WHEEL_ZONE_PX = 72;
const CHART_PRICE_WHEEL_ZOOM_SPEED = 0.0015;
const ORDER_PERCENT_STEP = 1;
const ORDER_PERCENT_TICKS = [0, 25, 50, 75, 100];
const NOTIFICATION_HIDE_MS = 4000;
const ACCOUNT_REFRESH_INTERVAL_MS = 5000;
const POSITION_ACTION_CONFIRM_INTERVAL_MS = 500;
const POSITION_ACTION_CONFIRM_TIMEOUT_MS = 30000;
const POSITION_SIZE_EPSILON = 1e-8;
const CHART_TIMEZONE_OPTIONS = [
  { value: 'local', label: 'Local', timeZone: undefined },
  { value: 'UTC', label: 'UTC', timeZone: 'UTC' },
  { value: 'America/New_York', label: 'New York', timeZone: 'America/New_York' },
  { value: 'Europe/London', label: 'London', timeZone: 'Europe/London' },
  { value: 'Asia/Shanghai', label: 'Shanghai', timeZone: 'Asia/Shanghai' },
  { value: 'Asia/Tokyo', label: 'Tokyo', timeZone: 'Asia/Tokyo' }
] as const;
type ChartTimezoneValue = (typeof CHART_TIMEZONE_OPTIONS)[number]['value'];

type IndicatorState = {
  volume: boolean;
  ema: boolean;
  sma: boolean;
  emaLength: number;
  smaLength: number;
};

type DesktopVerticalPanGesture = {
  pointerId: number;
  startY: number;
  height: number;
  range: { from: number; to: number };
  active: boolean;
};

type OrderTicketPrefill = {
  id: number;
  side: 'long' | 'short';
  orderType: 'market' | 'limit';
  margin: string;
  leverage: number;
  limitPrice: string;
  reduceOnly: boolean;
  message: string;
};

type RefreshHandler = () => void | Promise<void>;
type NotificationKind = 'pending' | 'success' | 'error';
type GlobalNotification = {
  id: number;
  kind: NotificationKind;
  message: string;
};
type NotifyHandler = (kind: NotificationKind, message: string) => void;

type PositionTradeAction = 'limit' | 'market' | 'reverse';
type FavoriteMetricMode = 'price' | 'change';
type OrderAmountMode = 'margin' | 'size';
type MobilePrimaryTab = 'chart' | 'trade';
type ChartCandleDetails = {
  time?: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  ema?: number;
  sma?: number;
};

const DEFAULT_INDICATORS: IndicatorState = {
  volume: false,
  ema: false,
  sma: false,
  emaLength: 20,
  smaLength: 50
};
const INDICATOR_PERIOD_MIN = 2;
const INDICATOR_PERIOD_MAX = 500;
const INDICATOR_EMA_COLOR = '#38bdf8';
const INDICATOR_SMA_COLOR = '#f59e0b';

const MOBILE_PRIMARY_TABS: { id: MobilePrimaryTab; label: string }[] = [
  { id: 'chart', label: 'Chart' },
  { id: 'trade', label: 'Trade' }
];

function usd(value: number | undefined): string {
  return `$${Number(value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function usdPrecise(value: number | undefined, maximumFractionDigits = 8): string {
  const amount = Number(value || 0);
  return `$${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits })}`;
}

function usdPreciseOrDash(value: number | undefined, maximumFractionDigits = 8): string {
  return typeof value === 'number' && Number.isFinite(value) ? usdPrecise(value, maximumFractionDigits) : '-';
}

function significantNumber(value: number | undefined, significantDigits = 5): string {
  const amount = Number(value || 0);
  if (!Number.isFinite(amount)) return significantNumber(0, significantDigits);
  const digits = Math.max(1, Math.floor(significantDigits || 1));
  const rounded = Number(amount.toPrecision(digits));
  const abs = Math.abs(rounded);
  let fractionDigits = digits - 1;
  if (abs > 0 && abs < 1) {
    fractionDigits = digits;
  } else if (abs >= 1) {
    const integerDigits = Math.floor(Math.log10(abs)) + 1;
    fractionDigits = Math.max(0, Math.min(digits, digits - integerDigits));
  }
  const formatted = rounded.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits
  });
  return rounded !== 0 && Number(formatted.replace(/,/g, '')) === 0 ? rounded.toExponential(digits - 1) : formatted;
}

function price(value: number | undefined): string {
  return significantNumber(value, 5);
}

function chartPriceFormat() {
  return {
    type: 'custom' as const,
    minMove: 0.00000001,
    formatter: (value: BarPrice) => price(Number(value)),
    tickmarksFormatter: (values: BarPrice[]) => values.map((value) => price(Number(value)))
  };
}

function priceOrDash(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? price(value) : '-';
}

function signedPrice(value: number | undefined): string {
  const amount = Number(value || 0);
  if (amount > 0) return `+${price(amount)}`;
  if (amount < 0) return `-${price(Math.abs(amount))}`;
  return price(0);
}

function quotePrice(value: number | undefined): string {
  return `$${price(value)}`;
}

function orderBookNumber(value: number | undefined): string {
  return significantNumber(value, 5);
}

function pct(value: number | undefined): string {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function qty(value: number | undefined): string {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function qtyPrecise(value: number | undefined): string {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function qtyPreciseOrDash(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? qtyPrecise(value) : '-';
}

function usdcAmount(value: number | undefined): string {
  return `${Number(value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDC`;
}

function usdcPrecise(value: number | undefined, minimumFractionDigits = 4, maximumFractionDigits = 8): string {
  const amount = Number(value || 0);
  return `${amount.toLocaleString(undefined, { minimumFractionDigits, maximumFractionDigits })} USDC`;
}

function usdcPreciseOrDash(value: number | undefined, minimumFractionDigits = 4, maximumFractionDigits = 8): string {
  return typeof value === 'number' && Number.isFinite(value) ? usdcPrecise(value, minimumFractionDigits, maximumFractionDigits) : '-';
}

function inputNumber(value: number | undefined): string {
  const amount = Number(value || 0);
  return Number.isFinite(amount) && amount > 0 ? String(Number(amount.toFixed(6))) : '';
}

function inputSizeNumber(value: number | undefined): string {
  const amount = Number(value || 0);
  return Number.isFinite(amount) && amount > 0 ? String(Number(amount.toFixed(8))) : '';
}

function signedUsd(value: number | undefined): string {
  const amount = Number(value || 0);
  if (amount < 0) return `-${usd(Math.abs(amount))}`;
  return usd(amount);
}

function signedUsdPrecise(value: number | undefined, maximumFractionDigits = 8): string {
  const amount = Number(value || 0);
  if (amount < 0) return `-${usdPrecise(Math.abs(amount), maximumFractionDigits)}`;
  return usdPrecise(amount, maximumFractionDigits);
}

function signedUsdPreciseOrDash(value: number | undefined, maximumFractionDigits = 8): string {
  return typeof value === 'number' && Number.isFinite(value) ? signedUsdPrecise(value, maximumFractionDigits) : '-';
}

function signedUsdWithPlus(value: number | undefined): string {
  const amount = Number(value || 0);
  if (amount > 0) return `+${usd(amount)}`;
  if (amount < 0) return `-${usd(Math.abs(amount))}`;
  return usd(0);
}

function signedPct(value: number | undefined): string {
  const amount = Number(value || 0) * 100;
  if (amount > 0) return `+${amount.toFixed(2)}%`;
  if (amount < 0) return `${amount.toFixed(2)}%`;
  return '0.00%';
}

function pctPrecise(value: number | undefined): string {
  const amount = Number(value || 0) * 100;
  return `${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}%`;
}

function fundingRateLabel(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
  return `${(value * 100).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 })}%`;
}

function rawText(row: Record<string, unknown> | undefined, ...keys: string[]): string {
  if (!row) return '-';
  for (const key of keys) {
    const value = row[key];
    if (value !== undefined && value !== null && value !== '') return String(value);
  }
  return '-';
}

function rawNumber(row: Record<string, unknown> | undefined, ...keys: string[]): number | undefined {
  const text = rawText(row, ...keys);
  const value = Number(text);
  return Number.isFinite(value) ? value : undefined;
}

function rowTimeMs(row: Record<string, unknown> | undefined, ...keys: string[]): number | undefined {
  const value = rawNumber(row, ...keys);
  if (!value || value <= 0) return undefined;
  return value > 10_000_000_000 ? value : value * 1000;
}

function historyRowTimeMs(row: Record<string, unknown>): number {
  return rowTimeMs(row, 'statusTimestamp', 'timestamp', 'time', 'startTime', 'endTime') || 0;
}

function sortedHistoryRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  return rows.slice().sort((left, right) => historyRowTimeMs(right) - historyRowTimeMs(left));
}

function timeLabel(value: number | undefined, timezone: ChartTimezoneValue = 'local'): string {
  if (!value) return '-';
  const timeZone = chartTimezoneName(timezone);
  return new Date(value).toLocaleString(undefined, {
    timeZone,
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function tradeTimeLabel(value: number | undefined, timezone: ChartTimezoneValue = 'local'): string {
  if (!value) return '-';
  const timeZone = chartTimezoneName(timezone);
  return new Date(value).toLocaleTimeString(undefined, {
    timeZone,
    hour: '2-digit',
    minute: '2-digit'
  });
}

function historyFullTimeLabel(value: number | undefined, timezone: ChartTimezoneValue = 'local'): string {
  if (!value) return '-';
  const timeZone = chartTimezoneName(timezone);
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23'
  }).formatToParts(new Date(value));
  const part = (type: string) => parts.find((item) => item.type === type)?.value || '00';
  return `${part('month')}/${part('day')}/${part('year')} - ${part('hour')}:${part('minute')}:${part('second')}`;
}

function chartCandleTimeLabel(value: number | undefined, timezone: ChartTimezoneValue): string {
  if (!value) return '-';
  return chartTimeFormatter(timezone, 'crosshair')(value as UTCTimestamp);
}

function candleDetailsFromCandle(candle: Candle): ChartCandleDetails {
  return {
    time: Number(candle.time),
    open: Number(candle.open),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
    volume: Number(candle.volume || 0)
  };
}

function candleDetailsFromCrosshair(raw: unknown, fallback?: ChartCandleDetails): ChartCandleDetails | null {
  if (!raw || typeof raw !== 'object') return fallback || null;
  const row = raw as Record<string, unknown>;
  const open = Number(row.open ?? fallback?.open);
  const high = Number(row.high ?? fallback?.high);
  const low = Number(row.low ?? fallback?.low);
  const close = Number(row.close ?? fallback?.close);
  if ([open, high, low, close].some((item) => !Number.isFinite(item))) return fallback || null;
  const time = Number(row.time ?? fallback?.time);
  const volume = Number(row.volume ?? row.value ?? fallback?.volume ?? 0);
  const ema = Number(row.ema ?? fallback?.ema);
  const sma = Number(row.sma ?? fallback?.sma);
  const details: ChartCandleDetails = {
    time: Number.isFinite(time) ? time : fallback?.time,
    open,
    high,
    low,
    close,
    volume: Number.isFinite(volume) ? volume : fallback?.volume
  };
  if (Number.isFinite(ema)) details.ema = ema;
  if (Number.isFinite(sma)) details.sma = sma;
  return details;
}

function chartDataPriceRange(
  data: CandlestickData[],
  logicalRange?: { from: number; to: number } | null
): { from: number; to: number } | null {
  if (data.length === 0) return null;
  const rawFrom = Number(logicalRange?.from);
  const rawTo = Number(logicalRange?.to);
  const start = Number.isFinite(rawFrom) ? Math.max(0, Math.floor(rawFrom)) : 0;
  const end = Number.isFinite(rawTo) ? Math.min(data.length - 1, Math.ceil(rawTo)) : data.length - 1;
  const visibleData = start <= end ? data.slice(start, end + 1) : data;
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (const candle of visibleData) {
    const candleLow = Number(candle.low);
    const candleHigh = Number(candle.high);
    if (Number.isFinite(candleLow)) low = Math.min(low, candleLow);
    if (Number.isFinite(candleHigh)) high = Math.max(high, candleHigh);
  }
  if (!Number.isFinite(low) || !Number.isFinite(high)) return null;
  if (high > low) return { from: low, to: high };
  const padding = Math.max(Math.abs(high) * 0.01, 1e-8);
  return { from: low - padding, to: high + padding };
}

function sideLabel(value: string): string {
  if (value === 'B') return 'Buy';
  if (value === 'A') return 'Sell';
  return value || '-';
}

function booleanText(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes'].includes(normalized)) return true;
    if (['false', '0', 'no'].includes(normalized)) return false;
  }
  if (typeof value === 'number') return value !== 0;
  return undefined;
}

function tradeValue(row: Record<string, unknown>): number | undefined {
  const explicit = rawNumber(row, 'value', 'notional', 'ntl', 'tradeValue', 'trade_value');
  if (typeof explicit === 'number') return explicit;
  const fillPrice = rawNumber(row, 'px', 'price');
  const fillSize = rawNumber(row, 'sz', 'size');
  if (typeof fillPrice !== 'number' || typeof fillSize !== 'number') return undefined;
  return fillPrice * fillSize;
}

function tradeFeeLabel(row: Record<string, unknown>): string {
  const fee = rawNumber(row, 'fee');
  if (typeof fee !== 'number') return '-';
  const token = rawText(row, 'feeToken', 'fee_token');
  return token === '-' ? usdPrecise(fee) : `${usdPrecise(fee)} ${token}`;
}

function isLiquidationFill(row: Record<string, unknown>): boolean {
  const liquidation = row.liquidation ?? row.liq;
  if (liquidation && typeof liquidation === 'object' && !Array.isArray(liquidation)) return Object.keys(liquidation).length > 0;
  if (typeof liquidation === 'boolean') return liquidation;
  if (typeof liquidation === 'string' && liquidation.trim()) return true;
  const direction = rawText(row, 'dir', 'direction', 'type', 'orderType', 'order_type').toLowerCase();
  return direction.includes('liquidation') || direction.includes('liquidated');
}

function tradeDirectionLabel(row: Record<string, unknown>): string {
  const direction = rawText(row, 'dir', 'direction');
  if (!isLiquidationFill(row)) return direction;
  return direction === '-' ? 'Market Order Liquidation:' : `Market Order Liquidation: ${direction}`;
}

function liquidityLabel(row: Record<string, unknown>): string {
  const direct = rawText(row, 'liquidity', 'liquidityType', 'liquidity_type');
  if (direct !== '-') return direct;
  const crossed = booleanText(row.crossed);
  if (crossed === true) return 'Taker';
  if (crossed === false) return 'Maker';
  return '-';
}

function readableErrorMessage(exc: unknown, fallback: string): string {
  const raw = exc instanceof Error ? exc.message : String(exc || '');
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown; message?: unknown };
    const detail = parsed.detail ?? parsed.message;
    if (typeof detail === 'string' && detail.trim()) return detail;
  } catch {
    // The error is already plain text.
  }
  return raw;
}

function leverageFailureMessage(result: Record<string, unknown>): string {
  const message = rawText(result, 'message', 'detail');
  if (message !== '-') return message;
  const stage = rawText(result, 'stage');
  if (stage !== '-') return `Leverage adjustment was not applied (${stage}).`;
  return 'Leverage adjustment was not applied.';
}

function operationFailureMessage(result: Record<string, unknown>, fallback: string): string {
  const message = rawText(result, 'message', 'detail', 'error');
  if (message !== '-') return message;
  const stage = rawText(result, 'stage', 'status');
  if (stage !== '-') return `${fallback} (${stage}).`;
  return fallback;
}

function payloadHasOrderStatus(payload: unknown, statusKey: 'filled' | 'resting'): boolean {
  if (Array.isArray(payload)) return payload.some((item) => payloadHasOrderStatus(item, statusKey));
  if (!payload || typeof payload !== 'object') return false;
  const row = payload as Record<string, unknown>;
  const direct = row[statusKey];
  if (direct && typeof direct === 'object' && !Array.isArray(direct) && Object.keys(direct).length > 0) return true;
  return Object.values(row).some((value) => payloadHasOrderStatus(value, statusKey));
}

function acceptedOrderState(result: Record<string, unknown>, orderType: 'market' | 'limit'): 'completed' | 'submitted' {
  if (payloadHasOrderStatus(result, 'filled')) return 'completed';
  if (orderType === 'market' && !payloadHasOrderStatus(result, 'resting')) return 'completed';
  return 'submitted';
}

function positionFromLeverageResult(result: Record<string, unknown>): Position | null {
  const value = result.position;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const position = value as Partial<Position>;
  if (typeof position.symbol !== 'string' || !position.symbol.trim()) return null;
  return position as Position;
}

function mergeAccountPosition(current: Account | null, nextPosition: Position): Account | null {
  if (!current) return current;
  let replaced = false;
  const positions = current.positions || [];
  const nextPositions = positions.map((position) => {
    if (position.symbol !== nextPosition.symbol) return position;
    replaced = true;
    return { ...position, ...nextPosition };
  });
  return {
    ...current,
    positions: replaced ? nextPositions : [...nextPositions, nextPosition]
  };
}

function mergedHistoryDelta(row: Record<string, unknown>): Record<string, unknown> {
  const delta = row.delta;
  return delta && typeof delta === 'object' && !Array.isArray(delta)
    ? { ...row, ...(delta as Record<string, unknown>) }
    : row;
}

function positionMarketParts(symbol: string): { dex: string; market: string } {
  const value = String(symbol || '').trim();
  if (!value.includes(':')) return { dex: '', market: value };
  const [dex, ...rest] = value.split(':');
  return { dex, market: rest.join(':') || value };
}

function stripDexPrefix(value: string): string {
  const text = String(value || '').trim();
  if (!text.includes(':')) return text;
  return text.split(':').slice(1).join(':') || text;
}

function fundingMarketParts(row: Record<string, unknown>): { dex: string; market: string; asset: string } {
  const symbol = rawText(row, 'coin', 'symbol');
  const parts = positionMarketParts(symbol === '-' ? '' : symbol);
  const dex = rawText(row, 'dex', 'marketDex');
  const marketName = rawText(row, 'market', 'market_name', 'asset');
  const market = marketName !== '-' ? marketName : parts.market || symbol;
  const asset = stripDexPrefix(market).replace(/-USDC$/, '') || stripDexPrefix(symbol).replace(/-USDC$/, '') || '-';
  return { dex: dex !== '-' ? dex : parts.dex, market: asset, asset };
}

function fundingSizeLabel(row: Record<string, unknown>): string {
  const size = rawNumber(row, 'sz', 'szi', 'size');
  if (typeof size !== 'number') return '-';
  return `${qtyPrecise(Math.abs(size))} ${fundingMarketParts(row).asset}`;
}

function fundingSideLabel(row: Record<string, unknown>): string {
  const direct = rawText(row, 'side', 'direction');
  if (direct !== '-') return direct;
  const dir = rawText(row, 'dir').toLowerCase();
  if (dir.includes('short')) return 'Short';
  if (dir.includes('long')) return 'Long';
  const signedSize = rawNumber(row, 'szi', 'signed_size');
  if (typeof signedSize === 'number') return signedSize < 0 ? 'Short' : 'Long';
  return '-';
}

function positionSizeLabel(position: Position): string {
  const { market } = positionMarketParts(position.symbol);
  return `${qty(Math.abs(Number(position.size || 0)))} ${market || position.symbol}`;
}

function positionActionSizing(position: Position): { leverage: number; marginUsd: number } {
  const maxLeverage = Number(position.max_leverage || 50) || 50;
  const leverage = Math.max(1, Math.min(Number(position.leverage || 1), maxLeverage));
  const notionalMargin = Math.abs(Number(position.notional_usd || 0)) / leverage;
  const fallbackMargin = Number(position.lifecycle_roi_basis_usd || position.margin_used || 0);
  return { leverage, marginUsd: notionalMargin || fallbackMargin };
}

function oppositePositionSide(position: Position): 'long' | 'short' {
  return position.side === 'long' ? 'short' : 'long';
}

function accountPositionForSymbol(account: Account | null, symbol: string): Position | undefined {
  if (!account?.positions?.length) return undefined;
  const parts = positionMarketParts(symbol);
  const keys = new Set([symbol, stripDexPrefix(symbol), parts.market, parts.dex && parts.market ? `${parts.dex}:${parts.market}` : ''].filter(Boolean));
  return account.positions.find((position) => positionLiveKeys(position).some((key) => keys.has(key)));
}

function positionHasEffectiveSize(position: Position | undefined): boolean {
  return Math.abs(Number(position?.size || 0)) > POSITION_SIZE_EPSILON;
}

function positionActionConfirmed(account: Account | null, position: Position, action: Exclude<PositionTradeAction, 'limit'>): boolean {
  const current = accountPositionForSymbol(account, position.symbol);
  if (action === 'market') {
    return !positionHasEffectiveSize(current) || String(current?.side || '').toLowerCase() !== String(position.side || '').toLowerCase();
  }
  return positionHasEffectiveSize(current) && String(current?.side || '').toLowerCase() === oppositePositionSide(position);
}

function waitMs(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function marketAssetLabel(market: Market | null | undefined): string {
  if (!market) return '-';
  const display = stripDexPrefix(market.display_name || market.market_name || market.symbol);
  return (display || market.symbol).replace(/-USDC$/, '') || market.symbol;
}

function currentPositionLabel(position: Position | undefined, market: Market | null): string {
  const rawSize = Number(position?.size || 0);
  const side = String(position?.side || '').toLowerCase();
  const signedSize = side === 'short' && rawSize > 0 ? -rawSize : rawSize;
  const absoluteText = Math.abs(signedSize).toLocaleString(undefined, {
    minimumFractionDigits: 3,
    maximumFractionDigits: 6
  });
  const sign = signedSize < 0 ? '-' : '';
  const fallbackAsset = position ? positionMarketParts(position.symbol).market || position.symbol : '-';
  return `${sign}${absoluteText} ${marketAssetLabel(market) || fallbackAsset}`;
}

function clampOrderPercent(value: number): number {
  const amount = Math.min(100, Math.max(0, Number(value || 0)));
  return Math.round(amount);
}

function positionPnlLabel(position: Position): string {
  const pnl = position.synthetic_pnl_usd ?? position.unrealized_pnl;
  const pnlPct = position.synthetic_pnl_pct ?? position.return_on_equity;
  return `${signedUsdWithPlus(pnl)} (${signedPct(pnlPct)})`;
}

function positionFundingLabel(position: Position): string {
  if (typeof position.funding_since_open_usd !== 'number') return '-';
  return signedUsd(position.funding_since_open_usd);
}

function positionMarginLabel(position: Position): string {
  const mode = position.only_isolated ? 'Isolated' : 'Cross';
  return `${usd(position.lifecycle_roi_basis_usd || position.margin_used)} (${mode})`;
}

function nestedOrder(row: Record<string, unknown>): Record<string, unknown> {
  const nested = row.order;
  return nested && typeof nested === 'object' && !Array.isArray(nested) ? (nested as Record<string, unknown>) : row;
}

function orderMatchesPosition(order: Record<string, unknown>, position: Position): boolean {
  const source = nestedOrder(order);
  const coin = rawText(source, 'coin', 'symbol');
  const { market } = positionMarketParts(position.symbol);
  return coin === position.symbol || coin === market;
}

function orderTriggerPrice(order: Record<string, unknown>): string {
  const source = nestedOrder(order);
  const value = rawNumber(source, 'triggerPx', 'trigger_price', 'price', 'px', 'limitPx');
  return value ? price(value) : '--';
}

function positionTpSlLabel(position: Position, openOrders: Record<string, unknown>[]): string {
  let takeProfit = '--';
  let stopLoss = '--';
  for (const order of openOrders) {
    if (!orderMatchesPosition(order, position)) continue;
    const source = nestedOrder(order);
    const tpsl = rawText(source, 'tpsl', 'tpSl', 'tpslType').toLowerCase();
    const orderType = rawText(source, 'orderType', 'order_type', 'type').toLowerCase();
    if (tpsl === 'tp' || orderType.includes('take profit')) {
      takeProfit = orderTriggerPrice(order);
    }
    if (tpsl === 'sl' || orderType.includes('stop loss') || orderType.includes('stop market')) {
      stopLoss = orderTriggerPrice(order);
    }
  }
  return `${takeProfit} / ${stopLoss}`;
}

function marketLabel(market: Market): string {
  return stripDexPrefix(market.display_name || `${market.symbol}-USDC`);
}

function marketSearchText(market: Market): string {
  return [market.symbol, market.execution_symbol, market.display_name, market.dex, market.market_name]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

function marketReferencePrice(market: Market): number {
  const candidates = [market.mid_price, market.mark_price];
  for (const candidate of candidates) {
    const value = Number(candidate || 0);
    if (Number.isFinite(value) && value > 0) return value;
  }
  return 0;
}

function orderReferencePrice(market: Market | null, orderType: 'market' | 'limit', limitPrice: string): number {
  const limit = Number(limitPrice || 0);
  if (orderType === 'limit' && Number.isFinite(limit) && limit > 0) return limit;
  return market ? marketReferencePrice(market) : 0;
}

function marginToOrderSize(marginUsd: string | number, leverage: number, referencePrice: number): number {
  const margin = Number(marginUsd || 0);
  const priceValue = Number(referencePrice || 0);
  return margin > 0 && leverage > 0 && priceValue > 0 ? (margin * leverage) / priceValue : 0;
}

function orderSizeToMargin(size: string | number, leverage: number, referencePrice: number): number {
  const amount = Number(size || 0);
  const priceValue = Number(referencePrice || 0);
  return amount > 0 && leverage > 0 && priceValue > 0 ? (amount * priceValue) / leverage : 0;
}

function marketChangePct(market: Market): number | undefined {
  const current = marketReferencePrice(market);
  const previous = Number(market.prev_day_price || 0);
  if (!Number.isFinite(current) || !Number.isFinite(previous) || current <= 0 || previous <= 0) return undefined;
  return (current - previous) / previous;
}

function favoriteMetricLabel(market: Market, mode: FavoriteMetricMode): string {
  if (mode === 'change') {
    const change = marketChangePct(market);
    return typeof change === 'number' ? signedPct(change) : '-';
  }
  const current = marketReferencePrice(market);
  return current > 0 ? quotePrice(current) : '-';
}

function marketLiveKeys(market: Market): string[] {
  return [market.ws_symbol, market.execution_symbol, market.symbol, market.display_name?.replace(/-USDC$/, '')]
    .filter((item): item is string => Boolean(item))
    .map((item) => item.trim())
    .filter(Boolean);
}

function marketDex(market: Market | null | undefined): string {
  if (!market) return '';
  const explicit = String(market.dex || '').trim().toLowerCase();
  if (explicit) return explicit;
  return positionMarketParts(market.symbol).dex;
}

function liveMidForMarket(market: Market | null | undefined, mids: Record<string, number>): number | undefined {
  if (!market) return undefined;
  for (const key of marketLiveKeys(market)) {
    const value = mids[key];
    if (Number.isFinite(value) && value > 0) return value;
  }
  return undefined;
}

function applyLiveMids(markets: Market[], mids: Record<string, number>): Market[] {
  let changed = false;
  const nextMarkets = markets.map((market) => {
    const mid = liveMidForMarket(market, mids);
    if (!mid || Number(market.mid_price || 0) === mid) return market;
    changed = true;
    return { ...market, mid_price: mid };
  });
  return changed ? nextMarkets : markets;
}

function positionLiveKeys(position: Position): string[] {
  const parts = positionMarketParts(position.symbol);
  return [
    position.symbol,
    stripDexPrefix(position.symbol),
    parts.market,
    parts.dex && parts.market ? `${parts.dex}:${parts.market}` : ''
  ]
    .filter((item): item is string => Boolean(item))
    .map((item) => item.trim())
    .filter(Boolean);
}

function selectedPositionForMarket(account: Account | null, market: Market | null): Position | undefined {
  if (!account?.positions?.length || !market) return undefined;
  const marketKeys = new Set(
    [...marketLiveKeys(market), stripDexPrefix(market.symbol), market.market_name || '']
      .map((item) => item.trim())
      .filter(Boolean)
  );
  return account.positions.find((position) => positionLiveKeys(position).some((key) => marketKeys.has(key)));
}

function liveMidForPosition(position: Position, mids: Record<string, number>): number | undefined {
  for (const key of positionLiveKeys(position)) {
    const value = mids[key];
    if (Number.isFinite(value) && value > 0) return value;
  }
  return undefined;
}

function applyLiveMidToPosition(position: Position, mid: number): Position {
  const size = Math.abs(Number(position.size || 0));
  const side = String(position.side || '').toLowerCase();
  const direction = side === 'short' ? -1 : side === 'long' ? 1 : 0;
  const entry = Number(position.entry_price || position.display_entry_price || 0);
  const next: Position = { ...position, mid_price: mid };
  if (size > 0) {
    next.notional_usd = size * mid;
  }
  if (size > 0 && direction !== 0 && entry > 0) {
    const unrealized = (mid - entry) * size * direction;
    const carried = Number(position.carried_realized_pnl_usd || 0);
    const syntheticPnl = carried + unrealized;
    const basis = Number(position.lifecycle_roi_basis_usd || position.margin_used || 0);
    const marginUsed = Number(position.margin_used || 0);
    next.unrealized_pnl = unrealized;
    next.synthetic_pnl_usd = syntheticPnl;
    if (basis > 0) next.synthetic_pnl_pct = syntheticPnl / basis;
    if (marginUsed > 0) next.return_on_equity = unrealized / marginUsed;
  }
  return next;
}

function applyLiveMidsToAccount(account: Account | null, mids: Record<string, number>): Account | null {
  if (!account?.positions?.length) return account;
  let changed = false;
  const positions = account.positions.map((position) => {
    const mid = liveMidForPosition(position, mids);
    if (!mid || Number(position.mid_price || 0) === mid) return position;
    changed = true;
    return applyLiveMidToPosition(position, mid);
  });
  return changed ? { ...account, positions } : account;
}

function GlobalNotificationToast({ notification }: { notification: GlobalNotification | null }) {
  if (!notification) return null;
  const isError = notification.kind === 'error';
  return (
    <div
      className={`global-notification ${notification.kind}`}
      key={notification.id}
      role={isError ? 'alert' : 'status'}
    >
      {notification.kind === 'pending' && <span className="notification-spinner" aria-label="Pending" />}
      <span>{notification.message}</span>
    </div>
  );
}

function normalizeAllMids(raw: unknown): Record<string, number> {
  if (!raw || typeof raw !== 'object') return {};
  const payload = raw as Record<string, unknown>;
  const mids = payload.mids && typeof payload.mids === 'object' ? (payload.mids as Record<string, unknown>) : payload;
  return Object.fromEntries(
    Object.entries(mids)
      .map(([key, value]) => [key, Number(value)] as const)
      .filter(([, value]) => Number.isFinite(value) && value > 0)
  );
}

function intervalWindowSeconds(interval: string): number {
  return WINDOW_SECONDS_BY_INTERVAL[interval] || 3600;
}

function intervalResolution(interval: string): string {
  return (
    {
      '1m': '1',
      '5m': '5',
      '15m': '15',
      '30m': '30',
      '1h': '60',
      '4h': '240',
      '1d': '1D'
    }[interval] || '1'
  );
}

function normalizeIndicatorPeriod(value: unknown, fallback: number): number {
  const next = Math.round(Number(value));
  if (!Number.isFinite(next)) return fallback;
  return Math.min(INDICATOR_PERIOD_MAX, Math.max(INDICATOR_PERIOD_MIN, next));
}

function parseIndicatorPeriodDraft(value: string): number | null {
  if (value.trim() === '') return 0;
  const next = Math.round(Number(value));
  if (!Number.isFinite(next)) return null;
  return Math.min(INDICATOR_PERIOD_MAX, Math.max(0, next));
}

function readStoredIndicators(): IndicatorState {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHART_INDICATORS_KEY) || '{}') as Partial<IndicatorState> & {
      ema20?: boolean;
      sma50?: boolean;
    };
    return {
      volume: typeof parsed.volume === 'boolean' ? parsed.volume : DEFAULT_INDICATORS.volume,
      ema: typeof parsed.ema === 'boolean' ? parsed.ema : typeof parsed.ema20 === 'boolean' ? parsed.ema20 : DEFAULT_INDICATORS.ema,
      sma: typeof parsed.sma === 'boolean' ? parsed.sma : typeof parsed.sma50 === 'boolean' ? parsed.sma50 : DEFAULT_INDICATORS.sma,
      emaLength: normalizeIndicatorPeriod(parsed.emaLength, DEFAULT_INDICATORS.emaLength),
      smaLength: normalizeIndicatorPeriod(parsed.smaLength, DEFAULT_INDICATORS.smaLength)
    };
  } catch {
    return DEFAULT_INDICATORS;
  }
}

function storeIndicators(indicators: IndicatorState): void {
  localStorage.setItem(CHART_INDICATORS_KEY, JSON.stringify(indicators));
}

function readStoredChartTimezone(): ChartTimezoneValue {
  const stored = localStorage.getItem(CHART_TIMEZONE_KEY);
  return CHART_TIMEZONE_OPTIONS.some((option) => option.value === stored) ? (stored as ChartTimezoneValue) : 'local';
}

function storeChartTimezone(timezone: ChartTimezoneValue): void {
  localStorage.setItem(CHART_TIMEZONE_KEY, timezone);
}

function getMediaQueryMatches(query: string): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(query).matches
    : false;
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => getMediaQueryMatches(query));

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const mediaQuery = window.matchMedia(query);
    const update = () => setMatches(mediaQuery.matches);
    update();
    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', update);
      return () => mediaQuery.removeEventListener('change', update);
    }
    mediaQuery.addListener(update);
    return () => mediaQuery.removeListener(update);
  }, [query]);

  return matches;
}

function chartTimezoneName(value: ChartTimezoneValue): string | undefined {
  return CHART_TIMEZONE_OPTIONS.find((option) => option.value === value)?.timeZone;
}

function dateFromChartTime(time: Time): Date {
  if (typeof time === 'number') return new Date(time * 1000);
  if (typeof time === 'string') return new Date(time);
  return new Date(Date.UTC(time.year, time.month - 1, time.day));
}

function chartTimeFormatter(
  timezone: ChartTimezoneValue,
  tickMode: 'axis' | 'crosshair',
  tickType?: TickMarkType
): (time: Time, tickMarkType?: TickMarkType, locale?: string) => string {
  const timeZone = chartTimezoneName(timezone);
  return (time, incomingTickType, locale = navigator.language) => {
    const markType = tickType ?? incomingTickType;
    const date = dateFromChartTime(time);
    if (tickMode === 'crosshair') {
      return new Intl.DateTimeFormat(locale, {
        timeZone,
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      }).format(date);
    }
    if (markType === TickMarkType.Year) {
      return new Intl.DateTimeFormat(locale, { timeZone, year: '2-digit' }).format(date);
    }
    if (markType === TickMarkType.Month || markType === TickMarkType.DayOfMonth) {
      return new Intl.DateTimeFormat(locale, { timeZone, month: '2-digit', day: '2-digit' }).format(date);
    }
    return new Intl.DateTimeFormat(locale, { timeZone, hour: '2-digit', minute: '2-digit', hour12: false }).format(date);
  };
}

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

function upsertCandles(current: Candle[], incoming: Candle[]): Candle[] {
  const byTime = new Map<number, Candle>();
  for (const candle of current) byTime.set(Number(candle.time), candle);
  for (const candle of incoming) byTime.set(Number(candle.time), candle);
  return Array.from(byTime.values())
    .sort((left, right) => Number(left.time) - Number(right.time))
    .slice(-5000);
}

function movingAverage(candles: Candle[], length: number): LineData[] {
  const rows: LineData[] = [];
  let rolling = 0;
  candles.forEach((candle, index) => {
    rolling += Number(candle.close);
    if (index >= length) rolling -= Number(candles[index - length].close);
    if (index >= length - 1) {
      rows.push({ time: candle.time as UTCTimestamp, value: rolling / length });
    }
  });
  return rows;
}

function exponentialMovingAverage(candles: Candle[], length: number): LineData[] {
  const rows: LineData[] = [];
  const multiplier = 2 / (length + 1);
  let current = 0;
  candles.forEach((candle, index) => {
    const close = Number(candle.close);
    current = index === 0 ? close : close * multiplier + current * (1 - multiplier);
    if (index >= length - 1) rows.push({ time: candle.time as UTCTimestamp, value: current });
  });
  return rows;
}

function normalizeBookSide(levels: unknown): OrderBookLevel[] {
  let total = 0;
  const rows: OrderBookLevel[] = [];
  for (const item of Array.isArray(levels) ? levels : []) {
    const row = item as Record<string, unknown>;
    const priceValue = Number(row.price ?? row.px ?? 0);
    const sizeValue = Number(row.size ?? row.sz ?? 0);
    if (!Number.isFinite(priceValue) || !Number.isFinite(sizeValue) || priceValue <= 0 || sizeValue <= 0) continue;
    total += sizeValue;
    rows.push({
      price: priceValue,
      size: sizeValue,
      total,
      orders: Number(row.orders ?? row.n ?? 0)
    });
  }
  return rows;
}

function normalizeOrderBook(raw: Record<string, unknown>): OrderBook {
  const levels = Array.isArray(raw.levels) ? raw.levels : [];
  const bids = normalizeBookSide(levels[0]).slice(0, 16);
  const asks = normalizeBookSide(levels[1]).slice(0, 16);
  const midPrice = bids[0] && asks[0] ? (bids[0].price + asks[0].price) / 2 : Number(raw.mid_price || 0);
  return {
    symbol: String(raw.symbol ?? raw.coin ?? ''),
    time: Number(raw.time || 0),
    mid_price: Number.isFinite(midPrice) ? midPrice : 0,
    bids,
    asks
  };
}

function normalizeTrade(raw: Record<string, unknown>): MarketTrade | null {
  const tradePrice = Number(raw.price ?? raw.px ?? 0);
  const tradeSize = Number(raw.size ?? raw.sz ?? 0);
  const timestamp = Number(raw.time ?? raw.t ?? 0);
  if (!Number.isFinite(tradePrice) || !Number.isFinite(tradeSize) || !Number.isFinite(timestamp)) return null;
  if (tradePrice <= 0 || tradeSize <= 0 || timestamp <= 0) return null;
  return {
    coin: String(raw.coin ?? raw.symbol ?? ''),
    side: String(raw.side ?? ''),
    price: tradePrice,
    size: tradeSize,
    hash: raw.hash ? String(raw.hash) : undefined,
    time: timestamp > 10_000_000_000 ? Math.floor(timestamp) : Math.floor(timestamp * 1000)
  };
}

function tradeKey(trade: MarketTrade): string {
  return trade.hash || `${trade.coin}:${trade.side}:${trade.price}:${trade.size}:${trade.time}`;
}

function upsertTrades(current: MarketTrade[], incoming: MarketTrade[]): MarketTrade[] {
  const byKey = new Map<string, MarketTrade>();
  for (const trade of current) byKey.set(tradeKey(trade), trade);
  for (const trade of incoming) byKey.set(tradeKey(trade), trade);
  return Array.from(byKey.values())
    .sort((left, right) => Number(right.time) - Number(left.time))
    .slice(0, 40);
}

function TokenGate({ onUnlock }: { onUnlock: (token: string) => void }) {
  const [token, setToken] = useState('');
  return (
    <main className="auth-shell">
      <form
        className="auth-panel"
        onSubmit={(event) => {
          event.preventDefault();
          if (token.trim()) onUnlock(token.trim());
        }}
      >
        <Lock size={22} />
        <h1>Private Trade</h1>
        <label htmlFor="admin-token">Admin Token</label>
        <input
          id="admin-token"
          type="password"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          autoFocus
        />
        <button type="submit">Unlock</button>
      </form>
    </main>
  );
}

function LightweightChartPanel({
  snapshot,
  currentPosition,
  interval,
  wsStatus,
  showResetButton,
  resetKey,
  indicators,
  chartTimezone,
  onChartTimezoneChange,
  onIndicatorsChange,
  onLoadEarlier,
  onIntervalChange
}: {
  snapshot: MarketSnapshot | null;
  currentPosition?: Position;
  interval: string;
  wsStatus: string;
  showResetButton: boolean;
  resetKey: string;
  indicators: IndicatorState;
  chartTimezone: ChartTimezoneValue;
  onChartTimezoneChange: (timezone: ChartTimezoneValue) => void;
  onIndicatorsChange: (indicators: IndicatorState) => void;
  onLoadEarlier: () => Promise<number>;
  onIntervalChange: (interval: string) => void;
}) {
  const chartHostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const emaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const smaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const entryPriceLineRef = useRef<{ series: ISeriesApi<'Candlestick'>; line: IPriceLine } | null>(null);
  const liquidationPriceLineRef = useRef<{ series: ISeriesApi<'Candlestick'>; line: IPriceLine } | null>(null);
  const desktopVerticalPanRef = useRef<DesktopVerticalPanGesture | null>(null);
  const priceScaleFreezeTimerRef = useRef<number | null>(null);
  const lastFitKeyRef = useRef('');
  const loadingEarlierRef = useRef(false);
  const historyExhaustedRef = useRef(false);
  const onLoadEarlierRef = useRef(onLoadEarlier);
  const chartMenuRef = useRef<HTMLDivElement | null>(null);
  const indicatorMenuRef = useRef<HTMLDivElement | null>(null);
  const indicatorPopoverRef = useRef<HTMLDivElement | null>(null);
  const candleDetailsByTimeRef = useRef<Map<number, ChartCandleDetails>>(new Map());
  const [indicatorMenuOpen, setIndicatorMenuOpen] = useState(false);
  const [indicatorPopoverStyle, setIndicatorPopoverStyle] = useState<CSSProperties>({});
  const [chartContextMenu, setChartContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [hoveredCandle, setHoveredCandle] = useState<ChartCandleDetails | null>(null);
  const candles = snapshot?.candles || [];
  const emaLength = normalizeIndicatorPeriod(indicators.emaLength, DEFAULT_INDICATORS.emaLength);
  const smaLength = normalizeIndicatorPeriod(indicators.smaLength, DEFAULT_INDICATORS.smaLength);
  const chartTimeOptions = useMemo(
    () => ({
      localization: {
        timeFormatter: chartTimeFormatter(chartTimezone, 'crosshair')
      },
      timeScale: {
        tickMarkFormatter: chartTimeFormatter(chartTimezone, 'axis')
      }
    }),
    [chartTimezone]
  );
  const chartData = useMemo<CandlestickData[]>(
    () =>
      candles.map((candle) => ({
        time: candle.time as UTCTimestamp,
        open: Number(candle.open),
        high: Number(candle.high),
        low: Number(candle.low),
        close: Number(candle.close)
      })),
    [candles]
  );
  const volumeData = useMemo<HistogramData[]>(
    () =>
      candles.map((candle) => ({
        time: candle.time as UTCTimestamp,
        value: Number(candle.volume || 0),
        color: Number(candle.close) >= Number(candle.open) ? 'rgba(34, 197, 94, 0.34)' : 'rgba(239, 68, 68, 0.34)'
      })),
    [candles]
  );
  const emaData = useMemo(() => exponentialMovingAverage(candles, emaLength), [candles, emaLength]);
  const smaData = useMemo(() => movingAverage(candles, smaLength), [candles, smaLength]);
  const candleDetailsByTime = useMemo(() => {
    const emaByTime = new Map(emaData.map((row) => [Number(row.time), Number(row.value)]));
    const smaByTime = new Map(smaData.map((row) => [Number(row.time), Number(row.value)]));
    return new Map(
      candles.map((candle) => {
        const time = Number(candle.time);
        const details = candleDetailsFromCandle(candle);
        const ema = emaByTime.get(time);
        const sma = smaByTime.get(time);
        if (typeof ema === 'number' && Number.isFinite(ema)) details.ema = ema;
        if (typeof sma === 'number' && Number.isFinite(sma)) details.sma = sma;
        return [time, details];
      })
    );
  }, [candles, emaData, smaData]);
  const latestCandle = useMemo(() => {
    if (candles.length === 0) return null;
    const latestTime = Number(candles[candles.length - 1].time);
    return candleDetailsByTime.get(latestTime) || candleDetailsFromCandle(candles[candles.length - 1]);
  }, [candles, candleDetailsByTime]);
  const displayedCandle = useMemo(() => {
    const current = hoveredCandle || latestCandle;
    if (!current) return null;
    const detailTime = Number(current.time);
    const detail = Number.isFinite(detailTime) ? candleDetailsByTime.get(detailTime) : undefined;
    if (!detail) return current;
    return {
      ...detail,
      ...current,
      volume: current.volume ?? detail.volume,
      ema: current.ema ?? detail.ema,
      sma: current.sma ?? detail.sma
    };
  }, [hoveredCandle, latestCandle, candleDetailsByTime]);
  const displayedCandleChange = displayedCandle ? displayedCandle.close - displayedCandle.open : 0;
  const displayedCandleChangePct =
    displayedCandle && displayedCandle.open > 0 ? displayedCandleChange / displayedCandle.open : 0;
  const entryPrice = Number(currentPosition?.display_entry_price || currentPosition?.entry_price || 0);
  const liquidationPrice = Number(currentPosition?.liquidation_price || 0);

  function clearEntryPriceLine() {
    const current = entryPriceLineRef.current;
    if (!current) return;
    current.series.removePriceLine(current.line);
    entryPriceLineRef.current = null;
  }

  function clearLiquidationPriceLine() {
    const current = liquidationPriceLineRef.current;
    if (!current) return;
    current.series.removePriceLine(current.line);
    liquidationPriceLineRef.current = null;
  }

  useEffect(() => {
    onLoadEarlierRef.current = onLoadEarlier;
  }, [onLoadEarlier]);

  useEffect(() => {
    candleDetailsByTimeRef.current = candleDetailsByTime;
  }, [candleDetailsByTime]);

  useEffect(() => {
    historyExhaustedRef.current = false;
    setHoveredCandle(null);
  }, [resetKey]);

  useEffect(() => {
    if (!chartContextMenu) return undefined;
    function closeOnPointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (chartMenuRef.current?.contains(target)) return;
      setChartContextMenu(null);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') setChartContextMenu(null);
    }
    document.addEventListener('pointerdown', closeOnPointerDown);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOnPointerDown);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [chartContextMenu]);

  function updateIndicatorPopoverPosition() {
    if (!indicatorMenuRef.current || typeof window === 'undefined') return;
    const rect = indicatorMenuRef.current.getBoundingClientRect();
    const popoverWidth = indicatorPopoverRef.current?.offsetWidth || 190;
    const popoverHeight = indicatorPopoverRef.current?.offsetHeight || 146;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1024;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 768;
    const maxLeft = Math.max(8, viewportWidth - popoverWidth - 8);
    const left = Math.min(Math.max(8, rect.right - popoverWidth), maxLeft);
    const belowTop = rect.bottom + 8;
    const top =
      belowTop + popoverHeight > viewportHeight - 8 && rect.top > popoverHeight + 14
        ? rect.top - popoverHeight - 8
        : Math.min(belowTop, Math.max(8, viewportHeight - popoverHeight - 8));
    setIndicatorPopoverStyle({ left, top });
  }

  useLayoutEffect(() => {
    if (indicatorMenuOpen) updateIndicatorPopoverPosition();
  }, [indicatorMenuOpen]);

  useEffect(() => {
    if (!indicatorMenuOpen) return undefined;
    function closeOnPointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (indicatorMenuRef.current?.contains(target) || indicatorPopoverRef.current?.contains(target)) return;
      setIndicatorMenuOpen(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') setIndicatorMenuOpen(false);
    }
    updateIndicatorPopoverPosition();
    window.addEventListener('resize', updateIndicatorPopoverPosition);
    window.addEventListener('scroll', updateIndicatorPopoverPosition, true);
    document.addEventListener('pointerdown', closeOnPointerDown);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      window.removeEventListener('resize', updateIndicatorPopoverPosition);
      window.removeEventListener('scroll', updateIndicatorPopoverPosition, true);
      document.removeEventListener('pointerdown', closeOnPointerDown);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [indicatorMenuOpen]);

  useEffect(() => {
    const host = chartHostRef.current;
    if (!host || !resetKey.split(':')[0]) return;
    const chartHost = host;
    lastFitKeyRef.current = '';
    loadingEarlierRef.current = false;
    historyExhaustedRef.current = false;
    const chart = createChart(host, {
      width: Math.max(host.clientWidth, 320),
      height: Math.max(host.clientHeight, 320),
      layout: {
        background: { type: ColorType.Solid, color: '#121720' },
        textColor: '#9ba7b6'
      },
      grid: {
        vertLines: { color: '#1d2633' },
        horzLines: { color: '#1d2633' }
      },
      crosshair: {
        mode: CrosshairMode.Normal
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true
      },
      rightPriceScale: { borderColor: '#2d3748' },
      localization: chartTimeOptions.localization,
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: CHART_RIGHT_OFFSET_BARS,
        ...chartTimeOptions.timeScale
      }
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      priceFormat: chartPriceFormat()
    }, 0);
    chart.priceScale('').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 }
    });
    chartRef.current = chart;
    seriesRef.current = series;

    function handlePriceAxisWheel(event: WheelEvent) {
      const rect = chartHost.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      const priceScale = series.priceScale();
      const axisWidth = Math.max(CHART_PRICE_AXIS_WHEEL_ZONE_PX, Number(priceScale.width?.() || 0));
      const overRightPriceAxis =
        localX >= rect.width - axisWidth && localX <= rect.width && localY >= 0 && localY <= rect.height;
      if (!overRightPriceAxis) return;
      const range = priceScale.getVisibleRange?.();
      const from = Number(range?.from);
      const to = Number(range?.to);
      if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return;
      event.preventDefault();
      event.stopPropagation();
      priceScale.setAutoScale(false);
      const span = to - from;
      const pointerRatio = rect.height > 0 ? Math.min(1, Math.max(0, localY / rect.height)) : 0.5;
      const anchor = to - span * pointerRatio;
      const zoomFactor = Math.min(10, Math.max(0.1, Math.exp(event.deltaY * CHART_PRICE_WHEEL_ZOOM_SPEED)));
      priceScale.setVisibleRange({
        from: anchor - (anchor - from) * zoomFactor,
        to: anchor + (to - anchor) * zoomFactor
      });
    }

    function handleCrosshairMove(param: { point?: unknown; time?: Time; seriesData?: Map<unknown, unknown> }) {
      if (!param.point) {
        setHoveredCandle(null);
        return;
      }
      const rawCandle = param.seriesData?.get(series);
      const rawTime =
        rawCandle && typeof rawCandle === 'object'
          ? Number((rawCandle as Record<string, unknown>).time ?? param.time)
          : Number(param.time);
      const fallback = Number.isFinite(rawTime) ? candleDetailsByTimeRef.current.get(rawTime) : undefined;
      setHoveredCandle(candleDetailsFromCrosshair(rawCandle, fallback));
    }

    const handleVisibleRangeChange = (range: { from: number; to: number } | null) => {
      if (!range || range.from >= 8 || loadingEarlierRef.current || historyExhaustedRef.current) return;
      loadingEarlierRef.current = true;
      const previousRange = chart.timeScale().getVisibleLogicalRange?.() || range;
      void onLoadEarlierRef
        .current()
        .then((addedCount) => {
          if (addedCount <= 0) {
            historyExhaustedRef.current = true;
            return;
          }
          chart.timeScale().setVisibleLogicalRange?.({
            from: Number(previousRange.from) + addedCount,
            to: Number(previousRange.to) + addedCount
          });
        })
        .finally(() => {
          loadingEarlierRef.current = false;
        });
    };

    chart.subscribeCrosshairMove(handleCrosshairMove);
    chart.timeScale().subscribeVisibleLogicalRangeChange?.(handleVisibleRangeChange);

    function resize() {
      if (!chartHostRef.current) return;
      chart.applyOptions({
        width: Math.max(chartHostRef.current.clientWidth, 320),
        height: Math.max(chartHostRef.current.clientHeight, 320)
      });
    }

    resize();
    window.addEventListener('resize', resize);
    chartHost.addEventListener('wheel', handlePriceAxisWheel, { capture: true, passive: false });
    return () => {
      window.removeEventListener('resize', resize);
      chartHost.removeEventListener('wheel', handlePriceAxisWheel, { capture: true });
      clearScheduledPriceScaleFreeze();
      chart.unsubscribeCrosshairMove(handleCrosshairMove);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange?.(handleVisibleRangeChange);
      clearEntryPriceLine();
      clearLiquidationPriceLine();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      volumeSeriesRef.current = null;
      emaSeriesRef.current = null;
      smaSeriesRef.current = null;
    };
  }, [resetKey]);

  useEffect(() => {
    chartRef.current?.applyOptions(chartTimeOptions);
  }, [chartTimeOptions]);

  function resetPriceScales() {
    seriesRef.current?.priceScale().setAutoScale(true);
    volumeSeriesRef.current?.priceScale().setAutoScale(true);
  }

  function clearScheduledPriceScaleFreeze() {
    if (priceScaleFreezeTimerRef.current === null) return;
    window.clearTimeout(priceScaleFreezeTimerRef.current);
    priceScaleFreezeTimerRef.current = null;
  }

  function freezeMainPriceScaleForPan() {
    clearScheduledPriceScaleFreeze();
    priceScaleFreezeTimerRef.current = window.setTimeout(() => {
      priceScaleFreezeTimerRef.current = null;
      const priceScale = seriesRef.current?.priceScale();
      if (!priceScale) return;
      const range = priceScale.getVisibleRange?.();
      const from = Number(range?.from);
      const to = Number(range?.to);
      if (Number.isFinite(from) && Number.isFinite(to) && to > from) {
        priceScale.setVisibleRange({ from, to });
      } else {
        priceScale.setAutoScale(false);
      }
    }, 0);
  }

  function resetChartViewport() {
    const chart = chartRef.current;
    if (!chart || chartData.length === 0) return;
    resetPriceScales();
    const targetTo = Math.max(CHART_RIGHT_OFFSET_BARS, chartData.length - 1 + CHART_RIGHT_OFFSET_BARS);
    chart.timeScale().setVisibleLogicalRange?.({
      from: targetTo - CHART_RESET_VISIBLE_BARS,
      to: targetTo
    });
    freezeMainPriceScaleForPan();
    lastFitKeyRef.current = resetKey;
  }

  useEffect(() => {
    clearEntryPriceLine();
    const series = seriesRef.current;
    if (!series || !Number.isFinite(entryPrice) || entryPrice <= 0) return undefined;
    const line = series.createPriceLine({
      price: entryPrice,
      color: '#f59e0b',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: `Entry ${price(entryPrice)}`
    });
    entryPriceLineRef.current = { series, line };
    return () => clearEntryPriceLine();
  }, [entryPrice, resetKey]);

  useEffect(() => {
    clearLiquidationPriceLine();
    const series = seriesRef.current;
    if (!series || !Number.isFinite(liquidationPrice) || liquidationPrice <= 0) return undefined;
    const line = series.createPriceLine({
      price: liquidationPrice,
      color: '#fb7185',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: `Liq ${price(liquidationPrice)}`
    });
    liquidationPriceLineRef.current = { series, line };
    return () => clearLiquidationPriceLine();
  }, [liquidationPrice, resetKey]);

  function configurePanes() {
    const chart = chartRef.current;
    if (!chart) return;
    const panes = chart.panes();
    panes[0]?.setStretchFactor(4);
    panes[1]?.setStretchFactor(1);
  }

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (indicators.volume && !volumeSeriesRef.current) {
      if (chart.panes().length < 2) chart.addPane();
      configurePanes();
      volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        color: 'rgba(148, 163, 184, 0.32)',
        lastValueVisible: false,
        priceLineVisible: false
      }, 1);
      volumeSeriesRef.current.priceScale().setAutoScale(true);
    }
    if (!indicators.volume && volumeSeriesRef.current) {
      chart.removeSeries(volumeSeriesRef.current);
      volumeSeriesRef.current = null;
      if (chart.panes().length > 1) chart.removePane(1);
      chart.panes()[0]?.setStretchFactor(1);
    }
    if (indicators.ema && !emaSeriesRef.current) {
      emaSeriesRef.current = chart.addSeries(LineSeries, {
        color: INDICATOR_EMA_COLOR,
        lineWidth: 2,
        priceFormat: chartPriceFormat(),
        priceLineVisible: false,
        lastValueVisible: true
      }, 0);
    }
    if (indicators.ema && emaSeriesRef.current) {
      emaSeriesRef.current.applyOptions({
        color: INDICATOR_EMA_COLOR,
        priceFormat: chartPriceFormat(),
        priceLineVisible: false,
        lastValueVisible: true
      });
    }
    if (!indicators.ema && emaSeriesRef.current) {
      chart.removeSeries(emaSeriesRef.current);
      emaSeriesRef.current = null;
    }
    if (indicators.sma && !smaSeriesRef.current) {
      smaSeriesRef.current = chart.addSeries(LineSeries, {
        color: INDICATOR_SMA_COLOR,
        lineWidth: 2,
        priceFormat: chartPriceFormat(),
        priceLineVisible: false,
        lastValueVisible: true
      }, 0);
    }
    if (indicators.sma && smaSeriesRef.current) {
      smaSeriesRef.current.applyOptions({
        color: INDICATOR_SMA_COLOR,
        priceFormat: chartPriceFormat(),
        priceLineVisible: false,
        lastValueVisible: true
      });
    }
    if (!indicators.sma && smaSeriesRef.current) {
      chart.removeSeries(smaSeriesRef.current);
      smaSeriesRef.current = null;
    }
  }, [indicators, resetKey]);

  useEffect(() => {
    seriesRef.current?.setData(chartData);
    volumeSeriesRef.current?.setData(volumeData);
    emaSeriesRef.current?.setData(emaData);
    smaSeriesRef.current?.setData(smaData);
    if (chartData.length > 0 && lastFitKeyRef.current !== resetKey) {
      resetChartViewport();
    }
  }, [chartData, emaData, indicators, resetKey, smaData, volumeData]);

  function resetChartView() {
    resetChartViewport();
  }

  function openChartContextMenu(event: ReactMouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    setChartContextMenu({
      x: Math.max(6, event.clientX - rect.left),
      y: Math.max(6, event.clientY - rect.top)
    });
  }

  function currentChartPriceRange() {
    const priceScale = seriesRef.current?.priceScale();
    const visibleRange = priceScale?.getVisibleRange?.();
    const fallbackRange = chartDataPriceRange(chartData, chartRef.current?.timeScale().getVisibleLogicalRange?.());
    const resolvedRange =
      Number.isFinite(Number(visibleRange?.from)) && Number.isFinite(Number(visibleRange?.to)) ? visibleRange : fallbackRange;
    const from = Number(resolvedRange?.from);
    const to = Number(resolvedRange?.to);
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return null;
    return { from, to };
  }

  function beginDesktopVerticalPan(event: ReactPointerEvent<HTMLDivElement>) {
    if ((event.pointerType && event.pointerType !== 'mouse') || event.button !== 0 || chartContextMenu) return;
    if (event.target instanceof Element && event.target.closest('.chart-context-menu')) return;
    const range = currentChartPriceRange();
    if (!range) return;
    const rect = event.currentTarget.getBoundingClientRect();
    try {
      event.currentTarget.setPointerCapture?.(event.pointerId);
    } catch {
      // Pointer capture is best-effort; losing it only cancels this desktop drag.
    }
    desktopVerticalPanRef.current = {
      pointerId: event.pointerId,
      startY: event.clientY,
      height: Math.max(80, rect.height || 160),
      range,
      active: false
    };
  }

  function updateDesktopVerticalPan(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = desktopVerticalPanRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaY = event.clientY - drag.startY;
    if (Math.abs(deltaY) < 2) return;
    event.preventDefault();
    event.stopPropagation();
    const priceScale = seriesRef.current?.priceScale();
    if (!priceScale) return;
    if (!drag.active) {
      priceScale.setAutoScale(false);
      drag.active = true;
    }
    const span = drag.range.to - drag.range.from;
    const shift = (deltaY / drag.height) * span;
    priceScale.setVisibleRange({
      from: drag.range.from + shift,
      to: drag.range.to + shift
    });
  }

  function endDesktopVerticalPan(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = desktopVerticalPanRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    try {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    } catch {
      // Ignore release failures when capture was not available.
    }
    desktopVerticalPanRef.current = null;
  }

  const indicatorPopover = (
    <div
      aria-label="Chart indicators"
      className="indicator-popover floating-popover"
      ref={indicatorPopoverRef}
      role="dialog"
      style={indicatorPopoverStyle}
    >
      <label>
        <input
          checked={indicators.volume}
          onChange={(event) => onIndicatorsChange({ ...indicators, volume: event.target.checked })}
          type="checkbox"
        />
        Volume
      </label>
      <div className="indicator-period-row">
        <label>
          <input
            checked={indicators.ema}
            onChange={(event) => onIndicatorsChange({ ...indicators, ema: event.target.checked })}
            type="checkbox"
          />
          EMA
        </label>
        <input
          aria-label="EMA period"
          min={INDICATOR_PERIOD_MIN}
          max={INDICATOR_PERIOD_MAX}
          onBlur={() => onIndicatorsChange({ ...indicators, emaLength })}
          onChange={(event) => {
            const nextLength = parseIndicatorPeriodDraft(event.target.value);
            if (nextLength !== null) onIndicatorsChange({ ...indicators, emaLength: nextLength });
          }}
          step={1}
          type="number"
          value={indicators.emaLength > 0 ? indicators.emaLength : ''}
        />
      </div>
      <div className="indicator-period-row">
        <label>
          <input
            checked={indicators.sma}
            onChange={(event) => onIndicatorsChange({ ...indicators, sma: event.target.checked })}
            type="checkbox"
          />
          SMA
        </label>
        <input
          aria-label="SMA period"
          min={INDICATOR_PERIOD_MIN}
          max={INDICATOR_PERIOD_MAX}
          onBlur={() => onIndicatorsChange({ ...indicators, smaLength })}
          onChange={(event) => {
            const nextLength = parseIndicatorPeriodDraft(event.target.value);
            if (nextLength !== null) onIndicatorsChange({ ...indicators, smaLength: nextLength });
          }}
          step={1}
          type="number"
          value={indicators.smaLength > 0 ? indicators.smaLength : ''}
        />
      </div>
    </div>
  );

  return (
    <section className="chart-panel">
      <div className="panel-title split">
        <div className="panel-title-left">
          <BarChart3 size={16} />
          <span className={`ws-state ${wsStatus}`}>{wsStatus}</span>
          {showResetButton && (
            <button
              aria-label="Reset chart view"
              className="chart-reset-button"
              onClick={resetChartView}
              title="Reset chart view"
              type="button"
            >
              <RefreshCw size={13} />
            </button>
          )}
        </div>
        <div className="interval-tabs" aria-label="Candle interval">
          <select
            aria-label="Chart timezone"
            className="timezone-select"
            onChange={(event) => onChartTimezoneChange(event.target.value as ChartTimezoneValue)}
            value={chartTimezone}
          >
            {CHART_TIMEZONE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <div className="indicator-menu" ref={indicatorMenuRef}>
            <button aria-expanded={indicatorMenuOpen} onClick={() => setIndicatorMenuOpen((open) => !open)} type="button">
              <SlidersHorizontal size={13} />
              Indicators
            </button>
            {indicatorMenuOpen && typeof document !== 'undefined' ? createPortal(indicatorPopover, document.body) : null}
          </div>
          {CHART_INTERVALS.map((item) => (
            <button
              className={item === interval ? 'active' : ''}
              key={item}
              onClick={() => onIntervalChange(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
      </div>
      <div
        className="chart-surface"
        aria-label="chart"
        onContextMenu={openChartContextMenu}
        onPointerCancel={endDesktopVerticalPan}
        onPointerDown={beginDesktopVerticalPan}
        onPointerLeave={endDesktopVerticalPan}
        onPointerMoveCapture={updateDesktopVerticalPan}
        onPointerUp={endDesktopVerticalPan}
        ref={chartHostRef}
      >
        {displayedCandle && (
          <div className="chart-ohlc-strip" aria-label="Selected candle details">
            <span>{chartCandleTimeLabel(displayedCandle.time, chartTimezone)}</span>
            <span>O {price(displayedCandle.open)}</span>
            <span>H {price(displayedCandle.high)}</span>
            <span>L {price(displayedCandle.low)}</span>
            <span>C {price(displayedCandle.close)}</span>
            <span className={displayedCandleChange >= 0 ? 'positive' : 'negative'}>
              {signedPrice(displayedCandleChange)} ({signedPct(displayedCandleChangePct)})
            </span>
            <span>Vol {qty(displayedCandle.volume)}</span>
            {indicators.ema && <span style={{ color: INDICATOR_EMA_COLOR }}>EMA {emaLength} {priceOrDash(displayedCandle.ema)}</span>}
            {indicators.sma && <span style={{ color: INDICATOR_SMA_COLOR }}>SMA {smaLength} {priceOrDash(displayedCandle.sma)}</span>}
          </div>
        )}
        {candles.length === 0 && <div className="chart-empty">Waiting for candle data</div>}
        {chartContextMenu && (
          <div
            className="chart-context-menu"
            ref={chartMenuRef}
            role="menu"
            style={{ left: chartContextMenu.x, top: chartContextMenu.y }}
          >
            <button
              onClick={() => {
                resetChartView();
                setChartContextMenu(null);
              }}
              role="menuitem"
              type="button"
            >
              Reset chart view
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

function ChartPanel({
  selectedMarket,
  currentPosition,
  snapshot,
  interval,
  wsStatus,
  showResetButton,
  chartTimezone,
  indicators,
  onChartTimezoneChange,
  onIndicatorsChange,
  onLoadEarlier,
  onIntervalChange
}: {
  selectedMarket: Market | null;
  currentPosition?: Position;
  snapshot: MarketSnapshot | null;
  interval: string;
  wsStatus: string;
  showResetButton: boolean;
  chartTimezone: ChartTimezoneValue;
  indicators: IndicatorState;
  onChartTimezoneChange: (timezone: ChartTimezoneValue) => void;
  onIndicatorsChange: (indicators: IndicatorState) => void;
  onLoadEarlier: () => Promise<number>;
  onIntervalChange: (interval: string) => void;
}) {
  return (
    <LightweightChartPanel
      snapshot={snapshot}
      currentPosition={currentPosition}
      interval={interval}
      wsStatus={wsStatus}
      showResetButton={showResetButton}
      indicators={indicators}
      chartTimezone={chartTimezone}
      onChartTimezoneChange={onChartTimezoneChange}
      onIndicatorsChange={onIndicatorsChange}
      onLoadEarlier={onLoadEarlier}
      resetKey={`${selectedMarket?.symbol || snapshot?.symbol || ''}:${interval}`}
      onIntervalChange={onIntervalChange}
    />
  );
}

function MarketSearch({
  markets,
  favoriteSymbols,
  selectedSymbol,
  onToggleFavorite,
  onSelect
}: {
  markets: Market[];
  favoriteSymbols: string[];
  selectedSymbol: string;
  onToggleFavorite: (symbol: string) => void;
  onSelect: (symbol: string) => void;
}) {
  const pickerRef = useRef<HTMLDivElement | null>(null);
  const selectedMarket = markets.find((market) => market.symbol === selectedSymbol) || markets[0] || null;
  const [query, setQuery] = useState(selectedMarket ? marketLabel(selectedMarket) : '');
  const [open, setOpen] = useState(false);
  const selectedIsFavorite = Boolean(selectedMarket && favoriteSymbols.includes(selectedMarket.symbol));

  useEffect(() => {
    if (selectedMarket && !open) setQuery(marketLabel(selectedMarket));
  }, [open, selectedMarket?.symbol]);

  useEffect(() => {
    function closeOnOutsideClick(event: PointerEvent) {
      if (!pickerRef.current?.contains(event.target as Node)) setOpen(false);
    }

    document.addEventListener('pointerdown', closeOnOutsideClick);
    return () => document.removeEventListener('pointerdown', closeOnOutsideClick);
  }, []);

  const filteredMarkets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle || selectedMarket?.display_name === query || marketLabel(selectedMarket || { symbol: '' }) === query) {
      return markets.slice(0, 60);
    }
    return markets.filter((market) => marketSearchText(market).includes(needle)).slice(0, 60);
  }, [markets, query, selectedMarket]);

  return (
    <div className="market-picker" ref={pickerRef}>
      <div className="market-input-row">
        <input
          aria-label="Market"
          aria-controls="market-options"
          aria-expanded={open}
          autoComplete="off"
          role="combobox"
          value={query}
          onFocus={() => setOpen(true)}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
        />
        {selectedMarket && (
          <button
            aria-label={`${selectedIsFavorite ? 'Unfavorite' : 'Favorite'} ${marketLabel(selectedMarket)}`}
            className={selectedIsFavorite ? 'market-current-favorite active' : 'market-current-favorite'}
            onClick={() => onToggleFavorite(selectedMarket.symbol)}
            title={`${selectedIsFavorite ? 'Unfavorite' : 'Favorite'} ${marketLabel(selectedMarket)}`}
            type="button"
          >
            <Star size={14} fill={selectedIsFavorite ? 'currentColor' : 'none'} />
          </button>
        )}
      </div>
      {open && (
        <div className="market-options" id="market-options" role="listbox">
          {filteredMarkets.map((market) => (
            <div
              aria-selected={market.symbol === selectedSymbol}
              key={market.symbol}
              role="option"
              className="market-option-row"
            >
              <button
                className="market-option-select"
                type="button"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  setQuery(marketLabel(market));
                  setOpen(false);
                  onSelect(market.symbol);
                }}
              >
                <span>{marketLabel(market)}</span>
                <span>{market.max_leverage ? `${market.max_leverage}x` : ''}</span>
              </button>
              <button
                aria-label={`${favoriteSymbols.includes(market.symbol) ? 'Unfavorite' : 'Favorite'} ${marketLabel(market)}`}
                className={favoriteSymbols.includes(market.symbol) ? 'market-option-favorite active' : 'market-option-favorite'}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => onToggleFavorite(market.symbol)}
                title={`${favoriteSymbols.includes(market.symbol) ? 'Unfavorite' : 'Favorite'} ${marketLabel(market)}`}
                type="button"
              >
                <Star size={13} fill={favoriteSymbols.includes(market.symbol) ? 'currentColor' : 'none'} />
              </button>
            </div>
          ))}
          {filteredMarkets.length === 0 && <div className="market-empty">No markets</div>}
        </div>
      )}
    </div>
  );
}

function DepthRow({ level, side, maxTotal }: { level: OrderBookLevel; side: 'bid' | 'ask'; maxTotal: number }) {
  const depthPct = Math.min(100, Math.max(0, (Number(level.total || 0) / Math.max(maxTotal, 1e-12)) * 100));
  return (
    <div className={`depth-row ${side}`}>
      <div className="depth-fill" style={{ width: `${depthPct}%` }} />
      <span className="depth-price">{orderBookNumber(level.price)}</span>
      <span>{orderBookNumber(level.size)}</span>
      <span>{orderBookNumber(level.total)}</span>
    </div>
  );
}

function OrderBookPanel({ book, midPrice, wsStatus }: { book: OrderBook | null; midPrice?: number; wsStatus: string }) {
  const asks = (book?.asks || []).slice(0, ORDER_BOOK_VISIBLE_LEVELS).reverse();
  const bids = (book?.bids || []).slice(0, ORDER_BOOK_VISIBLE_LEVELS);
  const maxTotal = Math.max(...asks.map((item) => item.total), ...bids.map((item) => item.total), 1);

  return (
    <section className="book-panel" aria-label="Order Book">
      <div className="panel-title split">
        <div className="panel-title-left">
          <BookOpen size={16} />
          <span>Order Book</span>
        </div>
        <span className={`ws-state ${wsStatus}`}>{wsStatus}</span>
      </div>
      <div className="book-head">
        <span>Price</span>
        <span>Size</span>
        <span>Total</span>
      </div>
      <div className="book-side asks">
        {asks.map((level) => (
          <DepthRow key={`ask-${level.price}`} level={level} side="ask" maxTotal={maxTotal} />
        ))}
      </div>
      <div className="book-mid">Mid {quotePrice(midPrice || book?.mid_price)}</div>
      <div className="book-side bids">
        {bids.map((level) => (
          <DepthRow key={`bid-${level.price}`} level={level} side="bid" maxTotal={maxTotal} />
        ))}
      </div>
      {asks.length === 0 && bids.length === 0 && <div className="book-empty">Waiting for order book</div>}
    </section>
  );
}

function TradesPanel({ trades, chartTimezone }: { trades: MarketTrade[]; chartTimezone: ChartTimezoneValue }) {
  return (
    <section className="trades-panel" aria-label="Trades">
      <div className="panel-title">
        <Activity size={16} />
        <span>Trades</span>
      </div>
      <div className="trades-head">
        <span>Price</span>
        <span>Size</span>
        <span>Time</span>
      </div>
      <div className="trades-list">
        {trades.map((trade) => (
          <div className={`trade-row ${trade.side === 'B' ? 'buy' : 'sell'}`} key={tradeKey(trade)}>
            <span>{price(trade.price)}</span>
            <span>{qty(trade.size)}</span>
            <span>{tradeTimeLabel(trade.time, chartTimezone)}</span>
          </div>
        ))}
        {trades.length === 0 && <div className="trades-empty">Waiting for trades</div>}
      </div>
    </section>
  );
}

function leveragePresets(maxLeverage: number): number[] {
  const max = Math.max(1, Math.floor(maxLeverage || 1));
  return Array.from(new Set([1, 2, 3, 5, 10, 20, max].filter((item) => item <= max))).sort((left, right) => left - right);
}

function LeverageSelector({
  label,
  value,
  max,
  busy,
  confirm,
  allowSameCommit,
  onChange
}: {
  label: string;
  value: number;
  max: number;
  busy?: boolean;
  confirm?: boolean;
  allowSameCommit?: boolean;
  onChange: (value: number) => void | Promise<void>;
}) {
  const maxLeverage = Math.max(1, Math.floor(max || 1));
  const clampedValue = Math.max(1, Math.min(Math.floor(value || 1), maxLeverage));
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(clampedValue);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [popoverStyle, setPopoverStyle] = useState<CSSProperties>({});
  const leverageProgress = maxLeverage <= 1 ? 0 : ((draft - 1) / (maxLeverage - 1)) * 100;
  const leverageRangeStyle = { '--leverage-progress': `${leverageProgress}%` } as CSSProperties;

  useEffect(() => {
    setDraft(clampedValue);
  }, [clampedValue, maxLeverage]);

  function updatePopoverPosition() {
    if (!triggerRef.current || typeof window === 'undefined') return;
    const rect = triggerRef.current.getBoundingClientRect();
    const popoverWidth = popoverRef.current?.offsetWidth || 220;
    const popoverHeight = popoverRef.current?.offsetHeight || (confirm ? 184 : 146);
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1024;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 768;
    const maxLeft = Math.max(8, viewportWidth - popoverWidth - 8);
    const left = Math.min(Math.max(8, rect.right - popoverWidth), maxLeft);
    const belowTop = rect.bottom + 6;
    const top =
      belowTop + popoverHeight > viewportHeight - 8 && rect.top > popoverHeight + 14
        ? rect.top - popoverHeight - 6
        : Math.min(belowTop, Math.max(8, viewportHeight - popoverHeight - 8));
    setPopoverStyle({ left, top });
  }

  useLayoutEffect(() => {
    if (open) updatePopoverPosition();
  }, [open, confirm, label]);

  useEffect(() => {
    if (!open) return undefined;
    function handleKeydown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    function handlePointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || popoverRef.current?.contains(target)) return;
      setOpen(false);
    }
    updatePopoverPosition();
    window.addEventListener('resize', updatePopoverPosition);
    window.addEventListener('scroll', updatePopoverPosition, true);
    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeydown);
    return () => {
      window.removeEventListener('resize', updatePopoverPosition);
      window.removeEventListener('scroll', updatePopoverPosition, true);
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeydown);
    };
  }, [open, confirm, label]);

  async function commit(nextValue = draft) {
    const clean = Math.max(1, Math.min(Math.floor(nextValue || 1), maxLeverage));
    await onChange(clean);
    setOpen(false);
  }

  function choose(nextValue: number) {
    setDraft(nextValue);
    if (!confirm) void commit(nextValue);
  }

  const popover = (
    <div
      aria-label={`${label} options`}
      className="leverage-popover floating-popover"
      ref={popoverRef}
      role="dialog"
      style={popoverStyle}
    >
      <div className="leverage-popover-head">
        <span>{label}</span>
        <strong>{draft}x</strong>
      </div>
      <input
        aria-label={`${label} slider`}
        className="leverage-range"
        max={maxLeverage}
        min="1"
        onChange={(event) => setDraft(Number(event.target.value || 1))}
        style={leverageRangeStyle}
        type="range"
        value={draft}
      />
      <div className="leverage-presets">
        {leveragePresets(maxLeverage).map((item) => (
          <button className={draft === item ? 'active' : ''} key={item} onClick={() => choose(item)} type="button">
            {item}x
          </button>
        ))}
      </div>
      {confirm && (
        <button
          className="leverage-apply"
          disabled={busy || (!allowSameCommit && draft === clampedValue)}
          onClick={() => void commit()}
          type="button"
        >
          Apply Leverage
        </button>
      )}
    </div>
  );

  return (
    <>
      <div className="leverage-selector">
        <button
          aria-expanded={open}
          aria-label={`${label} ${clampedValue}x Max ${maxLeverage}x`}
          className="leverage-trigger"
          disabled={busy}
          onClick={() => setOpen((current) => !current)}
          ref={triggerRef}
          type="button"
        >
          <span>{clampedValue}x</span>
          <small>Max {maxLeverage}x</small>
        </button>
      </div>
      {open && typeof document !== 'undefined' ? createPortal(popover, document.body) : null}
    </>
  );
}

function OrderTicket({
  token,
  selectedMarket,
  account,
  prefill,
  onNotify,
  refresh
}: {
  token: string;
  selectedMarket: Market | null;
  account: Account | null;
  prefill: OrderTicketPrefill | null;
  onNotify: NotifyHandler;
  refresh: () => void;
}) {
  const [side, setSide] = useState<'long' | 'short'>('long');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [amountMode, setAmountMode] = useState<OrderAmountMode>('margin');
  const [margin, setMargin] = useState('');
  const [sizeDraft, setSizeDraft] = useState('');
  const [leverage, setLeverage] = useState(1);
  const [limitPrice, setLimitPrice] = useState('');
  const [reduceOnly, setReduceOnly] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const maxLeverage = selectedMarket?.max_leverage || 50;
  const availableToTrade = Number(account?.available_margin_usd ?? account?.remaining_capital_usd ?? 0);
  const selectedPosition = selectedPositionForMarket(account, selectedMarket);
  const referencePrice = orderReferencePrice(selectedMarket, orderType, limitPrice);
  const amountInputLabel = amountMode === 'margin' ? 'Margin' : 'Size';
  const amountInputValue = amountMode === 'margin' ? margin : sizeDraft;
  const amountAsset = selectedMarket ? marketAssetLabel(selectedMarket) : '';
  const positionValue = Number(margin || 0) * leverage;
  const orderPercent =
    availableToTrade > 0 ? clampOrderPercent((Number(margin || 0) / availableToTrade) * 100) : 0;
  const orderPercentStyle = { '--order-percent-progress': `${orderPercent}%` } as CSSProperties;

  useEffect(() => {
    setLeverage((current) => Math.max(1, Math.min(current, maxLeverage)));
  }, [maxLeverage]);

  useEffect(() => {
    if (!prefill) return;
    const prefillReferencePrice = orderReferencePrice(selectedMarket, prefill.orderType, prefill.limitPrice);
    setSide(prefill.side);
    setOrderType(prefill.orderType);
    setMargin(prefill.margin);
    setSizeDraft(inputSizeNumber(marginToOrderSize(prefill.margin, prefill.leverage || 1, prefillReferencePrice)));
    setLeverage(Math.max(1, Math.min(prefill.leverage || 1, maxLeverage)));
    setLimitPrice(prefill.limitPrice);
    setReduceOnly(prefill.reduceOnly);
    setMessage(prefill.message);
    setError('');
  }, [maxLeverage, prefill, selectedMarket?.symbol]);

  useEffect(() => {
    if (amountMode !== 'size') return;
    if (!sizeDraft) {
      setMargin('');
      return;
    }
    setMargin(inputNumber(orderSizeToMargin(sizeDraft, leverage, referencePrice)));
  }, [amountMode, leverage, referencePrice, sizeDraft]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!selectedMarket) return;
    setBusy(true);
    setMessage('');
    setError('');
    onNotify('pending', 'Order pending');
    try {
      const result = await postOrder(token, {
        symbol: selectedMarket.symbol,
        order_type: orderType,
        side,
        margin_usd: Number(margin || 0),
        leverage,
        limit_price: orderType === 'limit' ? Number(limitPrice || 0) : 0,
        reduce_only: reduceOnly
      });
      if (result.accepted === false) {
        const rejectedMessage = operationFailureMessage(result, 'Order rejected');
        setMessage('Rejected');
        onNotify('error', rejectedMessage);
      } else {
        const nextState = acceptedOrderState(result, orderType);
        const partialFill = booleanText(result.partial_fill) === true;
        const resultMessage = rawText(result, 'message');
        setMessage(partialFill ? 'Partial fill' : nextState === 'completed' ? 'Completed' : 'Submitted');
        onNotify(
          'success',
          partialFill && resultMessage !== '-' ? resultMessage : nextState === 'completed' ? 'Order completed' : 'Order submitted'
        );
      }
      refresh();
    } catch (exc) {
      const failure = readableErrorMessage(exc, 'Order failed.');
      setError(failure);
      onNotify('error', failure);
    } finally {
      setBusy(false);
    }
  }

  function applyOrderPercent(nextPercent: number) {
    const percent = clampOrderPercent(nextPercent);
    const nextMargin = inputNumber((availableToTrade * percent) / 100);
    setMargin(nextMargin);
    if (amountMode === 'size') setSizeDraft(inputSizeNumber(marginToOrderSize(nextMargin, leverage, referencePrice)));
  }

  function switchAmountMode(nextMode: OrderAmountMode) {
    setAmountMode(nextMode);
    if (nextMode === 'size') setSizeDraft(inputSizeNumber(marginToOrderSize(margin, leverage, referencePrice)));
  }

  function updateAmountInput(value: string) {
    if (amountMode === 'margin') {
      setMargin(value);
      return;
    }
    setSizeDraft(value);
    setMargin(value ? inputNumber(orderSizeToMargin(value, leverage, referencePrice)) : '');
  }

  return (
    <form className="order-ticket" onSubmit={submit}>
      <div className="panel-title">
        <Activity size={16} />
        <span>Order</span>
      </div>
      <div className="segmented">
        <button className={side === 'long' ? 'active buy' : ''} onClick={() => setSide('long')} type="button">
          Long
        </button>
        <button className={side === 'short' ? 'active sell' : ''} onClick={() => setSide('short')} type="button">
          Short
        </button>
      </div>
      <label>
        Type
        <select value={orderType} onChange={(event) => setOrderType(event.target.value as 'market' | 'limit')}>
          <option value="market">Market</option>
          <option value="limit">Limit</option>
        </select>
      </label>
      <div className="ticket-control amount-control">
        <div className="amount-control-head">
          <span>{amountMode === 'margin' ? 'Margin' : `Size${amountAsset ? ` (${amountAsset})` : ''}`}</span>
          <div className="amount-mode-toggle" aria-label="Order unit">
            <button
              className={amountMode === 'margin' ? 'active' : ''}
              onClick={() => switchAmountMode('margin')}
              type="button"
            >
              Margin
            </button>
            <button
              className={amountMode === 'size' ? 'active' : ''}
              onClick={() => switchAmountMode('size')}
              type="button"
            >
              Size
            </button>
          </div>
        </div>
        <input
          aria-label={amountInputLabel}
          type="number"
          min="0"
          step={amountMode === 'margin' ? '0.000001' : '0.00000001'}
          placeholder="0.00"
          value={amountInputValue}
          onChange={(event) => updateAmountInput(event.target.value)}
        />
      </div>
      {orderType === 'limit' && (
        <label>
          Limit Price
          <input
            type="number"
            min="0"
            step="0.000001"
            placeholder="0.00"
            value={limitPrice}
            onChange={(event) => setLimitPrice(event.target.value)}
          />
        </label>
      )}
      <div className="ticket-control">
        <span>Leverage</span>
        <LeverageSelector label="Leverage" max={maxLeverage} value={leverage} onChange={setLeverage} />
      </div>
      <div className="ticket-metric">
        <span>Position Value</span>
        <strong>{usd(positionValue)}</strong>
      </div>
      <div className="order-account-summary" aria-label="Order account summary">
        <div className="order-account-row">
          <span>Available to Trade</span>
          <strong>{usdcAmount(availableToTrade)}</strong>
        </div>
        <div className="order-account-row">
          <span>Current Position</span>
          <strong>{currentPositionLabel(selectedPosition, selectedMarket)}</strong>
        </div>
      </div>
      <div className="order-percent-control">
        <div className="order-percent-head">
          <span>Order Size</span>
          <strong>{orderPercent}%</strong>
        </div>
        <input
          aria-label="Order size percentage"
          disabled={availableToTrade <= 0}
          list="order-percent-ticks"
          max="100"
          min="0"
          onChange={(event) => applyOrderPercent(Number(event.target.value))}
          step={ORDER_PERCENT_STEP}
          style={orderPercentStyle}
          type="range"
          value={orderPercent}
        />
        <datalist id="order-percent-ticks">
          {ORDER_PERCENT_TICKS.map((tick) => (
            <option key={tick} value={tick} />
          ))}
        </datalist>
        <div className="order-percent-labels">
          {ORDER_PERCENT_TICKS.map((tick) => (
            <span key={tick}>{tick}%</span>
          ))}
        </div>
      </div>
      <label className="checkbox-line">
        <input type="checkbox" checked={reduceOnly} onChange={(event) => setReduceOnly(event.target.checked)} />
        Reduce Only
      </label>
      <button className={side === 'long' ? 'primary buy' : 'primary sell'} type="submit" disabled={busy || !selectedMarket}>
        Place {side === 'long' ? 'Long' : 'Short'}
      </button>
      {message && <div className="ticket-message">{message}</div>}
      {error && <div className="ticket-message error">{error}</div>}
    </form>
  );
}

function FavoriteMarketsBar({
  favorites,
  selectedMarket,
  metricMode,
  onMetricModeChange,
  onSelect,
  onToggle
}: {
  favorites: Market[];
  selectedMarket: Market | null;
  metricMode: FavoriteMetricMode;
  onMetricModeChange: (mode: FavoriteMetricMode) => void;
  onSelect: (symbol: string) => void;
  onToggle: (symbol: string) => void;
}) {
  const selectedLabel = selectedMarket ? marketLabel(selectedMarket) : '';
  const selectedIsFavorite = Boolean(selectedMarket && favorites.some((market) => market.symbol === selectedMarket.symbol));
  const sortedFavorites = useMemo(
    () => favorites.slice().sort((left, right) => marketLabel(left).localeCompare(marketLabel(right))),
    [favorites]
  );

  return (
    <section className="favorites-bar" aria-label="Favorite markets">
      <div className="favorite-bar-left">
        <div className="favorite-metric-toggle" aria-label="Favorite metric">
          <button className={metricMode === 'price' ? 'active' : ''} onClick={() => onMetricModeChange('price')} type="button">
            Price
          </button>
          <button className={metricMode === 'change' ? 'active' : ''} onClick={() => onMetricModeChange('change')} type="button">
            24h%
          </button>
        </div>
        {selectedMarket && (
          <button
            aria-label={`${selectedIsFavorite ? 'Unfavorite' : 'Favorite'} ${selectedLabel}`}
            className={selectedIsFavorite ? 'favorite-toggle active' : 'favorite-toggle'}
            onClick={() => onToggle(selectedMarket.symbol)}
            title={`${selectedIsFavorite ? 'Unfavorite' : 'Favorite'} ${selectedLabel}`}
            type="button"
          >
            <Star size={14} fill={selectedIsFavorite ? 'currentColor' : 'none'} />
          </button>
        )}
      </div>
      <div className="favorites-list horizontal">
        {sortedFavorites.map((market) => (
          <div className={market.symbol === selectedMarket?.symbol ? 'favorite-row active' : 'favorite-row'} key={market.symbol}>
            <button className="favorite-market" onClick={() => onSelect(market.symbol)} type="button">
              <span>{marketLabel(market)}</span>
              <span className={metricMode === 'change' && Number(marketChangePct(market) || 0) < 0 ? 'sell-text' : metricMode === 'change' ? 'buy-text' : ''}>
                {favoriteMetricLabel(market, metricMode)}
              </span>
            </button>
            <button
              aria-label={`Unfavorite ${market.symbol}`}
              className="favorite-remove"
              onClick={() => onToggle(market.symbol)}
              title={`Unfavorite ${marketLabel(market)}`}
              type="button"
            >
              <Star size={13} fill="currentColor" />
            </button>
          </div>
        ))}
        {sortedFavorites.length === 0 && <div className="favorites-empty">No favorites</div>}
      </div>
    </section>
  );
}

function MarginModal({
  token,
  position,
  limits,
  direction,
  onClose,
  onNotify,
  onUpdated
}: {
  token: string;
  position: Position;
  limits: MarginLimits;
  direction: 'add' | 'remove';
  onClose: () => void;
  onNotify: NotifyHandler;
  onUpdated: () => void;
}) {
  const [amount, setAmount] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const title = direction === 'add' ? 'Add Margin' : 'Remove Margin';
  const maxAmount = direction === 'add' ? Number(limits.max_add_margin_usd || 0) : Number(limits.max_remove_margin_usd || 0);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    onNotify('pending', direction === 'add' ? 'Add margin pending' : 'Remove margin pending');
    try {
      const result = await postMargin(token, position.symbol, { direction, amount_usd: Number(amount || 0) });
      if (result.accepted === false) {
        const failure = operationFailureMessage(result, 'Margin update failed');
        setError(failure);
        onNotify('error', failure);
        return;
      }
      onNotify('success', direction === 'add' ? 'Margin added' : 'Margin removed');
      onUpdated();
      onClose();
    } catch (exc) {
      const failure = readableErrorMessage(exc, 'Margin update failed.');
      setError(failure);
      onNotify('error', failure);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form className="modal" onSubmit={submit}>
        <h2>{title}</h2>
        <p>
          {position.symbol} {position.side} · Max {usd(maxAmount)}
        </p>
        <label htmlFor="margin-amount">Amount</label>
        <div className="inline-input">
          <input
            id="margin-amount"
            type="number"
            min="0"
            step="0.000001"
            value={amount}
            onChange={(event) => setAmount(event.target.value)}
          />
          <button type="button" onClick={() => setAmount(String(maxAmount))}>
            Max
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" disabled={busy}>
            Confirm
          </button>
        </div>
      </form>
    </div>
  );
}

function TpslModal({
  token,
  position,
  onClose,
  onNotify,
  onUpdated
}: {
  token: string;
  position: Position;
  onClose: () => void;
  onNotify: NotifyHandler;
  onUpdated: () => void;
}) {
  const [takeProfit, setTakeProfit] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    onNotify('pending', 'TP/SL pending');
    try {
      const result = await postPositionTpsl(token, position.symbol, {
        take_profit_price: Number(takeProfit || 0),
        stop_loss_price: Number(stopLoss || 0)
      });
      if (result.accepted === false) {
        const failure = operationFailureMessage(result, 'TP/SL update failed');
        setError(failure);
        onNotify('error', failure);
        return;
      }
      onNotify('success', 'TP/SL updated');
      onUpdated();
      onClose();
    } catch (exc) {
      const failure = readableErrorMessage(exc, 'TP/SL update failed.');
      setError(failure);
      onNotify('error', failure);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form className="modal" onSubmit={submit}>
        <h2>Set TP/SL</h2>
        <p>
          {position.symbol} {position.side} · Size {positionSizeLabel(position)}
        </p>
        <label htmlFor="take-profit-price">Take Profit</label>
        <input
          id="take-profit-price"
          min="0"
          step="0.000001"
          type="number"
          value={takeProfit}
          onChange={(event) => setTakeProfit(event.target.value)}
        />
        <label htmlFor="stop-loss-price">Stop Loss</label>
        <input
          id="stop-loss-price"
          min="0"
          step="0.000001"
          type="number"
          value={stopLoss}
          onChange={(event) => setStopLoss(event.target.value)}
        />
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" disabled={busy}>
            Confirm
          </button>
        </div>
      </form>
    </div>
  );
}

function PositionsTable({
  positions,
  token,
  account,
  onTradeAction,
  onSelectMarket,
  onPositionUpdated,
  onNotify,
  refresh
}: {
  positions: Position[];
  token: string;
  account: Account | null;
  onTradeAction: (position: Position, action: PositionTradeAction) => void;
  onSelectMarket: (symbol: string) => void;
  onPositionUpdated: (position: Position) => void;
  onNotify: NotifyHandler;
  refresh: RefreshHandler;
}) {
  const [modal, setModal] = useState<{ direction: 'add' | 'remove'; position: Position; limits: MarginLimits } | null>(null);
  const [tpslPosition, setTpslPosition] = useState<Position | null>(null);
  const [leverageBusy, setLeverageBusy] = useState('');
  const [leverageError, setLeverageError] = useState<{ symbol: string; message: string } | null>(null);
  const maxAdd = Number(account?.available_margin_usd || 0);

  function fallbackLimits(position: Position): MarginLimits {
    return {
      enabled: Boolean(position.only_isolated),
      max_add_margin_usd: maxAdd,
      max_remove_margin_usd: Math.max(0, Number(position.margin_used || 0))
    };
  }

  async function openMargin(position: Position, direction: 'add' | 'remove') {
    setModal({ direction, position, limits: position.margin_limits || fallbackLimits(position) });
    try {
      const limits = await fetchMarginLimits(token, position.symbol);
      setModal((current) =>
        current?.position.symbol === position.symbol && current.direction === direction
          ? { direction, position, limits }
          : current
      );
    } catch {
      // Keep the account snapshot limit visible if the realtime limit request fails.
    }
  }

  async function applyLeverage(position: Position, value: number) {
    const leverage = Number(value || 0);
    if (!leverage) return;
    setLeverageBusy(position.symbol);
    setLeverageError(null);
    onNotify('pending', 'Leverage pending');
    try {
      const result = await postLeverage(token, position.symbol, leverage);
      if (result.accepted === false) {
        const failure = leverageFailureMessage(result);
        setLeverageError({ symbol: position.symbol, message: failure });
        onNotify('error', failure);
        return;
      }
      const updatedPosition = positionFromLeverageResult(result);
      await refresh();
      if (updatedPosition) onPositionUpdated(updatedPosition);
      onNotify('success', 'Leverage adjusted');
    } catch (exc) {
      const failure = readableErrorMessage(exc, 'Leverage adjustment failed.');
      setLeverageError({
        symbol: position.symbol,
        message: failure
      });
      onNotify('error', failure);
    } finally {
      setLeverageBusy('');
    }
  }

  return (
    <div className="account-table-scroll positions-table-scroll">
      <table>
        <thead>
          <tr>
            <th>Market</th>
            <th>Side</th>
            <th>Leverage</th>
            <th>Value</th>
            <th>Size</th>
            <th>Entry</th>
            <th>Mark</th>
            <th>PnL</th>
            <th>ROE</th>
            <th>Funding</th>
            <th>Liq</th>
            <th>Margin</th>
            <th>Actions</th>
            <th>TP/SL</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((position) => {
            const marketParts = positionMarketParts(position.symbol);
            return (
              <tr key={position.symbol}>
                <td data-label="Market">
                  <div className="position-market">
                    <button
                      className="position-market-select"
                      onClick={() => onSelectMarket(position.symbol)}
                      type="button"
                      aria-label={`Select ${marketParts.market || position.symbol}-USDC`}
                    >
                      <strong>{marketParts.market || position.symbol}-USDC</strong>
                    </button>
                  </div>
                </td>
                <td className={position.side === 'long' ? 'buy-text' : 'sell-text'} data-label="Side">
                  {position.side}
                </td>
                <td data-label="Leverage">
                  <LeverageSelector
                    label={`${position.symbol} leverage`}
                    value={Number(position.leverage || 1)}
                    max={Number(position.max_leverage || 1)}
                    busy={leverageBusy === position.symbol}
                    confirm
                    allowSameCommit
                    onChange={(nextLeverage) => applyLeverage(position, nextLeverage)}
                  />
                  {leverageError?.symbol === position.symbol && (
                    <div className="position-inline-error" role="alert">
                      {leverageError.message}
                    </div>
                  )}
                </td>
                <td data-label="Value">{usd(position.notional_usd)}</td>
                <td data-label="Size">{positionSizeLabel(position)}</td>
                <td data-label="Entry">{price(position.display_entry_price || position.entry_price)}</td>
                <td data-label="Mark">{price(position.mid_price)}</td>
                <td
                  className={Number(position.synthetic_pnl_usd ?? position.unrealized_pnl ?? 0) >= 0 ? 'buy-text' : 'sell-text'}
                  data-label="PnL"
                >
                  {positionPnlLabel(position)}
                </td>
                <td data-label="ROE">{signedPct(position.synthetic_pnl_pct ?? position.return_on_equity)}</td>
                <td data-label="Funding">{positionFundingLabel(position)}</td>
                <td data-label="Liq">{price(position.liquidation_price)}</td>
                <td data-label="Margin">
                  <div className="position-margin-cell">
                    <span>{positionMarginLabel(position)}</span>
                    <div className="margin-actions">
                      <button type="button" onClick={() => void openMargin(position, 'add')}>
                        <PlusCircle size={14} /> Add Margin
                      </button>
                      <button type="button" onClick={() => void openMargin(position, 'remove')}>
                        <MinusCircle size={14} /> Remove
                      </button>
                    </div>
                  </div>
                </td>
                <td data-label="Actions">
                  <div className="position-trade-actions">
                    <button onClick={() => onTradeAction(position, 'limit')} title="Prefill a limit close order" type="button">
                      Limit
                    </button>
                    <button onClick={() => onTradeAction(position, 'market')} title="Close the full position at market" type="button">
                      Market
                    </button>
                    <button onClick={() => onTradeAction(position, 'reverse')} title="Reverse the full position at market" type="button">
                      Reverse
                    </button>
                  </div>
                </td>
                <td data-label="TP/SL">
                  <div className="position-tpsl-cell">
                    <span>{positionTpSlLabel(position, account?.open_orders || [])}</span>
                    <button type="button" onClick={() => setTpslPosition(position)}>
                      Set TP/SL
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {positions.length === 0 && <div className="empty-row">No open positions</div>}
      {modal && (
        <MarginModal
          token={token}
          position={modal.position}
          direction={modal.direction}
          limits={modal.limits}
          onClose={() => setModal(null)}
          onNotify={onNotify}
          onUpdated={refresh}
        />
      )}
      {tpslPosition && (
        <TpslModal
          token={token}
          position={tpslPosition}
          onClose={() => setTpslPosition(null)}
          onNotify={onNotify}
          onUpdated={refresh}
        />
      )}
    </div>
  );
}

type AccountTab = 'balances' | 'positions' | 'openOrders' | 'tradeHistory' | 'fundingHistory' | 'orderHistory';

const ACCOUNT_TABS: { id: AccountTab; label: string }[] = [
  { id: 'balances', label: 'Balances' },
  { id: 'positions', label: 'Positions' },
  { id: 'openOrders', label: 'Open Orders' },
  { id: 'tradeHistory', label: 'Trade History' },
  { id: 'fundingHistory', label: 'Funding History' },
  { id: 'orderHistory', label: 'Order History' }
];

function accountTabNeedsHistory(tab: AccountTab): boolean {
  return tab === 'tradeHistory' || tab === 'fundingHistory' || tab === 'orderHistory';
}

function accountTabCount(tab: AccountTab, account: Account | null, history: AccountHistory | null): number | undefined {
  if (tab === 'positions') return account?.positions?.length || 0;
  if (tab === 'openOrders') return account?.open_orders?.length || 0;
  return undefined;
}

function AccountTabs({
  activeTab,
  account,
  history,
  onSelect
}: {
  activeTab: AccountTab;
  account: Account | null;
  history: AccountHistory | null;
  onSelect: (tab: AccountTab) => void;
}) {
  return (
    <nav className="account-tabs top-account-tabs" aria-label="Account sections">
      {ACCOUNT_TABS.map((tab) => {
        const count = accountTabCount(tab.id, account, history);
        return (
          <button
            aria-label={typeof count === 'number' ? `${tab.label} ${count}` : tab.label}
            aria-pressed={activeTab === tab.id}
            className={activeTab === tab.id ? 'active' : ''}
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            type="button"
          >
            <span className="account-tab-label">{tab.label}</span>
            {typeof count === 'number' && <span className="account-tab-count">{count}</span>}
          </button>
        );
      })}
    </nav>
  );
}

function AccountPanel({
  account,
  token,
  activeTab,
  history,
  historyLoading,
  historyError,
  chartTimezone,
  onPositionTradeAction,
  onSelectMarket,
  onPositionUpdated,
  onNotify,
  refresh
}: {
  account: Account | null;
  token: string;
  activeTab: AccountTab;
  history: AccountHistory | null;
  historyLoading: boolean;
  historyError: string;
  chartTimezone: ChartTimezoneValue;
  onPositionTradeAction: (position: Position, action: PositionTradeAction) => void;
  onSelectMarket: (symbol: string) => void;
  onPositionUpdated: (position: Position) => void;
  onNotify: NotifyHandler;
  refresh: RefreshHandler;
}) {
  const positions = account?.positions || [];
  const openOrders = account?.open_orders || [];
  const needsHistory = accountTabNeedsHistory(activeTab);

  function renderBalances() {
    const metrics = [
      ['Account Value', account?.account_equity_usd ?? account?.perp_account_equity_usd],
      ['Available', account?.available_margin_usd ?? account?.remaining_capital_usd],
      ['Withdrawable', account?.withdrawable_usd],
      ['Margin Used', account?.total_margin_used_usd],
      ['Spot USDC', account?.spot_usdc_total],
      ['Spot Available', account?.spot_available_usdc]
    ];
    return (
      <div className="balance-grid">
        {metrics.map(([label, value]) => (
          <div className="balance-metric" key={String(label)}>
            <span>{label}</span>
            <strong>{typeof value === 'number' ? usd(value) : '-'}</strong>
          </div>
        ))}
      </div>
    );
  }

  function renderOpenOrders() {
    return (
      <div className="account-table-scroll open-orders-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Market</th>
              <th>Side</th>
              <th>Type</th>
              <th>Price</th>
              <th>Size</th>
              <th>Reduce Only</th>
              <th>OID</th>
            </tr>
          </thead>
          <tbody>
            {openOrders.map((order, index) => (
              <tr key={`${rawText(order, 'oid')}-${index}`}>
                <td>{timeLabel(rowTimeMs(order, 'timestamp', 'time'), chartTimezone)}</td>
                <td>{rawText(order, 'coin', 'symbol')}</td>
                <td>{sideLabel(rawText(order, 'side'))}</td>
                <td>{rawText(order, 'orderType', 'order_type', 'type')}</td>
                <td>{rawText(order, 'limitPx', 'triggerPx', 'price', 'px')}</td>
                <td>{rawText(order, 'sz', 'origSz', 'size')}</td>
                <td>{String(Boolean(order.reduceOnly ?? order.reduce_only))}</td>
                <td>{rawText(order, 'oid')}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {openOrders.length === 0 && <div className="empty-row">No open orders</div>}
      </div>
    );
  }

  function renderTradeHistory() {
    const rows = sortedHistoryRows(history?.trade_history || []);
    return (
      <div className="account-table-scroll history-table-scroll trade-history-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Market</th>
              <th>Direction</th>
              <th>Side</th>
              <th>Price</th>
              <th>Size</th>
              <th>Value</th>
              <th>Fee</th>
              <th>Closed PnL</th>
              <th>Start Pos</th>
              <th>Liquidity</th>
              <th>OID</th>
              <th>TID</th>
              <th>Hash</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${rawText(row, 'hash', 'oid')}-${index}`}>
                <td>{timeLabel(rowTimeMs(row, 'time'), chartTimezone)}</td>
                <td>{rawText(row, 'coin', 'symbol')}</td>
                <td>{tradeDirectionLabel(row)}</td>
                <td>{sideLabel(rawText(row, 'side'))}</td>
                <td>{rawNumber(row, 'px', 'price') ? price(rawNumber(row, 'px', 'price')) : '-'}</td>
                <td>{qtyPreciseOrDash(rawNumber(row, 'sz', 'size'))}</td>
                <td>{usdPreciseOrDash(tradeValue(row))}</td>
                <td>{tradeFeeLabel(row)}</td>
                <td>{signedUsdPreciseOrDash(rawNumber(row, 'closedPnl', 'closed_pnl'))}</td>
                <td>{qtyPreciseOrDash(rawNumber(row, 'startPosition', 'start_position'))}</td>
                <td>{liquidityLabel(row)}</td>
                <td>{rawText(row, 'oid')}</td>
                <td>{rawText(row, 'tid')}</td>
                <td>{rawText(row, 'hash')}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <div className="empty-row">No trade history</div>}
      </div>
    );
  }

  function renderFundingHistory() {
    const rows = sortedHistoryRows(history?.funding_history || []);
    return (
      <div className="account-table-scroll history-table-scroll funding-history-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Market</th>
              <th>Size</th>
              <th>Side</th>
              <th>Funding</th>
              <th>Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const source = mergedHistoryDelta(row);
              const marketParts = fundingMarketParts(source);
              return (
                <tr key={`${rawText(source, 'coin')}-${rawText(row, 'time')}-${index}`}>
                  <td>{historyFullTimeLabel(rowTimeMs(row, 'time'), chartTimezone)}</td>
                  <td>
                    <span className="position-market">
                      <strong>{marketParts.market}</strong>
                      {marketParts.dex && <span>{marketParts.dex}</span>}
                    </span>
                  </td>
                  <td>{fundingSizeLabel(source)}</td>
                  <td>{fundingSideLabel(source)}</td>
                  <td>{usdcPreciseOrDash(rawNumber(source, 'usdc', 'funding'), 4, 8)}</td>
                  <td>{fundingRateLabel(rawNumber(source, 'fundingRate', 'funding_rate', 'rate'))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && <div className="empty-row">No funding history</div>}
      </div>
    );
  }

  function renderOrderHistory() {
    const rows = sortedHistoryRows(history?.order_history || []);
    return (
      <div className="account-table-scroll history-table-scroll order-history-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Market</th>
              <th>Side</th>
              <th>Type</th>
              <th>Price</th>
              <th>Size</th>
              <th>Status</th>
              <th>OID</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const order = row.order && typeof row.order === 'object' ? (row.order as Record<string, unknown>) : row;
              return (
                <tr key={`${rawText(order, 'oid')}-${index}`}>
                  <td>
                    {timeLabel(
                      rowTimeMs(row, 'statusTimestamp', 'timestamp', 'time') || rowTimeMs(order, 'timestamp', 'time'),
                      chartTimezone
                    )}
                  </td>
                  <td>{rawText(order, 'coin', 'symbol')}</td>
                  <td>{sideLabel(rawText(order, 'side'))}</td>
                  <td>{rawText(order, 'orderType', 'order_type', 'type')}</td>
                  <td>{rawText(order, 'limitPx', 'triggerPx', 'price', 'px')}</td>
                  <td>{rawText(order, 'sz', 'origSz', 'size')}</td>
                  <td>{rawText(row, 'status')}</td>
                  <td>{rawText(order, 'oid')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && <div className="empty-row">No order history</div>}
      </div>
    );
  }

  function renderActiveTab() {
    if (activeTab === 'balances') return renderBalances();
    if (activeTab === 'positions') {
      return (
        <PositionsTable
          positions={positions}
          token={token}
          account={account}
          onTradeAction={onPositionTradeAction}
          onSelectMarket={onSelectMarket}
          onPositionUpdated={onPositionUpdated}
          onNotify={onNotify}
          refresh={refresh}
        />
      );
    }
    if (activeTab === 'openOrders') return renderOpenOrders();
    if (needsHistory && historyLoading && !history) return <div className="empty-row">Loading history</div>;
    if (historyError) return <div className="empty-row error">{historyError}</div>;
    if (activeTab === 'tradeHistory') return renderTradeHistory();
    if (activeTab === 'fundingHistory') return renderFundingHistory();
    return renderOrderHistory();
  }

  return (
    <section className="account-panel" aria-label="Account">
      <div className="account-panel-body">{renderActiveTab()}</div>
    </section>
  );
}

export default function App() {
  const [token, setToken] = useState(getStoredToken());
  const [session, setSession] = useState<Session | null>(null);
  const [markets, setMarkets] = useState<Market[]>([]);
  const [favoriteMarkets, setFavoriteMarkets] = useState<Market[]>([]);
  const [favoriteMetricMode, setFavoriteMetricMode] = useState<FavoriteMetricMode>('price');
  const [chartIndicators, setChartIndicators] = useState<IndicatorState>(() => readStoredIndicators());
  const [chartTimezone, setChartTimezone] = useState<ChartTimezoneValue>(() => readStoredChartTimezone());
  const [account, setAccount] = useState<Account | null>(null);
  const [activeAccountTab, setActiveAccountTab] = useState<AccountTab>('positions');
  const [history, setHistory] = useState<AccountHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const accountLoadingRef = useRef(false);
  const historyLoadingRef = useRef(false);
  const [snapshot, setSnapshot] = useState<MarketSnapshot | null>(null);
  const [book, setBook] = useState<OrderBook | null>(null);
  const [trades, setTrades] = useState<MarketTrade[]>([]);
  const [chartInterval, setChartInterval] = useState('1m');
  const [liveMid, setLiveMid] = useState<number | undefined>(undefined);
  const [wsStatus, setWsStatus] = useState('idle');
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [orderPrefill, setOrderPrefill] = useState<OrderTicketPrefill | null>(null);
  const [mobilePrimaryTab, setMobilePrimaryTab] = useState<MobilePrimaryTab>('chart');
  const [error, setError] = useState('');
  const [notification, setNotification] = useState<GlobalNotification | null>(null);
  const notificationIdRef = useRef(0);
  const mobileLayout = useMediaQuery('(max-width: 980px)');
  const selectedMarket = useMemo(
    () => markets.find((market) => market.symbol === selectedSymbol) || markets[0] || null,
    [markets, selectedSymbol]
  );
  const displayMid = liveMid || snapshot?.mid_price || selectedMarket?.mid_price;
  const favoriteDexKey = favoriteMarkets
    .map((market) => `${market.symbol}:${marketDex(market)}`)
    .sort()
    .join('|');
  const allMidsDexes = useMemo(() => {
    const dexes = new Set<string>();
    if (selectedMarket) dexes.add(marketDex(selectedMarket));
    favoriteMarkets.forEach((market) => dexes.add(marketDex(market)));
    if (dexes.size === 0) dexes.add('');
    return Array.from(dexes).sort();
  }, [favoriteDexKey, selectedMarket?.dex, selectedMarket?.symbol]);

  async function load(currentToken = token, nextSymbol = selectedSymbol, nextInterval = chartInterval) {
    if (!currentToken) return;
    setError('');
    try {
      const [nextSession, nextMarkets, nextAccount, nextFavorites] = await Promise.all([
        fetchSession(currentToken),
        fetchMarkets(currentToken),
        fetchAccount(currentToken),
        fetchFavoriteMarkets(currentToken).catch(() => [])
      ]);
      setSession(nextSession);
      setMarkets(nextMarkets);
      setAccount(nextAccount);
      setFavoriteMarkets(nextFavorites);
      const symbol = nextSymbol || nextMarkets[0]?.symbol || nextAccount.positions[0]?.symbol || '';
      if (symbol) {
        setSelectedSymbol(symbol);
        setTrades([]);
        const nextSnapshot = await fetchMarketSnapshot(
          currentToken,
          symbol,
          nextInterval,
          intervalWindowSeconds(nextInterval)
        );
        setSnapshot(nextSnapshot);
        setLiveMid(nextSnapshot.mid_price);
        try {
          setBook(await fetchMarketBook(currentToken, symbol));
        } catch {
          setBook(null);
        }
      }
      if (history || accountTabNeedsHistory(activeAccountTab)) {
        void loadAccountHistory(currentToken);
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      clearToken();
      setToken('');
    }
  }

  useEffect(() => {
    void load();
    // selectedSymbol changes are handled through explicit selection below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function selectMarket(symbol: string) {
    setSelectedSymbol(symbol);
    setSnapshot(null);
    setBook(null);
    setTrades([]);
    setLiveMid(undefined);
    if (token) {
      const nextSnapshot = await fetchMarketSnapshot(token, symbol, chartInterval, intervalWindowSeconds(chartInterval));
      setSnapshot(nextSnapshot);
      setLiveMid(nextSnapshot.mid_price);
      try {
        setBook(await fetchMarketBook(token, symbol));
      } catch {
        setBook(null);
      }
    }
  }

  async function loadAccountHistory(currentToken = token) {
    if (!currentToken || historyLoadingRef.current) return;
    historyLoadingRef.current = true;
    setHistoryLoading(true);
    setHistoryError('');
    try {
      setHistory(await fetchAccountHistory(currentToken, 90));
    } catch (exc) {
      setHistoryError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      historyLoadingRef.current = false;
      setHistoryLoading(false);
    }
  }

  async function loadAccountSnapshot(currentToken = token) {
    if (!currentToken || accountLoadingRef.current) return;
    accountLoadingRef.current = true;
    try {
      setAccount(await fetchAccount(currentToken));
    } catch {
      // Keep the current account snapshot if a background refresh fails.
    } finally {
      accountLoadingRef.current = false;
    }
  }

  function selectAccountTab(tab: AccountTab) {
    setActiveAccountTab(tab);
    if (accountTabNeedsHistory(tab)) {
      void loadAccountHistory();
    }
  }

  useEffect(() => {
    if (!token || !accountTabNeedsHistory(activeAccountTab)) return undefined;
    const timer = window.setInterval(() => {
      void loadAccountHistory(token);
    }, 15000);
    return () => window.clearInterval(timer);
    // loadAccountHistory uses a ref for in-flight state; this interval only needs token/tab changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeAccountTab, token]);

  useEffect(() => {
    if (!token) return undefined;
    const timer = window.setInterval(() => {
      void loadAccountSnapshot(token);
    }, ACCOUNT_REFRESH_INTERVAL_MS);
    return () => window.clearInterval(timer);
    // loadAccountSnapshot uses a ref for in-flight state; this interval only needs token changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function prefillPositionTrade(position: Position, action: PositionTradeAction) {
    const nextSide = oppositePositionSide(position);
    const sizing = positionActionSizing(position);
    const limitReference = Number(position.mid_price || position.display_entry_price || position.entry_price || 0);

    await selectMarket(position.symbol);
    setOrderPrefill({
      id: Date.now(),
      side: nextSide,
      orderType: 'limit',
      margin: inputNumber(sizing.marginUsd),
      leverage: sizing.leverage,
      limitPrice: action === 'limit' ? inputNumber(limitReference) : '',
      reduceOnly: true,
      message: `Limit close ${position.symbol} prepared`
    });
  }

  async function submitPositionMarketAction(position: Position, action: Exclude<PositionTradeAction, 'limit'>) {
    if (!token) return;
    const sizing = positionActionSizing(position);
    if (sizing.marginUsd <= 0) {
      notify('error', `${position.symbol} position size is too small.`);
      return;
    }
    const isClose = action === 'market';
    const pendingMessage = isClose ? 'Market close pending' : 'Reverse pending';
    const successMessage = isClose ? 'Market close completed' : 'Reverse completed';
    notify('pending', pendingMessage);
    try {
      const result = await postOrder(token, {
        symbol: position.symbol,
        order_type: 'market',
        side: oppositePositionSide(position),
        margin_usd: sizing.marginUsd,
        leverage: sizing.leverage,
        limit_price: 0,
        reduce_only: isClose,
        close_all: isClose,
        position_action: action === 'reverse' ? 'reverse' : undefined
      });
      if (result.accepted === false) {
        notify('error', operationFailureMessage(result, isClose ? 'Market close rejected' : 'Reverse rejected'));
        return;
      }
      const confirmed = await waitForPositionActionConfirmation(position, action);
      if (confirmed) notify('success', successMessage);
    } catch (exc) {
      notify('error', readableErrorMessage(exc, isClose ? 'Market close failed.' : 'Reverse failed.'));
    }
  }

  async function waitForPositionActionConfirmation(position: Position, action: Exclude<PositionTradeAction, 'limit'>): Promise<boolean> {
    const startedAt = Date.now();
    while (Date.now() - startedAt < POSITION_ACTION_CONFIRM_TIMEOUT_MS) {
      const nextAccount = await fetchAccount(token);
      setAccount(nextAccount);
      if (positionActionConfirmed(nextAccount, position, action)) return true;
      await waitMs(POSITION_ACTION_CONFIRM_INTERVAL_MS);
    }
    return false;
  }

  function handlePositionTradeAction(position: Position, action: PositionTradeAction) {
    if (action === 'limit') {
      void prefillPositionTrade(position, action);
      return;
    }
    void submitPositionMarketAction(position, action);
  }

  async function changeInterval(nextInterval: string) {
    setChartInterval(nextInterval);
    if (!token || !selectedSymbol) return;
    setSnapshot(null);
    const nextSnapshot = await fetchMarketSnapshot(token, selectedSymbol, nextInterval, intervalWindowSeconds(nextInterval));
    setSnapshot(nextSnapshot);
    setLiveMid(nextSnapshot.mid_price);
  }

  function updateChartIndicators(nextIndicators: IndicatorState) {
    setChartIndicators(nextIndicators);
    storeIndicators(nextIndicators);
  }

  function changeChartTimezone(nextTimezone: ChartTimezoneValue) {
    setChartTimezone(nextTimezone);
    storeChartTimezone(nextTimezone);
  }

  async function loadEarlierCandles(): Promise<number> {
    if (!token || !selectedSymbol || !snapshot?.candles?.length) return 0;
    const earliest = Math.min(...snapshot.candles.map((candle) => Number(candle.time)));
    if (!Number.isFinite(earliest) || earliest <= 0) return 0;
    const result = await fetchMarketBars(token, selectedSymbol, intervalResolution(chartInterval), undefined, earliest, 500);
    const existingTimes = new Set(snapshot.candles.map((candle) => Number(candle.time)));
    const newCandles = result.bars.filter((candle) => !existingTimes.has(Number(candle.time)));
    if (newCandles.length === 0) return 0;
    setSnapshot((current) => ({
      symbol: current?.symbol || selectedSymbol,
      interval: current?.interval || chartInterval,
      mid_price: current?.mid_price,
      candles: upsertCandles(current?.candles || [], newCandles)
    }));
    return newCandles.length;
  }

  async function toggleFavorite(symbol: string) {
    if (!token) return;
    const currentSymbols = favoriteMarkets.map((market) => market.symbol);
    const nextSymbols = currentSymbols.includes(symbol)
      ? currentSymbols.filter((item) => item !== symbol)
      : [...currentSymbols, symbol];
    const nextFavorites = await putFavoriteMarkets(token, nextSymbols);
    setFavoriteMarkets(nextFavorites);
  }

  function updateAccountPosition(position: Position) {
    setAccount((current) => mergeAccountPosition(current, position));
  }

  function notify(kind: NotificationKind, message: string) {
    const clean = String(message || '').trim();
    if (!clean) return;
    notificationIdRef.current += 1;
    setNotification({ id: notificationIdRef.current, kind, message: clean });
  }

  useEffect(() => {
    if (!notification || notification.kind === 'pending') return undefined;
    const timer = window.setTimeout(() => {
      setNotification((current) => (current?.id === notification.id ? null : current));
    }, NOTIFICATION_HIDE_MS);
    return () => window.clearTimeout(timer);
  }, [notification]);

  useEffect(() => {
    const wsSymbol = selectedMarket?.ws_symbol || selectedMarket?.execution_symbol || selectedMarket?.symbol || '';
    if (!token || !session?.network || !wsSymbol) return;
    const network = session.network;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let stopped = false;

    function subscribe(nextSocket: WebSocket) {
      allMidsDexes.forEach((dex) => {
        const subscription: { type: 'allMids'; dex?: string } = { type: 'allMids' };
        if (dex) subscription.dex = dex;
        nextSocket.send(JSON.stringify({ method: 'subscribe', subscription }));
      });
      nextSocket.send(JSON.stringify({ method: 'subscribe', subscription: { type: 'l2Book', coin: wsSymbol } }));
      nextSocket.send(JSON.stringify({ method: 'subscribe', subscription: { type: 'trades', coin: wsSymbol } }));
      nextSocket.send(
        JSON.stringify({ method: 'subscribe', subscription: { type: 'candle', coin: wsSymbol, interval: chartInterval } })
      );
    }

    function connect() {
      setWsStatus('connecting');
      socket = new WebSocket(wsUrlForNetwork(network));
      socket.onopen = () => {
        setWsStatus('live');
        if (socket) subscribe(socket);
      };
      socket.onmessage = (event) => {
        let payload: { channel?: string; data?: unknown };
        try {
          payload = JSON.parse(String(event.data));
        } catch {
          return;
        }
        if (payload.channel === 'l2Book' && payload.data && typeof payload.data === 'object') {
          const nextBook = normalizeOrderBook(payload.data as Record<string, unknown>);
          setBook(nextBook);
          if (nextBook.mid_price) {
            const midPrice = nextBook.mid_price;
            setLiveMid(midPrice);
            const selectedKeys = [
              selectedMarket?.ws_symbol,
              selectedMarket?.execution_symbol,
              selectedMarket?.symbol,
              selectedSymbol,
              nextBook.symbol
            ].filter((item): item is string => Boolean(item));
            const selectedMids = Object.fromEntries(selectedKeys.map((key) => [key, midPrice]));
            setMarkets((current) => applyLiveMids(current, selectedMids));
            setFavoriteMarkets((current) => applyLiveMids(current, selectedMids));
            setAccount((current) => applyLiveMidsToAccount(current, selectedMids));
          }
        }
        if (payload.channel === 'allMids' && payload.data && typeof payload.data === 'object') {
          const nextMids = normalizeAllMids(payload.data);
          if (Object.keys(nextMids).length > 0) {
            setMarkets((current) => applyLiveMids(current, nextMids));
            setFavoriteMarkets((current) => applyLiveMids(current, nextMids));
            setAccount((current) => applyLiveMidsToAccount(current, nextMids));
            const selectedMid = liveMidForMarket(selectedMarket, nextMids);
            if (selectedMid) setLiveMid(selectedMid);
          }
        }
        if (payload.channel === 'trades' && payload.data) {
          const rawItems = Array.isArray(payload.data) ? payload.data : [payload.data];
          const nextTrades = rawItems
            .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
            .map(normalizeTrade)
            .filter((item): item is MarketTrade => Boolean(item));
          if (nextTrades.length > 0) {
            setTrades((current) => upsertTrades(current, nextTrades));
          }
        }
        if (payload.channel === 'candle' && payload.data) {
          const rawItems = Array.isArray(payload.data) ? payload.data : [payload.data];
          const nextCandles = rawItems
            .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
            .map(normalizeRealtimeCandle)
            .filter((item): item is Candle => Boolean(item));
          if (nextCandles.length > 0) {
            setSnapshot((current) => ({
              symbol: selectedSymbol,
              interval: chartInterval,
              mid_price: current?.mid_price,
              candles: upsertCandles(current?.candles || [], nextCandles)
            }));
          }
        }
      };
      socket.onerror = () => {
        if (!stopped) setWsStatus('reconnecting');
      };
      socket.onclose = () => {
        if (stopped) return;
        setWsStatus('reconnecting');
        reconnectTimer = window.setTimeout(connect, 1500);
      };
    }

    connect();
    return () => {
      stopped = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [
    token,
    session?.network,
    selectedMarket?.ws_symbol,
    selectedMarket?.execution_symbol,
    selectedMarket?.symbol,
    selectedSymbol,
    chartInterval,
    allMidsDexes
  ]);

  if (!token) {
    return (
      <>
        <TokenGate
          onUnlock={(nextToken) => {
            storeToken(nextToken);
            setToken(nextToken);
          }}
        />
        {error && <div className="toast error">{error}</div>}
      </>
    );
  }

  const chartPanel = (
    <ChartPanel
      selectedMarket={selectedMarket}
      currentPosition={selectedPositionForMarket(account, selectedMarket)}
      snapshot={snapshot}
      interval={chartInterval}
      wsStatus={wsStatus}
      showResetButton={mobileLayout}
      chartTimezone={chartTimezone}
      indicators={chartIndicators}
      onChartTimezoneChange={changeChartTimezone}
      onIndicatorsChange={updateChartIndicators}
      onLoadEarlier={loadEarlierCandles}
      onIntervalChange={(nextInterval) => void changeInterval(nextInterval)}
    />
  );
  const accountTabs = <AccountTabs activeTab={activeAccountTab} account={account} history={history} onSelect={selectAccountTab} />;
  const accountPanel = (
    <AccountPanel
      account={account}
      token={token}
      activeTab={activeAccountTab}
      history={history}
      historyLoading={historyLoading}
      historyError={historyError}
      chartTimezone={chartTimezone}
      onPositionTradeAction={handlePositionTradeAction}
      onSelectMarket={(symbol) => void selectMarket(symbol)}
      onPositionUpdated={updateAccountPosition}
      onNotify={notify}
      refresh={() => load()}
    />
  );
  const favoritesBar = (
    <FavoriteMarketsBar
      favorites={favoriteMarkets}
      selectedMarket={selectedMarket}
      metricMode={favoriteMetricMode}
      onMetricModeChange={setFavoriteMetricMode}
      onSelect={(symbol) => void selectMarket(symbol)}
      onToggle={(symbol) => void toggleFavorite(symbol)}
    />
  );
  const accountBand = (
    <div className="market-control-band">
      {accountTabs}
      {accountPanel}
      {!mobileLayout && favoritesBar}
    </div>
  );
  const orderTicket = (
    <OrderTicket
      token={token}
      selectedMarket={selectedMarket}
      account={account}
      prefill={orderPrefill}
      onNotify={notify}
      refresh={() => void load()}
    />
  );
  const mobileBottomTabs = (
    <nav className="mobile-bottom-tabs" role="tablist" aria-label="Mobile primary views">
      {MOBILE_PRIMARY_TABS.map((tab) => (
        <button
          aria-selected={mobilePrimaryTab === tab.id}
          className={mobilePrimaryTab === tab.id ? 'active' : ''}
          key={tab.id}
          onClick={() => setMobilePrimaryTab(tab.id)}
          role="tab"
          type="button"
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
  const mobileTradingStack = (
    <section className="mobile-trade-stack" aria-label="Mobile trading">
      {favoritesBar}
      <div className="mobile-trade-top">
        {orderTicket}
        <div className="mobile-market-stream-stack">
          <OrderBookPanel book={book} midPrice={displayMid} wsStatus={wsStatus} />
          <TradesPanel trades={trades} chartTimezone={chartTimezone} />
        </div>
      </div>
      {accountBand}
    </section>
  );

  return (
    <main className="terminal">
      <GlobalNotificationToast notification={notification} />
      <header className="topbar">
        <div className="brand">Private Hyperliquid</div>
        <MarketSearch
          markets={markets}
          favoriteSymbols={favoriteMarkets.map((market) => market.symbol)}
          selectedSymbol={selectedSymbol}
          onToggleFavorite={(symbol) => void toggleFavorite(symbol)}
          onSelect={(symbol) => void selectMarket(symbol)}
        />
        <div className={session?.live_trading ? 'mode live' : 'mode dry'}>{session?.live_trading ? 'LIVE' : 'DRY RUN'}</div>
        <div className="account-chip">{session?.account_address_masked}</div>
        <button className="icon-button" type="button" onClick={() => void load()}>
          <RefreshCw size={16} />
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={() => {
            clearToken();
            setToken('');
          }}
        >
          <LogOut size={16} />
        </button>
      </header>
      {error && <div className="toast error">{error}</div>}
      {mobileLayout ? (
        <>
          <section className="mobile-workspace">
            {mobilePrimaryTab === 'chart' ? (
              <div className="mobile-chart-view">
                {favoritesBar}
                {chartPanel}
                {accountBand}
              </div>
            ) : (
              mobileTradingStack
            )}
          </section>
          {mobileBottomTabs}
        </>
      ) : (
        <>
          {accountBand}
          <section className="workspace desktop-workspace">
            {chartPanel}
            <div className="book-stack">
              <OrderBookPanel book={book} midPrice={displayMid} wsStatus={wsStatus} />
              <TradesPanel trades={trades} chartTimezone={chartTimezone} />
            </div>
            <div className="right-rail">{orderTicket}</div>
          </section>
        </>
      )}
    </main>
  );
}
