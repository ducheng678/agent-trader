export type Session = {
  network: string;
  account_address: string | null;
  account_address_masked: string;
  live_trading: boolean;
};

export type Market = {
  symbol: string;
  execution_symbol?: string;
  ws_symbol?: string;
  display_name?: string;
  dex?: string;
  market_name?: string;
  max_leverage?: number;
  only_isolated?: boolean;
  mid_price?: number;
  mark_price?: number;
  prev_day_price?: number;
  day_volume_usd?: number;
};

export type MarginLimits = {
  enabled: boolean;
  reason?: string;
  safety_buffer_usd?: number;
  current_position_value_usd?: number;
  isolated_position_equity_usd?: number;
  required_remaining_margin_usd?: number;
  max_add_margin_usd: number;
  max_remove_margin_usd: number;
};

export type Position = {
  symbol: string;
  side: string;
  size?: number;
  notional_usd: number;
  leverage: number;
  max_leverage: number;
  display_entry_price?: number;
  entry_price?: number;
  mid_price?: number;
  synthetic_pnl_usd?: number;
  synthetic_pnl_pct?: number;
  carried_realized_pnl_usd?: number;
  unrealized_pnl?: number;
  return_on_equity?: number;
  funding_since_open_usd?: number;
  funding_all_time_usd?: number;
  funding_since_change_usd?: number;
  liquidation_price?: number;
  margin_used?: number;
  lifecycle_roi_basis_usd?: number;
  only_isolated?: boolean;
  margin_limits?: MarginLimits;
};

export type Account = {
  available_margin_usd?: number;
  withdrawable_usd?: number;
  account_equity_usd?: number;
  perp_account_equity_usd?: number;
  spot_usdc_total?: number;
  spot_available_usdc?: number;
  total_margin_used_usd?: number;
  remaining_capital_usd?: number;
  positions: Position[];
  open_orders: Record<string, unknown>[];
};

export type AccountHistory = {
  window_days: number;
  start_time_ms?: number;
  end_time_ms?: number;
  trade_history: Record<string, unknown>[];
  funding_history: Record<string, unknown>[];
  order_history: Record<string, unknown>[];
};

export type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type MarketSnapshot = {
  symbol: string;
  interval?: string;
  candles?: Candle[];
  mid_price?: number;
};

export type MarketBars = {
  symbol: string;
  resolution: string;
  interval: string;
  bars: Candle[];
  no_data: boolean;
};

export type OrderBookLevel = {
  price: number;
  size: number;
  total: number;
  orders?: number;
};

export type OrderBook = {
  symbol?: string;
  time?: number;
  mid_price?: number;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
};

export type MarketTrade = {
  coin: string;
  side: 'B' | 'A' | string;
  price: number;
  size: number;
  hash?: string;
  time: number;
};

export type OrderPayload = {
  symbol: string;
  order_type: 'market' | 'limit';
  side: 'long' | 'short';
  margin_usd: number;
  leverage: number;
  limit_price?: number;
  reduce_only?: boolean;
  close_all?: boolean;
  position_action?: 'open' | 'reverse';
};
