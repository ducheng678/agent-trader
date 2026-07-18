import type { Account, AccountHistory, MarginLimits, Market, MarketBars, MarketSnapshot, OrderBook, OrderPayload, Session } from './types';

const TOKEN_KEY = 'webTradeAdminToken';

function readStorage(storage: Storage): string {
  try {
    return storage.getItem(TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

function writeStorage(storage: Storage, token: string): void {
  try {
    storage.setItem(TOKEN_KEY, token);
  } catch {
    // Ignore storage failures, for example private browsing quota restrictions.
  }
}

function removeStorage(storage: Storage): void {
  try {
    storage.removeItem(TOKEN_KEY);
  } catch {
    // Ignore storage failures, for example private browsing quota restrictions.
  }
}

export function getStoredToken(): string {
  const persistentToken = readStorage(localStorage);
  if (persistentToken) return persistentToken;
  const sessionToken = readStorage(sessionStorage);
  if (sessionToken) writeStorage(localStorage, sessionToken);
  return sessionToken;
}

export function storeToken(token: string): void {
  writeStorage(localStorage, token);
  writeStorage(sessionStorage, token);
}

export function clearToken(): void {
  removeStorage(localStorage);
  removeStorage(sessionStorage);
}

async function requestJson<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...(init.headers || {})
    }
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchSession(token: string): Promise<Session> {
  return requestJson<Session>('/api/session', token);
}

export function fetchMarkets(token: string): Promise<Market[]> {
  return requestJson<Market[]>('/api/markets', token);
}

export function fetchAccount(token: string): Promise<Account> {
  return requestJson<Account>('/api/account', token);
}

export function fetchAccountHistory(token: string, windowDays = 90): Promise<AccountHistory> {
  const params = new URLSearchParams({ window_days: String(windowDays) });
  return requestJson<AccountHistory>(`/api/account/history?${params.toString()}`, token);
}

export function fetchMarketSnapshot(
  token: string,
  symbol: string,
  interval = '1m',
  windowSeconds = 3600
): Promise<MarketSnapshot> {
  const params = new URLSearchParams({ interval, window_seconds: String(windowSeconds) });
  return requestJson<MarketSnapshot>(`/api/market/${encodeURIComponent(symbol)}/snapshot?${params.toString()}`, token);
}

export function fetchMarketBook(token: string, symbol: string): Promise<OrderBook> {
  return requestJson<OrderBook>(`/api/market/${encodeURIComponent(symbol)}/book`, token);
}

export function fetchMarketBars(
  token: string,
  symbol: string,
  resolution: string,
  from?: number,
  to?: number,
  countBack?: number
): Promise<MarketBars> {
  const params = new URLSearchParams({ resolution });
  if (typeof from === 'number') params.set('from', String(from));
  if (typeof to === 'number') params.set('to', String(to));
  if (typeof countBack === 'number') params.set('count_back', String(countBack));
  return requestJson<MarketBars>(`/api/market/${encodeURIComponent(symbol)}/bars?${params.toString()}`, token);
}

export function fetchFavoriteMarkets(token: string): Promise<Market[]> {
  return requestJson<Market[]>('/api/favorites/markets', token);
}

export function putFavoriteMarkets(token: string, symbols: string[]): Promise<Market[]> {
  return requestJson<Market[]>('/api/favorites/markets', token, {
    method: 'PUT',
    body: JSON.stringify({ symbols })
  });
}

export function fetchMarginLimits(token: string, symbol: string): Promise<MarginLimits> {
  return requestJson<MarginLimits>(`/api/positions/${encodeURIComponent(symbol)}/margin-limits`, token);
}

export function postMargin(
  token: string,
  symbol: string,
  payload: { direction: 'add' | 'remove'; amount_usd: number }
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/positions/${encodeURIComponent(symbol)}/margin`, token, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function postPositionTpsl(
  token: string,
  symbol: string,
  payload: { take_profit_price: number; stop_loss_price: number }
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/positions/${encodeURIComponent(symbol)}/tpsl`, token, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function postLeverage(
  token: string,
  symbol: string,
  leverage: number
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/positions/${encodeURIComponent(symbol)}/leverage`, token, {
    method: 'POST',
    body: JSON.stringify({ leverage })
  });
}

export function postOrder(token: string, payload: OrderPayload): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>('/api/orders', token, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
