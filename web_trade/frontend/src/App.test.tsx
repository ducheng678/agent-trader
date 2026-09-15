import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import App from './App';
import { clearToken, getStoredToken, storeToken } from './api';

const styles = readFileSync('src/styles.css', 'utf8');

const chartMocks = vi.hoisted(() => ({
  createChart: vi.fn(),
  setData: vi.fn(),
  addSeries: vi.fn(),
  addPane: vi.fn(),
  panes: vi.fn(),
  removePane: vi.fn(),
  removeSeries: vi.fn(),
  remove: vi.fn(),
  applyOptions: vi.fn(),
  seriesApplyOptions: vi.fn(),
  priceScaleSetAutoScale: vi.fn(),
  priceScaleGetVisibleRange: vi.fn(),
  priceScaleSetVisibleRange: vi.fn(),
  priceScaleWidth: vi.fn(),
  createPriceLine: vi.fn(),
  removePriceLine: vi.fn(),
  priceLineApplyOptions: vi.fn(),
  priceLineOptions: vi.fn(),
  timeScaleFitContent: vi.fn(),
  timeScaleResetTimeScale: vi.fn(),
  timeScaleScrollToRealTime: vi.fn(),
  timeScaleGetVisibleLogicalRange: vi.fn(),
  timeScaleSetVisibleLogicalRange: vi.fn(),
  timeScaleSubscribeVisibleLogicalRangeChange: vi.fn(),
  timeScaleUnsubscribeVisibleLogicalRangeChange: vi.fn(),
  subscribeCrosshairMove: vi.fn(),
  unsubscribeCrosshairMove: vi.fn(),
  visibleLogicalRangeHandler: undefined as ((range: { from: number; to: number } | null) => void) | undefined,
  crosshairMoveHandler: undefined as ((param: { seriesData: Map<unknown, unknown>; time?: number; point?: { x: number; y: number } }) => void) | undefined,
  candlestickSeries: undefined as unknown
}));

vi.mock('lightweight-charts', () => ({
  CandlestickSeries: 'CandlestickSeries',
  HistogramSeries: 'HistogramSeries',
  LineSeries: 'LineSeries',
  ColorType: { Solid: 'Solid' },
  CrosshairMode: { Normal: 0, Magnet: 1, Hidden: 2, MagnetOHLC: 3 },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2, LargeDashed: 3, SparseDotted: 4 },
  createChart: chartMocks.createChart.mockImplementation(() => ({
    addSeries: chartMocks.addSeries.mockImplementation((seriesType: string) => {
      const series = {
        setData: chartMocks.setData,
        applyOptions: chartMocks.seriesApplyOptions,
        priceScale: vi.fn(() => ({
          setAutoScale: chartMocks.priceScaleSetAutoScale,
          getVisibleRange: chartMocks.priceScaleGetVisibleRange,
          setVisibleRange: chartMocks.priceScaleSetVisibleRange,
          width: chartMocks.priceScaleWidth
        })),
        createPriceLine: chartMocks.createPriceLine.mockImplementation(() => ({
          applyOptions: chartMocks.priceLineApplyOptions,
          options: chartMocks.priceLineOptions
        })),
        removePriceLine: chartMocks.removePriceLine
      };
      if (seriesType === 'CandlestickSeries') chartMocks.candlestickSeries = series;
      return series;
    }),
    addPane: chartMocks.addPane,
    panes: chartMocks.panes,
    applyOptions: chartMocks.applyOptions,
    remove: chartMocks.remove,
    removePane: chartMocks.removePane,
    removeSeries: chartMocks.removeSeries,
    priceScale: vi.fn(() => ({ applyOptions: vi.fn() })),
    subscribeCrosshairMove: chartMocks.subscribeCrosshairMove.mockImplementation(
      (handler: (param: { seriesData: Map<unknown, unknown>; time?: number; point?: { x: number; y: number } }) => void) => {
        chartMocks.crosshairMoveHandler = handler;
      }
    ),
    unsubscribeCrosshairMove: chartMocks.unsubscribeCrosshairMove,
    timeScale: vi.fn(() => ({
      fitContent: chartMocks.timeScaleFitContent,
      resetTimeScale: chartMocks.timeScaleResetTimeScale,
      scrollToRealTime: chartMocks.timeScaleScrollToRealTime,
      getVisibleLogicalRange: chartMocks.timeScaleGetVisibleLogicalRange,
      setVisibleLogicalRange: chartMocks.timeScaleSetVisibleLogicalRange,
      subscribeVisibleLogicalRangeChange: chartMocks.timeScaleSubscribeVisibleLogicalRangeChange.mockImplementation(
        (handler: (range: { from: number; to: number } | null) => void) => {
          chartMocks.visibleLogicalRangeHandler = handler;
        }
      ),
      unsubscribeVisibleLogicalRangeChange: chartMocks.timeScaleUnsubscribeVisibleLogicalRangeChange
    }))
  }))
}));

const session = {
  network: 'testnet',
  account_address: null,
  account_address_masked: '0x1234...5678',
  live_trading: true
};

let favoriteSymbols: string[] = [];
let leverageResponse: Record<string, unknown> | null = null;
let orderResponse: Record<string, unknown> | null = null;
let leveragePromise: Promise<Response> | null = null;
let orderPromise: Promise<Response> | null = null;
let btcSnapshotCandles: Array<{ time: number; open: number; high: number; low: number; close: number; volume: number }> = [];

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readyState = 1;
  sent: unknown[] = [];
  url: string;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    queueMicrotask(() => this.onopen?.(new Event('open')));
  }

  send(payload: string) {
    this.sent.push(JSON.parse(payload));
  }

  close() {
    this.readyState = 3;
    this.onclose?.(new Event('close'));
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function mockMatchMedia(matches: boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  );
}

const account = {
  available_margin_usd: 500,
  withdrawable_usd: 480,
  account_equity_usd: 1200,
  spot_usdc_total: 200,
  spot_available_usdc: 180,
  total_margin_used_usd: 1000,
  positions: [
    {
      symbol: 'BTC',
      side: 'long',
      size: 0.1,
      notional_usd: 10000,
      leverage: 10,
      max_leverage: 40,
      display_entry_price: 100000,
      mid_price: 101000,
      synthetic_pnl_usd: 120,
      synthetic_pnl_pct: 0.12,
      return_on_equity: 0.12,
      funding_since_open_usd: -0.004,
      liquidation_price: 92000,
      margin_used: 1000,
      lifecycle_roi_basis_usd: 1000,
      only_isolated: true,
      margin_limits: {
        enabled: true,
        max_add_margin_usd: 480,
        max_remove_margin_usd: 250
      }
    }
  ],
  open_orders: [
    {
      coin: 'BTC',
      side: 'B',
      orderType: 'Limit',
      limitPx: '102000',
      sz: '0.05',
      reduceOnly: false,
      oid: 777,
      timestamp: 1700000000000
    }
  ]
};
let accountOverride: typeof account | null = null;

describe('App', () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    vi.clearAllMocks();
    MockWebSocket.instances = [];
    favoriteSymbols = ['ETH'];
    accountOverride = null;
    leverageResponse = null;
    orderResponse = null;
    leveragePromise = null;
    orderPromise = null;
    btcSnapshotCandles = [{ time: 1700000000, open: 100, high: 110, low: 95, close: 105, volume: 12 }];
    chartMocks.visibleLogicalRangeHandler = undefined;
    chartMocks.crosshairMoveHandler = undefined;
    chartMocks.candlestickSeries = undefined;
    chartMocks.timeScaleGetVisibleLogicalRange.mockReturnValue({ from: 0, to: 20 });
    chartMocks.priceScaleGetVisibleRange.mockReturnValue({ from: 95, to: 115 });
    chartMocks.priceScaleWidth.mockReturnValue(72);
    chartMocks.panes.mockReturnValue([
      { setStretchFactor: vi.fn(), setHeight: vi.fn() },
      { setStretchFactor: vi.fn(), setHeight: vi.fn() }
    ]);
    mockMatchMedia(false);
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const auth = init?.headers && (init.headers as Record<string, string>).Authorization;
        if (auth !== 'Bearer secret-token') {
          return new Response(JSON.stringify({ detail: 'Unauthorized' }), { status: 401 });
        }
        if (url.endsWith('/api/session')) {
          return Response.json(session);
        }
        if (url.endsWith('/api/markets')) {
          return Response.json([
            {
              symbol: 'BTC',
              execution_symbol: 'BTC',
              ws_symbol: 'BTC',
              display_name: 'BTC-USDC',
              mid_price: 101000,
              prev_day_price: 100000,
              mark_price: 101050,
              day_volume_usd: 123456,
              max_leverage: 40,
              only_isolated: true
            },
            {
              symbol: 'ETH',
              execution_symbol: 'ETH',
              ws_symbol: 'ETH',
              display_name: 'ETH-USDC',
              mid_price: 2050,
              prev_day_price: 2000,
              mark_price: 2051,
              day_volume_usd: 654321,
              max_leverage: 25,
              only_isolated: false
            },
            {
              symbol: 'xyz:ARM',
              execution_symbol: 'xyz:ARM',
              ws_symbol: 'xyz:ARM',
              display_name: 'ARM-USDC',
              dex: 'xyz',
              market_name: 'ARM',
              mid_price: 332.5,
              prev_day_price: 330,
              max_leverage: 10,
              only_isolated: true
            },
            {
              symbol: 'xyz:DRAM',
              execution_symbol: 'xyz:DRAM',
              ws_symbol: 'xyz:DRAM',
              display_name: 'DRAM-USDC',
              dex: 'xyz',
              market_name: 'DRAM',
              mid_price: 72.2,
              prev_day_price: 70,
              max_leverage: 20,
              only_isolated: true
            },
            {
              symbol: 'testdex:BTC',
              execution_symbol: 'testdex:BTC',
              ws_symbol: 'testdex:BTC',
              display_name: 'testdex:BTC-USDC',
              dex: 'testdex',
              mid_price: 99000,
              prev_day_price: 98000,
              max_leverage: 10,
              only_isolated: true
            }
          ]);
        }
        if (url.endsWith('/api/account')) {
          return Response.json(accountOverride || account);
        }
        if (url.includes('/api/account/history')) {
          return Response.json({
            window_days: 90,
            trade_history: [
              {
                coin: 'BTC',
                dir: 'Open Long',
                side: 'B',
                px: '100500',
                sz: '0.03',
                fee: '1.23',
                feeToken: 'USDC',
                closedPnl: '4.56',
                startPosition: '0.02',
                crossed: true,
                hash: '0xabcdef1234567890',
                oid: 888,
                tid: 123456,
                time: 1700000000000
              },
              {
                coin: 'BTC',
                dir: 'Close Long',
                side: 'A',
                px: '101000',
                sz: '0.01',
                fee: '0.41',
                feeToken: 'USDC',
                closedPnl: '10.00',
                startPosition: '0.03',
                crossed: false,
                hash: '0xrecentfill',
                oid: 889,
                tid: 123457,
                time: 1700000060000
              },
              {
                coin: 'BTC',
                dir: 'Close Long',
                side: 'A',
                px: '92000',
                sz: '0.02',
                fee: '0.20',
                feeToken: 'USDC',
                closedPnl: '-160.00',
                startPosition: '0.02',
                crossed: true,
                liquidation: { liquidatedUser: '0x1234', markPx: '92000', method: 'market' },
                hash: '0xliquidation',
                oid: 890,
                tid: 123458,
                time: 1700000120000
              }
            ],
            funding_history: [
              {
                coin: 'BTC',
                usdc: '0.0002',
                fundingRate: '0.00001',
                sz: '0.01',
                side: 'Long',
                time: Date.UTC(2026, 5, 27, 9, 0, 0)
              },
              {
                coin: 'xyz:ARM',
                usdc: '0.0005',
                fundingRate: '-0.00004',
                sz: '0.04',
                side: 'Long',
                time: Date.UTC(2026, 5, 27, 8, 0, 0)
              }
            ],
            order_history: [
              {
                order: { coin: 'BTC', side: 'B', orderType: 'Limit', limitPx: '101000', sz: '0.01', oid: 999 },
                status: 'filled',
                statusTimestamp: 1700000000000
              }
            ]
          });
        }
        if (url.endsWith('/api/favorites/markets') && init?.method !== 'PUT') {
          const allMarkets = [
            {
              symbol: 'BTC',
              execution_symbol: 'BTC',
              ws_symbol: 'BTC',
              display_name: 'BTC-USDC',
              mid_price: 101000,
              prev_day_price: 100000,
              max_leverage: 40,
              only_isolated: true
            },
            {
              symbol: 'ETH',
              execution_symbol: 'ETH',
              ws_symbol: 'ETH',
              display_name: 'ETH-USDC',
              mid_price: 2050,
              prev_day_price: 2000,
              max_leverage: 25,
              only_isolated: false
            },
            {
              symbol: 'xyz:ARM',
              execution_symbol: 'xyz:ARM',
              ws_symbol: 'xyz:ARM',
              display_name: 'ARM-USDC',
              dex: 'xyz',
              market_name: 'ARM',
              mid_price: 332.5,
              prev_day_price: 330,
              max_leverage: 10,
              only_isolated: true
            },
            {
              symbol: 'xyz:DRAM',
              execution_symbol: 'xyz:DRAM',
              ws_symbol: 'xyz:DRAM',
              display_name: 'DRAM-USDC',
              dex: 'xyz',
              market_name: 'DRAM',
              mid_price: 72.2,
              prev_day_price: 70,
              max_leverage: 20,
              only_isolated: true
            }
          ];
          return Response.json(allMarkets.filter((market) => favoriteSymbols.includes(market.symbol)));
        }
        if (url.endsWith('/api/favorites/markets') && init?.method === 'PUT') {
          favoriteSymbols = JSON.parse(String(init.body)).symbols;
          return Response.json(
            favoriteSymbols
              .filter((symbol) => ['BTC', 'ETH', 'xyz:ARM', 'xyz:DRAM'].includes(symbol))
              .map((symbol) => ({
                symbol,
                execution_symbol: symbol,
                ws_symbol: symbol,
                display_name: `${symbol.includes(':') ? symbol.split(':')[1] : symbol}-USDC`,
                dex: symbol.includes(':') ? symbol.split(':')[0] : undefined,
                market_name: symbol.includes(':') ? symbol.split(':')[1] : symbol,
                mid_price: symbol === 'BTC' ? 101000 : symbol === 'ETH' ? 2050 : symbol === 'xyz:ARM' ? 332.5 : 72.2,
                prev_day_price: symbol === 'BTC' ? 100000 : symbol === 'ETH' ? 2000 : symbol === 'xyz:ARM' ? 330 : 70,
                max_leverage: symbol === 'BTC' ? 40 : symbol === 'ETH' ? 25 : symbol === 'xyz:ARM' ? 10 : 20,
                only_isolated: true
              }))
          );
        }
        if (url.includes('/api/market/BTC/snapshot')) {
          return Response.json({
            symbol: 'BTC',
            candles: btcSnapshotCandles,
            mid_price: 101000
          });
        }
        if (url.includes('/api/market/ETH/snapshot')) {
          return Response.json({
            symbol: 'ETH',
            candles: [{ time: 1700000000, open: 2000, high: 2100, low: 1980, close: 2050, volume: 9 }],
            mid_price: 2050
          });
        }
        if (url.includes('/api/market/BTC/bars')) {
          return Response.json({
            symbol: 'BTC',
            resolution: '1',
            interval: '1m',
            bars: [{ time: 1699999940, open: 98, high: 102, low: 96, close: 100, volume: 3 }],
            no_data: false
          });
        }
        if (url.endsWith('/api/positions/BTC/margin-limits')) {
          return Response.json(account.positions[0].margin_limits);
        }
        if (url.endsWith('/api/positions/BTC/margin')) {
          return Response.json({ accepted: true, applied_amount_usd: 250, position: account.positions[0] });
        }
        if (url.endsWith('/api/positions/BTC/leverage')) {
          if (leveragePromise) return leveragePromise;
          return Response.json(leverageResponse || { accepted: true, target_leverage: JSON.parse(String(init?.body)).leverage });
        }
        if (url.endsWith('/api/positions/BTC/tpsl')) {
          return Response.json({ accepted: true });
        }
        if (url.endsWith('/api/orders')) {
          if (orderPromise) return orderPromise;
          return Response.json(orderResponse || { accepted: true });
        }
        return new Response('{}', { status: 404 });
      })
    );
  });

  test('persists the admin token across browser sessions', () => {
    storeToken('secret-token');

    expect(localStorage.getItem('webTradeAdminToken')).toBe('secret-token');
    expect(getStoredToken()).toBe('secret-token');

    clearToken();
    sessionStorage.setItem('webTradeAdminToken', 'legacy-token');

    expect(getStoredToken()).toBe('legacy-token');
    expect(localStorage.getItem('webTradeAdminToken')).toBe('legacy-token');
  });

  test('requires a token before loading private account data', async () => {
    render(<App />);

    expect(screen.getByLabelText(/admin token/i)).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/admin token/i), 'secret-token');
    await userEvent.click(screen.getByRole('button', { name: /unlock/i }));

    await waitFor(() => expect(screen.getByText('0x1234...5678')).toBeInTheDocument());
    expect(screen.getByText('LIVE')).toBeInTheDocument();
    expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0);
    expect(screen.getByText('+$120.00 (+12.00%)')).toBeInTheDocument();
  });

  test('uses max add margin amount in isolated margin modal', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /add margin/i }));
    await userEvent.click(screen.getByRole('button', { name: /^max$/i }));

    expect(screen.getByLabelText(/amount/i)).toHaveValue(480);
  });

  test('closes margin modal when clicking the backdrop', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /add margin/i }));
    expect(screen.getByRole('heading', { name: /add margin/i })).toBeInTheDocument();

    const backdrop = document.querySelector('.modal-backdrop');
    expect(backdrop).toBeTruthy();
    await userEvent.click(backdrop as Element);

    await waitFor(() => expect(screen.queryByRole('heading', { name: /add margin/i })).not.toBeInTheDocument());
  });

  test('renders account tabs with balances and open orders like the official panel', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole('button', { name: /balances/i }));

    expect(screen.getByText('Account Value')).toBeInTheDocument();
    expect(screen.getByText('$1,200.00')).toBeInTheDocument();
    expect(screen.getByText('Available')).toBeInTheDocument();
    expect(screen.getByText('$500.00')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /open orders 1/i }));
    expect(screen.getByText('102000')).toBeInTheDocument();
    expect(screen.getByText('777')).toBeInTheDocument();
  });

  test('polls the account snapshot so another client order updates positions automatically', async () => {
    const intervals: Array<{ handler: TimerHandler; timeout?: number }> = [];
    const setIntervalSpy = vi.spyOn(window, 'setInterval').mockImplementation((handler: TimerHandler, timeout?: number) => {
      intervals.push({ handler, timeout });
      return intervals.length as unknown as number;
    });
    const clearIntervalSpy = vi.spyOn(window, 'clearInterval').mockImplementation(() => undefined);

    try {
      sessionStorage.setItem('webTradeAdminToken', 'secret-token');
      render(<App />);

      await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
      accountOverride = {
        ...account,
        positions: [
          ...account.positions,
          {
            ...account.positions[0],
            symbol: 'ETH',
            size: 1,
            notional_usd: 2050,
            max_leverage: 25,
            mid_price: 2050
          }
        ]
      };

      const accountPoll = intervals.find((entry) => entry.timeout === 5000)?.handler;
      expect(typeof accountPoll).toBe('function');
      await act(async () => {
        if (typeof accountPoll === 'function') accountPoll();
      });

      await waitFor(() => expect(screen.getByRole('button', { name: /positions 2/i })).toBeInTheDocument());
      expect(within(screen.getByRole('region', { name: /account/i })).getByRole('row', { name: /ETH-USDC/i })).toBeInTheDocument();
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  test('places account tabs, active account content, and the favorite market bar above the chart', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const tabs = await screen.findByRole('navigation', { name: /account sections/i });
    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    const chart = screen.getByLabelText('chart');
    const accountPanel = screen.getByRole('region', { name: /account/i });

    expect(tabs.compareDocumentPosition(accountPanel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(accountPanel.compareDocumentPosition(favorites) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(favorites.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(accountPanel.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(accountPanel).getByRole('row', { name: /BTC-USDC/i })).toBeInTheDocument();
    expect(accountPanel).not.toContainElement(tabs);
  });

  test('selects a position market for charting and order entry when its symbol is clicked', async () => {
    accountOverride = {
      ...account,
      positions: [
        ...account.positions,
        {
          ...account.positions[0],
          symbol: 'ETH',
          size: 2,
          notional_usd: 4100,
          leverage: 5,
          max_leverage: 25,
          display_entry_price: 2000,
          mid_price: 2050,
          synthetic_pnl_usd: 100,
          synthetic_pnl_pct: 0.025,
          liquidation_price: 1600,
          margin_used: 820,
          margin_limits: {
            enabled: true,
            max_add_margin_usd: 480,
            max_remove_margin_usd: 120
          }
        }
      ]
    };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByDisplayValue('BTC-USDC')).toBeInTheDocument());
    const accountPanel = screen.getByRole('region', { name: /account/i });
    const ethRow = within(accountPanel).getByRole('row', { name: /ETH-USDC/i });

    await userEvent.click(within(ethRow).getByRole('button', { name: /select ETH-USDC/i }));

    await waitFor(() => expect(screen.getByDisplayValue('ETH-USDC')).toBeInTheDocument());
    const orderTicket = document.querySelector('.order-ticket') as HTMLElement;
    expect(within(orderTicket).getByRole('button', { name: /leverage 1x max 25x/i })).toBeInTheDocument();
    expect(within(orderTicket).getByText('2.000 ETH')).toBeInTheDocument();
  });

  test('uses mobile bottom tabs for chart viewing and the full trading stack', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const chart = await screen.findByLabelText('chart');
    const accountPanel = await screen.findByRole('region', { name: /account/i });
    const mobileTabs = screen.getByRole('tablist', { name: /mobile primary views/i });

    expect(chart.compareDocumentPosition(accountPanel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.queryByText('K-Line')).not.toBeInTheDocument();
    expect(within(mobileTabs).getByRole('tab', { name: /^chart$/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('region', { name: /mobile trading/i })).not.toBeInTheDocument();

    await userEvent.click(within(mobileTabs).getByRole('tab', { name: /^trade$/i }));
    const mobileTrading = screen.getByRole('region', { name: /mobile trading/i });

    expect(within(mobileTabs).getByRole('tab', { name: /^trade$/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByLabelText('chart')).not.toBeInTheDocument();
    expect(within(mobileTrading).getByRole('region', { name: /account/i })).toBeInTheDocument();
    expect(within(mobileTrading).getByRole('button', { name: /place long/i })).toBeInTheDocument();
    expect(within(mobileTrading).getByRole('region', { name: /order book/i })).toBeInTheDocument();
    expect(within(mobileTrading).getByRole('region', { name: /^trades$/i })).toBeInTheDocument();
  });

  test('marks account tables for compact mobile account layouts', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const accountPanel = await screen.findByRole('region', { name: /account/i });
    const positionsTable = accountPanel.querySelector('.positions-table-scroll');

    expect(positionsTable).toBeTruthy();
    expect(positionsTable?.querySelector('td[data-label="Market"]')).toBeTruthy();
    expect(positionsTable?.querySelector('td[data-label="TP/SL"]')).toBeTruthy();

    await userEvent.click(screen.getByRole('button', { name: /trade history/i }));
    await waitFor(() => expect(screen.getByText('Close Long')).toBeInTheDocument());
    expect(accountPanel.querySelector('.history-table-scroll.trade-history-table-scroll')).toBeTruthy();
  });

  test('keeps the mobile favorite market bar above the chart', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    const chart = await screen.findByLabelText('chart');
    const accountPanel = await screen.findByRole('region', { name: /account/i });

    expect(favorites.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(chart.compareDocumentPosition(accountPanel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(favorites).getByRole('button', { name: /ETH-USDC/i })).toBeInTheDocument();
  });

  test('places mobile order entry beside the order book above trades', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await userEvent.click(await screen.findByRole('tab', { name: /^trade$/i }));
    const mobileTrading = screen.getByRole('region', { name: /mobile trading/i });
    const tradeTop = mobileTrading.querySelector('.mobile-trade-top');
    expect(tradeTop).toBeTruthy();
    const tradeTopElement = tradeTop as HTMLElement;
    const streamStack = mobileTrading.querySelector('.mobile-market-stream-stack');
    expect(streamStack).toBeTruthy();
    const streamStackElement = streamStack as HTMLElement;

    const orderButton = within(mobileTrading).getByRole('button', { name: /place long/i });
    const orderBook = within(mobileTrading).getByRole('region', { name: /order book/i });
    const trades = within(mobileTrading).getByRole('region', { name: /^trades$/i });

    expect(tradeTopElement).toContainElement(orderButton);
    expect(streamStackElement).toContainElement(orderBook);
    expect(streamStackElement).toContainElement(trades);
    expect(orderBook.compareDocumentPosition(trades) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test('keeps the mobile favorite market bar above the trade tab content', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await userEvent.click(await screen.findByRole('tab', { name: /^trade$/i }));

    const mobileTrading = screen.getByRole('region', { name: /mobile trading/i });
    const favorites = within(mobileTrading).getByRole('region', { name: /favorite markets/i });
    const tradeTop = mobileTrading.querySelector('.mobile-trade-top') as HTMLElement;
    const accountPanel = within(mobileTrading).getByRole('region', { name: /account/i });

    expect(within(favorites).getByRole('button', { name: /ETH-USDC/i })).toBeInTheDocument();
    expect(within(favorites).getByText('$2,050.0')).toBeInTheDocument();
    expect(favorites.compareDocumentPosition(tradeTop) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(tradeTop.compareDocumentPosition(accountPanel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test('anchors the mobile order size slider at both endpoints', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await userEvent.click(await screen.findByRole('tab', { name: /^trade$/i }));

    const slider = screen.getByRole('slider', { name: /order size percentage/i });
    expect(slider).toHaveValue('0');
    expect(slider.getAttribute('style')).toContain('--order-percent-progress: 0%');

    fireEvent.change(slider, { target: { value: '100' } });

    await waitFor(() => expect(slider).toHaveValue('100'));
    expect(slider.getAttribute('style')).toContain('--order-percent-progress: 100%');
  });

  test('renders positions with official-style perps fields and action labels', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    const accountPanel = screen.getByRole('region', { name: /account/i });
    const headers = within(accountPanel)
      .getAllByRole('columnheader')
      .map((header) => header.textContent);

    expect(headers).toEqual([
      'Market',
      'Side',
      'Leverage',
      'Value',
      'Size',
      'Entry',
      'Mark',
      'PnL',
      'ROE',
      'Funding',
      'Liq',
      'Margin',
      'Actions',
      'TP/SL'
    ]);
    const row = within(accountPanel).getByRole('row', { name: /BTC-USDC/i });

    expect(within(row).getByText('0.1 BTC')).toBeInTheDocument();
    expect(within(row).getByText('101,000')).toBeInTheDocument();
    expect(within(row).getByText('+$120.00 (+12.00%)')).toBeInTheDocument();
    expect(within(row).getByText('-$0.00')).toBeInTheDocument();
    expect(within(row).getByText('$1,000.00 (Isolated)')).toBeInTheDocument();
    expect(within(row).getByRole('button', { name: /limit/i })).toBeInTheDocument();
    expect(within(row).getByRole('button', { name: /^market$/i })).toBeInTheDocument();
    expect(within(row).getByRole('button', { name: /reverse/i })).toBeInTheDocument();
    expect(within(row).getByText('-- / --')).toBeInTheDocument();
    expect(within(row).getByRole('button', { name: /set tp\/sl/i })).toBeInTheDocument();
  });

  test('sets TP and SL trigger prices from the position row', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /set tp\/sl/i }));

    await userEvent.type(screen.getByLabelText(/take profit/i), '105000');
    await userEvent.type(screen.getByLabelText(/stop loss/i), '99000');
    await userEvent.click(screen.getByRole('button', { name: /^confirm$/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith('/api/positions/BTC/tpsl') &&
            init?.method === 'POST' &&
            JSON.stringify(JSON.parse(String(init.body))) ===
              JSON.stringify({ take_profit_price: 105000, stop_loss_price: 99000 })
        )
      ).toBe(true)
    );
  });

  test('position limit action prefills the order ticket', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /^limit$/i }));

    await waitFor(() => expect(screen.getByLabelText(/^type$/i)).toHaveValue('limit'));
    expect(screen.getByLabelText(/^margin$/i)).toHaveValue(1000);
    expect(screen.getByLabelText(/limit price/i)).toHaveValue(101000);
    expect(screen.getByLabelText(/reduce only/i)).toBeChecked();
    expect(screen.getByRole('button', { name: /place short/i })).toBeInTheDocument();
    expect(screen.getByText('Limit close BTC prepared')).toBeInTheDocument();
  });

  test('position market action submits a full reduce-only market close', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /^market$/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url, init]) => {
          if (!String(url).endsWith('/api/orders') || init?.method !== 'POST') return false;
          const body = JSON.parse(String(init.body));
          return (
            body.symbol === 'BTC' &&
            body.order_type === 'market' &&
            body.side === 'short' &&
            body.margin_usd === 1000 &&
            body.leverage === 10 &&
            body.reduce_only === true &&
            body.close_all === true
          );
        })
      ).toBe(true)
    );
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Market close pending'));
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    expect(screen.getByRole('status')).toHaveTextContent('Market close pending');

    accountOverride = { ...account, positions: [] };
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Market close completed'));
  });

  test('position reverse action submits an opposite market target as a full reverse', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /^reverse$/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url, init]) => {
          if (!String(url).endsWith('/api/orders') || init?.method !== 'POST') return false;
          const body = JSON.parse(String(init.body));
          return (
            body.symbol === 'BTC' &&
            body.order_type === 'market' &&
            body.side === 'short' &&
            body.margin_usd === 1000 &&
            body.leverage === 10 &&
            body.reduce_only === false &&
            body.close_all === false &&
            body.position_action === 'reverse'
          );
        })
      ).toBe(true)
    );
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Reverse pending'));
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    expect(screen.getByRole('status')).toHaveTextContent('Reverse pending');

    accountOverride = {
      ...account,
      positions: [{ ...account.positions[0], side: 'short', size: 0.1, notional_usd: 10000 }]
    };
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Reverse completed'));
  });

  test('shows available balance, current position, and a percent slider in the order ticket', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());

    expect(screen.getByText('Available to Trade')).toBeInTheDocument();
    expect(screen.getByText('500.00 USDC')).toBeInTheDocument();
    expect(screen.getByText('Current Position')).toBeInTheDocument();
    expect(screen.getByText('0.100 BTC')).toBeInTheDocument();

    const slider = screen.getByRole('slider', { name: /order size percentage/i });
    expect(slider).toHaveValue('0');
    expect(slider).toHaveAttribute('step', '1');
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    fireEvent.change(slider, { target: { value: '26' } });

    expect(screen.getByLabelText(/^margin$/i)).toHaveValue(130);
    expect(slider).toHaveValue('26');
    expect(screen.getByText('26%')).toBeInTheDocument();
  });

  test('lazy loads and renders trade funding and order history tabs', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /trade history/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole('button', { name: /trade history/i }));

    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/account/history?window_days=90'))).toBe(true)
    );
    await waitFor(() =>
      expect(within(screen.getByRole('region', { name: /account/i })).getAllByRole('row').length).toBeGreaterThan(3)
    );
    expect(screen.getByText('Market Order Liquidation: Close Long')).toBeInTheDocument();
    const tradeRows = within(screen.getByRole('region', { name: /account/i })).getAllByRole('row');
    expect(tradeRows[1]).toHaveTextContent('Market Order Liquidation: Close Long');
    expect(tradeRows[2]).toHaveTextContent('Close Long');
    expect(tradeRows[3]).toHaveTextContent('Open Long');
    expect(screen.getByText('Open Long')).toBeInTheDocument();
    expect(screen.getByText('Buy')).toBeInTheDocument();
    expect(screen.getByText('$3,015.00')).toBeInTheDocument();
    expect(screen.getByText('$1.23 USDC')).toBeInTheDocument();
    expect(screen.getAllByText('0.02').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Taker').length).toBeGreaterThan(0);
    expect(screen.getByText('0xabcdef1234567890')).toBeInTheDocument();
    expect(screen.getByText('888')).toBeInTheDocument();
    expect(screen.getByText('123456')).toBeInTheDocument();
    const tabs = screen.getByRole('navigation', { name: /account sections/i });
    expect(within(tabs).getByRole('button', { name: /^trade history$/i })).toBeInTheDocument();
    expect(within(tabs).getByRole('button', { name: /^funding history$/i })).toBeInTheDocument();
    expect(within(tabs).getByRole('button', { name: /^order history$/i })).toBeInTheDocument();
    expect(within(tabs).queryByRole('button', { name: /trade history 2/i })).not.toBeInTheDocument();
    expect(within(tabs).queryByRole('button', { name: /funding history 1/i })).not.toBeInTheDocument();
    expect(within(tabs).queryByRole('button', { name: /order history 1/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /funding history/i }));
    const fundingRow = screen.getByRole('row', { name: /0\.0005 USDC/i });
    expect(within(fundingRow).getByText('6/27/2026 - 08:00:00')).toBeInTheDocument();
    expect(within(fundingRow).getByText('ARM')).toBeInTheDocument();
    expect(within(fundingRow).getByText('xyz')).toBeInTheDocument();
    expect(within(fundingRow).getByText('0.04 ARM')).toBeInTheDocument();
    expect(within(fundingRow).getByText('Long')).toBeInTheDocument();
    expect(within(fundingRow).getByText('0.0005 USDC')).toBeInTheDocument();
    expect(within(fundingRow).getByText('-0.0040%')).toBeInTheDocument();
    expect(screen.queryByText('$0.0005')).not.toBeInTheDocument();
    const btcFundingRow = screen.getByRole('row', { name: /0\.0002 USDC/i });
    expect(within(btcFundingRow).getByText('BTC')).toBeInTheDocument();
    expect(within(btcFundingRow).queryByText('-')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /order history/i }));
    expect(screen.getByText('filled')).toBeInTheDocument();
    expect(screen.getByText('999')).toBeInTheDocument();
  });

  test('refreshes account history when an already loaded history tab is selected again', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /order history/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole('button', { name: /order history/i }));
    await waitFor(() => expect(screen.getByText('filled')).toBeInTheDocument());
    await waitFor(
      () => expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/account/history?window_days=90'))).toHaveLength(1)
    );

    await userEvent.click(screen.getByRole('button', { name: /positions/i }));
    await userEvent.click(screen.getByRole('button', { name: /order history/i }));

    await waitFor(
      () => expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/account/history?window_days=90'))).toHaveLength(2)
    );
  });

  test('submits a completed market order from the order ticket', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    expect(screen.queryByRole('spinbutton', { name: /^leverage$/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /leverage 1x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^20x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    await waitFor(() => expect(screen.getByText('Completed')).toBeInTheDocument());
    expect(await screen.findByRole('status')).toHaveTextContent('Order completed');
    const orderCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/orders'));
    expect(orderCall).toBeDefined();
    expect(JSON.parse(String(orderCall?.[1]?.body))).toMatchObject({
      symbol: 'BTC',
      order_type: 'market',
      side: 'long',
      margin_usd: 250,
      leverage: 20
    });
  });

  test('shows the exchange partial fill message for a market order', async () => {
    orderResponse = {
      accepted: true,
      partial_fill: true,
      message: 'Market order partially filled: filled 0.0002 / requested 0.00034 BTC.'
    };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    await waitFor(() => expect(screen.getByText('Partial fill')).toBeInTheDocument());
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(
        'Market order partially filled: filled 0.0002 / requested 0.00034 BTC.'
      )
    );
  });

  test('shows submitted for a resting limit order from the order ticket', async () => {
    orderResponse = {
      accepted: true,
      result: {
        actions: [
          {
            entry_limit: {
              response: {
                data: {
                  statuses: [{ resting: { oid: 123 } }]
                }
              }
            }
          }
        ]
      }
    };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.selectOptions(screen.getByLabelText(/^type$/i), 'limit');
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.type(screen.getByLabelText(/limit price/i), '99000');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    await waitFor(() => expect(screen.getByText('Submitted')).toBeInTheDocument());
    expect(await screen.findByRole('status')).toHaveTextContent('Order submitted');
  });

  test('can switch the order amount input from margin to size and submit the converted margin', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /leverage 1x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^20x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /^size$/i }));
    await userEvent.type(screen.getByLabelText(/^size$/i), '0.05');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    await waitFor(() => expect(screen.getByText('Completed')).toBeInTheDocument());
    const orderCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/orders'));
    expect(JSON.parse(String(orderCall?.[1]?.body))).toMatchObject({
      symbol: 'BTC',
      order_type: 'market',
      side: 'long',
      margin_usd: 252.5,
      leverage: 20
    });
  });

  test('shows size in the order amount input when the percentage slider changes in size mode', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /^size$/i }));
    fireEvent.change(screen.getByRole('slider', { name: /order size percentage/i }), { target: { value: '50' } });

    expect(screen.getByLabelText(/^size$/i)).toHaveValue(0.00247525);
    await waitFor(() => expect(screen.getByText('$250.00')).toBeInTheDocument());
  });

  test('shows a top-left alert when an order is rejected', async () => {
    orderResponse = { accepted: false, message: 'Order rejected by exchange.' };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Order rejected by exchange.');
  });

  test('shows a top-left pending notification while an order is submitting', async () => {
    let resolveOrder: (response: Response) => void = () => undefined;
    orderPromise = new Promise((resolve) => {
      resolveOrder = resolve;
    });
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    expect(await screen.findByRole('status')).toHaveTextContent('Order pending');

    resolveOrder(Response.json({ accepted: true }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Order completed'));
  });

  test('keeps pending order status visible with a spinner until the request finishes', async () => {
    let resolveOrder: (response: Response) => void = () => undefined;
    orderPromise = new Promise((resolve) => {
      resolveOrder = resolve;
    });
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    expect(await screen.findByRole('status')).toHaveTextContent('Order pending');
    await new Promise((resolve) => window.setTimeout(resolve, 4300));
    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('Order pending');
    expect(within(status).getByLabelText(/pending/i)).toBeInTheDocument();

    resolveOrder(Response.json({ accepted: true }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Order completed'));
  }, 8000);

  test('auto hides the top-left order status after a short delay', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.type(screen.getByLabelText(/^margin$/i), '250');
    await userEvent.click(screen.getByRole('button', { name: /place long/i }));

    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Order completed'));
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument(), { timeout: 5000 });
  });

  test('renders the order leverage slider progress at the visual minimum and maximum', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /leverage 1x max 40x/i }));
    const minSlider = screen.getByLabelText(/leverage slider/i);

    expect(minSlider).toHaveValue('1');
    expect(minSlider).toHaveStyle('--leverage-progress: 0%');

    await userEvent.click(screen.getByRole('button', { name: /^40x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /leverage 40x max 40x/i }));
    const maxSlider = screen.getByLabelText(/leverage slider/i);

    expect(maxSlider).toHaveValue('40');
    expect(maxSlider).toHaveStyle('--leverage-progress: 100%');
  });

  test('filters markets by typed search text and exposes row favorite controls', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const marketSearch = screen.getByRole('combobox', { name: /market/i });
    await userEvent.clear(marketSearch);
    await userEvent.type(marketSearch, 'eth');

    expect(screen.getByRole('option', { name: /ETH-USDC/i })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /^BTC-USDC$/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /unfavorite ETH-USDC/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith('/api/favorites/markets') &&
            init?.method === 'PUT' &&
            JSON.stringify(JSON.parse(String(init.body)).symbols) === JSON.stringify([])
        )
      ).toBe(true)
    );
    expect(screen.getByRole('combobox', { name: /market/i })).toHaveValue('eth');
  });

  test('does not overwrite typed market search text during live market updates', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const marketSearch = screen.getByRole('combobox', { name: /market/i });
    await userEvent.clear(marketSearch);
    await userEvent.type(marketSearch, 'sp');
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));

    MockWebSocket.instances[0].emit({
      channel: 'allMids',
      data: { mids: { BTC: '101200', ETH: '2060' } }
    });

    await waitFor(() => expect(screen.getByText('$2,060.0')).toBeInTheDocument());
    await waitFor(() => expect(marketSearch).toHaveValue('sp'));
    expect(document.querySelector('.market-stat')).toBeNull();
  });

  test('hides dex prefixes from market search results while keeping the internal symbol', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const marketSearch = screen.getByRole('combobox', { name: /market/i });
    await userEvent.clear(marketSearch);
    await userEvent.type(marketSearch, 'testdex');

    const listbox = screen.getByRole('listbox');
    expect(within(listbox).queryByText('testdex:BTC-USDC')).not.toBeInTheDocument();
    expect(within(listbox).getByRole('option', { name: /BTC-USDC/i })).toBeInTheDocument();
  });

  test('hides xyz dex labels in the positions table', async () => {
    const originalSymbol = account.positions[0].symbol;
    account.positions[0].symbol = 'xyz:BTC';
    try {
      sessionStorage.setItem('webTradeAdminToken', 'secret-token');
      render(<App />);

      await waitFor(() => expect(screen.getByRole('button', { name: /positions 1/i })).toBeInTheDocument());
      const accountPanel = screen.getByRole('region', { name: /account/i });
      const row = within(accountPanel).getByRole('row', { name: /BTC-USDC/i });

      expect(within(row).getByText('BTC-USDC')).toBeInTheDocument();
      expect(within(row).queryByText('xyz')).not.toBeInTheDocument();
      expect(within(row).queryByText('xyz:BTC-USDC')).not.toBeInTheDocument();
    } finally {
      account.positions[0].symbol = originalSymbol;
    }
  });

  test('closes the market search dropdown when clicking outside it', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const marketSearch = screen.getByRole('combobox', { name: /market/i });
    await userEvent.click(marketSearch);

    expect(screen.getByRole('option', { name: /ETH-USDC/i })).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText('chart'));

    expect(screen.queryByRole('option', { name: /ETH-USDC/i })).not.toBeInTheDocument();
  });

  test('renders normalized candles without default indicators and only shows chosen indicators', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    expect(chartMocks.addSeries).not.toHaveBeenCalledWith('HistogramSeries', expect.any(Object));
    expect(chartMocks.addSeries).not.toHaveBeenCalledWith('LineSeries', expect.any(Object));

    await userEvent.click(screen.getByRole('button', { name: /indicators/i }));
    await userEvent.click(screen.getByRole('checkbox', { name: /volume/i }));
    await userEvent.click(screen.getByRole('checkbox', { name: /^ema$/i }));

    await waitFor(() => expect(chartMocks.addSeries).toHaveBeenCalledWith('HistogramSeries', expect.any(Object), 1));
    const emaCall = chartMocks.addSeries.mock.calls.find(
      ([seriesType, options]) => seriesType === 'LineSeries' && (options as { color?: string }).color === '#38bdf8'
    );
    const emaOptions = emaCall?.[1] as { title?: string; lastValueVisible?: boolean; priceLineVisible?: boolean; priceFormat?: { formatter?: (value: number) => string } };
    expect(emaOptions).toEqual(
      expect.objectContaining({
        color: '#38bdf8',
        lastValueVisible: true,
        priceLineVisible: false
      })
    );
    expect(emaOptions.title).toBeUndefined();
    expect(emaOptions.priceFormat?.formatter?.(72.2)).toBe('72.200');
    expect(chartMocks.addSeries).not.toHaveBeenCalledWith('LineSeries', expect.objectContaining({ color: '#f59e0b' }), 0);
  });

  test('lets users configure EMA and SMA indicator periods', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    await userEvent.click(screen.getByRole('button', { name: /indicators/i }));
    const emaPeriod = screen.getByRole('spinbutton', { name: /ema period/i });
    const smaPeriod = screen.getByRole('spinbutton', { name: /sma period/i });

    expect(emaPeriod).toHaveValue(20);
    expect(smaPeriod).toHaveValue(50);

    fireEvent.change(emaPeriod, { target: { value: '12' } });
    await userEvent.click(screen.getByRole('checkbox', { name: /^ema$/i }));

    await waitFor(() => expect(chartMocks.addSeries).toHaveBeenCalledWith('LineSeries', expect.objectContaining({ color: '#38bdf8' }), 0));
    expect(JSON.parse(localStorage.getItem('webTradeChartIndicators') || '{}')).toEqual(
      expect.objectContaining({ ema: true, emaLength: 12, smaLength: 50 })
    );
  });

  test('closes the indicators menu when clicking outside it', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createChart).toHaveBeenCalled());

    await userEvent.click(screen.getByRole('button', { name: /indicators/i }));
    expect(screen.getByRole('spinbutton', { name: /ema period/i })).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('chart'));

    expect(screen.queryByRole('spinbutton', { name: /ema period/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /indicators/i })).toHaveAttribute('aria-expanded', 'false');
  });

  test('renders the indicators menu outside the mobile interval scroller', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createChart).toHaveBeenCalled());

    await userEvent.click(screen.getByRole('button', { name: /indicators/i }));

    const popover = screen.getByRole('dialog', { name: /chart indicators/i });
    expect(popover).toHaveClass('floating-popover');
    expect(popover.parentElement).toBe(document.body);
    expect(screen.getByRole('spinbutton', { name: /ema period/i })).toBeInTheDocument();
  });

  test('keeps the chart crosshair following the mouse instead of snapping to candle prices', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createChart).toHaveBeenCalled());

    expect(chartMocks.createChart).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.objectContaining({
        crosshair: { mode: 0 },
        handleScroll: expect.objectContaining({
          horzTouchDrag: true,
          vertTouchDrag: true
        }),
        handleScale: expect.objectContaining({
          pinch: true
        })
      })
    );
  });

  test('keeps right-side breathing room when a new chart is loaded', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createChart).toHaveBeenCalled());

    expect(chartMocks.createChart).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.objectContaining({
        timeScale: expect.objectContaining({ rightOffset: expect.any(Number) })
      })
    );
    const [, options] = chartMocks.createChart.mock.calls[0];
    expect((options as { timeScale?: { rightOffset?: number } }).timeScale?.rightOffset).toBeGreaterThan(0);
  });

  test('resets the chart viewport on initial candle load without animated scrolling', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );

    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 });
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
  });

  test('formats the chart price axis with five significant digits', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.addSeries).toHaveBeenCalledWith('CandlestickSeries', expect.any(Object), 0));
    const candleCall = chartMocks.addSeries.mock.calls.find(([series]) => series === 'CandlestickSeries');
    const priceFormat = (candleCall?.[1] as { priceFormat?: { formatter?: (value: number) => string } } | undefined)
      ?.priceFormat;

    expect(priceFormat?.formatter?.(72.2)).toBe('72.200');
    expect(priceFormat?.formatter?.(100)).toBe('100.00');
    expect(priceFormat?.formatter?.(0.021)).toBe('0.02100');
    expect(priceFormat?.formatter?.(0.021)).not.toBe('0.021000');
  });

  test('leaves mobile touch panning to the native Lightweight Charts handler', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    const chart = screen.getByLabelText('chart');
    chart.getBoundingClientRect = vi.fn(() => ({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 800,
      bottom: 160,
      width: 800,
      height: 160,
      toJSON: () => ({})
    }));
    chartMocks.priceScaleSetAutoScale.mockClear();
    chartMocks.priceScaleSetVisibleRange.mockClear();

    fireEvent.touchStart(chart, {
      touches: [{ clientX: 320, clientY: 200 }],
      changedTouches: [{ clientX: 320, clientY: 200 }]
    });
    fireEvent.touchMove(chart, {
      touches: [{ clientX: 326, clientY: 224 }],
      changedTouches: [{ clientX: 326, clientY: 224 }]
    });
    fireEvent.touchEnd(chart, {
      touches: [],
      changedTouches: [{ clientX: 326, clientY: 224 }]
    });

    expect(chartMocks.priceScaleSetAutoScale).not.toHaveBeenCalled();
    expect(chartMocks.priceScaleSetVisibleRange).not.toHaveBeenCalled();
  });

  test('freezes the autoscaled price range after initial candles so mobile vertical panning works immediately', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );

    await waitFor(() => expect(chartMocks.priceScaleSetVisibleRange).toHaveBeenCalledWith({ from: 95, to: 115 }));
  });

  test('keeps desktop mouse dragging able to pan the price range vertically', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    const chart = screen.getByLabelText('chart');
    chart.getBoundingClientRect = vi.fn(() => ({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 800,
      bottom: 160,
      width: 800,
      height: 160,
      toJSON: () => ({})
    }));
    chartMocks.priceScaleSetAutoScale.mockClear();
    chartMocks.priceScaleSetVisibleRange.mockClear();

    fireEvent.pointerDown(chart, { button: 0, clientX: 320, clientY: 200, pointerId: 1, pointerType: 'mouse' });
    fireEvent.pointerMove(chart, { clientX: 320, clientY: 224, pointerId: 1, pointerType: 'mouse' });
    fireEvent.pointerUp(chart, { pointerId: 1, pointerType: 'mouse' });

    expect(chartMocks.priceScaleSetAutoScale).toHaveBeenCalledWith(false);
    expect(chartMocks.priceScaleSetVisibleRange).toHaveBeenCalledWith({ from: 98, to: 118 });
  });

  test('zooms the price scale when the mouse wheel is over the right price axis', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    const chart = screen.getByLabelText('chart');
    chart.getBoundingClientRect = vi.fn(() => ({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 800,
      bottom: 160,
      width: 800,
      height: 160,
      toJSON: () => ({})
    }));
    chartMocks.priceScaleSetAutoScale.mockClear();
    chartMocks.priceScaleSetVisibleRange.mockClear();

    fireEvent.wheel(chart, { clientX: 760, clientY: 80, deltaY: -100 });

    expect(chartMocks.priceScaleSetAutoScale).toHaveBeenCalledWith(false);
    expect(chartMocks.priceScaleSetVisibleRange).toHaveBeenCalledTimes(1);
    const [nextRange] = chartMocks.priceScaleSetVisibleRange.mock.calls[0];
    expect(nextRange.from).toBeGreaterThan(95);
    expect(nextRange.to).toBeLessThan(115);

    chartMocks.priceScaleSetVisibleRange.mockClear();
    fireEvent.wheel(chart, { clientX: 320, clientY: 80, deltaY: -100 });

    expect(chartMocks.priceScaleSetVisibleRange).not.toHaveBeenCalled();
  });

  test('shows OHLC details for the candle under the crosshair', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.crosshairMoveHandler).toBeDefined());
    act(() => {
      chartMocks.crosshairMoveHandler?.({
        time: 1700000000,
        point: { x: 120, y: 80 },
        seriesData: new Map([
          [
            chartMocks.candlestickSeries,
            { time: 1700000000, open: 100, high: 110, low: 95, close: 105 }
          ]
        ])
      });
    });

    const details = await screen.findByLabelText(/selected candle details/i);
    expect(details).toHaveTextContent('O 100.00');
    expect(details).toHaveTextContent('H 110.00');
    expect(details).toHaveTextContent('L 95.000');
    expect(details).toHaveTextContent('C 105.00');
    expect(details).toHaveTextContent('Vol 12');
  });

  test('shows enabled indicator values for the candle under the crosshair', async () => {
    btcSnapshotCandles = [
      { time: 1700000000, open: 99, high: 101, low: 98, close: 100, volume: 1 },
      { time: 1700000060, open: 109, high: 111, low: 108, close: 110, volume: 2 },
      { time: 1700000120, open: 118, high: 125, low: 117, close: 120, volume: 3 }
    ];
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.crosshairMoveHandler).toBeDefined());
    await userEvent.click(screen.getByRole('button', { name: /indicators/i }));
    fireEvent.change(screen.getByRole('spinbutton', { name: /ema period/i }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: /sma period/i }), { target: { value: '3' } });
    await userEvent.click(screen.getByRole('checkbox', { name: /^ema$/i }));
    await userEvent.click(screen.getByRole('checkbox', { name: /^sma$/i }));

    await waitFor(() => expect(chartMocks.addSeries).toHaveBeenCalledWith('LineSeries', expect.objectContaining({ color: '#38bdf8' }), 0));
    await waitFor(() => expect(chartMocks.addSeries).toHaveBeenCalledWith('LineSeries', expect.objectContaining({ color: '#f59e0b' }), 0));

    act(() => {
      chartMocks.crosshairMoveHandler?.({
        time: 1700000120,
        point: { x: 140, y: 90 },
        seriesData: new Map([
          [
            chartMocks.candlestickSeries,
            { time: 1700000120, open: 118, high: 125, low: 117, close: 120 }
          ]
        ])
      });
    });

    const details = await screen.findByLabelText(/selected candle details/i);
    expect(details).toHaveTextContent('EMA 2 115.56');
    expect(details).toHaveTextContent('SMA 3 110.00');
    expect(screen.getByText('EMA 2 115.56')).toHaveStyle({ color: '#38bdf8' });
    expect(screen.getByText('SMA 3 110.00')).toHaveStyle({ color: '#f59e0b' });
  });

  test('draws the current position entry line on the chart', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createPriceLine).toHaveBeenCalled());
    expect(chartMocks.createPriceLine).toHaveBeenCalledWith(
      expect.objectContaining({
        price: 100000,
        title: 'Entry 100,000'
      })
    );
  });

  test('draws the current position liquidation line on the chart', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createPriceLine).toHaveBeenCalled());
    expect(chartMocks.createPriceLine).toHaveBeenCalledWith(
      expect.objectContaining({
        price: 92000,
        title: 'Liq 92,000'
      })
    );
  });

  test('resets the chart view from a right-click chart menu without animated scrolling', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    chartMocks.timeScaleResetTimeScale.mockClear();
    chartMocks.timeScaleScrollToRealTime.mockClear();
    fireEvent.contextMenu(screen.getByLabelText('chart'), { clientX: 120, clientY: 160 });
    await userEvent.click(screen.getByRole('menuitem', { name: /reset chart view/i }));

    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 });
    expect(chartMocks.timeScaleResetTimeScale).not.toHaveBeenCalled();
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
    expect(chartMocks.priceScaleSetAutoScale).toHaveBeenCalledWith(true);

    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    chartMocks.timeScaleGetVisibleLogicalRange.mockReturnValue({ from: -500, to: 500 });
    fireEvent.contextMenu(screen.getByLabelText('chart'), { clientX: 140, clientY: 180 });
    await userEvent.click(screen.getByRole('menuitem', { name: /reset chart view/i }));

    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 });
  });

  test('shows a tappable chart reset button on mobile', async () => {
    mockMatchMedia(true);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    chartMocks.timeScaleScrollToRealTime.mockClear();

    await userEvent.click(screen.getByRole('button', { name: /reset chart view/i }));

    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 });
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
  });

  test('changes the chart time axis timezone from the toolbar without recreating the chart', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.createChart).toHaveBeenCalledTimes(1));
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /chart timezone/i }), 'UTC');

    expect(screen.getByRole('combobox', { name: /chart timezone/i })).toHaveValue('UTC');
    expect(chartMocks.createChart).toHaveBeenCalledTimes(1);
    expect(chartMocks.applyOptions).toHaveBeenCalledWith(
      expect.objectContaining({
        localization: expect.objectContaining({ timeFormatter: expect.any(Function) }),
        timeScale: expect.objectContaining({ tickMarkFormatter: expect.any(Function) })
      })
    );
  });

  test('uses the selected chart timezone for account history timestamps', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('combobox', { name: /chart timezone/i })).toBeInTheDocument());
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /chart timezone/i }), 'Asia/Shanghai');
    await userEvent.click(screen.getByRole('button', { name: /trade history/i }));

    const expectedTime = new Intl.DateTimeFormat(undefined, {
      timeZone: 'Asia/Shanghai',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(1700000000000));
    expect(await screen.findByText(expectedTime)).toBeInTheDocument();
  });

  test('changes candle interval and reloads the chart snapshot', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /^5m$/i })).toBeInTheDocument());
    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    chartMocks.timeScaleScrollToRealTime.mockClear();
    await userEvent.click(screen.getByRole('button', { name: /^5m$/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) => {
          const value = String(url);
          return value.includes('/api/market/BTC/snapshot') && value.includes('interval=5m');
        })
      ).toBe(true)
    );
    await waitFor(() => expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 }));
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
  });

  test('does not offer unsupported 3m candle interval', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getByRole('button', { name: /^1m$/i })).toBeInTheDocument());

    expect(screen.queryByRole('button', { name: /^3m$/i })).not.toBeInTheDocument();
  });

  test('renders favorite markets in the top horizontal bar and switches markets from the list', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    expect(within(favorites).getByText('$2,050.0')).toBeInTheDocument();
    await userEvent.click(within(favorites).getByRole('button', { name: /ETH-USDC/i }));

    await waitFor(() => expect(screen.getByRole('combobox', { name: /market/i })).toHaveValue('ETH-USDC'));
  });

  test('sorts favorite markets alphabetically in the horizontal bar', async () => {
    favoriteSymbols = ['ETH', 'BTC'];
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    const labels = within(favorites)
      .getAllByRole('button')
      .map((button) => button.textContent || '')
      .filter((text) => text.includes('-USDC'));

    expect(labels).toEqual(['BTC-USDC$101,000', 'ETH-USDC$2,050.0']);
  });

  test('updates the selected favorite market price from realtime order book mid price', async () => {
    favoriteSymbols = ['BTC', 'ETH'];
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    expect(within(favorites).getByText('$101,000')).toBeInTheDocument();
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    MockWebSocket.instances[0].emit({
      channel: 'l2Book',
      data: {
        coin: 'BTC',
        time: 1700000001000,
        levels: [
          [{ px: '101100', sz: '2', n: 3 }],
          [{ px: '101200', sz: '1.5', n: 2 }]
        ]
      }
    });

    await waitFor(() => expect(within(favorites).getByText('$101,150')).toBeInTheDocument());
  });

  test('updates all favorite market prices from realtime all mids stream', async () => {
    favoriteSymbols = ['BTC', 'ETH'];
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    expect(within(favorites).getByText('$2,050.0')).toBeInTheDocument();
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    MockWebSocket.instances[0].emit({
      channel: 'allMids',
      data: { mids: { BTC: '101150', ETH: '2060' } }
    });

    await waitFor(() => expect(within(favorites).getByText('$2,060.0')).toBeInTheDocument());
  });

  test('subscribes to favorite dex all mids and updates unselected dex favorite prices', async () => {
    favoriteSymbols = ['xyz:DRAM', 'xyz:ARM'];
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    expect(within(favorites).getByText('$332.50')).toBeInTheDocument();
    expect(within(favorites).getByText('$72.200')).toBeInTheDocument();
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    const socket = MockWebSocket.instances[0];

    await waitFor(() =>
      expect(socket.sent).toEqual(
        expect.arrayContaining([{ method: 'subscribe', subscription: { type: 'allMids', dex: 'xyz' } }])
      )
    );

    socket.emit({
      channel: 'allMids',
      data: { mids: { 'xyz:ARM': '333.75', 'xyz:DRAM': '72.4' } }
    });

    await waitFor(() => expect(within(favorites).getByText('$333.75')).toBeInTheDocument());
  });

  test('toggles the favorite market bar between price and 24h percent change', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    expect(within(favorites).getByText('$2,050.0')).toBeInTheDocument();

    await userEvent.click(within(favorites).getByRole('button', { name: /24h%/i }));

    expect(within(favorites).getByText('+2.50%')).toBeInTheDocument();
  });

  test('recreates the chart when switching symbols so the price axis resets to the new market', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    chartMocks.timeScaleScrollToRealTime.mockClear();
    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    await userEvent.click(within(favorites).getByRole('button', { name: /ETH-USDC/i }));

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 2000, high: 2100, low: 1980, close: 2050 }])
    );
    expect(chartMocks.remove).toHaveBeenCalled();
    expect(chartMocks.createChart).toHaveBeenCalledTimes(2);
    expect(chartMocks.priceScaleSetAutoScale).toHaveBeenCalledWith(true);
    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: -112, to: 8 });
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
  });

  test('persists the current market when favorite toggle is clicked', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const favorites = await screen.findByRole('region', { name: /favorite markets/i });
    await userEvent.click(within(favorites).getByRole('button', { name: /favorite BTC-USDC/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith('/api/favorites/markets') &&
            init?.method === 'PUT' &&
            JSON.stringify(JSON.parse(String(init.body)).symbols) === JSON.stringify(['ETH', 'BTC'])
        )
      ).toBe(true)
    );
  });

  test('keeps chart zoom stable when realtime candles update the same symbol and interval', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledTimes(1));
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
    expect(chartMocks.timeScaleFitContent).not.toHaveBeenCalled();
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    chartMocks.timeScaleSetVisibleLogicalRange.mockClear();
    MockWebSocket.instances[0].emit({
      channel: 'candle',
      data: { t: 1700000060000, o: '105', h: '112', l: '101', c: '109', v: '4' }
    });

    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([
        { time: 1700000000, open: 100, high: 110, low: 95, close: 105 },
        { time: 1700000060, open: 105, high: 112, low: 101, close: 109 }
      ])
    );
    expect(chartMocks.timeScaleSetVisibleLogicalRange).not.toHaveBeenCalled();
    expect(chartMocks.timeScaleScrollToRealTime).not.toHaveBeenCalled();
    expect(chartMocks.timeScaleFitContent).not.toHaveBeenCalled();
  });

  test('loads earlier candles when the chart is dragged to the left history edge', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(chartMocks.visibleLogicalRangeHandler).toBeDefined());
    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([{ time: 1700000000, open: 100, high: 110, low: 95, close: 105 }])
    );
    chartMocks.visibleLogicalRangeHandler?.({ from: 4, to: 25 });

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) => {
          const value = String(url);
          return value.includes('/api/market/BTC/bars') && value.includes('to=1700000000');
        })
      ).toBe(true)
    );
    await waitFor(() =>
      expect(chartMocks.setData).toHaveBeenCalledWith([
        { time: 1699999940, open: 98, high: 102, low: 96, close: 100 },
        { time: 1700000000, open: 100, high: 110, low: 95, close: 105 }
      ])
    );
    expect(chartMocks.timeScaleSetVisibleLogicalRange).toHaveBeenCalledWith({ from: 1, to: 21 });
  });

  test('updates existing position leverage through a max-aware selector', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    expect(screen.queryByRole('spinbutton', { name: /BTC leverage/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^5x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /apply leverage/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith('/api/positions/BTC/leverage') &&
            init?.method === 'POST' &&
            JSON.parse(String(init.body)).leverage === 5
        )
      ).toBe(true)
    );
  });

  test('shows a top-left pending notification while leverage is adjusting', async () => {
    let resolveLeverage: (response: Response) => void = () => undefined;
    leveragePromise = new Promise((resolve) => {
      resolveLeverage = resolve;
    });
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^5x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /apply leverage/i }));

    expect(await screen.findByRole('status')).toHaveTextContent('Leverage pending');

    resolveLeverage(Response.json({ accepted: true, target_leverage: 5 }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Leverage adjusted'));
  });

  test('keeps the updated leverage response visible when the account refresh is still stale', async () => {
    leverageResponse = {
      accepted: true,
      position: {
        ...account.positions[0],
        leverage: 5,
        notional_usd: 5000,
        size: 0.05,
        margin_used: 1000,
        lifecycle_roi_basis_usd: 1000
      }
    };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^5x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /apply leverage/i }));

    await waitFor(() => expect(within(row).getByRole('button', { name: /BTC leverage 5x max 40x/i })).toBeInTheDocument());
    expect(within(row).getByText('$5,000.00')).toBeInTheDocument();
    expect(within(row).getByText('0.05 BTC')).toBeInTheDocument();
  });

  test('allows applying the current position leverage again to rebalance precision drift', async () => {
    const fetchMock = vi.mocked(fetch);
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /apply leverage/i }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith('/api/positions/BTC/leverage') &&
            init?.method === 'POST' &&
            JSON.parse(String(init.body)).leverage === 10
        )
      ).toBe(true)
    );
  });

  test('shows a visible message when an existing position leverage update is rejected', async () => {
    leverageResponse = {
      accepted: false,
      stage: 'target_below_min_trade_notional',
      message: 'Leverage rebalance skipped because the target notional is below the exchange minimum.'
    };
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    const row = screen.getByRole('row', { name: /BTC-USDC/i });
    await userEvent.click(within(row).getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^5x$/i }));
    await userEvent.click(screen.getByRole('button', { name: /apply leverage/i }));

    expect(await within(row).findByRole('alert')).toHaveTextContent(
      'Leverage rebalance skipped because the target notional is below the exchange minimum.'
    );
    await waitFor(() =>
      expect(
        screen
          .getAllByRole('alert')
          .some(
            (item) =>
              item.classList.contains('global-notification') &&
              item.textContent?.includes('Leverage rebalance skipped because the target notional is below the exchange minimum.')
          )
      ).toBe(true)
    );
  });

  test('renders the position leverage popover outside the account panel so it is not clipped', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /BTC leverage 10x max 40x/i }));

    const accountPanel = screen.getByRole('region', { name: /account/i });
    const popover = screen.getByRole('dialog', { name: /BTC leverage options/i });

    expect(popover).toHaveClass('floating-popover');
    expect(accountPanel).not.toContainElement(popover);
  });

  test('closes the position leverage popover when clicking outside it', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    expect(screen.getByRole('dialog', { name: /BTC leverage options/i })).toBeInTheDocument();

    await userEvent.click(screen.getByText('Private Hyperliquid'));

    await waitFor(() => expect(screen.queryByRole('dialog', { name: /BTC leverage options/i })).not.toBeInTheDocument());
  });

  test('renders the position leverage slider progress at the selected maximum', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(screen.getAllByText('BTC-USDC').length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole('button', { name: /BTC leverage 10x max 40x/i }));
    await userEvent.click(screen.getByRole('button', { name: /^40x$/i }));
    const slider = screen.getByLabelText(/BTC leverage slider/i);

    expect(slider).toHaveValue('40');
    expect(slider).toHaveStyle('--leverage-progress: 100%');
  });

  test('subscribes to live market data and renders the l2 order book', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    const socket = MockWebSocket.instances[0];
    await waitFor(() =>
      expect(socket.sent).toEqual(
        expect.arrayContaining([
          { method: 'subscribe', subscription: { type: 'allMids' } },
          { method: 'subscribe', subscription: { type: 'l2Book', coin: 'BTC' } },
          { method: 'subscribe', subscription: { type: 'trades', coin: 'BTC' } },
          { method: 'subscribe', subscription: { type: 'candle', coin: 'BTC', interval: '1m' } }
        ])
      )
    );

    socket.emit({
      channel: 'l2Book',
      data: {
        coin: 'BTC',
        time: 1700000001000,
        levels: [
          [
            { px: '100', sz: '2', n: 3 },
            { px: '99.5', sz: '0.021', n: 1 }
          ],
          [{ px: '101', sz: '1.5', n: 2 }]
        ]
      }
    });

    const book = await screen.findByRole('region', { name: /order book/i });
    expect(within(book).getByText('100.00')).toBeInTheDocument();
    expect(within(book).getByText('101.00')).toBeInTheDocument();
    expect(within(book).getAllByText('2.0000').length).toBeGreaterThan(0);
    expect(within(book).getAllByText('1.5000').length).toBeGreaterThan(0);
    expect(within(book).getByText('99.500')).toBeInTheDocument();
    expect(within(book).getByText('0.02100')).toBeInTheDocument();
    expect(within(book).getByText('Mid $100.50')).toBeInTheDocument();
  });

  test('updates the position mark price from realtime mid prices', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    const accountPanel = await screen.findByRole('region', { name: /account/i });
    const row = within(accountPanel).getByRole('row', { name: /BTC-USDC/i });
    expect(within(row).getByText('101,000')).toBeInTheDocument();
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));

    MockWebSocket.instances[0].emit({
      channel: 'l2Book',
      data: {
        coin: 'BTC',
        time: 1700000001000,
        levels: [
          [{ px: '102000', sz: '2', n: 3 }],
          [{ px: '102200', sz: '1.5', n: 2 }]
        ]
      }
    });

    await waitFor(() => expect(within(row).getByText('102,100')).toBeInTheDocument());
  });

  test('renders realtime trades below the order book', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    MockWebSocket.instances[0].emit({
      channel: 'trades',
      data: [
        {
          coin: 'BTC',
          side: 'B',
          px: '100.5',
          sz: '0.25',
          hash: '0xabc',
          time: 1700000002000
        }
      ]
    });

    const trades = await screen.findByRole('region', { name: /^trades$/i });
    expect(within(trades).getByText('100.50')).toBeInTheDocument();
    expect(within(trades).getByText('0.25')).toBeInTheDocument();
    const tradeTime = trades.querySelector('.trade-row span:nth-child(3)')?.textContent || '';
    expect(tradeTime).toMatch(/\d{1,2}:\d{2}/);
    expect(tradeTime).not.toMatch(/\d{1,2}\/\d{1,2}/);
  });

  test('keeps order book compact while allocating remaining market height to trades', () => {
    expect(styles).toContain('--book-panel-height: 284px;');
    expect(styles).toContain('grid-template-rows: var(--book-panel-height) minmax(0, 1fr);');
    expect(styles).not.toContain('grid-template-rows: minmax(0, 1fr) 138px;');
  });

  test('aligns order book and trades stream columns on the same horizontal grid', () => {
    expect(styles).toContain(
      '--market-stream-columns: minmax(78px, 1fr) minmax(64px, 0.82fr) minmax(116px, 1.25fr);'
    );
    expect(styles).toContain('.depth-row span:not(.depth-price)');
    expect(styles).not.toContain('.depth-row span:not(:first-child)');
    expect(styles.match(/grid-template-columns: var\(--market-stream-columns\);/g)).toHaveLength(3);
  });

  test('clips the trades list without showing its own scrollbar', () => {
    expect(styles).toMatch(/\.trades-list\s*\{[^}]*overflow: hidden;/s);
    expect(styles).not.toMatch(/\.trades-list\s*\{[^}]*overflow: auto;/s);
  });

  test('uses larger readable typography in market and account display areas without enlarging compact controls', () => {
    expect(styles).toContain('--display-text-size: 14px;');
    expect(styles).toContain('--display-heading-size: 15px;');
    expect(styles).toContain('--display-small-size: 13px;');
    expect(styles).toContain('font-size: var(--display-text-size);');
    expect(styles).toContain('font-size: var(--display-heading-size);');
    expect(styles).toContain('font-size: var(--display-small-size);');
    expect(styles).toContain('.favorite-metric-toggle button');
    expect(styles).toContain('font-size: 12px;');
    expect(styles).toContain('.order-ticket label');
    expect(styles).toContain('font-size: 13px;');
  });

  test('keeps trade timestamps on one line after display typography is enlarged', () => {
    expect(styles).toContain('grid-template-columns: var(--market-stream-columns);');
    expect(styles).toContain('white-space: nowrap;');
  });

  test('centers icons inside square icon-only buttons', () => {
    expect(styles).toContain('.icon-button,');
    expect(styles).toContain('.market-current-favorite,');
    expect(styles).toContain('.market-option-favorite,');
    expect(styles).toContain('.favorite-toggle,');
    expect(styles).toContain('.favorite-remove');
    expect(styles).toMatch(/\.icon-button,[\s\S]*\.favorite-remove\s*\{[\s\S]*display: inline-flex;[\s\S]*align-items: center;[\s\S]*justify-content: center;[\s\S]*line-height: 1;/);
    expect(styles).toMatch(/\.icon-button svg,[\s\S]*\.favorite-remove svg\s*\{[\s\S]*display: block;/);
  });

  test('makes mobile chart controls scrollable and pins the mobile primary tabs to the bottom', () => {
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.interval-tabs\s*\{[\s\S]*overflow-x: auto;/);
    expect(styles).toContain('.mobile-bottom-tabs');
    expect(styles).toContain('.mobile-trade-stack');
    expect(styles).toContain('.mobile-trade-top');
    expect(styles).toContain('.mobile-market-stream-stack');
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top\s*\{[\s\S]*grid-template-columns: minmax\(0, 0\.96fr\) minmax\(172px, 0\.84fr\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-market-stream-stack\s*\{[\s\S]*height: 100%;[\s\S]*grid-template-rows: minmax\(0, 1fr\) calc\(30px \+ 22px \+ \(3 \* 22px\)\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-market-stream-stack \.trades-list\s*\{[\s\S]*max-height: calc\(3 \* 22px\);[\s\S]*overflow: hidden;/);
    expect(styles).toContain('.mobile-chart-view');
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view\s*\{[\s\S]*grid-template-rows: max-content max-content max-content;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.market-control-band\s*\{[\s\S]*grid-template-rows: max-content minmax/);
    expect(styles).toMatch(/\.mobile-bottom-tabs\s*\{[^}]*position: sticky;[^}]*bottom: 0;/s);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.brand\s*\{[\s\S]*grid-row: 1;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mode\s*\{[\s\S]*grid-row: 1;/);
  });

  test('compacts mobile favorites order ticket and aligns mobile trades with order book columns', () => {
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorites-bar,[\s\S]*\.mobile-trade-stack \.favorites-bar\s*\{[\s\S]*grid-template-columns: auto minmax\(0, 1fr\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorite-bar-left,[\s\S]*\.mobile-trade-stack \.favorite-bar-left\s*\{[\s\S]*justify-content: flex-start;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorites-list\.horizontal \.favorite-row,[\s\S]*\.mobile-trade-stack \.favorites-list\.horizontal \.favorite-row\s*\{[\s\S]*min-width: 128px;[\s\S]*max-width: 148px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorites-list\.horizontal \.favorite-market,[\s\S]*\.mobile-trade-stack \.favorites-list\.horizontal \.favorite-market\s*\{[\s\S]*flex-direction: column;[\s\S]*align-items: flex-start;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorite-market span:first-child,[\s\S]*\.mobile-chart-view \.favorite-market span:last-child,[\s\S]*\.mobile-trade-stack \.favorite-market span:first-child,[\s\S]*\.mobile-trade-stack \.favorite-market span:last-child\s*\{[\s\S]*max-width: 100%;[\s\S]*white-space: nowrap;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-chart-view \.favorite-market span:last-child,[\s\S]*\.mobile-trade-stack \.favorite-market span:last-child\s*\{[\s\S]*font-size: 11px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.order-ticket\s*\{[\s\S]*height: auto;[\s\S]*overflow: visible;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top\s*\{[\s\S]*grid-template-columns: minmax\(0, 0\.96fr\) minmax\(172px, 0\.84fr\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-market-stream-stack\s*\{[^}]*--market-stream-columns: minmax\(50px, 0\.9fr\) minmax\(42px, 0\.74fr\) minmax\(60px, 1fr\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-market-stream-stack \.trades-head,[\s\S]*\.mobile-market-stream-stack \.trade-row\s*\{[\s\S]*grid-template-columns: var\(--market-stream-columns\);[\s\S]*gap: 4px;[\s\S]*padding: 0 4px;[\s\S]*font-size: 12px;/);
    expect(styles).not.toMatch(/\.mobile-market-stream-stack \.trades-head span:nth-child\(3\),[\s\S]*\.mobile-market-stream-stack \.trade-row span:nth-child\(3\)\s*\{[\s\S]*display: none;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.book-head,[\s\S]*\.mobile-trade-top \.depth-row\s*\{[\s\S]*gap: 4px;[\s\S]*padding: 0 4px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.book-head span:first-child,[\s\S]*\.mobile-trade-top \.depth-row \.depth-price,[\s\S]*\.mobile-trade-stack \.trades-head span:first-child,[\s\S]*\.mobile-trade-stack \.trade-row span:first-child\s*\{[\s\S]*text-align: left;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.book-head span:nth-child\(2\),[\s\S]*\.mobile-trade-top \.depth-row span:nth-of-type\(2\),[\s\S]*\.mobile-trade-stack \.trades-head span:nth-child\(2\),[\s\S]*\.mobile-trade-stack \.trade-row span:nth-child\(2\)\s*\{[\s\S]*text-align: center;/);
  });

  test('keeps mobile order ticket rows single-line and compact', () => {
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.order-ticket\s*\{[\s\S]*gap: 4px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.order-ticket \.panel-title\s*\{[\s\S]*height: 28px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.amount-control-head,[\s\S]*\.mobile-trade-top \.ticket-metric,[\s\S]*\.mobile-trade-top \.order-account-row,[\s\S]*\.mobile-trade-top \.order-percent-head,[\s\S]*\.mobile-trade-top \.order-percent-labels,[\s\S]*\.mobile-trade-top \.checkbox-line\s*\{[\s\S]*white-space: nowrap;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.order-account-row strong,[\s\S]*\.mobile-trade-top \.ticket-metric strong\s*\{[\s\S]*white-space: nowrap;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.mobile-trade-top \.order-account-summary\s*\{[\s\S]*gap: 3px;[\s\S]*padding: 5px 7px;/);
  });

  test('wraps mobile positions and compacts mobile history tables', () => {
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll\s*\{[\s\S]*overflow-x: hidden;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll table\s*\{[\s\S]*min-width: 0;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll thead\s*\{[\s\S]*display: none;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll tr\s*\{[\s\S]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll td::before\s*\{[\s\S]*content: attr\(data-label\);/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.positions-table-scroll \.position-margin-cell,[\s\S]*\.positions-table-scroll \.position-trade-actions,[\s\S]*\.positions-table-scroll \.position-tpsl-cell\s*\{[\s\S]*flex-wrap: wrap;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.history-table-scroll th,[\s\S]*\.history-table-scroll td\s*\{[\s\S]*padding: 3px 4px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.funding-history-table-scroll table\s*\{[\s\S]*min-width: 460px;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.trade-history-table-scroll th:nth-child\(n \+ 10\),[\s\S]*\.trade-history-table-scroll td:nth-child\(n \+ 10\)\s*\{[\s\S]*display: none;/);
    expect(styles).toMatch(/@media \(max-width: 980px\)[\s\S]*\.order-history-table-scroll th:nth-child\(8\),[\s\S]*\.order-history-table-scroll td:nth-child\(8\)\s*\{[\s\S]*display: none;/);
  });

  test('uses full-width custom geometry for the order size slider', () => {
    expect(styles).toMatch(/\.order-percent-control input\[type='range'\]\s*\{[\s\S]*--order-percent-progress: 0%;[\s\S]*--order-percent-thumb-size: 14px;[\s\S]*width: calc\(100% \+ var\(--order-percent-thumb-size\)\);[\s\S]*margin: 0 calc\(var\(--order-percent-thumb-size\) \/ -2\);[\s\S]*appearance: none;[\s\S]*background: transparent;/);
    expect(styles).toMatch(/\.order-percent-control input\[type='range'\]::-webkit-slider-runnable-track\s*\{[\s\S]*width: 100%;[\s\S]*height: var\(--order-percent-track-height\);[\s\S]*var\(--order-percent-progress\)/);
    expect(styles).toMatch(/\.order-percent-control input\[type='range'\]::-webkit-slider-thumb\s*\{[\s\S]*width: var\(--order-percent-thumb-size\);[\s\S]*height: var\(--order-percent-thumb-size\);[\s\S]*margin-top: calc\(\(var\(--order-percent-track-height\) - var\(--order-percent-thumb-size\)\) \/ 2\);/);
    expect(styles).toMatch(/\.order-percent-control input\[type='range'\]::-moz-range-track\s*\{[\s\S]*height: var\(--order-percent-track-height\);/);
    expect(styles).toMatch(/\.order-percent-control input\[type='range'\]::-moz-range-progress\s*\{[\s\S]*background: #22c55e;/);
  });

  test('keeps the top search bar close to account tabs without extra vertical whitespace', () => {
    expect(styles).toContain('gap: 1px;');
    expect(styles).toContain('grid-template-rows: max-content max-content minmax(0, 1fr);');
    expect(styles).toContain('--account-panel-height: clamp(240px, 30dvh, 340px);');
    expect(styles).not.toContain('grid-template-rows: auto auto var(--market-panel-height);');
    expect(styles).toContain('padding: 2px 12px 1px;');
    expect(styles).toContain('width: 28px;');
    expect(styles).toContain('height: 28px;');
    expect(styles).toContain('padding: 5px 10px;');
    expect(styles).toContain('padding: 1px 8px;');
    expect(styles).toContain('padding: 3px 8px;');
    expect(styles).toContain('padding: 5px 10px;');
    expect(styles).toContain('grid-template-rows: max-content minmax(0, var(--account-panel-height)) max-content;');
    expect(styles).not.toContain('--account-panel-height: clamp(104px, 12dvh, 132px);');
    expect(styles).not.toContain('--account-panel-height: clamp(180px, 22dvh, 260px);');
  });

  test('keeps isolated margin value and margin actions on one row', () => {
    expect(styles).toContain('.position-margin-cell');
    expect(styles).toContain('display: flex;');
    expect(styles).toContain('align-items: center;');
    expect(styles).toContain('white-space: nowrap;');
    expect(styles).not.toContain('.position-margin-cell {\n  display: grid;');
  });

  test('limits the compact order book to four visible levels per side', async () => {
    sessionStorage.setItem('webTradeAdminToken', 'secret-token');
    render(<App />);

    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
    const bids = Array.from({ length: 10 }, (_, index) => ({ px: String(100 - index), sz: '1', n: 1 }));
    const asks = Array.from({ length: 10 }, (_, index) => ({ px: String(101 + index), sz: '1', n: 1 }));
    MockWebSocket.instances[0].emit({
      channel: 'l2Book',
      data: {
        coin: 'BTC',
        time: 1700000001000,
        levels: [bids, asks]
      }
    });

    const book = await screen.findByRole('region', { name: /order book/i });
    expect(within(book).getByText('100.00')).toBeInTheDocument();
    expect(within(book).getByText('97.000')).toBeInTheDocument();
    expect(within(book).queryByText('96.000')).not.toBeInTheDocument();
    expect(within(book).getByText('101.00')).toBeInTheDocument();
    expect(within(book).getByText('104.00')).toBeInTheDocument();
    expect(within(book).queryByText('105.00')).not.toBeInTheDocument();
  });
});
