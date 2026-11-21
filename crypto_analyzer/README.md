# Cryptocurrency Technical Analysis Tool — Version 2

A command-line (CLI) tool for automated technical analysis of crypto pairs:
indicators, candlestick and chart patterns, support/resistance levels,
stop-loss suggestions, and most importantly — the **historical success rate**
of each pattern, based on backtesting over real past data.

## 🆕 What does version 2 add?

- **Binance data source (default):** native candles for each timeframe
  straight from Binance's public endpoint — **no API key needed**. Long
  history (1H: 1500 candles, 4H: 1500, 1D: 1000, 1W: 400) and real exchange
  trading volume. The version 1 CoinGecko path is still available as a
  fallback with `--source coingecko`.
- **Market scanner and today's suggestion:** 20 coins × 2 timeframes (4h and
  1d) are scanned and ranked with a transparent score from 0 to 100 (trend,
  momentum, pattern, reward-to-risk, volume), and the best candidate is
  suggested with its reasoning, stop loss and target.
- **Chart image (PNG):** a candlestick chart with EMA20/50, Bollinger Bands,
  support/resistance lines and stop losses is saved in the `charts/` folder.
- **Persian numbered menu:** running it with no arguments brings up a simple
  menu.
- **Fixes for the version 1 audit findings:** chart patterns
  (double/head-and-shoulders/triangle/wedge) only count as "completed" after
  a **break of the neckline/boundary**; support/resistance is built only from
  the last 250 candles; MACD is fully computed on all timeframes (since the
  history is no longer short); Binance errors are returned with the real
  message and without pointless retries.

## ⚠️ Read this first

- This tool **does not predict prices**. The percentages it shows ("such-and
  such pattern has historically succeeded X% of the time") are statistics
  from the past, not a guarantee of the future. The market can change its
  behavior.
- When you see `⚠️ نمونه کم` next to a pattern, it means the number of
  historical occurrences was so small (fewer than 10) that the percentage
  figure is effectively meaningless — do not trust it.
- This tool is not a financial advisor and should not replace your own
  decision-making.

## Features

| Section | Includes |
|---|---|
| **Indicators** | SMA, EMA, MACD, ADX (trend) \| RSI, Stochastic, CCI, Williams %R (momentum) \| Bollinger, ATR, Keltner (volatility) \| OBV (volume) — all shown in the CLI report |
| **Levels** | Fibonacci Retracement/Extension, Pivot Points, automatic Support/Resistance (based on swings) |
| **Candlestick patterns** | 21 patterns — Doji, Hammer, Engulfing, Harami, Morning/Evening Star, Three White Soldiers, etc. |
| **Chart patterns** | Double Top/Bottom, Head & Shoulders (and inverse), Triangle (ascending/descending/symmetrical), Wedge (rising/falling) |
| **Risk** | Stop loss based on ATR and based on price structure (last high/low) |
| **Backtest** | For each pattern, computes the historical success rate over all available data |
| **Data source** | Binance Public API (version 2 default, no API key) + CoinGecko (fallback, with `--source coingecko`) |
| **Market scanner** | Scans 20 coins on 2 timeframes, scores 0 to 100 and suggests the best candidate (`--scan`) |
| **Chart image** | PNG output of the chart with indicators and levels (`--chart` or from the menu) |

## Installation (Termux / Android)

```bash
pkg update && pkg upgrade
pkg install python
pip install -r requirements.txt
```

If installing `pandas`/`numpy` with pip is slow or gives a compile error, try this:
```bash
pkg install python-numpy python-pandas
```

## Installation (Linux / Windows / Mac)

```bash
pip install -r requirements.txt
```

## Usage

### Numbered menu (no arguments):
```bash
python main.py
```
The menu has three options: 1) Analyze one coin (asks for the symbol and
timeframe; defaults are BTC and 4h — the report is printed and the chart
image is also saved) 2) Market scan and today's suggestion 3) Exit.

### With direct arguments:
```bash
python main.py --symbol btc --timeframe 4h
python main.py --symbol btc --timeframe 1d --chart
python main.py --scan
python main.py -s doge -t 1h
python main.py --symbol bitcoin --vs usd --timeframe 4h --source coingecko
```

- The default data source is **Binance** (`--source binance`); pairs are USDT
  and the symbol can be written in any form: `btc`, `BTC`, `BTCUSDT`,
  `BTC/USDT`.
- `--chart` saves the chart image in the `charts/` folder and prints its
  path.
- `--scan` runs the same market scan without the menu: a table of the top 5
  candidates, the details of the first suggestion and its chart. A full scan
  of 40 combinations takes about 1.5 to 2 minutes.

Supported timeframes: `1h`, `4h`, `1d`, `1w`

### How is the scanner score computed?
| Component | Cap | Logic |
|---|---|---|
| Trend | 25 | Alignment of price with EMA50/EMA200, scaled by ADX — its sign determines the direction (long/short) |
| Momentum | 20 | MACD + RSI pointing the same way in the healthy zone for that direction (extremes are penalized) |
| Pattern | 25 | Backtest success rate of the best recent confirmed pattern, on the same coin/timeframe |
| Reward-to-risk | 20 | Distance to the first opposing level divided by distance to the stop loss |
| Volume | 10 | OBV trend pointing the same way as the direction |

All weights and constants are in `config.py` (the Scanner section) and can be
changed.

### Offline test (no internet, no hitting CoinGecko's rate limit):
To quickly see what the report looks like, or to test after a code change,
use synthetic data instead of fetching real data:
```bash
python tests/run_offline_demo.py
```

## Project structure

```
crypto_analyzer/
├── config.py                  ← all settings are hardcoded here (not .env)
├── main.py                    ← CLI entry point + numbered menu
├── scanner.py                 ← multi-coin scanning and scoring/suggestion (version 2)
├── charting.py                ← drawing the PNG chart image (version 2)
├── data/
│   ├── binance_client.py      ← fetching candles from Binance (version 2 default source)
│   └── coingecko_client.py    ← fetching candles from CoinGecko (fallback source)
├── indicators/
│   ├── trend.py               ← SMA, EMA, MACD, ADX
│   ├── momentum.py            ← RSI, Stochastic, CCI, Williams %R
│   ├── volatility.py          ← Bollinger, ATR, Keltner, OBV
│   └── levels.py              ← Fibonacci, Pivot Points, S/R
├── patterns/
│   ├── candlestick.py         ← 21 candlestick patterns
│   └── chart_patterns.py      ← geometric chart patterns
├── risk/
│   └── stop_loss.py           ← stop-loss calculation
├── backtest/
│   └── engine.py              ← historical success-rate calculation
├── analysis/
│   └── analyzer.py            ← coordinator of the whole process
└── tests/
    ├── sample_data.py         ← synthetic data generation for tests
    └── run_offline_demo.py    ← full pipeline run with no internet
```

The architecture is **modular**: the analysis core (`analysis/analyzer.py`)
is completely separate from the user interface (`main.py`). That means when
you want to build a Telegram bot on the same logic, you only need to call
`run_full_analysis()` from a new file (e.g. `telegram_bot.py`) — there is no
need to change the analysis core.

## How the backtest works (important note)

For each pattern, when it occurred in the past:
1. A "target" (based on ATR or the pattern's neckline) and an "invalidation
   point" are determined.
2. It moves forward up to 10 candles later (for a candlestick pattern) or 20
   candles later (for a chart pattern) to see whether price reached the
   target first (**success**), touched the invalidation point first
   (**failure**), or neither happened (**timeout**).
3. Success rate = (number of successes) ÷ (total occurrences) × 100

This is exactly what was explained in the initial conversation: "out of the
10 times the pattern was seen, it worked out 4 times."

## Known limitations (honest)

- **The default path (Binance) does not have the version 1 limitations:**
  candles are native to the timeframe (no resampling), the 1h timeframe works
  correctly, history runs to hundreds or thousands of candles, and volume is
  real. Only USDT pairs are supported, and the symbol must be listed on
  Binance.
- **The limitations below apply only to the CoinGecko fallback path
  (`--source coingecko`):**
  CoinGecko's raw candle resolution is determined automatically by `days` and
  cannot be overridden (on the free plan, sending the `interval` parameter
  gives a 400 error): 1–2 days → 30-minute, 3–30 days → 4-hour, over 30 days
  → **4-day** (not daily). The history cap on the free plan is 365 days, and
  the `1d` timeframe on this path yields at most about 30 real daily candles
  — which is why, on the CoinGecko path, the backtest sample size often stays
  small, especially for chart patterns, and the "low sample" warning shows up
  a lot. (On the Binance path too, low-occurrence patterns may have small
  samples; take the warning seriously.)
- **A full market scan is time-consuming** (40 combinations, each with a full
  backtest over 1000 to 1500 candles; about 1.5 to 2 minutes). The list of
  coins and timeframes scanned can be trimmed or extended in `config.py`.
- **Chart patterns are detected with a windowed scan**, which can sometimes
  identify a real pattern several times (in overlapping windows) or with a
  slight delay. The code tries to remove this overlap, but it is not perfect.
- **For backtesting chart patterns, the simulation start point is
  deliberately placed a few candles after the pattern completion point** (by
  `SWING_LOOKBACK_CANDLES`), because detecting a high/low itself needs a few
  following candles to be confirmed — without this adjustment, the backtest
  would be slightly more optimistic than reality (a kind of future
  information leakage).
- **Volume on the CoinGecko path is fetched from a separate endpoint
  (`market_chart`) and, for each candle, the volume of all points inside that
  candle's time range is summed** — more accurate than single-point matching,
  but approximate. On the Binance path, volume is the real volume of the
  candle itself and this limitation does not exist.

## Suggested next step

Once you are happy with the CLI results, the same logic
(`analysis/analyzer.py`) can be connected to a Telegram bot with a few lines
of code (receive the user's message → call `run_full_analysis()` → send the
report as a message). Let me know whenever you want us to go for it.
