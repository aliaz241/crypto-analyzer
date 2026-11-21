# Forex Module — FOREX.md

An add-on on top of the same crypto pipeline: the same scoring, the same
final stack (accepted trailing + regime/MTF + EV gate), the same decision
layer. Crypto behavior has not changed at all (regression and reproduction
of the matrix numbers are in the "Verification" section below).

## Data source and feed verification

Two sources, both free and with no API key, both verified from this very
environment (2026-11-07):

### Primary source: Yahoo Finance (chart API)
`https://query1.finance.yahoo.com/v8/finance/chart/{YAHOO_SYMBOL}`
(symbol mapping in `config.FOREX_YAHOO_SYMBOLS`; gold = the GC=F futures
contract). Actual depth delivered: 1h up to 1 year (≈6,200 hours → exactly
1500 4h candles) and 1d up to 5 years (exactly 1000 daily candles after
removing artificial Sundays) — so both full-matrix depth targets were met.
Live sample: EURUSD spot price 1.1143 with the last hourly candle
2026-11-06 22:00 UTC at 1.11433.

Yahoo's constraints (honest):
- **Volume for FX pairs is zero** (the feed itself returns zero/None). So
  `obv_trend` becomes neutral and the volume component of the score always
  gets the neutral score (half the cap) — volume adds no information to the
  forex signal. For XAUUSD (GC=F) there is real contract volume.
- Daily candles come back labeled with ET midnight; the label was normalized
  to the UTC calendar date of the same day. Over 12 shared days with
  Dukascopy candles, the mean absolute OHLC difference for this very
  alignment was 0.0011 — the remaining difference is of the trading-day
  convention kind (17:00 New York close vs. the UTC day) and feed
  differences, not a labeling error.
- **Daily "Sunday" candles are artificial** (all four OHLC values equal to
  Friday's close; the market is closed) and are removed. In the hourly
  series only the Sunday 23:00 opening hour is real and stays; Sunday-night
  4h candles are the week's actual reopening trades (reported separately in
  the matrix review).

### Second/backup source: Dukascopy (hourly .bi5 tick files)
`https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`
(the month in the URL is zero-based). The tick format was read from real
data and confirmed: each tick is 20 bytes big-endian — millisecond offset
from the start of the hour, integer ask, integer bid, ask volume, bid
volume. Point: 1e5 for most pairs, 1e3 for JPY and XAUUSD
(`config.FOREX_POINT`). Sample: EURUSD, hour 10:00 UTC on 2026-11-05: 3469
ticks, first bid 1.12314 and ask 1.12317.
Candles are built from the **bid** price; volume = the sum of tick volumes
("tick activity", not real spot market volume, which does not exist).
**Important constraint:** this feed is extremely unstable from this
environment — long periods of 503/timeout/disconnection, with scattered
short windows of success (a patient multi-hour fetch campaign managed to
write only 300 hours of EURUSD to disk: 2026-10-21 to 2026-11-06; cache in
`data/forex_cache/`). That is why its role is backup: if Yahoo errors out,
`get_candles` falls back to the Dukascopy cache/fetch and returns whatever
really exists — depth below the target is never fabricated.

- Alternative examined and rejected: Stooq daily CSV — it returns a
  JavaScript challenge page, not a CSV; unusable.
- Day boundaries on both paths are UTC (a documented simplification; the
  common forex trading day is 17:00 New York). No weekend candle is ever
  fabricated.

## Symbols

EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, EURGBP and **XAUUSD, which
is gold — a commodity representative, not a currency pair** — and it appears
with that label in all outputs. (`config.FOREX_SYMBOLS`)

## "Current" price and paper

There is no real spot price in either feed: `latest_price` first takes the
close of the last 1h candle from Yahoo (and on error goes to the Dukascopy
cache) — **delayed** (usually until the current hour closes; variable).
Forex paper trades are allowed **manually only**, with this same price and
with a "delayed price" label in the position display; automatic mode stays
crypto (and per the EV rules, with no n≥30 and positive expectancy, no forex
BUY is fired either).

## Cost model (different from crypto)

The forex backtest does not use crypto's percentage fee: the cost of each
trade in price units = (round-trip spread + 0.2 pip slippage) × pip size,
and it is subtracted from the price profit and divided by the risk
(`backtest/forex_backtest.forex_trade_r`). Spreads
(`config.FOREX_SPREAD_PIPS`): EURUSD 1.0, GBPUSD 1.5, USDJPY 1.2, USDCHF
1.8, AUDUSD 1.4, USDCAD 1.8, EURGBP 1.5 pips; XAUUSD 25 cents (0.25 price
units). Pip: 0.0001, and 0.01 for JPY and 0.1 for gold.
**Overnight swap/rollover is not counted** (an explicit constraint; holding
up to 40 candles). Implemented as a wrapper over the engine's stack output
so the crypto cost formula stays untouched; the engine's provisional crypto
R is repriced in forex pricing with the spread cost.

## Live paths

- `analyzer.fetch_market(symbol, timeframe, market=None)`: a forex symbol →
  the forex feed; anything else → Binance, as before. `run_full_analysis`
  with `data_source="forex"|"market"` takes the same path.
- `scanner.scan_forex_market()`: a scan of 8 symbols × [4h, 1d] with the
  same score_candidate and the regime/MTF defaults and the EV layer. A
  combination with no entry in the EV cache input gets NO_TRADE with an
  insufficient-sample reason.
- CLI: menu option 5 and `--scan-forex`; single-symbol analysis also accepts
  forex symbols.
- Telegram: the "💱 اسکن فارکس" (Forex scan) button with the same
  decision/EV display; single-coin analysis for forex symbols comes through
  the forex path. (The bot was not switched on in this build.)

## The forex matrix and the verdict on the combinations

The full matrix of 8 symbols × (4h, 1d) with the final stack and spread cost
was built in `matrix_forex/` (Yahoo; 4h exactly 1500 candles and 178–303
evaluations, 1d exactly 1000 candles and 178 evaluations; zero evaluation
errors) and its @60 out-of-sample threshold was merged into the EV cache
with source="matrix_forex" (16 entries; total cache entries 56).

@60 out-of-sample verdict — the full reality:
- **No combination reached a meaningful sample (n≥10)**; the largest sample
  is XAUUSD 1d with n=7. So **all 16 combinations are NO_TRADE with an
  insufficient-sample reason, and the number of WAIT-or-better combinations
  is: zero.**
- Two small positives: USDJPY 1d (n=6, 83.3% win rate, +0.663R, total
  +3.98R) and XAUUSD 1d (n=7, 71.4% win rate, +0.219R, total +1.54R). Both
  are below the noise threshold; especially USDJPY, whose profit comes from
  5 wins out of 6 trades and whose in-sample sample was also only 6 trades —
  interesting to keep an eye on, not to draw conclusions from.
- The negatives: GBPUSD 1d (n=6, −0.179R), EURGBP 1d (n=4, −0.509R), AUDUSD
  4h (n=1, −1.043R), USDCAD 1d (n=1, −1.009R).
- **9 of the 16 combinations did not produce even one trade out-of-sample**
  — the stack (gate + regime/MTF) is nearly silent on forex, just like the
  crypto result after the improvement program.
- Weekend check: there is no Saturday entry in the entire matrix; there are
  only 2 entries labeled Sunday (USDCHF 4h, hour 20:00 UTC), which are real
  Sunday-night market reopening trades, not fake candles.

Honest summary: the forex module is engineering-complete and on the same
track as crypto, but the current data shows no reliable edge; exactly the
result the EV layer was designed for: silence.

## Verification (this build)

- Snapshot regression on the frozen baseline DFs: zero changes/deletions in
  the previous fields (only the known additive fields of the earlier layers)
  and the chart PNG hash byte-for-byte identical (62286dba…).
- Crypto number reproduction: `run_system_backtest` on the frozen
  2026-11-06 candles for BTC 4h at threshold 60: out-of-sample 25 trades,
  16.0% win rate, expectancy −0.627R, total −15.67R — exactly the original
  matrix numbers.
- Tests: the previous suites (paper 26, ported 26, EV 45, trailing 12,
  improvement 21, offline demo) + the forex tests, 30 checks (synthetic bi5
  parsing, candles, resample and weekend gaps, cost model with a hand-checked
  answer, mock routing, scan over a synthetic cache, forex paper, Yahoo
  parsing with mocks).
- Engine equivalence on forex: the matrix was built with the same
  stack_engine as the improvement program; Saturday entries zero, and Sunday
  entries only the 20:00 UTC reopening (above).
