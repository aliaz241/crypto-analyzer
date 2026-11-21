"""
config.py
Project settings — all values are hardcoded directly here (no .env file needed)
"""

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = ""  # optional: your own CoinGecko API key
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_DELAY_SECONDS = 2

# The CoinGecko free plan's request rate is about 10-30 requests per minute.
# This delay between requests keeps us from hitting the limit.
MIN_SECONDS_BETWEEN_REQUESTS = 2.0

# ---------------------------------------------------------------------------
# Default analysis settings
# ---------------------------------------------------------------------------
DEFAULT_VS_CURRENCY = "usd"

# Supported time ranges and the number of days to request from CoinGecko.
#
# Important note: on the free plan, the /coins/{id}/ohlc endpoint determines the raw candle
# resolution itself from the days value (it cannot be overridden, because the interval
# parameter returns a 400 error on the free plan):
#   1-2 days   -> 30-minute candles
#   3-30 days  -> 4-hour candles
#   31+ days   -> 4-day candles   (not daily!)
# The free plan also covers only the past 365 days.
# Therefore, days is chosen for each timeframe so that an appropriate raw resolution
# (finer than or equal to the target timeframe) is returned, making resampling meaningful:
TIMEFRAME_DAYS_MAP = {
    "1h": 2,     # Raw: 30-minute -> resampled to 1-hour (~48 candles)
    "4h": 30,    # Raw: 4-hour (native, no resampling needed) (~180 candles)
    "1d": 30,    # Raw: 4-hour -> resampled to daily (~30 candles)
    "1w": 365,   # Raw: 4-day (maximum allowed history) -> resampled to weekly (~52 candles)
}

# Minimum number of candles needed only for the indicators to warm up and for something
# to be available to display. A small sample for backtesting is warned about separately
# with the MIN_RELIABLE_SAMPLE_SIZE flag (below), rather than rejecting
# the entire request.
MIN_CANDLES_REQUIRED = 30

# ---------------------------------------------------------------------------
# Pattern detection sensitivity
# ---------------------------------------------------------------------------
# For detecting peaks/troughs (swing high/low) in chart patterns
SWING_LOOKBACK_CANDLES = 3

# Maximum distance (as a percentage) between two levels for them to be considered "equal"
# (for example, in a Double Top the two peaks must be at roughly the same height)
LEVEL_EQUALITY_TOLERANCE_PCT = 1.5

# ---------------------------------------------------------------------------
# Risk management defaults
# ---------------------------------------------------------------------------
ATR_PERIOD = 14
ATR_STOP_LOSS_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# Backtest settings
# ---------------------------------------------------------------------------
# After a pattern is identified, how many subsequent candles to check for reaching the target
BACKTEST_FORWARD_WINDOW_CANDLES = 10

# The price target, expressed as a multiple of ATR from the pattern identification point
BACKTEST_TARGET_ATR_MULTIPLIER = 2.0

# Fewer historical occurrences than this = a very small statistical sample; the success rate is unreliable
MIN_RELIABLE_SAMPLE_SIZE = 10

# ---------------------------------------------------------------------------
# Binance settings (the primary data source since version 2)
# ---------------------------------------------------------------------------
# Binance public market endpoint — no API key is required for candle data.
# data-api.binance.vision is the dedicated public-data domain and is tried first;
# api.binance.com returns a 451 error in some regions.
BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]
BINANCE_REQUEST_TIMEOUT_SECONDS = 15
BINANCE_MAX_RETRIES = 3            # Only for 429/5xx and network errors — no retry on a 400
BINANCE_PAGE_LIMIT = 1000          # Maximum candles per /klines request

# Default number of candles fetched from Binance for each timeframe (native candles, no resampling)
BINANCE_CANDLES_PER_TIMEFRAME = {
    "1h": 1500,
    "4h": 1500,
    "1d": 1000,
    "1w": 400,
}

# ---------------------------------------------------------------------------
# Support/Resistance recency (fix for audit issue no. 4)
# ---------------------------------------------------------------------------
# Only swings from this many recent candles participate in support/resistance clustering, so that
# on long Binance histories, stale levels far from the current price do not spoil the result.
RECENT_SR_CANDLES = 250

# ---------------------------------------------------------------------------
# Scanner settings (multi-symbol scan and daily suggestion)
# ---------------------------------------------------------------------------
SCAN_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK",
                "AVAX", "NEAR", "ARB", "INJ", "SUI", "PEPE", "WIF", "TAO",
                "FET", "ONDO", "ENA", "JUP"]
SCAN_TIMEFRAMES = ["4h", "1d"]
SCAN_TOP_N = 5

# Cap for each scanner component's score (total = 100) — all weights are here for transparency
SCORE_TREND_MAX = 25.0
SCORE_MOMENTUM_MAX = 20.0
SCORE_PATTERN_MAX = 25.0
SCORE_RR_MAX = 20.0
SCORE_VOLUME_MAX = 10.0

# If the absolute trend score is below this, the trend counts as "neutral" and the direction is taken from MACD
SCORE_TREND_WASH_THRESHOLD = 6.0
# Pattern success-rate-to-score mapping: 30% -> zero points, 70% -> the maximum score (linear)
SCORE_PATTERN_RATE_FLOOR = 30.0
SCORE_PATTERN_RATE_CEIL = 70.0
# When the opposing level (target) or stop loss is unavailable, this fixed score is given for the risk/reward component
SCORE_RR_MISSING = 8.0

# ---------------------------------------------------------------------------
# System backtest settings (system backtest — backtest/system_backtest.py module)
# ---------------------------------------------------------------------------
# These constants are read only by the new system-backtest module and have no effect on any
# existing calculation.
SYSBT_MIN_HISTORY = 250        # Minimum historical candles before the first evaluation (indicator warm-up)
SYSBT_STEP = 4                 # Score once every this many candles (to reduce computation cost)
SYSBT_MAX_HOLD_CANDLES = 40    # Maximum holding period for a simulated trade (after that, exit at market price)
SYSBT_FEE_PCT = 0.10           # Round-trip fee, as a percentage of the trade value
SYSBT_SLIPPAGE_PCT = 0.05      # Slippage per side, as a percentage
SYSBT_INSAMPLE_FRAC = 0.6      # Share of the first part of the data for in-sample (the rest is out-of-sample)

# ---------------------------------------------------------------------------
# MTF confluence (multi-timeframe alignment — step 2, fully optional/additive)
# ---------------------------------------------------------------------------
# It takes effect only when higher_tf_direction is passed to score_candidate; the
# default path (None) is exactly the previous behavior.
MTF_CONFLICT_FACTOR = 0.6      # Penalty factor applied to the total score when the higher-timeframe direction conflicts

# ---------------------------------------------------------------------------
# Regime filter (market-regime filter — step 3, fully optional/additive)
# ---------------------------------------------------------------------------
# It takes effect only when apply_regime=True is passed to score_candidate.
REGIME_ADX_TREND_MIN = 25.0               # ADX above this = a trending market (subject to the EMA200 slope)
REGIME_BBW_WINDOW = 100                   # Candle window for the Bollinger Band width percentile
REGIME_BBW_VOLATILE_PERCENTILE = 80.0     # Band-width percentile above this = extreme volatility (volatile)
REGIME_EMA200_SLOPE_LOOKBACK = 20         # The EMA200 slope is measured over this many candles
REGIME_EMA200_SLOPE_MIN_PCT = 0.1         # Minimum absolute slope (percent) to count as a "trend"
REGIME_RANGE_TREND_SCALE = 0.6            # In a range: multiplier for the trend component
REGIME_RANGE_MOMENTUM_SCALE = 1.15        # In a range: multiplier for the momentum component (capped by its own maximum)
REGIME_VOLATILE_PATTERN_SCALE = 0.7       # In extreme volatility: multiplier for the pattern component

# ---------------------------------------------------------------------------
# Structure component (divergence and market structure — step 4, active only with extras)
# ---------------------------------------------------------------------------
STRUCTURE_MAX = 10.0           # Maximum score for the structure component; only when extras is passed
STRUCTURE_RSI_DIV_POINTS = 4.0
STRUCTURE_MACD_DIV_POINTS = 3.0
STRUCTURE_EVENT_POINTS = 3.0
STRUCTURE_RECENT_CANDLES = 100  # Divergence is valid only when the second swing is within this recent window

# ---------------------------------------------------------------------------
# Position sizing (position size — step 6, only an additive key in the scanner output)
# ---------------------------------------------------------------------------
# ⚠️ ACCOUNT_SIZE_USDT is only a placeholder for displaying the formula, not the user's
# real balance; it must be replaced with real capital before any real use.
ACCOUNT_SIZE_USDT = 1000.0
RISK_PER_TRADE_PCT = 1.0       # Allowed risk percentage per trade from the placeholder account

# ---------------------------------------------------------------------------
# Chart settings (chart image)
# ---------------------------------------------------------------------------
CHART_OUTPUT_DIR = "charts"
CHART_CANDLES = 120

# ---------------------------------------------------------------------------
# Paper trading (paper/hypothetical exchange — paper_trading.py module, fully additive)
# ---------------------------------------------------------------------------
# These constants are read only by the new paper-trading module and have no effect on any
# existing calculation (indicators/patterns/scanner/backtest).
PAPER_START_BALANCE_USDT = 10_000.0   # Starting virtual balance for each paper account
PAPER_FEE_PCT = 0.10                  # Fee per side (percentage of the position amount) — the same model as the backtest
PAPER_PRICE_CACHE_SECONDS = 3         # Binance spot-price cache
# Default settings for auto mode (testing the main bot on a paper account)
PAPER_AUTO_TIMEFRAME = "4h"
PAPER_AUTO_THRESHOLD = 60.0           # Minimum scanner score for automatic opening
PAPER_AUTO_RISK_PCT = 1.0             # Risk percentage of equity in each automatic trade
PAPER_AUTO_MAX_OPEN = 3               # Maximum simultaneous open positions in auto mode
PAPER_AUTO_COOLDOWN_CANDLES = 12      # After a position closes, the same symbol is not reopened for this many candles
PAPER_AUTO_MIN_AMOUNT_USDT = 10.0     # Minimum automatic position amount (smaller amounts are rejected)

# ---------------------------------------------------------------------------
# EV decision layer (EV decision layer — analysis/ev.py module, fully additive)
# ---------------------------------------------------------------------------
# These constants are read only by the new EV layer and have no effect on any existing
# calculation (scanner score, backtest, manual paper trading). The EV of each symbol/
# timeframe combination is computed only from the "out-of-sample" statistics of the system
# backtest of that same combination and at the EV_BACKTEST_THRESHOLD threshold; in-sample
# statistics never reach a live decision.
EV_BACKTEST_THRESHOLD = 60            # Score threshold underlying the EV statistics (the same threshold as the 40-item matrix)
EV_SAMPLE_INSUFFICIENT_MAX = 10       # n below this = insufficient sample → NO_TRADE
EV_SAMPLE_OK_MIN = 30                 # n at least this = a "reliable" sample (between the two = low confidence)
EV_STRONG_SCORE = 80.0                # Score required for a strong decision (STRONG_*)
EV_BUY_SCORE = 60.0                   # Score required for a buy/sell decision (BUY/SELL)
EV_CACHE_FILENAME = "ev_cache.json"   # EV cache file next to the package (data derived from backtests; can be reset)

# ---------------------------------------------------------------------------
# Trailing stop / profit lock (trailing stop — system backtest, additive)
# ---------------------------------------------------------------------------
# Backtest trailing parameters are defined in R (the trade's risk unit = |entry − initial
# stop loss|) — not as a price percentage — because R is the common unit across all
# trades in this engine. ⚠️ The semantics of paper_trading.py differ: there,
# trailing_pct is a percentage of "price" and is active from the moment of entry (with no
# activation threshold). The two were deliberately not unified; the documented deviation is in
# TRAILING_AB_FA.md and the system_backtest module docstring.
SYSBT_TRAILING_GRID = (   # Small, pre-announced A/B test grid (activation × distance, in R)
    {"activation_r": 0.5, "trail_r": 0.5},
    {"activation_r": 0.5, "trail_r": 1.0},
    {"activation_r": 1.0, "trail_r": 0.5},
    {"activation_r": 1.0, "trail_r": 1.0},
)
# A/B test result (TRAILING_AB_FA.md): the selected in-sample variant at both
# thresholds was {activation 1.0R, distance 0.5R}, and out-of-sample, aggregated across 5 symbols,
# it improved on the baseline (threshold 60: −0.057R versus −0.115R; XRP stayed positive)
# → trailing was "adopted" and this constant was set. ⚠️ Interpretation caution: adoption
# means "as a usable default option in backtests and the basis for
# paper alignment" — the mathematical expectancy is still negative, and this constant alone
# does not automatically activate any path; run_system_backtest takes effect only when trailing
# is passed explicitly or read from this constant, and with
# trailing=None the previous behavior is always reproduced exactly.
SYSBT_TRAILING_ADOPTED = {"activation_r": 1.0, "trail_r": 0.5}   # A/B selection: activation 1.0R, distance 0.5R

# ---------------------------------------------------------------------------
# Improvement program (improvement program — IMPROVEMENT_PROGRAM_FA.md, additive)
# ---------------------------------------------------------------------------
# These constants are the outputs of the improvement program stages and describe only the
# mechanisms that were adopted in their own stage's A/B test. A value of None means
# "that stage was rejected / there is nothing to wire up" and the default behavior remains
# exactly as before. These constants are filled in at the end of the program (after stage 4)
# and are recorded in IMPROVEMENT_PROGRAM_FA.md.
IMPROVE_GATE_MODE = "on"              # Walk-forward EV gate (stage 1): None=rejected | "on" | "smooth" — adopted: "on"
IMPROVE_GATE_N_MIN = 8                # Minimum prior realized trades for the gate to activate
IMPROVE_GATE_SMOOTH_EDGE_R = -0.2     # Smooth-mode edge: suppress only if prior expectancy < this value (R)
IMPROVE_REGIME_DEFAULT = True         # Regime filter as a scanner default (stage 2 — adopted)
IMPROVE_MTF_DEFAULT = True            # MTF gate as a scanner default (stage 2 — adopted)
IMPROVE_SYSBT_MAX_HOLD = None         # Adopted maximum holding period in the system backtest (stage 3): None=same as SYSBT_MAX_HOLD_CANDLES

# ---------------------------------------------------------------------------
# Forex market (forex module — additive; crypto behavior does not change at all)
# ---------------------------------------------------------------------------
# Data source: free Dukascopy historical feed (hourly .bi5 tick files).
# XAUUSD represents a commodity (gold), not a currency pair — in the forex scan it appears
# with the same label. Feed details, latency, and constraints are in FOREX_FA.md.
FOREX_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF",
                 "AUDUSD", "USDCAD", "EURGBP", "XAUUSD"]
FOREX_TIMEFRAMES = ["4h", "1d"]
# Point size for each symbol (price denominator for Dukascopy ticks): JPY and gold 1e3, the rest 1e5
FOREX_POINT = {"EURUSD": 100000, "GBPUSD": 100000, "USDJPY": 1000,
               "USDCHF": 100000, "AUDUSD": 100000, "USDCAD": 100000,
               "EURGBP": 100000, "XAUUSD": 1000}
# Pip size in price units (for JPY, 0.01, and for gold, 0.1 as the conventional "pip")
FOREX_PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
                  "USDCHF": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
                  "EURGBP": 0.0001, "XAUUSD": 0.1}
# Typical round-trip spread for each symbol, in pips (forex backtest cost model;
# for XAUUSD, 25 cents = 0.25 price units = 2.5 "pips" of 0.1)
FOREX_SPREAD_PIPS = {"EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.2,
                     "USDCHF": 1.8, "AUDUSD": 1.4, "USDCAD": 1.8,
                     "EURGBP": 1.5, "XAUUSD": 2.5}
FOREX_SLIPPAGE_PIPS = 0.2      # Total round-trip slippage, in pips
FOREX_CANDLES_PER_TIMEFRAME = {"4h": 1500, "1d": 1000}   # Target; the actual depth depends on the feed (FOREX_FA.md)
FOREX_WINDOW_END = "2026-11-07T00:00:00+00:00"  # Fixed end of the data window (the feed's last fully verified day; matrix determinism)
FOREX_MAX_FETCH_HOURS = 30000  # Maximum fetchable hours for building history (≈ 4.8 years of trading days)
FOREX_FETCH_RETRIES = 6        # Attempts per hourly file (the feed sometimes returns 503/dropouts)
FOREX_FETCH_TIMEOUT = 30       # Seconds, per request
FOREX_FETCH_PAUSE = 0.35       # Pause between successful requests (seconds) — the feed is rate-sensitive
FOREX_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
# Second source (additive): Yahoo Finance chart API (no key). Verified from this environment
# (2026-11-07): provides 1h up to 1 year and 1d up to 5 years; deeper
# than Dukascopy for the matrix, but its FX volume is zero (OBV becomes neutral
# — FOREX_FA.md) and its daily timestamps are at ET midnight.
# Each pair's Yahoo symbol: EURUSD=X, etc.; gold is the GC=F futures contract.
FOREX_YAHOO_SYMBOLS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
                       "USDJPY": "JPY=X", "USDCHF": "CHF=X",
                       "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
                       "EURGBP": "EURGBP=X", "XAUUSD": "GC=F"}
FOREX_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
FOREX_YAHOO_TIMEOUT = 25       # Seconds, per request
FOREX_PRIMARY = "yahoo"        # Primary candle source (matrix); Dukascopy is the backup/comparison

# ---------------------------------------------------------------------------
# LLM decision layer (LLM review layer — the analysis/llm_client.py
# and analysis/llm_engine.py modules and the scanner.attach_llm_analysis function, fully additive)
# ---------------------------------------------------------------------------
# This layer is only a "reasoning review" of candidates that have passed the EV gate;
# it never changes an EV decision, never overrides NO_TRADE, and without
# a key it is completely invisible (exactly the previous behavior). Model identifiers are
# deliberately not repeated here; the single source of model constants is in analysis/llm_engine.py
# (LLM_PRIMARY_MODEL / LLM_CROSSCHECK_MODEL / LLM_TIEBREAK_MODEL).
# ⚠️ Paste the real key only here (or set the NVIDIA_API_KEY environment variable);
# never write it in chat or in any other file. Full guide: LLM_LAYER_FA.md.
NVIDIA_API_KEY = ""  # optional: your own NVIDIA Build API key (for the LLM layer)
LLM_ENABLED_DEFAULT = False    # Scanner default: the LLM layer is off; it is enabled with llm=True or the --llm flag
LLM_SCAN_TOP_K = 3             # Maximum candidates per scan that receive an LLM review (cost/time control)
LLM_VOTES = 3                  # Number of runs of the primary model for a majority vote (= length of LLM_PRIMARY_TEMPERATURES in llm_engine)
