"""
analysis/structure.py
Divergences and market structure — add-on module (step 4).

This module changes none of the existing computations and build_report does
not use it either. Its output (analyze_structure) only has an effect when it
is explicitly passed as extras to scanner.score_candidate; in that case the
"structure" component (up to ±10 points) is added to the score and the scale
becomes /110. The scanner's default path stays without this module and with
the /100 scale.

Definitions:
- swing high/low: a point that is larger/smaller than SWING_LOOKBACK_CANDLES
  neighbors on both sides (the same constant as the project's chart patterns).
- Regular RSI/MACD divergence: between the last two same-type swings of price;
  a higher price high with a lower oscillator high = bearish; a lower price low
  with a higher oscillator low = bullish. The second swing must be within the
  last STRUCTURE_RECENT_CANDLES candles, otherwise the divergence does not
  count as recent.
- Market structure: comparing the last two swing highs and two swing lows
  (HH+HL = uptrend, LH+LL = downtrend, otherwise sideways).
- BOS/CHoCH event (documented simplification): after the last swing is
  confirmed, the first close beyond that level is a break; in the direction of
  the structure = BOS, against the structure = CHoCH. (Full tracking of
  intermediate events is not done; only the last level and the first break
  after it.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from indicators import momentum as momentum_ind
from indicators import trend as trend_ind


def _swing_indices(values, lookback: int, kind: str) -> list:
    """Swing indices on a numeric array (strict comparison with neighbors)."""
    out = []
    v = np.asarray(values, dtype=float)
    for i in range(lookback, len(v) - lookback):
        if np.isnan(v[i]):
            continue
        seg = v[i - lookback:i + lookback + 1]
        if kind == "high" and np.all(v[i] > np.delete(seg, lookback)):
            out.append(i)
        elif kind == "low" and np.all(v[i] < np.delete(seg, lookback)):
            out.append(i)
    return out


def _last_divergence(price, osc, lookback: int, recent: int):
    """
    The last regular divergence between price and an oscillator.
    Output: "bullish" | "bearish" | None
    """
    n = len(price)
    highs = _swing_indices(price, lookback, "high")
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if b >= n - recent and not (np.isnan(osc[a]) or np.isnan(osc[b])):
            if price[b] > price[a] and osc[b] < osc[a]:
                return "bearish"
    lows = _swing_indices(price, lookback, "low")
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if b >= n - recent and not (np.isnan(osc[a]) or np.isnan(osc[b])):
            if price[b] < price[a] and osc[b] > osc[a]:
                return "bullish"
    return None


def _market_structure(df, lookback: int):
    """(structure, last event) — the definition and simplification are given at the top of the file."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    sh = _swing_indices(highs, lookback, "high")
    sl = _swing_indices(lows, lookback, "low")

    structure = "sideways"
    if len(sh) >= 2 and len(sl) >= 2:
        if highs[sh[-1]] > highs[sh[-2]] and lows[sl[-1]] > lows[sl[-2]]:
            structure = "uptrend"
        elif highs[sh[-1]] < highs[sh[-2]] and lows[sl[-1]] < lows[sl[-2]]:
            structure = "downtrend"

    breaks = []
    if sh:
        level = highs[sh[-1]]
        for pos in range(sh[-1] + lookback, len(closes)):
            if closes[pos] > level:
                breaks.append((pos, "bullish", level))
                break
    if sl:
        level = lows[sl[-1]]
        for pos in range(sl[-1] + lookback, len(closes)):
            if closes[pos] < level:
                breaks.append((pos, "bearish", level))
                break

    event = None
    if breaks:
        pos, bias, level = max(breaks, key=lambda b: b[0])
        if structure == "uptrend":
            etype = "BOS" if bias == "bullish" else "CHoCH"
        elif structure == "downtrend":
            etype = "BOS" if bias == "bearish" else "CHoCH"
        else:
            etype = "BOS"
        event = {"type": etype, "direction": bias,
                 "index": str(df.index[pos]), "level": float(level)}
    return structure, event


def analyze_structure(df) -> dict:
    """
    Ready output for passing to score_candidate(..., extras=...):
    {"rsi_divergence", "macd_divergence", "structure", "last_event"}
    """
    lookback = config.SWING_LOOKBACK_CANDLES
    recent = config.STRUCTURE_RECENT_CANDLES
    close = df["close"].to_numpy(dtype=float)
    rsi = momentum_ind.rsi(df).to_numpy(dtype=float)
    macd_line = trend_ind.macd(df)["macd"].to_numpy(dtype=float)
    structure, event = _market_structure(df, lookback)
    return {
        "rsi_divergence": _last_divergence(close, rsi, lookback, recent),
        "macd_divergence": _last_divergence(close, macd_line, lookback, recent),
        "structure": structure,
        "last_event": event,
    }
