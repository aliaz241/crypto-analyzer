"""
scanner.py
Multi-symbol market scanner: for each (symbol × timeframe) pair in config.SCAN_SYMBOLS,
it fetches data from Binance, builds the same full analyzer report, and computes a
transparent score from 0 to 100 so candidates can be ranked and the best one can be
shown as "today's pick".

Score components (all caps and constants are in config.py):
- Trend (25): alignment of price with EMA50 and EMA200 + the direction of EMA50
  relative to EMA200; scaled by trend strength (ADX). The sign of this component
  determines the trade direction.
- Momentum (20): the MACD histogram aligned with the direction + RSI located in the
  healthy zone for that direction; extreme RSI values (overbought/oversold) are penalized.
- Pattern (25): the best recent confirmed pattern (candlestick or chart) aligned with
  the direction, scored from the backtest success rate of that same pattern on the
  same symbol/timeframe.
- Risk/reward (20): the distance to the first opposing level (target) relative to
  the distance to the suggested stop loss.
- Volume (10): the OBV trend aligned with the trade direction.

⚠️ A high score is not a "guarantee of profit"; it only means that, at this moment,
among the scanned options, this symbol/timeframe/direction combination looks better
under these mechanical rules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
from analysis import analyzer
from analysis import ev as ev_mod
from analysis import llm_client as llm_client_mod
from analysis import llm_engine as llm_engine_mod
from indicators import trend as trend_ind
from indicators import volatility as vol_ind


# Cache of the most recent scan: (symbol, timeframe) -> (df, report)
# Lets main render the top pick's chart without fetching the data again.
_SCAN_CACHE = {}


def get_scan_cache(symbol: str, timeframe: str):
    """Return the stored (df, report) for a candidate from the most recent scan, or None."""
    return _SCAN_CACHE.get((symbol.upper(), timeframe))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def _trend_signed_score(df, ind) -> float:
    """
    Signed trend score: positive = long bias, negative = short bias, range [-25, +25].
    Three alignment conditions (each ±1): price above EMA50, price above EMA200,
    EMA50 above EMA200. It is then scaled by trend strength (ADX): ADX >= 25 gets the
    full scale, a lower ADX gets a smaller linear scale (floor 0.5), and a missing ADX
    uses a scale of 0.8.
    """
    price = ind["price"]
    ema50 = trend_ind.ema(df, 50).iloc[-1]
    ema200 = trend_ind.ema(df, 200).iloc[-1]
    if price is None or ema50 != ema50 or ema200 != ema200:  # NaN check
        return 0.0

    align = 0.0
    align += 1 if price > ema50 else (-1 if price < ema50 else 0)
    align += 1 if price > ema200 else (-1 if price < ema200 else 0)
    align += 1 if ema50 > ema200 else (-1 if ema50 < ema200 else 0)
    raw = (align / 3.0) * config.SCORE_TREND_MAX

    adx = ind.get("adx")
    if adx is None:
        factor = 0.8
    elif adx >= 25:
        factor = 1.0
    else:
        factor = 0.5 + 0.5 * (adx / 25.0)
    return raw * factor


def _momentum_score(ind, direction: str) -> float:
    """Momentum score from 0 to 20 for the given direction (aligned MACD 12 + RSI zone up to 8)."""
    score = 0.0
    hist = ind["macd"].get("histogram")
    if hist is None:
        score += config.SCORE_MOMENTUM_MAX * 0.3  # Unknown: half of the MACD portion
    elif (direction == "long" and hist > 0) or (direction == "short" and hist < 0):
        score += config.SCORE_MOMENTUM_MAX * 0.6

    rsi = ind.get("rsi")
    if rsi is None:
        score += 4.0
    else:
        # For shorts, mirror the RSI so a single table works for both directions
        eff = rsi if direction == "long" else 100.0 - rsi
        if 45 <= eff <= 65:
            score += 8.0    # Healthy trend zone
        elif 40 <= eff < 45 or 65 < eff <= 70:
            score += 5.0
        elif 35 <= eff < 40 or 70 < eff <= 75:
            score += 3.0
        elif eff > 75:
            score += 0.0    # Extreme saturation against opening a new position in this direction
        else:  # eff < 35
            score += 2.0
    return min(score, config.SCORE_MOMENTUM_MAX)


def _pattern_score(report, direction: str):
    """
    Find the best recent pattern aligned with the direction and score it using its own
    backtest on the same symbol/timeframe. Returns: (score, pattern info or None).
    Success-rate mapping: 30% -> 0 and 70% -> 25 (linear), multiplied by
    min(1, occurrences/10).
    """
    want_bias = "bullish" if direction == "long" else "bearish"
    best = None

    def _consider(pattern, stats, kind):
        nonlocal best
        if not stats or not stats.get("total_occurrences"):
            return
        rate = stats["success_rate_pct"]
        occ = stats["total_occurrences"]
        span = config.SCORE_PATTERN_RATE_CEIL - config.SCORE_PATTERN_RATE_FLOOR
        pts = config.SCORE_PATTERN_MAX * _clamp(
            (rate - config.SCORE_PATTERN_RATE_FLOOR) / span, 0.0, 1.0)
        pts *= min(1.0, occ / 10.0)
        if best is None or pts > best[0]:
            best = (pts, {
                "name": pattern["name"], "fa": pattern["fa"], "kind": kind,
                "bias": pattern["bias"],
                "success_rate_pct": rate,
                "total_occurrences": occ,
                "reliable_sample": stats.get("reliable_sample", False),
            })

    bt = report.get("backtest", {})
    for p in report.get("latest_chart_patterns", []):
        if p["bias"] == want_bias:
            _consider(p, bt.get("chart", {}).get(p["name"]), "chart")
    for p in report.get("latest_candlestick_patterns", []):
        if p["bias"] == want_bias:
            _consider(p, bt.get("candlestick", {}).get(p["name"]), "candlestick")

    if best is None:
        return 0.0, None
    return best[0], best[1]


def _rr_score(report, direction: str, price: float):
    """
    Risk/reward score from 0 to 20 + (computed ratio, target level).
    Target = the first level opposing the price (resistance for long, support for short);
    risk = the distance to the suggested stop loss for that direction. If either is
    unavailable, the fixed score config.SCORE_RR_MISSING is used.
    """
    sr = report.get("support_resistance", {})
    if direction == "long":
        levels = sr.get("resistance") or []
        stop_data = (report.get("stop_loss_long") or {}).get("recommended") or {}
    else:
        levels = sr.get("support") or []
        stop_data = (report.get("stop_loss_short") or {}).get("recommended") or {}

    target = levels[0]["level"] if levels else None
    stop = stop_data.get("stop_loss")

    if target is None or stop is None:
        return config.SCORE_RR_MISSING, None, target

    if direction == "long":
        reward, risk = target - price, price - stop
    else:
        reward, risk = price - target, stop - price
    if reward <= 0 or risk <= 0:
        return config.SCORE_RR_MISSING, None, target

    rr = reward / risk
    pts = config.SCORE_RR_MAX * _clamp((rr - 0.5) / 2.0, 0.0, 1.0)
    return pts, rr, target


def _volume_score(ind, direction: str) -> float:
    """Aligned OBV trend: 10 | unknown: 5 | opposed: 0."""
    obv = ind.get("obv_trend")
    if obv is None:
        return config.SCORE_VOLUME_MAX / 2.0
    if (direction == "long" and obv == "up") or (direction == "short" and obv == "down"):
        return config.SCORE_VOLUME_MAX
    return 0.0


# ---------------------------------------------------------------------------
# Step 3 — Market-regime classification (new function; used only with apply_regime=True)
# ---------------------------------------------------------------------------
def classify_regime(df, report) -> str:
    """
    Return the market regime: "trend" | "range" | "volatile".

    Rules (all constants are in config):
    - volatile: the percentile of the current Bollinger Band width within the window of
      the most recent REGIME_BBW_WINDOW candles is at least
      REGIME_BBW_VOLATILE_PERCENTILE. This is checked first, because during extreme
      volatility even trends are unreliable.
    - trend: ADX is at least REGIME_ADX_TREND_MIN and the EMA200 slope over
      REGIME_EMA200_SLOPE_LOOKBACK candles has an absolute value of at least
      REGIME_EMA200_SLOPE_MIN_PCT percent. If EMA200 does not yet have enough data,
      ADX alone is the criterion.
    - range: all other cases.
    """
    ind = (report or {}).get("indicators") or {}

    try:
        bb = vol_ind.bollinger_bands(df)
        width = ((bb["upper"] - bb["lower"]) / bb["middle"]).dropna()
        if len(width) >= 20:
            window = width.tail(config.REGIME_BBW_WINDOW)
            pct = float((window <= width.iloc[-1]).mean() * 100.0)
            if pct >= config.REGIME_BBW_VOLATILE_PERCENTILE:
                return "volatile"
    except Exception:
        pass

    adx = ind.get("adx")
    if adx is not None and adx >= config.REGIME_ADX_TREND_MIN:
        try:
            ema200 = trend_ind.ema(df, 200).dropna()
            lb = config.REGIME_EMA200_SLOPE_LOOKBACK
            if len(ema200) > lb and ema200.iloc[-1 - lb] > 0:
                slope_pct = (ema200.iloc[-1] / ema200.iloc[-1 - lb] - 1.0) * 100.0
                if abs(slope_pct) < config.REGIME_EMA200_SLOPE_MIN_PCT:
                    return "range"
        except Exception:
            pass
        return "trend"
    return "range"


# ---------------------------------------------------------------------------
# Step 4 — Structure-component score (only when extras is passed to score_candidate)
# ---------------------------------------------------------------------------
def _structure_score(extras: dict, direction: str) -> float:
    """
    Build a score in the range [-STRUCTURE_MAX, +STRUCTURE_MAX] from the output of
    analysis.structure.analyze_structure: alignment of each structural signal with the
    candidate direction gives positive points and opposition gives negative points
    (RSI divergence: ±4, MACD divergence: ±3, latest BOS/CHoCH event: ±3 — the
    constants are in config).
    """
    def _aligned(bias) -> float:
        if bias is None:
            return 0.0
        want = "bullish" if direction == "long" else "bearish"
        return 1.0 if bias == want else -1.0

    pts = 0.0
    pts += _aligned(extras.get("rsi_divergence")) * config.STRUCTURE_RSI_DIV_POINTS
    pts += _aligned(extras.get("macd_divergence")) * config.STRUCTURE_MACD_DIV_POINTS
    event = extras.get("last_event") or {}
    pts += _aligned(event.get("direction")) * config.STRUCTURE_EVENT_POINTS
    return max(-config.STRUCTURE_MAX, min(config.STRUCTURE_MAX, pts))


# ---------------------------------------------------------------------------
# Step 6 — Position size (an additive calculation based on existing outputs only)
# ---------------------------------------------------------------------------
def _position_sizing(report, entry, stop):
    """
    Suggested size using fixed risk: risk amount = RISK_PER_TRADE_PCT percent of
    ACCOUNT_SIZE_USDT (a placeholder — not the user's real balance), and
    size = risk amount / |entry − stop loss|. The report's ATR is included alongside
    for information. Returns None if entry/stop loss is missing.
    """
    if entry is None or stop is None:
        return None
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return None
    risk_amount = config.ACCOUNT_SIZE_USDT * config.RISK_PER_TRADE_PCT / 100.0
    size = risk_amount / risk_per_unit
    return {
        "account_size_usdt": config.ACCOUNT_SIZE_USDT,
        "risk_pct": config.RISK_PER_TRADE_PCT,
        "risk_amount_usdt": round(risk_amount, 2),
        "size_units": round(size, 6),
        "notional_usdt": round(size * entry, 2),
        "atr": (report.get("indicators") or {}).get("atr"),
    }


# ---------------------------------------------------------------------------
# Scoring a candidate
# ---------------------------------------------------------------------------
def score_candidate(df, report, symbol: str, timeframe: str,
                    higher_tf_direction: str = None,
                    apply_regime: bool = False,
                    extras: dict = None) -> dict:
    """
    Additive parameters (all default to "off", and the default path exactly preserves
    the previous behavior):
    - higher_tf_direction (step 2): the higher-timeframe direction ("long"/"short");
      if it opposes the candidate direction, the total score is multiplied by
      MTF_CONFLICT_FACTOR and the "mtf" key is added to the output.
    - apply_regime (step 3): if True, the components are scaled according to
      classify_regime and the "regime" key is added.
    - extras (step 4): the output of analysis.structure.analyze_structure; if provided,
      the "structure" component (up to ±10) is added to the score — in this case the
      score scale is /110, not /100. The default path remains /100.
    """
    ind = report["indicators"]
    price = ind["price"]

    trend_signed = _trend_signed_score(df, ind)
    if abs(trend_signed) >= config.SCORE_TREND_WASH_THRESHOLD:
        direction = "long" if trend_signed > 0 else "short"
    else:
        # The trend is neutral; the direction is taken from the sign of the MACD histogram
        hist = ind["macd"].get("histogram")
        if hist is not None and hist != 0:
            direction = "long" if hist > 0 else "short"
        else:
            ema50 = trend_ind.ema(df, 50).iloc[-1]
            direction = "long" if price >= ema50 else "short"

    dir_sign = 1.0 if direction == "long" else -1.0
    trend_pts = max(0.0, trend_signed * dir_sign)
    momentum_pts = _momentum_score(ind, direction)
    pattern_pts, pattern_info = _pattern_score(report, direction)
    rr_pts, rr_value, target = _rr_score(report, direction, price)
    volume_pts = _volume_score(ind, direction)

    stop_data = (report.get("stop_loss_long" if direction == "long"
                            else "stop_loss_short") or {}).get("recommended") or {}
    stop = stop_data.get("stop_loss")

    # --- Step 3: regime scaling (only with apply_regime=True; off by default) ---
    regime_name = None
    if apply_regime:
        regime_name = classify_regime(df, report)
        if regime_name == "range":
            trend_pts = trend_pts * config.REGIME_RANGE_TREND_SCALE
            momentum_pts = min(momentum_pts * config.REGIME_RANGE_MOMENTUM_SCALE,
                               config.SCORE_MOMENTUM_MAX)
        elif regime_name == "volatile":
            pattern_pts = pattern_pts * config.REGIME_VOLATILE_PATTERN_SCALE

    components = {
        "trend": round(trend_pts, 1),
        "momentum": round(momentum_pts, 1),
        "pattern": round(pattern_pts, 1),
        "risk_reward": round(rr_pts, 1),
        "volume": round(volume_pts, 1),
    }
    total = round(sum(components.values()), 1)

    # One-line reason: the two components with the largest relative share of their own caps
    fragments = {
        "trend": "روند هم‌سو با EMA50/200" + (f" (ADX {ind['adx']})" if ind.get("adx") is not None else ""),
        "momentum": "مومنتوم هم‌جهت (MACD و RSI)",
        "pattern": (f"الگوی {pattern_info['fa']} با موفقیت تاریخی "
                    f"{pattern_info['success_rate_pct']}٪ روی همین تایم‌فریم") if pattern_info else "",
        "risk_reward": (f"سودبه‌ریسک {rr_value:.1f} تا سطح بعدی") if rr_value else "سطح مقابل نزدیک",
        "volume": "تأیید حجم (OBV هم‌جهت)",
    }
    maxes = {"trend": config.SCORE_TREND_MAX, "momentum": config.SCORE_MOMENTUM_MAX,
             "pattern": config.SCORE_PATTERN_MAX, "risk_reward": config.SCORE_RR_MAX,
             "volume": config.SCORE_VOLUME_MAX}
    ranked = sorted(components.items(),
                    key=lambda kv: kv[1] / maxes[kv[0]] if maxes[kv[0]] else 0,
                    reverse=True)
    top_fragments = [fragments[k] for k, v in ranked[:2] if v > 0 and fragments[k]]
    reason = "دلیل: " + (" + ".join(top_fragments) if top_fragments else "سیگنال مشخصی ندارد")

    # --- Step 4: structure component (only when extras is provided; /110 scale) ---
    # Added after the "reason" is computed, so the reason-ranking logic (which knows only the five
    # main components) remains untouched.
    if extras is not None:
        components["structure"] = round(_structure_score(extras, direction), 1)
        total = round(total + components["structure"], 1)

    # --- Step 2: MTF gate (only when a higher-timeframe direction is provided) ---
    mtf_info = None
    if higher_tf_direction is not None:
        conflict = higher_tf_direction != direction
        if conflict:
            total = round(total * config.MTF_CONFLICT_FACTOR, 1)
        mtf_info = {"higher_tf_direction": higher_tf_direction,
                    "conflict": conflict,
                    "factor": config.MTF_CONFLICT_FACTOR if conflict else 1.0}

    result = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "direction": direction,
        "direction_fa": "لانگ (خرید)" if direction == "long" else "شورت (فروش)",
        "score": total,
        "components": components,
        "price": price,
        "entry": price,
        "stop": stop,
        "target": target,
        "risk_reward": round(rr_value, 2) if rr_value else None,
        "pattern": pattern_info,
        "reason": reason,
    }
    if mtf_info is not None:
        result["mtf"] = mtf_info
    if regime_name is not None:
        result["regime"] = regime_name
    if extras is not None:
        result["structure"] = extras
    # Step 6: position size — an additive key, calculated from the existing entry/stop
    result["sizing"] = _position_sizing(report, result["entry"], result["stop"])
    return result


# ---------------------------------------------------------------------------
# Altcoin correlation with BTC (additive — ported from an external pack; display/analysis only)
# ---------------------------------------------------------------------------
def _btc_corr_from_frames(alt_df, btc_df, window: int = 30):
    """
    Pearson correlation of "returns" (not raw prices) between the alt and BTC closes
    over the last `window` aligned returns. Correlation on raw prices is misleading
    because of the shared trend. Returns the value rounded to 3 decimals, or None if
    there is insufficient data. Near 1 means the alt is effectively a beta of BTC;
    near 0/negative means more independent behavior.
    """
    try:
        joined = pd.DataFrame({"alt": alt_df["close"].astype(float),
                               "btc": btc_df["close"].astype(float)}).dropna()
        if len(joined) < window + 1:
            return None
        rets = joined.pct_change().dropna().tail(window)
        if len(rets) < 10:
            return None
        if rets["alt"].std() == 0 or rets["btc"].std() == 0:
            return None
        return round(float(rets["alt"].corr(rets["btc"])), 3)
    except Exception:
        return None


def alt_btc_correlation(symbol: str, timeframe: str, window: int = 30):
    """
    Latest correlation between the returns of `symbol` and BTC on the same timeframe
    (window of 30). For BTC itself, returns 1.0. Returns None on any error
    (network/insufficient data).
    """
    try:
        if symbol.upper() == "BTC":
            return 1.0
        need = window + 60   # Margin for time alignment and pct_change
        alt_df = analyzer.fetch_candles_binance(symbol, timeframe,
                                                total_candles=need)
        btc_df = analyzer.fetch_candles_binance("BTC", timeframe,
                                                total_candles=need)
        return _btc_corr_from_frames(alt_df, btc_df, window)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# EV decision layer (additive — analysis/ev.py)
# ---------------------------------------------------------------------------
def _ev_gate_annotation(cand: dict):
    """
    Additive annotation for the walk-forward EV gate (improvement program, stage 1 —
    adopted with config.IMPROVE_GATE_MODE) on the "ev_gate" key.

    The live form of the gate: a combination's "trailing realized" statistics are the
    same as the cached out-of-sample EV (from the final stack matrix); the gate
    suppresses when n ≥ IMPROVE_GATE_N_MIN and realized expectancy is below the edge
    (on mode: < 0; smooth: < −0.2R). This annotation is informational, and
    decide_candidate makes the actual decision — its mandatory rules (insufficient
    sample n < 10 → NO_TRADE and expectancy ≤ 0 → NO_TRADE) already produce the same
    NO_TRADE at every point where the gate suppresses (the live layer is stricter than
    the gate on sample size: the gate activates from n ≥ 8, while the EV layer's buy
    decision requires n ≥ 10 for any decision and n ≥ 30 for BUY/STRONG). The
    difference between these two forms is recorded in IMPROVEMENT_PROGRAM_FA.md.
    """
    mode = getattr(config, "IMPROVE_GATE_MODE", None)
    if not mode:
        return
    ev = cand.get("ev") or {}
    n = ev.get("n_trades") or 0
    exp = ev.get("expectancy_r")
    edge = 0.0 if mode == "on" else config.IMPROVE_GATE_SMOOTH_EDGE_R
    suppressed = bool(exp is not None and n >= config.IMPROVE_GATE_N_MIN
                      and exp < edge)
    cand["ev_gate"] = {"mode": mode, "n_min": config.IMPROVE_GATE_N_MIN,
                       "trailing_n": n, "trailing_expectancy_r": exp,
                       "suppressed": suppressed}


def attach_ev_decision(cand: dict) -> dict:
    """
    Add the additive "ev" and "decision"/"decision_fa"/"decision_reason" keys to a
    candidate. EV is read only from the cache/saved files (allow_compute=False) — this
    function never runs a backtest, so scans and live paths do not slow down. If the
    candidate already has both keys (for example, a test stub or an upstream caller),
    it is left untouched. On any error: ev=None and a NO_TRADE decision with an
    explicit reason (fail-safe). The additive "ev_gate" key (annotation for stage 1
    of the improvement program) is added on both paths; when IMPROVE_GATE_MODE is
    empty, no key is added and the behavior is exactly as before.
    """
    _ev_gate_annotation(cand)
    if "ev" in cand and "decision" in cand:
        return cand
    try:
        ev = ev_mod.ev_for_candidate(cand["symbol"], cand["timeframe"],
                                     direction=cand.get("direction"),
                                     allow_compute=False)
    except Exception:
        ev = None
    cand["ev"] = ev
    try:
        dec = ev_mod.decide_candidate(cand, ev)
    except Exception:
        dec = {"decision": "NO_TRADE",
               "decision_fa": ev_mod.DECISION_FA["NO_TRADE"],
               "decision_reason": "رد: خطا در لایه‌ی تصمیم EV."}
    cand.update(dec)
    _ev_gate_annotation(cand)
    return cand


# ---------------------------------------------------------------------------
# LLM review layer (additive — analysis/llm_engine.py; LLM_LAYER_FA.md)
# ---------------------------------------------------------------------------
def llm_layer_active() -> bool:
    """True means the LLM layer can be used for a scan: the config default
    (LLM_ENABLED_DEFAULT) is on or the caller explicitly enabled it, and an NVIDIA
    key resolves. Without a key it is always False."""
    if not getattr(config, "LLM_ENABLED_DEFAULT", False):
        return False
    try:
        return bool(llm_client_mod.resolve_api_key())
    except Exception:
        return False


def attach_llm_analysis(cand: dict, report_data_builder=None,
                        client=None) -> dict:
    """
    Attach the LLM review as the additive "llm" key on a candidate and return the
    same candidate (the attach_ev_decision pattern).

    Guards (all required before the engine is called):
    - The candidate must already have an EV decision (attach_ev_decision has run on
      it) and its decision must not be NO_TRADE — hard floor: NO_TRADE means zero LLM
      calls (the engine itself enforces the same floor; this is a second guard here).
    - If no key resolves or the engine returns None (layer inactive), no key is added
      and the candidate is returned exactly as before.
    - report_data_builder (optional): a function called with the candidate itself
      that builds extra "computed report" data for the model prompt; if not provided,
      the candidate itself is the model's entire input. Builder errors are silently
      ignored (report None) so a scan never stalls because of it.
    - On any other error, the candidate is returned untouched (fail-safe).
    """
    if cand.get("decision") is None or cand.get("decision") == "NO_TRADE":
        return cand
    try:
        report_data = None
        if report_data_builder is not None:
            try:
                report_data = report_data_builder(cand)
            except Exception:
                report_data = None
        result = llm_engine_mod.analyze_candidate(cand, report_data,
                                                  client=client)
    except Exception:
        return cand
    if result is not None:
        cand["llm"] = result
    return cand


def _attach_llm_to_scan(out: dict, enabled: bool, client=None) -> dict:
    """
    LLM step for scan output (internal; both scan_market and scan_forex_market pass
    through here so the behavior is identical).

    - enabled=False (the default for all paths) → output untouched, without even one
      extra key.
    - enabled=True but no key resolves → adds the "llm_summary" key with
      active=False and the reason "key not configured", so the caller (CLI) can print
      a clear line; no candidate receives an "llm" key.
    - Otherwise, at most the first config.LLM_SCAN_TOP_K candidates in the display
      slice whose decision is not NO_TRADE are reviewed, and "llm_summary" reports
      the number reviewed and the distribution of LLM decisions. An LLM error skips
      only that candidate; the scan and the other candidates remain exactly as before.
    """
    if not enabled:
        return out
    try:
        has_key = bool(llm_client_mod.resolve_api_key()) \
            or client is not None
    except Exception:
        has_key = client is not None
    if not has_key:
        out["llm_summary"] = {"active": False, "reason": "no_api_key",
                              "analyzed": 0, "decisions": {}}
        return out
    top_k = getattr(config, "LLM_SCAN_TOP_K", 3)
    analyzed = 0
    decisions = {}
    for cand in out.get("candidates", []):
        if analyzed >= top_k:
            break
        if cand.get("decision") in (None, "NO_TRADE"):
            continue
        before = "llm" in cand
        try:
            attach_llm_analysis(cand, client=client)
        except Exception:
            continue
        if "llm" in cand and not before:
            analyzed += 1
            d = (cand["llm"] or {}).get("llm_decision")
            if d:
                decisions[d] = decisions.get(d, 0) + 1
    out["llm_summary"] = {"active": True, "analyzed": analyzed,
                          "decisions": decisions}
    return out


# ---------------------------------------------------------------------------
# Market scan
# ---------------------------------------------------------------------------
def scan_market(symbols: list = None, timeframes: list = None,
                limit: int = None, progress: bool = True,
                use_mtf: bool = None,
                include_btc_corr: bool = False,
                only_actionable: bool = False,
                apply_regime: bool = None,
                llm: bool = False,
                llm_client=None) -> dict:
    """
    Scan and rank all symbol × timeframe combinations.
    If one combination errors (for example, a nonexistent symbol), the error is
    recorded and the scan continues.
    Output: {"candidates": [...sorted descending, at most limit items],
            "errors": [{"symbol","timeframe","error"}...], "scanned": successful count}

    use_mtf (step 2): if True, for 4h candidates the daily-timeframe direction is
    computed with the same scoring and passed to score_candidate as the MTF gate
    (the daily direction for each symbol is fetched once and cached during the scan).
    An error while fetching the daily data leaves the candidate without a gate and is
    recorded in errors with the label "1d (mtf)".
    Default None = read from config.IMPROVE_MTF_DEFAULT (improvement program,
    stage 2: adopted and set to True); an explicit False always overrides the config
    and restores the previous behavior.

    apply_regime (improvement program, stage 2 — additive): if True, regime scaling
    (step 3 of score_candidate) is applied to candidates.
    Default None = read from config.IMPROVE_REGIME_DEFAULT (adopted: True);
    explicit False = off. The MTF daily direction is intentionally computed "without"
    the regime — the same semantics as the stage evaluation engine (the simple daily
    pipeline direction) and the previous behavior of this path.

    include_btc_corr (additive, default False = previous behavior): if True, each
    candidate receives a "btc_correlation" key (correlation of returns with BTC; for
    BTC itself it is 1.0, and None on error). The BTC dataframe for each timeframe is
    fetched once and cached during the scan.

    EV layer (additive, always active but read-only from the cache): each candidate
    receives the "ev" and "decision"/"decision_fa"/"decision_reason" keys, and the
    output has a "decision_summary" key (count of each status across all scanned
    combinations, before the limit slice). NO_TRADE candidates are never silently
    removed — they remain in the list with their rejection reason.

    only_actionable (additive, default False): if True, the output also has a separate
    "actionable" key: only candidates whose decision is BUY/SELL/STRONG_* (from the
    whole scan, not just the limit slice). The main "candidates" list is complete and
    without any hidden filter in both cases.

    llm (additive, default False = output exactly as before): if True and an NVIDIA
    key resolves, at most the first config.LLM_SCAN_TOP_K non-NO_TRADE candidates
    receive an LLM review ("llm" key on the candidate + "llm_summary" key on the
    output — _attach_llm_to_scan). llm_client is for tests only (a fake client); in
    the real path it remains None.
    """
    if symbols is None:
        symbols = config.SCAN_SYMBOLS
    if timeframes is None:
        timeframes = config.SCAN_TIMEFRAMES
    if limit is None:
        limit = config.SCAN_TOP_N
    if use_mtf is None:
        use_mtf = config.IMPROVE_MTF_DEFAULT
    if apply_regime is None:
        apply_regime = config.IMPROVE_REGIME_DEFAULT

    results = []
    errors = []
    daily_dir_cache = {}
    btc_df_cache = {}
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال اسکن {symbol} {timeframe} ...", flush=True)
            try:
                df = analyzer.fetch_candles_binance(symbol, timeframe)
                report = analyzer.build_report(df, symbol, timeframe)
                _SCAN_CACHE[(symbol.upper(), timeframe)] = (df, report)
                cand = score_candidate(df, report, symbol, timeframe,
                                       apply_regime=apply_regime)
                if use_mtf and timeframe == "4h":
                    try:
                        key = symbol.upper()
                        if key not in daily_dir_cache:
                            ddf = analyzer.fetch_candles_binance(symbol, "1d")
                            drep = analyzer.build_report(ddf, symbol, "1d")
                            daily_dir_cache[key] = score_candidate(
                                ddf, drep, symbol, "1d")["direction"]
                        cand = score_candidate(
                            df, report, symbol, timeframe,
                            higher_tf_direction=daily_dir_cache[key],
                            apply_regime=apply_regime)
                    except Exception as exc:
                        errors.append({"symbol": symbol, "timeframe": "1d (mtf)",
                                       "error": str(exc)})
                if include_btc_corr:
                    if symbol.upper() == "BTC":
                        cand["btc_correlation"] = 1.0
                    else:
                        try:
                            if timeframe not in btc_df_cache:
                                btc_df_cache[timeframe] = \
                                    analyzer.fetch_candles_binance("BTC", timeframe)
                            cand["btc_correlation"] = _btc_corr_from_frames(
                                df, btc_df_cache[timeframe])
                        except Exception:
                            cand["btc_correlation"] = None
                # EV layer: read from the cache only; no backtest is run
                attach_ev_decision(cand)
                results.append(cand)
            except Exception as exc:
                errors.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
                if progress:
                    print(f"  ⚠️ خطا در اسکن {symbol} {timeframe}: {exc}", flush=True)

    results.sort(key=lambda c: c["score"], reverse=True)
    # Decision summary across all scanned combinations (before applying the limit slice)
    decision_summary = {}
    for c in results:
        d = c.get("decision", "NO_TRADE")
        decision_summary[d] = decision_summary.get(d, 0) + 1
    out = {
        "market": "crypto",
        "candidates": results[:limit],
        "errors": errors,
        "scanned": len(results),
        "decision_summary": decision_summary,
    }
    if only_actionable:
        out["actionable"] = [c for c in results
                             if c.get("decision") in ev_mod.ACTIONABLE_DECISIONS]
    _attach_llm_to_scan(out, llm, client=llm_client)
    return out


# ---------------------------------------------------------------------------
# Forex scan (additive — FOREX_FA.md; the crypto scan_market is untouched)
# ---------------------------------------------------------------------------
def scan_forex_market(symbols: list = None, timeframes: list = None,
                      limit: int = None, progress: bool = True,
                      use_mtf: bool = None,
                      only_actionable: bool = False,
                      apply_regime: bool = None,
                      llm: bool = False,
                      llm_client=None) -> dict:
    """
    Scan forex pairs × [4h, 1d] with exactly the same score_candidate and the final
    crypto-scan stack (regime/MTF from the same config defaults, EV layer read-only
    from the cache). The only differences are the data source
    (analyzer.fetch_market with market="forex") and the absence of BTC correlation.
    Combinations with no entry in the EV cache (before the forex matrix) receive a
    NO_TRADE decision with the reason "insufficient sample" — the correct pre-matrix
    behavior. The output has the same shape as scan_market, plus the
    "market": "forex" key. XAUUSD in this scan represents a commodity (gold), not a
    currency pair.

    llm (additive, default False): exactly the same semantics as scan_market — LLM
    review only with llm=True and a resolved key (see _attach_llm_to_scan). Forex is
    the current product priority, so this path is the main consumer of the LLM layer.
    """
    if symbols is None:
        symbols = config.FOREX_SYMBOLS
    if timeframes is None:
        timeframes = config.FOREX_TIMEFRAMES
    if limit is None:
        limit = config.SCAN_TOP_N
    if use_mtf is None:
        use_mtf = config.IMPROVE_MTF_DEFAULT
    if apply_regime is None:
        apply_regime = config.IMPROVE_REGIME_DEFAULT

    results = []
    errors = []
    daily_dir_cache = {}
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال اسکن فارکس {symbol} {timeframe} ...", flush=True)
            try:
                df = analyzer.fetch_market(symbol, timeframe, market="forex")
                report = analyzer.build_report(df, symbol, timeframe)
                _SCAN_CACHE[(symbol.upper(), timeframe)] = (df, report)
                cand = score_candidate(df, report, symbol, timeframe,
                                       apply_regime=apply_regime)
                cand["market"] = "forex"
                if use_mtf and timeframe == "4h":
                    try:
                        key = symbol.upper()
                        if key not in daily_dir_cache:
                            ddf = analyzer.fetch_market(symbol, "1d",
                                                        market="forex")
                            drep = analyzer.build_report(ddf, symbol, "1d")
                            daily_dir_cache[key] = score_candidate(
                                ddf, drep, symbol, "1d")["direction"]
                        cand = score_candidate(
                            df, report, symbol, timeframe,
                            higher_tf_direction=daily_dir_cache[key],
                            apply_regime=apply_regime)
                        cand["market"] = "forex"
                    except Exception as exc:
                        errors.append({"symbol": symbol,
                                       "timeframe": "1d (mtf)",
                                       "error": str(exc)})
                attach_ev_decision(cand)
                results.append(cand)
            except Exception as exc:
                errors.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
                if progress:
                    print(f"  ⚠️ خطا در اسکن فارکس {symbol} {timeframe}: {exc}",
                          flush=True)

    results.sort(key=lambda c: c["score"], reverse=True)
    decision_summary = {}
    for c in results:
        d = c.get("decision", "NO_TRADE")
        decision_summary[d] = decision_summary.get(d, 0) + 1
    out = {
        "market": "forex",
        "candidates": results[:limit],
        "errors": errors,
        "scanned": len(results),
        "decision_summary": decision_summary,
    }
    if only_actionable:
        out["actionable"] = [c for c in results
                             if c.get("decision") in ev_mod.ACTIONABLE_DECISIONS]
    _attach_llm_to_scan(out, llm, client=llm_client)
    return out
