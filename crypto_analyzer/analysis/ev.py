"""
analysis/ev.py — Decision layer based on expected value (EV Decision Layer)

Fully additive: no existing computation (scanner score, backtest, patterns)
changes. This module is only a "consumer" of the system backtest output and,
for each symbol/timeframe combination, builds a net EV and, from it + the
scanner's current score, determines the decision state (decision).

Definition of EV in this system
---------------------
The EV of each combination = the expected value (mean R) of the "out-of-sample"
trades of that combination's system backtest, at the score threshold
config.EV_BACKTEST_THRESHOLD (default 60). The R values are computed inside the
system backtest itself after deducting round-trip fees and slippage on both
sides, so this EV is "net" and no separate cost should be deducted from it.

⚠️ Out-of-sample only: in-sample statistics are the very data the rules were
shaped on and are overfit for live decisions; this module deliberately never
reads in-sample — neither on the saved-files path nor on the fresh-compute path.

⚠️ Direction limitation (important and honest): the backtest trades are produced
by the pipeline itself, which chose each trade's direction at the time; so the
EV statistics are "the pipeline's overall performance on this combination", not
separate statistics for long and short. We do not have per-direction statistics
and do not pretend we do; the candidate's current direction comes only from the
scanner and EV is applied equally to both directions.

Data sources (in priority order in ev_for_candidate):
1) in-memory cache, 2) disk cache (ev_cache.json next to the package),
3) saved backtest result files (the matrix/ folder next to the package or
   btc4h_backtest_result.json) — from their trade lists only segment="out"
   is read and all metrics (win rate, average win/loss, profit factor,
   break-even win rate) are recomputed from those same real R values,
4) only when allow_compute=True and none of the above exist: a fresh run of
   run_system_backtest, stored in the cache. The scan path (scan_market) always
   calls with allow_compute=False so a 40-combination scan does not turn into
   40 backtests; cache reset is explicit: ev_for_candidate(..., refresh=True) or
   refresh_ev_cache(...) or the CLI flag --refresh-ev in main.py.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(_PACKAGE_DIR, config.EV_CACHE_FILENAME)

# In-memory cache: (SYMBOL, timeframe) -> cache entry (includes r_values_oos)
_MEM_CACHE = {}

_RELIABILITY_FA = {
    "insufficient": "نمونه ناکافی",
    "low": "اطمینان کم (نمونه کم)",
    "ok": "نمونه قابل اتکا",
}

DECISION_FA = {
    "STRONG_BUY": "خرید قوی",
    "STRONG_SELL": "فروش قوی",
    "BUY": "خرید",
    "SELL": "فروش",
    "WAIT": "انتظار — فعلاً ورود نکن",
    "NO_TRADE": "بدون معامله (NO TRADE)",
}

ACTIONABLE_DECISIONS = ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL")


# ---------------------------------------------------------------------------
# Computing EV from statistics/trades
# ---------------------------------------------------------------------------
def _reliability(n: int) -> str:
    if n < config.EV_SAMPLE_INSUFFICIENT_MAX:
        return "insufficient"
    if n < config.EV_SAMPLE_OK_MIN:
        return "low"
    return "ok"


def _extract_r_values(stats) -> list:
    """Extracts the list of R values from supported inputs (without segment filtering;
    out-of-sample filtering is the caller's job)."""
    if stats is None:
        return []
    if isinstance(stats, (list, tuple)):
        items = list(stats)
    elif isinstance(stats, dict):
        if "r_values" in stats:
            items = list(stats["r_values"])
        elif "rs" in stats:
            items = list(stats["rs"])
        elif "trades" in stats:
            items = list(stats["trades"])
        else:
            return []
    else:
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            r = it.get("r")
        else:
            r = it
        if r is None:
            continue
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(r):
            out.append(r)
    return out


def compute_ev(stats) -> dict:
    """
    Computes EV metrics from one of these inputs:
    - a list of R values (a number, or a trade dict with key "r")
    - a dict with key "r_values" / "rs" / "trades"
    - an aggregate dict with keys win_rate (fraction 0..1) and avg_win_r and
      avg_loss_r (absolute value of the average loss) and n_trades

    Output:
      win_rate (fraction) + win_rate_pct, avg_win_r, avg_loss_r (positive absolute value),
      expectancy_r (net expected value, i.e. the mean R), profit_factor
      (gross profit/|gross loss|; inf if there is no loss but there is profit, 0.0
      if there is no profit, None if there are no trades),
      break_even_win_rate = avg_loss / (avg_win + avg_loss) (fraction) + _pct,
      n_trades, reliability and reliability_fa.
    """
    # Ready aggregate path
    if isinstance(stats, dict) and "avg_win_r" in stats and "avg_loss_r" in stats \
            and "win_rate" in stats and "n_trades" in stats:
        n = int(stats["n_trades"])
        wr = float(stats["win_rate"])
        aw = float(stats["avg_win_r"])
        al = float(stats["avg_loss_r"])
        expectancy = wr * aw - (1.0 - wr) * al
        gross_profit = wr * n * aw
        gross_loss = (1.0 - wr) * n * al
        if n == 0:
            pf = None
        elif gross_loss <= 0:
            pf = float("inf") if gross_profit > 0 else None
        else:
            pf = gross_profit / gross_loss
        be = (al / (aw + al)) if (aw + al) > 0 else None
        rel = _reliability(n)
        return {
            "win_rate": round(wr, 4), "win_rate_pct": round(wr * 100.0, 1),
            "avg_win_r": round(aw, 4), "avg_loss_r": round(al, 4),
            "expectancy_r": round(expectancy, 4),
            "profit_factor": (round(pf, 4) if pf is not None and math.isfinite(pf) else pf),
            "break_even_win_rate": (round(be, 4) if be is not None else None),
            "break_even_win_rate_pct": (round(be * 100.0, 1) if be is not None else None),
            "gross_profit_r": round(gross_profit, 3),
            "gross_loss_r": round(gross_loss, 3),
            "n_trades": n, "reliability": rel,
            "reliability_fa": _RELIABILITY_FA[rel],
        }

    rs = _extract_r_values(stats)
    n = len(rs)
    rel = _reliability(n)
    if n == 0:
        return {
            "win_rate": None, "win_rate_pct": None,
            "avg_win_r": None, "avg_loss_r": None,
            "expectancy_r": None, "profit_factor": None,
            "break_even_win_rate": None, "break_even_win_rate_pct": None,
            "gross_profit_r": 0.0, "gross_loss_r": 0.0,
            "n_trades": 0, "reliability": rel,
            "reliability_fa": _RELIABILITY_FA[rel],
        }
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    expectancy = sum(rs) / n
    if gross_loss <= 0:
        pf = float("inf") if gross_profit > 0 else None
    else:
        pf = gross_profit / gross_loss
    be = (avg_loss / (avg_win + avg_loss)) if (avg_win + avg_loss) > 0 else None
    return {
        "win_rate": round(win_rate, 4), "win_rate_pct": round(win_rate * 100.0, 1),
        "avg_win_r": round(avg_win, 4), "avg_loss_r": round(avg_loss, 4),
        "expectancy_r": round(expectancy, 4),
        "profit_factor": (round(pf, 4) if pf is not None and math.isfinite(pf) else pf),
        "break_even_win_rate": (round(be, 4) if be is not None else None),
        "break_even_win_rate_pct": (round(be * 100.0, 1) if be is not None else None),
        "gross_profit_r": round(gross_profit, 3),
        "gross_loss_r": round(gross_loss, 3),
        "n_trades": n, "reliability": rel,
        "reliability_fa": _RELIABILITY_FA[rel],
    }


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.upper()}_{timeframe}"


def _load_disk_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": 1, "entries": {}}


def _save_disk_cache(data: dict):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, CACHE_FILE)


def _store_entry(symbol: str, timeframe: str, r_values: list,
                 source: str, split_at=None):
    entry = {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "threshold": config.EV_BACKTEST_THRESHOLD,
        "source": source,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "split_at": split_at,
        "r_values_oos": [round(float(r), 4) for r in r_values],
        "n_trades": len(r_values),
    }
    _MEM_CACHE[(symbol.upper(), timeframe)] = entry
    data = _load_disk_cache()
    data["entries"][_cache_key(symbol, timeframe)] = entry
    _save_disk_cache(data)
    return entry


def _entry_from_memory_or_disk(symbol: str, timeframe: str):
    key = (symbol.upper(), timeframe)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    entry = _load_disk_cache()["entries"].get(_cache_key(symbol, timeframe))
    if entry is not None:
        _MEM_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Saved result files (matrix / single result)
# ---------------------------------------------------------------------------
def _stored_result_paths(symbol: str, timeframe: str) -> list:
    sym = symbol.upper()
    parent = os.path.dirname(_PACKAGE_DIR)   # The crypto_analyzer_v2/ folder
    paths = [
        # Forex matrix (add-on): for forex symbols this is the only saved source;
        # for crypto these files do not exist and have no effect.
        os.path.join(parent, "matrix_forex", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix_forex", f"{sym}_{timeframe}.json"),
        # The final matrix of the improvement program (final stack) comes first: after
        # the program, this is the valid record for each combination; the main matrix remains only as a
        # historical fallback (if the final file does not exist).
        os.path.join(parent, "matrix_final", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix_final", f"{sym}_{timeframe}.json"),
        os.path.join(parent, "matrix", f"{sym}_{timeframe}.json"),
        os.path.join(_PACKAGE_DIR, "matrix", f"{sym}_{timeframe}.json"),
    ]
    if sym == "BTC" and timeframe == "4h":
        paths.append(os.path.join(parent, "btc4h_backtest_result.json"))
        paths.append(os.path.join(_PACKAGE_DIR, "btc4h_backtest_result.json"))
    return paths


def _oos_r_values_from_result(result: dict):
    """Only the segment="out" trades of the EV threshold — never in-sample."""
    th = str(config.EV_BACKTEST_THRESHOLD)
    data = (result.get("thresholds") or {}).get(th)
    if data is None:
        # The base threshold is not in the file; we do not touch any other threshold.
        return None
    trades = data.get("trades") or []
    return [float(t["r"]) for t in trades
            if t.get("segment") == "out" and t.get("r") is not None]


def _entry_from_stored_files(symbol: str, timeframe: str):
    for path in _stored_result_paths(symbol, timeframe):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(result.get("symbol", "")).upper() != symbol.upper() \
                or result.get("timeframe") != timeframe:
            continue
        r_values = _oos_r_values_from_result(result)
        if r_values is None:
            continue
        source = "stored_matrix" if "matrix" in path else "stored_result"
        return _store_entry(symbol, timeframe, r_values, source,
                            split_at=result.get("split_at"))
    return None


def _entry_from_fresh_backtest(symbol: str, timeframe: str):
    """Fresh run of the system backtest (expensive; only on explicit request)."""
    from backtest.system_backtest import run_system_backtest
    result = run_system_backtest(
        symbol, timeframe,
        thresholds=(config.EV_BACKTEST_THRESHOLD,))
    r_values = _oos_r_values_from_result(result)
    if r_values is None:
        r_values = []
    return _store_entry(symbol, timeframe, r_values, "system_backtest",
                        split_at=result.get("split_at"))


# ---------------------------------------------------------------------------
# Main EV API
# ---------------------------------------------------------------------------
def ev_for_candidate(symbol: str, timeframe: str, direction: str = None,
                     refresh: bool = False, allow_compute: bool = True):
    """
    Returns the net out-of-sample EV of one symbol/timeframe combination
    (compute_ev output + metadata). If no source exists and computation is not
    allowed either, returns None.

    direction is accepted for the caller's convenience, but per the limitation
    documented in the module docstring, we have no per-direction statistics: EV is
    the same for both directions and the key direction_specific=False states this
    explicitly.
    refresh=True skips the cache and saved files and takes a fresh backtest
    (explicit cache reset).
    """
    entry = None
    if refresh:
        entry = _entry_from_fresh_backtest(symbol, timeframe)
    else:
        entry = _entry_from_memory_or_disk(symbol, timeframe)
        if entry is None:
            entry = _entry_from_stored_files(symbol, timeframe)
        if entry is None and allow_compute:
            entry = _entry_from_fresh_backtest(symbol, timeframe)
    if entry is None:
        return None
    ev = compute_ev(entry.get("r_values_oos") or [])
    ev.update({
        "symbol": symbol.upper(), "timeframe": timeframe,
        "threshold": entry.get("threshold", config.EV_BACKTEST_THRESHOLD),
        "source": entry.get("source"),
        "computed_at": entry.get("computed_at"),
        "split_at": entry.get("split_at"),
        "direction_specific": False,
    })
    return ev


def refresh_ev_cache(symbols: list, timeframes: list, progress: bool = True) -> dict:
    """
    Explicit EV cache reset: for each combination a fresh backtest is run and
    stored in the cache (slow — about a minute per combination). Output: a
    success/error summary.
    """
    done, failed = [], []
    for symbol in symbols:
        for timeframe in timeframes:
            if progress:
                print(f"در حال بازنشانی EV برای {symbol} {timeframe} ...", flush=True)
            try:
                ev = ev_for_candidate(symbol, timeframe, refresh=True)
                done.append({"symbol": symbol.upper(), "timeframe": timeframe,
                             "n_trades": ev["n_trades"] if ev else 0,
                             "expectancy_r": ev["expectancy_r"] if ev else None})
            except Exception as exc:
                failed.append({"symbol": symbol, "timeframe": timeframe,
                               "error": str(exc)})
    return {"refreshed": done, "errors": failed}


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
def decide_candidate(candidate: dict, ev: dict = None) -> dict:
    """
    Determines a candidate's decision state from (score, direction, stop/target, EV).
    Precise, deterministic rules (all thresholds in config):

    1) No stop loss or target → NO_TRADE (no invalidation point/target = mandatory rejection)
    2) No EV available → NO_TRADE (we do not decide without out-of-sample statistics)
    3) Insufficient sample (n < EV_SAMPLE_INSUFFICIENT_MAX) → NO_TRADE
    4) Net EV ≤ 0 → NO_TRADE (mandatory rejection; overrides any score)
    5) Score ≥ EV_STRONG_SCORE and EV > 0 and a "reliable" sample →
       STRONG_BUY / STRONG_SELL (based on the candidate's direction)
    6) Score ≥ EV_BUY_SCORE and EV > 0 and a "reliable" sample → BUY / SELL
    7) All other EV > 0 cases (score below the buy threshold, or a "low"
       sample with score ≥ the buy threshold) → WAIT (the direction is plausible
       but the conditions are not complete)

    Output: {"decision", "decision_fa", "decision_reason"} — the reason is always
    in Persian and explicit, especially for NO_TRADE.
    """
    if ev is None:
        ev = candidate.get("ev")
    direction = candidate.get("direction") or "long"
    score = candidate.get("score") or 0.0

    def out(decision, reason):
        return {"decision": decision,
                "decision_fa": DECISION_FA[decision],
                "decision_reason": reason}

    if candidate.get("stop") is None or candidate.get("target") is None:
        return out("NO_TRADE",
                   "رد: حد ضرر یا هدف مشخص نیست — بدون نقطه‌ی ابطال و هدف، معامله تعریف نمی‌شود.")
    if ev is None:
        return out("NO_TRADE",
                   "رد: آمار EV خارج‌ازنمونه برای این ترکیب موجود نیست (کش EV خالی است) — "
                   "بدون آمار تصمیم نمی‌گیریم؛ کش را بازنشانی کنید.")
    n = ev.get("n_trades") or 0
    if ev.get("reliability") == "insufficient" or n < config.EV_SAMPLE_INSUFFICIENT_MAX:
        return out("NO_TRADE",
                   f"رد: نمونه‌ی خارج‌ازنمونه ناکافی است ({n} معامله؛ حداقل "
                   f"{config.EV_SAMPLE_INSUFFICIENT_MAX}) — آمار قابل اتکا نیست.")
    expectancy = ev.get("expectancy_r")
    if expectancy is None or expectancy <= 0:
        return out("NO_TRADE",
                   f"رد اجباری: امید ریاضی خالص خارج‌ازنمونه نامثبت است ({expectancy}R ≤ 0) — "
                   "امتیاز بالا این رد را خنثی نمی‌کند.")
    # From here on we have EV > 0 with at least a 'low' sample.
    if ev.get("reliability") == "ok" and score >= config.EV_STRONG_SCORE:
        dec = "STRONG_BUY" if direction == "long" else "STRONG_SELL"
        return out(dec,
                   f"EV خالص مثبت ({expectancy:+}R، {n} معامله‌ی خارج‌ازنمونه) و امتیاز "
                   f"{score}/100 در بازه‌ی قوی است.")
    if ev.get("reliability") == "ok" and score >= config.EV_BUY_SCORE:
        dec = "BUY" if direction == "long" else "SELL"
        return out(dec,
                   f"EV خالص مثبت ({expectancy:+}R، {n} معامله‌ی خارج‌ازنمونه) و امتیاز "
                   f"{score}/100 از آستانه‌ی خرید گذشته است.")
    if ev.get("reliability") != "ok":
        return out("WAIT",
                   f"EV مثبت است ({expectancy:+}R) ولی نمونه کم است ({n} معامله) — "
                   "برای تصمیم خرید/فروش کافی نیست؛ منتظر داده‌ی بیشتر بمانید.")
    return out("WAIT",
               f"EV مثبت است ({expectancy:+}R) ولی امتیاز فعلی ({score}/100) به آستانه‌ی "
               f"خرید ({config.EV_BUY_SCORE:.0f}) نرسیده است.")


def format_ev_fa(ev: dict) -> str:
    """A ready-to-display Persian line for the EV statistics (or a message that they are absent)."""
    if ev is None:
        return "EV: موجود نیست (کش EV خالی)"
    if ev.get("expectancy_r") is None:
        # Cache entry with zero trades (e.g. a fresh forex combination before its
        # backtest produces any trade): there are no statistics to show; before this guard,
        # the :+ formatting crashed on None (a display bug, not a computation bug).
        return (f"EV: هنوز معامله‌ی خارج‌ازنمونه‌ای برای این ترکیب ثبت "
                f"نشده است (نمونه: {ev.get('n_trades', 0)} معامله — "
                f"{ev.get('reliability_fa', 'نمونه ناکافی')})")
    pf = ev.get("profit_factor")
    pf_txt = "∞" if pf == float("inf") else ("—" if pf is None else f"{pf}")
    return (f"EV خالص (خارج‌ازنمونه): {ev['expectancy_r']:+}R | "
            f"نرخ برد: {ev['win_rate_pct']}٪ | نرخ برد سربه‌سر: "
            f"{ev['break_even_win_rate_pct']}٪ | پروفیت فاکتور: {pf_txt} | "
            f"نمونه: {ev['n_trades']} معامله ({ev['reliability_fa']})")
