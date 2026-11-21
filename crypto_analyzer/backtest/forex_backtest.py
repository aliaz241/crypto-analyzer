"""
backtest/forex_backtest.py
Forex cost model and running the forex system backtest with the same final
crypto stack (accepted trailing + both/regime score sets + EV gate).

Why a wrapper and not a parameter inside run_system_backtest?
The crypto system engine computes cost inside _trade_r with a percentage fee,
and tampering with it would put crypto behavior/numbers at risk. Here the
simulation of the stack engine itself (improvement_runs/stack_engine — the
same engine that built matrix_final) is reused, with the difference that each
trade's R is recomputed from the raw (entry, exit, risk) with the forex
spread cost:

    cost_price = spread_pips × pip_size  +  slippage_pips × pip_size
    R = (gross price profit − cost_price) / |entry − stop loss|

i.e. round-trip spread + slippage, in price units, is deducted from profit
exactly like the fee in the _trade_r formula. Each symbol's spread:
config.FOREX_SPREAD_PIPS (for XAUUSD the value 2.5 "pips" of 0.1 = 0.25 price
units = 25 cents). Overnight swap/rollover is deliberately not counted
(holding up to 40 candles; recorded as a constraint in FOREX_FA.md).

Crypto numbers are untouched: this module changes no existing function, and
the proof is in the tests + regression (reproducing matrix_final).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backtest.system_backtest import _metrics

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))            # The crypto_analyzer_v2/ folder
_IMPROVEMENT = os.path.join(_ROOT, "improvement_runs")


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def forex_cost_price(symbol: str) -> float:
    """Round-trip cost of one forex trade in price units (spread+slippage)."""
    sym = str(symbol).upper()
    pips = (config.FOREX_SPREAD_PIPS[sym]
            + getattr(config, "FOREX_SLIPPAGE_PIPS", 0.0))
    return pips * config.FOREX_PIP_SIZE[sym]


def forex_trade_r(direction: str, entry: float, exit_price: float,
                  risk: float, symbol: str) -> float:
    """Trade return in R with the forex spread cost (equivalent of _trade_r)."""
    if risk <= 0:
        raise ValueError("risk باید مثبت باشد.")
    gross = (exit_price - entry) if direction == "long" else (entry - exit_price)
    return (gross - forex_cost_price(symbol)) / risk


def reprice_records(records: list, symbol: str) -> list:
    """
    Re-prices the stack engine's compact records ({p, s, mh:{hold:{r, o, entry, exit,
    risk, dir}}}) with the forex cost: only the r key changes; the rest is
    exactly the input record.
    """
    out = []
    for rec in records:
        new = dict(rec)
        mh = {}
        for hold, sim in rec["mh"].items():
            r = forex_trade_r(sim["dir"], sim["entry"], sim["exit"],
                              sim["risk"], symbol)
            mh[hold] = {"r": round(r, 3), "o": sim["o"]}
        new["mh"] = mh
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# EV gate (same semantics as stage1_gate / stage4_matrix)
# ---------------------------------------------------------------------------
def _kept_bars(base_records: list, threshold: int, max_hold: int,
               bar_candles: int = 100) -> set:
    """The bars the gate keeps when on (each bar's statistics only from previous bars)."""
    def sim_of(rec):
        return rec["mh"].get(str(max_hold)) or rec["mh"].get(max_hold)
    by_bar = {}
    for rec in base_records:
        if rec["s"] >= threshold:
            by_bar.setdefault(rec["p"] // bar_candles, []).append(sim_of(rec)["r"])
    kept, prior = set(), []
    for b in sorted(by_bar):
        n = len(prior)
        exp = (sum(prior) / n) if n else None
        if n < config.IMPROVE_GATE_N_MIN or (exp is not None and exp >= 0):
            kept.add(b)
        prior.extend(by_bar[b])
    return kept


def build_forex_result(symbol: str, timeframe: str, sets: dict,
                       meta: dict, df, thresholds=(55, 60, 65),
                       max_hold: int = 40) -> dict:
    """
    From the score sets priced with the forex cost, builds a result in the same
    format as matrix_final: for each threshold, all/in/out metrics + the trade
    log with segment. Score set: 4h → "both", 1d → "regime"
    (exactly the stage4_matrix contract) and the gate on the "base" shadow.
    """
    symbol = symbol.upper()
    sets = {name: reprice_records(recs, symbol) for name, recs in sets.items()}
    score_set = "both" if timeframe == "4h" else "regime"
    split_pos = meta["split_pos"]
    out_thresholds = {}
    for th in thresholds:
        kb = _kept_bars(sets["base"], th, max_hold)
        trades = []
        for rec in sets[score_set]:
            if rec["s"] < th or (rec["p"] // 100) not in kb:
                continue
            sim = rec["mh"].get(str(max_hold)) or rec["mh"].get(max_hold)
            trades.append({
                "entry_index": rec["p"],
                "entry_time": str(df.index[rec["p"]]),
                "segment": "in" if rec["p"] < split_pos else "out",
                "score": rec["s"],
                "outcome": sim["o"],
                "r": sim["r"],
            })
        stripped = [{"r": t["r"], "outcome": t["outcome"]} for t in trades]
        out_thresholds[str(th)] = {
            "all": _metrics(stripped),
            "in_sample": _metrics([{"r": t["r"], "outcome": t["outcome"]}
                                   for t in trades if t["segment"] == "in"]),
            "out_sample": _metrics([{"r": t["r"], "outcome": t["outcome"]}
                                    for t in trades if t["segment"] == "out"]),
            "trades": trades,
        }
    return {
        "symbol": symbol, "timeframe": timeframe, "candles": len(df),
        "evaluations": meta["evaluations"], "eval_errors": meta["eval_errors"],
        "step": config.SYSBT_STEP, "max_hold": max_hold,
        "split_at": str(df.index[split_pos]),
        "stack": {"trailing": config.SYSBT_TRAILING_ADOPTED,
                  "gate": {"mode": "on", "n_min": config.IMPROVE_GATE_N_MIN,
                           "bar_candles": 100,
                           "bookkeeping": "base pipeline shadow trades, prior bars only"},
                  "score_set": score_set,
                  "cost_model": "forex spread (config.FOREX_SPREAD_PIPS) + "
                                "slippage, no swap",
                  "note": "final stack after improvement program, forex cost model"},
        "thresholds": out_thresholds,
    }


def evaluate_forex_coin(symbol: str, df_main, df_daily=None):
    """
    Evaluates one forex combination with the stack engine (improvement_runs/stack_engine)
    and enriches the records with raw fields (entry/exit/risk/dir) for
    forex pricing in build_forex_result.

    Method: the stack engine's _simulate_trade is replaced during evaluation
    with a wrapper that keeps the original signature exactly and only records
    (exit price, entry, risk, direction) of each simulation based on (entry
    index, max_hold); after the evaluation, the raw values are merged into the
    records. The engine's own computation (including the temporary crypto R)
    is untouched and is replaced in the reprice step with the forex cost.
    The stack engine lives in the development repo (improvement_runs), not
    inside the package; if it is not available it raises a clear error (live
    paths do not depend on this function).
    """
    if _IMPROVEMENT not in sys.path:
        sys.path.insert(0, _IMPROVEMENT)
    try:
        import stack_engine
    except ImportError as exc:
        raise RuntimeError(
            "موتور پشته (improvement_runs/stack_engine) در دسترس نیست؛ "
            "بک‌تست فارکس به آن نیاز دارد.") from exc

    raw = {}
    orig_sim = stack_engine._simulate_trade

    def recording_sim(df, entry_pos, direction, entry, stop, target,
                      max_hold, trailing=None):
        xp, oc = orig_sim(df, entry_pos, direction, entry, stop, target,
                          max_hold, trailing=trailing)
        raw[(entry_pos, max_hold)] = {"entry": entry, "exit": xp,
                                      "risk": abs(entry - stop),
                                      "dir": direction}
        return xp, oc

    stack_engine._simulate_trade = recording_sim
    try:
        sets, meta = stack_engine.evaluate_coin(symbol, df_main, df_daily)
    finally:
        stack_engine._simulate_trade = orig_sim

    missing = 0
    for recs in sets.values():
        for rec in recs:
            for hold, sim in rec["mh"].items():
                info = raw.get((rec["p"], int(hold)))
                if info is None:
                    missing += 1
                    continue
                sim.update(info)
    if missing:
        raise RuntimeError(
            f"{missing} شبیه‌سازی بدون فیلد خام ماند؛ قیمت‌گذاری فارکس "
            f"ناقص می‌شود و ادامه امن نیست.")
    return sets, meta
