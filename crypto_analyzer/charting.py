"""
charting.py
Drawing a chart image (PNG) from the candles and the analysis report: candlesticks, EMA20/50,
Bollinger Bands, support/resistance lines, suggested stop losses and a volume sub-chart.

- Dark background, Latin labels (DejaVu) so they stay readable without font/RTL rendering issues.
- Candles are drawn manually with Rectangles and lines (no dependency on mplfinance).
- Any layer whose data is not in the report (None) is skipped and the function does not crash.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from indicators import trend as trend_ind
from indicators import volatility as vol_ind
import scanner as _scanner


_BG = "#0d1117"
_PANEL = "#0d1117"
_GRID = "#30363d"
_TEXT = "#c9d1d9"
_UP = "#26a69a"
_DOWN = "#ef5350"
_EMA20_COLOR = "#ffa726"
_EMA50_COLOR = "#42a5f5"
_BB_COLOR = "#8b949e"
_SUPPORT_COLOR = "#26a69a"
_RESIST_COLOR = "#ef5350"
_SL_LONG_COLOR = "#ffca28"
_SL_SHORT_COLOR = "#ab47bc"


def _fmt_price(value) -> str:
    """Formats the price with suitable precision (also correct for cheap coins like PEPE)."""
    if value is None:
        return "—"
    v = float(value)
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.4f}"
    if v >= 0.01:
        return f"{v:.6f}"
    return f"{v:.10f}".rstrip("0")


def render_chart(df, report: dict, out_path: str, candles: int = None) -> str:
    """
    Draws the chart of the last `candles` candles of df with the report's layers and saves it to out_path.
    Output: the same out_path.
    """
    if candles is None:
        candles = getattr(config, "CHART_CANDLES", 120)
    data = df.tail(candles)
    n = len(data)
    if n == 0:
        raise ValueError("داده‌ای برای رسم نمودار وجود ندارد.")

    x = np.arange(n)
    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    volumes = data["volume"].values if "volume" in data else np.zeros(n)
    colors = [_UP if c >= o else _DOWN for o, c in zip(opens, closes)]

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(1400 / 150, 900 / 150), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    fig.patch.set_facecolor(_BG)
    for a in (ax, axv):
        a.set_facecolor(_PANEL)
        a.tick_params(colors=_TEXT, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(_GRID)
        a.grid(True, color=_GRID, alpha=0.25, linewidth=0.5)

    # --- Candles (manual) ---
    for i in range(n):
        ax.plot([i, i], [lows[i], highs[i]], color=colors[i], linewidth=0.8, zorder=2)
        bottom = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i])
        if height == 0:
            height = (highs[i] - lows[i]) * 0.02 or 1e-12
        ax.add_patch(Rectangle((i - 0.35, bottom), 0.7, height,
                               facecolor=colors[i], edgecolor=colors[i], zorder=3))

    # --- EMA20 / EMA50 (computed on the whole df so the warm-up is right, then tail) ---
    try:
        ema20 = trend_ind.ema(df, 20).tail(n).values
        ema50 = trend_ind.ema(df, 50).tail(n).values
        ax.plot(x, ema20, color=_EMA20_COLOR, linewidth=1.0, label="EMA 20", zorder=4)
        ax.plot(x, ema50, color=_EMA50_COLOR, linewidth=1.0, label="EMA 50", zorder=4)
    except Exception:
        pass

    # --- Bollinger Bands ---
    try:
        bb = vol_ind.bollinger_bands(df).tail(n)
        ax.plot(x, bb["upper"].values, color=_BB_COLOR, linewidth=0.7, alpha=0.8, zorder=4)
        ax.plot(x, bb["lower"].values, color=_BB_COLOR, linewidth=0.7, alpha=0.8, zorder=4)
        ax.fill_between(x, bb["lower"].values, bb["upper"].values,
                        color=_BB_COLOR, alpha=0.07, zorder=1)
    except Exception:
        pass

    # --- Support / resistance (at most 3 on each side, the nearest ones) ---
    sr = (report or {}).get("support_resistance") or {}
    for r in (sr.get("resistance") or [])[:3]:
        lvl = r.get("level")
        if lvl is None:
            continue
        ax.axhline(lvl, color=_RESIST_COLOR, linewidth=0.9, alpha=0.75, zorder=2)
        ax.text(n - 0.5, lvl, f" R {_fmt_price(lvl)}", color=_RESIST_COLOR,
                fontsize=8, va="bottom", ha="right")
    for s in (sr.get("support") or [])[:3]:
        lvl = s.get("level")
        if lvl is None:
            continue
        ax.axhline(lvl, color=_SUPPORT_COLOR, linewidth=0.9, alpha=0.75, zorder=2)
        ax.text(n - 0.5, lvl, f" S {_fmt_price(lvl)}", color=_SUPPORT_COLOR,
                fontsize=8, va="top", ha="right")

    # --- Suggested stop losses (dashed line + label) ---
    for key, color, label in (("stop_loss_long", _SL_LONG_COLOR, "SL Long"),
                              ("stop_loss_short", _SL_SHORT_COLOR, "SL Short")):
        rec = ((report or {}).get(key) or {}).get("recommended") or {}
        sl = rec.get("stop_loss")
        if sl is None:
            continue
        ax.axhline(sl, color=color, linewidth=1.0, linestyle="--", alpha=0.9, zorder=2)
        ax.text(0.5, sl, f" {label} {_fmt_price(sl)}", color=color,
                fontsize=8, va="bottom", ha="left")

    # --- Volume ---
    axv.bar(x, volumes, color=colors, width=0.7)
    axv.set_ylabel("Volume", color=_TEXT, fontsize=9)

    # --- Title and axes ---
    ind = (report or {}).get("indicators") or {}
    price = ind.get("price") if ind.get("price") is not None else closes[-1]
    symbol = str((report or {}).get("symbol", "")).upper()
    timeframe = (report or {}).get("timeframe", "")
    last_ts = str(data.index[-1])[:16]
    ax.set_title(f"{symbol}/USDT · {timeframe} · {_fmt_price(price)} · {last_ts} UTC",
                 color="white", fontsize=13, loc="left")
    ax.set_ylabel("Price (USDT)", color=_TEXT, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, facecolor=_BG, edgecolor=_GRID,
              labelcolor=_TEXT)
    ax.set_xlim(-1, n)

    tick_idx = np.linspace(0, n - 1, 6).astype(int)
    axv.set_xticks(tick_idx)
    axv.set_xticklabels([str(data.index[i])[:10] for i in tick_idx], color=_TEXT)

    # --- Probable-direction arrow + next target level (from the same scanner scoring) ---
    # We read the direction from trend/momentum, the next level from the facing support/resistance.
    # This "probability" is a mechanical bias, not a prediction; the label on the image says the same.
    try:
        cand = _scanner.score_candidate(df, report, symbol, timeframe)
        direction = cand.get("direction")
        target = cand.get("target")
        last_x, last_y = n - 1, float(closes[-1])
        arrow_color = _UP if direction == "long" else _DOWN
        reach = (highs.max() - lows.min()) * 0.10 or last_y * 0.01
        dy = reach if direction == "long" else -reach
        if target is not None:
            dy = float(np.clip(target - last_y, -reach * 1.6, reach * 1.6))
            tcol = _RESIST_COLOR if direction == "long" else _SUPPORT_COLOR
            ax.axhline(target, color=tcol, linewidth=1.1, alpha=0.95, zorder=5)
            ax.text(n - 0.5, target, f" NEXT {_fmt_price(target)}",
                    color=tcol, fontsize=9, va="bottom" if direction == "long" else "top",
                    ha="right", fontweight="bold")
        lbl = f"Likely: {'UP' if direction == 'long' else 'DOWN'} · score {cand['score']}/100"
        ax.annotate(lbl, xy=(last_x + n * 0.04, last_y + dy),
                    xytext=(last_x, last_y),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.6,
                                    mutation_scale=26), zorder=6,
                    color=arrow_color, fontsize=9, fontweight="bold", va="center")
    except Exception:
        pass

    # We limit the vertical range to the price range of the displayed candles (with margin) so that
    # when distant levels/stop losses are in the report, the candles themselves are not crushed; lines outside
    # this range are clipped but their labels are visible at the chart edge.
    pad = (highs.max() - lows.min()) * 0.10 or 1.0
    ax.set_ylim(lows.min() - pad, highs.max() + pad)
    ax.set_xlim(-1, n + n * 0.30)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
