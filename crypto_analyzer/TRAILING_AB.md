# Trailing-Stop (Profit Lock) A/B Test — Result and Decision

**Question:** Suggestion #1 from the public strategy benchmark was that the bot has no "trailing stop", and trades that moved in its favor return unprotected to the initial stop. Does adding trailing to the trading engine improve out-of-sample results?
**Method:** Trailing was added to `backtest/system_backtest.py` as an additive, default-off option; a small pre-declared grid (4 variants) was selected using **in-sample** data only, and the decision was made using **out-of-sample** data only. Every number in this document is the output of a real run (scripts and JSONs in `~/workspace/crypto_analyzer_v2/ab_runs/`).

---

## 1) Paper trailing semantics (existing) vs backtest (new)

| | `paper_trading.py` (pre-existing) | System backtest (this work) |
|---|---|---|
| Parameter | `trailing_pct`: a percentage of **price** | `activation_r` and `trail_r`: in **R** units (risk unit = \|entry − initial stop\|) |
| Activation | From the moment of entry, no threshold (best_price is tracked from entry) | Only when floating profit reaches `activation_r`×R |
| Distance | `best × (1 ∓ pct%)` | `trail_r`×R behind the best price seen |
| Direction | Only in the position's favor (max with the fixed stop) | Only in the trade's favor (ratchet), never worse than the initial stop |
| Timing | Live price (tick) in `evaluate` | Candle-based; updates take effect from the **next candle** (anti-lookahead) |
| Simultaneous conflict | — | Simultaneous touch of stop (fixed/trailing) and target in one candle = **stop first** (the engine's existing rule) |

**Deviation and its reason:** The backtest deliberately did not copy the paper semantics. A percentage distance in paper means the real distance, in R terms, contracts/expands as price moves away from entry, and activation from the moment of entry makes the stop tighter than the structural stop from the very first candle; for the R-based backtest engine, an R-based parameter with an activation threshold is both comparable with the rest of the outputs and implements the "first let the trade reach meaningful profit, then lock it" behavior. This deviation is also recorded in the module's docstring. Costs are the same existing model on both paths (round-trip fee 0.10% + slippage 0.05% per side, the `SYSBT_*` constants).

## 2) A/B protocol

- **Data:** BTC/ETH/XRP/SOL/BNB, 4h timeframe, 1500 Binance candles (2026-03-02 to 2026-11-07) — the same range and protocol as the matrix (evaluation from candle 250, step 4, max hold 40 candles, 60/40 in/out-of-sample split by entry candle index).
- **Thresholds:** 60 = headline (the matrix and EV-layer standard); 55 = robustness check.
- **Pre-declared grid** (in `config.SYSBT_TRAILING_GRID`): activation ∈ {0.5R, 1.0R} × distance ∈ {0.5R, 1.0R} — only 4 variants.
- **Selection rule:** best aggregated **in-sample** expectancy (5 coins, weighted by trade count). At most one variant.
- **Decision rule (pre-declared):** accept only if the selected variant's aggregated out-of-sample expectancy at threshold 60 beats the baseline **and** XRP 4h (the baseline's only positive coin) does not flip deeply negative.
- All variants were simulated on the same evaluations of a single run (entries identical; only exits differ) — so the differences are purely the effect of trailing.

**Reproducibility note (important):** Re-running the baseline today on a fresh candle fetch did not exactly reproduce the matrix number (BTC at threshold 60: −0.527R vs the matrix's −0.596R). The cause was traced and proven: the matrix fetch (Nov 6) and today's fetch are identical in the candles themselves (the only difference was the last, still-unclosed candle at the moment of the benchmark fetch), and the scores were reproduced on the same entry points (12 of 12); the difference comes from the **floating pattern success rates** in `build_report`, which are computed on the growing slice of data, so as "today" advances, the scores of a few borderline points — and therefore the trade set — shift slightly. The decisive control: re-simulating each of the 26 out-of-sample trades from the matrix log with this same engine and this same data returns exactly −0.596R, and the exit counts (8 target / 15 stop / 3 timeout) are identical. Conclusion: the A/B comparison must be **internal** — baseline and variants in one run on one dataset — which is what was done; do not mix this document's numbers with the matrix numbers.

## 3) In-sample selection

Aggregated in-sample expectancy (5 coins):

| Variant (activation/distance) | Threshold 55 (179 trades) | Threshold 60 (114 trades) |
|---|---|---|
| Baseline (no trailing) | −0.132R | −0.112R |
| 0.5R / 0.5R | −0.092R | −0.079R |
| 0.5R / 1.0R | −0.097R | −0.058R |
| **1.0R / 0.5R ← selected** | **−0.068R** | **−0.056R** |
| 1.0R / 1.0R | −0.086R | −0.064R |

All four variants beat the baseline in-sample at both thresholds, and **both thresholds independently selected the same variant: activation 1.0R, distance 0.5R** — the selection is not tied to one particular threshold. (Everything in-sample is still negative; selection means "least losing", not "profitable".)

## 4) Out-of-sample result — threshold 60 (headline)

Aggregated over 5 coins, 84 trades in both columns (identical entries):

| | Baseline | Trailing 1.0R/0.5R |
|---|---|---|
| Expectancy | −0.115R | **−0.057R** |
| Total | −9.63R | −4.81R |
| Win rate | 56.0% | 60.7% |
| Profit factor | 0.714 | (aggregate PF is not reported for variants; the variant log was not saved) |
| Exits | 49 target / 27 stop / 8 timeout | 47 target / 23 stop / 7 timeout / **7 trailing** |

Per coin (out-of-sample, threshold 60):

| Coin | Baseline: n, win, expectancy | Trailing: n, win, expectancy | Trailing exits |
|---|---|---|---|
| BTC | 27, 29.6%, −0.527R | 27, 40.7%, **−0.333R** | 9 target / 11 stop / 3 timeout / 4 trailing; max drawdown 14.29→9.84R; consecutive losses 11→7 |
| ETH | 18, 66.7%, +0.106R | 18, 66.7%, +0.106R (unchanged) | Trailing never activated |
| XRP | 13, 92.3%, +0.602R | 13, 92.3%, **+0.443R** | 10 target / 1 stop / 2 trailing |
| SOL | 12, 50%, −0.289R | 12, 50%, −0.289R (unchanged) | Trailing never activated |
| BNB | 14, 64.3%, −0.119R | 14, 71.4%, **−0.002R** | 10 target / 3 stop / 1 trailing |

Reading: the aggregate improvement comes from BTC (+0.194R per trade and cutting the loss tail) and BNB; ETH and SOL are unaffected (their price paths never gave 1R of floating profit to wake the trailing); XRP gives up −0.159R because two of its winning trades were closed earlier with the profit lock instead of the full target — but at +0.443R it remains the strongest coin in the field, and the "no deep flip to negative" condition passes with room to spare.

## 5) Robustness — threshold 55

Aggregated over 5 coins, 123 trades: baseline −0.137R (total −16.89R, PF 0.658) → trailing −0.067R (total −8.22R). Per coin: BTC −0.418→−0.272R, ETH −0.002→+0.076R, XRP +0.573→+0.465R (stayed positive), SOL unchanged −0.362R, BNB −0.260→−0.099R. The same pattern repeats at the second threshold.

## 6) Exploratory table (all 4 variants, out-of-sample) — for selection transparency only

Aggregated out-of-sample expectancy:

| Variant | Threshold 55 | Threshold 60 |
|---|---|---|
| Baseline | −0.137R | −0.115R |
| 0.5R / 0.5R | −0.057R | −0.059R |
| 0.5R / 1.0R | −0.101R | −0.101R |
| **1.0R / 0.5R (in-sample selected)** | −0.067R | **−0.057R** |
| 1.0R / 1.0R | −0.093R | −0.084R |

Statistical-honesty note: all four variants beat the baseline out-of-sample, so the result is not sensitive to picking one particular variant (it is a directional effect of the mechanism itself, not parameter hunting). And the in-sample selection is also the best out-of-sample at threshold 60; at 55 it is second (−0.067 vs −0.057 for the 0.5/0.5 variant), and the decision is reported based on the pre-selected variant, not the ex-post best.

## 7) Decision: accepted — with a precise interpretation

The pre-declared rule passed: (1) aggregated out-of-sample at threshold 60 beats the baseline (−0.057 vs −0.115R); (2) XRP stayed positive and strong (+0.443R).

**But this acceptance does not mean profitability:** expectancy is still negative in aggregate at both thresholds. Trailing roughly halves the loss and cuts the left tail of the distribution (BTC max drawdown and consecutive losses); it does not create a positive edge. Trailing entered as an "available default option", not as a buy signal.

### What changed in the live paths

1. `config.SYSBT_TRAILING_ADOPTED = {"activation_r": 1.0, "trail_r": 0.5}` — the named constant selected by the A/B.
2. `backtest/system_backtest.py` — additive `trailing` and `include_trailing_grid` parameters; with `trailing=None` (default) the previous behavior is reproduced exactly (verification: a direct run with the selected `trailing=` returned the grid number exactly; regression snapshot: zero change in previous fields, chart hash byte-for-byte identical).
3. `paper_trading.py` — limited, additive alignment: an **automatic** position that has a stop-loss and did not explicitly take a `trailing_pct` now takes the adopted distance as the equivalent percentage at the moment of entry (`0.5 × |entry − stop| / entry × 100`). Manual positions are exactly as before (no trailing unless explicitly requested). Paper's two semantic deviations (percentage-based, no activation threshold) remain and are documented in the `_effective_stop` docstring.
4. **The EV cache was not touched.** The `ev_cache.json` numbers were computed before trailing and are the basis of current EV decisions; after the first `main.py --refresh-ev`, the EVs will be recomputed with the trailing-capable engine and will most likely shift (expected direction: less negative). Until that refresh, the EV layer works with the old numbers — this gap was consciously accepted and recorded here; the cache was not silently rewritten.
5. The Telegram bot was off the whole time and was not turned on.

## 8) Reproduction

- `ab_fetch_candles.py` (fetch and cache candles), `ab_coin_runner.py <SYM>` (run each coin), `ab_aggregate.py` (aggregate and select) — in the root of `crypto_analyzer_v2/`; outputs in `ab_runs/`.
- Tests: `tests/test_trailing_ab.py` — 12 checks (8 trailing-math checks with hand-computed answers: activation, ratchet, anti-lookahead, stop-first conflict, short symmetry, tight distance; + the adopted constant and the paper alignment).
