# Trading Stack Improvement Program — The Complete File (4 Stages)

Execution date: 2026-11-07. Scope: only the backtest/scanner/EV layer; the
Telegram bot was off for the entire program and is off at the end too.
Version v1 was not touched.

## Overall protocol (announced before seeing any result)

- Each stage: at most 4 pre-announced variants; variant selection only on
  the in-sample segment; the acceptance decision only with the
  out-of-sample (OOS) segment.
- Acceptance rule for each stage: the aggregate OOS expectancy at @60 must
  be better than that stage's baseline (same run), and XRP (the only known
  positive coin) must not flip to deeply negative (operational threshold:
  XRP expectancy must not get worse than −0.1R).
- A stage being rejected means "the stack stays untouched, the result is
  recorded, the program continues" — rejection is also a result.
- Each stage's baseline is "the current stack": from the very start, the
  accepted trailing (activation 1.0R, trail 0.5R) is inside the
  simulations.
- Headline threshold: @60; robustness: @55.
- Floating-baseline warning (documented beforehand): pattern success rates
  inside `build_report` float as the data slice grows; so comparing
  absolute numbers across runs on different days is not valid — all
  comparisons in this file are "within the same run", and each stage has
  its own control.

## Measurement infrastructure (built and verified before stage 1)

- One shared evaluation pass per coin (`stack_engine.evaluate_coin`) which,
  at every evaluation point of the system engine, computes all four score
  sets (base/regime/mtf/both) and the simulation of all four `max_hold`
  values (40/30/20/10) with the accepted trailing; stages 1 to 3 all feed
  from this same cache (`improvement_runs/eval_cache.pkl`), so the entries
  of stages 1 and 3 are exactly identical across variants.
- Cache verification: for all 5 coins, the base score set at both
  thresholds and both segments (in/out of sample) matched the output of
  `run_system_backtest` itself (with trailing) exactly — `ALL MATCH` ×5.
  The regime/mtf/both score sets were also verified on sample points by
  calling `score_candidate` directly (`VERIFY ALL MATCH`).
- Two "behaviorally equivalent" speedups (scoring output byte-for-byte
  identical on multiple sample points, including boundary points): 1)
  pattern rates from a snapshot of the same slice instead of a repeated
  `backtest_all`; 2) `latest_patterns` only over the last 400 candles of
  the slice (the latest output is inherently the last 15 candles).
- Daily direction for MTF: the pipeline's own direction on the last
  "closed" day before the entry time (no future leakage); the daily candle
  still forming is never used. The daily frame is anchored to closed days
  up to 2026-11-06.

## Stage 1 — Walk-forward per-coin EV gate — ✅ Accepted (on mode)

### Design (pre-announced)
- Load = non-overlapping blocks of 100 candles; 15 loads over the
  1500-candle frame.
- Each load's statistics come only from the pipeline's realized trades in
  "the previous loads of the same coin" (shadow ledger: all pipeline
  trades with no gate, kept or not — the only form whose past is fully
  known at decision time).
- Modes: `off` (baseline), `on` (hard: if `n≥8` and the previous expectancy
  `<0` → all of the load's trades are suppressed), `smooth` (soft:
  suppression only if `n≥8` and the previous expectancy `<−0.2R`).
- The choice between `on`/`smooth` only on the "selection segment" = loads
  1..9; the final report on the "evaluation segment" = loads 10..14
  (candle ≥1000; exactly the last 40% of the loads). Note: stages 2 and 3
  use the engine split (candle 900 boundary); numbers are not
  interchangeable between stages — each stage's control is that stage's
  baseline.

### Results @60 (headline)
Selection segment (in-sample):

| Mode | Trades | Expectancy (R) | Total (R) |
|---|---|---|---|
| off | 139 | −0.085 | −11.79 |
| **on** | 104 | **+0.035** | +3.65 |
| smooth | 125 | −0.053 | −6.61 |

In-sample selection: **on** (the only mode with a positive expectancy).

Evaluation segment (out-of-sample):

| Mode | Kept | Suppressed | Expectancy (R) | Total (R) |
|---|---|---|---|---|
| off (baseline) | 59 | 0 | +0.010 | +0.61 |
| **on (chosen)** | 17 | 42 | **+0.064** | **+1.09** |
| smooth (exploratory) | 25 | 34 | +0.072 | +1.81 |

Robustness @55: the selection is again `on` (in-sample −0.037 versus
off −0.096 and smooth −0.123); evaluation: `off` −0.002R (91 trades)
versus `on` +0.137R (27 trades, +3.70R).

Per-coin detail in the evaluation segment @60 (baseline → gate on):

| Coin | Baseline: trades / expectancy / total | Gate: kept / expectancy / total |
|---|---|---|
| BTC | 17 / −0.203 / −3.46R | **0** / — / 0 (all suppressed) |
| ETH | 15 / +0.214 / +3.21R | 10 / −0.069 / −0.69R |
| XRP | 7 / +0.255 / +1.79R | 7 / +0.255 / +1.79R (all kept) |
| SOL | 10 / −0.137 / −1.36R | **0** / — / 0 (all suppressed) |
| BNB | 10 / +0.044 / +0.44R | 0 / — / 0 (all suppressed) |

### The pre-stated health check
The expectation was that the gate would "suppress most of BTC's damage and
keep most of XRP's trades" — and that is exactly what happened: BTC's
accumulated expectancy went negative by load 9 (−0.097R with n=30) and
stayed closed from load 9 to 14; XRP had a positive previous expectancy in
every load (+0.18 to +0.80) and none of its trades were suppressed.

### The honest parts
- `smooth` was slightly better than `on` on the evaluation segment
  (+0.072 versus +0.064); but the selection was made with the
  pre-announced rule from the in-sample segment, and `on` was the chosen
  one — changing the selection after seeing the OOS is exactly the
  overfitting this program was designed to avoid.
- The gate lost money on ETH in the evaluation segment: it suppressed 5
  winning trades and the survivors gave −0.69R (the baseline was +3.21R).
  The gate's profit came from zeroing out BTC and SOL, not from a uniform
  improvement.
- The gate also sacrificed BNB's small +0.44R profit (opportunity cost).
- The naive gate (hindsight, whole-period statistics including the load
  itself — the "overfitting upper bound"): `on` reaches +0.227R; that is,
  looking at the past makes the gate's effect look about 3.5 times larger
  than the walk-forward truth (+0.064R). The honest number is the
  walk-forward one.
- The kept volume is small (17 trades); the +0.054R edge over the baseline
  is narrow. The gate's main value is "removing damage", not building an
  edge.

### Decision
**Accepted: the gate in on mode** (`n≥8` and previous realized expectancy
`<0` → suppression). The XRP clause holds: +0.255R.

---

## Stage 2 — Regime filter + MTF gate — ✅ Accepted (both together)

### Design (pre-announced)
- Variants: `base` (baseline), `regime` (the regime scaling in
  `score_candidate`: trend/momentum in a ranging market, pattern in a
  volatile market), `mtf` (a conflict between the closed daily direction
  and the candidate's direction → score ×0.6), `both`.
- Stage 1's gate (on) was applied identically on all four variants; the
  gate decision is taken from the "base" realizations (the only leak-free
  form).
- ⚠️ Documented difference from stage 1: regime/MTF change the score, so
  the entry set differs between variants; an "identical entries"
  comparison is not possible. The data and evaluation code are the same
  for all.
- Segments: the engine split (in-sample candle <900, out-of-sample ≥900).

### Results @60
In-sample (selection basis): base +0.010 (95), regime −0.041 (62),
mtf +0.036 (49), **both +0.067 (33)** ← chosen: both.

Out-of-sample:

| Variant | Trades | Expectancy (R) | Total (R) | Win % |
|---|---|---|---|---|
| base (baseline) | 26 | +0.145 | +3.77 | 73.1 |
| regime | 21 | +0.114 | +2.39 | 71.4 |
| mtf (exploratory) | 23 | **+0.220** | **+5.07** | 82.6 |
| **both (chosen)** | 19 | **+0.191** | +3.62 | 78.9 |

Per-coin out-of-sample @60: after the gate, in the out-of-sample segment
only ETH and XRP have any trades at all (BTC/SOL/BNB stayed completely
outside the gate). XRP in both: +0.449R (12 trades, +5.39R) — the XRP
clause holds. ETH stayed negative in all four variants (base −0.153 to
both −0.253): the stack's out-of-sample profit in this window is
practically carried entirely by XRP, and ETH is a negative weight — stage
1's gate (whose track record on ETH was positive in-sample) left it open.

Robustness @55: the in-sample selection is again both (+0.086);
out-of-sample: base +0.150, regime +0.158, mtf +0.238, both +0.206 — the
direction of the result is the same at both thresholds.

### The honest parts
- `mtf` alone was better than `both` on the OOS at both thresholds; but
  the pre-announced selection (in-sample) was `both` at both thresholds,
  and the rule accepted that. `mtf-only` is recorded as an exploratory
  observation, not a decision.
- The samples are small (19 to 26 OOS trades), and the sequential
  selections of the stages stacked on each other carry their own stage-wise
  overfitting risk; the broader test is stage 4's final matrix over 20
  coins.

### Decision
**Accepted: both** — the regime filter and the MTF gate both default to on
(with an off switch in the config).

---

## Stage 3 — The timeout rule (max_hold) — ❌ Rejected (40 stays)

### Design (pre-announced)
- Variants: `max_hold ∈ {40 (baseline), 30, 20, 10}` candles; the entries
  are exactly identical (the timeout only changes the exit); the input
  stack = gate on + both scoring; the simulation of all four values had
  been done in the same cached pass.
- Motivation: the benchmark winner (PatternRecognition) took its profit
  from timeout exits after a favorable drift; the hypothesis was that our
  losers ride up to candle 40 and reach the stop.

### Results @60
In-sample: 40: +0.067, **30: +0.114** ← chosen, 20: +0.037, 10: −0.066
(all four over the same 33 trades).
Out-of-sample (the same 19 trades):

| max_hold | Expectancy (R) | Total (R) | Exits |
|---|---|---|---|
| **40 (baseline)** | **+0.191** | **+3.62** | target 13, stop 3, timeout 2, trailing 1 |
| 30 (chosen) | +0.179 | +3.39 | roughly the same mix |
| 20 | +0.158 | +3.00 | timeout 4 |
| 10 (exploratory) | +0.199 | +3.78 | timeout 4 |

@55 shows the same pattern: the in-sample chosen 30 (+0.122) lost on the
OOS (+0.198 versus the baseline's +0.206).

### Why the hypothesis did not transfer
This stack's trades (after the gate and both gates) mostly close before
candle 20 with a target/stop/trailing — in the OOS there are only 2
timeouts at `max_hold` 40; there is nothing to "rescue from riding to the
stop". The differences (at most 0.2R over 19 trades) are inside the noise.

### Decision
**Rejected.** `max_hold` stays at 40; `IMPROVE_SYSBT_MAX_HOLD` stays None
(i.e. the engine's previous constant). The in-sample chosen value (30) was
worse than the baseline on the OOS, and 10 is exploratory only (its
in-sample was the worst; a classic noise inversion).

---

## The final stack after stages 1 to 3

| Layer | Status |
|---|---|
| Scanner base score | Unchanged |
| Trailing stop (1.0R activation, 0.5R trail) | Previously accepted — active in backtest and paper-auto |
| Per-coin walk-forward EV gate | **Accepted (stage 1): on mode, `n≥8`, previous expectancy `<0` → suppression** |
| Regime filter | **Accepted (stage 2): default on** |
| MTF gate (daily conflict ×0.6) | **Accepted (stage 2): default on** |
| max_hold | 40 — unchanged (stage 3 rejected) |

---

## Stage 4 — Validation of the final stack on the full 20-coin matrix

### Execution
- Fresh data from today (2026-11-07) for all 20 coins: 4h (1500 candles)
  and 1d (1000; fewer for young listings) + the closed daily MTF frame
  (anchor 2026-11-06).
- Output format exactly the original matrix: a JSON per combination +
  SUMMARY in `matrix_final/`; 40 combinations, 0 errors. The `trades`
  field is a compact subset (entry_index/entry_time/segment/score/outcome/r)
  — documented in the runner header; the metrics use the engine's same
  `_metrics`.
- On 1d the final score set is "regime" (there is no higher timeframe for
  MTF); the gate was applied on both timeframes.

### Aggregate headline — out-of-sample @60, all 40 combinations together

| | Trades | Total (R) | Aggregate expectancy (R) |
|---|---|---|---|
| Original matrix (old stack) | 257 | **−18.56** | −0.072 |
| Final matrix (final stack) | 83 | **+3.85** | **+0.046** |

Breakdown: 4h went from −18.78R (204 trades) to +2.74R (44 trades); 1d from
+0.22R (53) to +1.11R (39).

### Combinations in detail (out-of-sample @60)
- Combinations with a meaningful sample (n≥10): in the original matrix, 2
  were positive (XRP +0.472 and ETH +0.022 borderline) and **7 negative**
  (including BTC −0.596 with −15.50R). In the final matrix: **1 positive
  (XRP +0.367R, 12 trades, +4.40R, 83.3% win rate) and zero negative**.
- Completely silent combinations (0 out-of-sample trades): rose from 4 to
  **17**. BTC, SOL, BNB, DOGE, AVAX, INJ, SUI, ONDO and the rest on 4h no
  longer take any trades at all — the worst combination in this bot's
  history (BTC −15.50R) reached complete silence.
- The raw count of "positive combinations" dropped from 19 to 15 — and
  this is not misleading if its composition is stated: most of the
  original matrix's positives were 1-to-6-trade noise on 1d; the final
  stack either silences them or shrinks their sample. The only positive
  that stood with a real sample in both matrices is XRP 4h.
- Opportunity costs and flips (honest): WIF 4h (original: +0.363, 8
  trades) went silent; ETH 4h flipped from +0.022 (19 trades) to −0.418 (7
  trades) — in the final run, ETH's surviving trades lost (in stage 2 too,
  ETH was negative in every variant; this coin is the stack's persistent
  weak point); FET 4h flipped from +0.428 (7) to −0.306 (5). All three are
  small samples.
- ⚠️ The standing caveat: this comparison is between two separate runs
  (fresh data + floating pattern rates) — it is "indicative", not exact.
  The rigorous evidence is the within-run A/Bs of stages 1 to 3; but the
  direction and size of the change (from −18.6R to +3.9R in aggregate, with
  the complete removal of high-sample negative combinations) are
  consistent with that evidence.

### Resetting the EV cache with the final stack
The EV cache was rebuilt from the out-of-sample logs of `matrix_final`
(source="matrix_final" for all 40 entries; only the R values with
segment="out" at @60 — never in-sample). Decision class at the combination
level, before ← after:

| Class | Before the program | After the program |
|---|---|---|
| NO_TRADE (non-positive expectancy) | 8 | 0 |
| NO_TRADE (insufficient sample) | 30 | 39 |
| WAIT | 2 | 1 |
| BUY-eligible (expectancy >0 and n≥30) | 0 | 0 |

- 9 combinations changed class — all on 4h: all 8 "proven loser"
  combinations (including BTC/SOL/BNB/AVAX/INJ/SUI) moved from
  "non-positive expectancy" to "insufficient sample", because the final
  stack takes 0 to 5 out-of-sample trades on them; ETH 4h lost its WAIT.
- The only remaining WAIT: **XRP 4h** (+0.367R, 12 trades).
- ⚠️ An important honest note: the gate no longer produces the sample that
  proved those combinations were bad; their current NO_TRADE rests on "no
  trades", not on fresh negative evidence. The negative evidence is in
  place in the original matrix files, and the cache is read from
  `matrix_final`; if the gate is ever opened, these combinations must be
  viewed with the same caution as before.
- No combination reached BUY-eligible: the final stack is inherently
  low-trade, and no combination has 30 out-of-sample trades. That means,
  in the current state, the automatic path (paper-auto) can in practice
  only open when a candidate from XRP 4h arrives with a score ≥60, and
  even then it goes no higher than WAIT — automatic buying under the
  current EV rules (BUY requires n≥30) is still not possible, and that is
  deliberate.

---

## Wiring (only the accepted mechanisms)

| Mechanism | Wiring |
|---|---|
| Gate (stage 1) | `config.IMPROVE_GATE_MODE = "on"` (+ the constants `N_MIN=8` and the smooth edge). Live form: the EV layer — each combination's realized statistics are read from the cache (final matrix), and the mandatory rules of `decide_candidate` (`n<10` or expectancy ≤0 → `NO_TRADE`) give the same result at all gate suppression points; the additive annotation `ev_gate` in `scanner.attach_ev_decision` explicitly records the gate's state on every candidate (mode/n_min/trailing_n/trailing_expectancy/suppressed). Difference between the forms: the backtest gate is load-by-load and walk-forward; the live form is combination-by-combination and static between cache resets, and is stricter in the sample-size dimension (n≥10/30 versus the gate's n≥8). This is the closest honest equivalent without future leakage; manual paper stayed ungated, as always, and auto inherits from the decision. |
| Regime + MTF (stage 2) | `IMPROVE_REGIME_DEFAULT = True` and `IMPROVE_MTF_DEFAULT = True`; `scan_market` now resolves `use_mtf=None`/`apply_regime=None` from the config, and an explicit False always turns them off. The MTF daily direction is computed without the regime (the semantics of the stages' engine). |
| Timeout (stage 3) | Rejected — `IMPROVE_SYSBT_MAX_HOLD = None` stayed; `max_hold` is the engine's constant 40. |
| EV file path | `matrix_final/` came first in `_stored_result_paths`, so the cache fallback reads from the final stack's record; the original matrix is the historical fallback. |

## Regression and tests
- Regression snapshot (`baseline_before.json`, the same cached candles):
  the `build_report` layer **zero change**; the chart PNG hash
  **byte-for-byte identical** (sha256: 62286dba…, 133,038 bytes);
  `score_candidate` directly: zero non-additive change (only the previously
  known `sizing` keys). The scan layer's differences are exactly and only
  these: the `ev/decision` keys (the EV layer, now from the `matrix_final`
  cache), the `ev_gate` key (stage 1), the `regime`/`mtf` keys and the
  score change of two 1d candidates (BTC 60.4→57.4 and ETH 26.3→24.2 — the
  `range` regime scaling; stage 2) — i.e. exactly the accepted mechanisms,
  one by one.
- Tests: paper 26/26, ported 26/26, EV 45/45, trailing 12/12, the
  improvement program's new tests **21/21**
  (`tests/test_improvement_program.py`: the accepted constants, the gate
  annotation's semantics and its consistency with the mandatory `NO_TRADE`,
  resolving `scan_market` defaults from the config and the explicit-off
  override), offline demo exit 0.

---

## Final verdict

**The stack, before and after (within-run A/B evidence):**
- Stage 1 (gate): the evaluation segment @60 went from +0.010R (59 trades)
  to +0.064R (17 kept trades); BTC's and SOL's damage in that segment was
  completely removed; @55 the direction was even stronger (−0.002 →
  +0.137R).
- Stage 2 (regime+MTF on the gate): out-of-sample @60 from +0.145R to
  +0.191R (the in-sample chosen one); `mtf` alone was exploratorily better
  (+0.220R) and, per the rule, was not selected.
- Stage 3 (timeout): rejected — the 40-candle baseline stayed best.
- Stage 4 (wide matrix): the out-of-sample aggregate of all 40 combinations
  went from −18.56R to +3.85R; high-sample negative combinations: 7 →
  **0**.

**Did the program create a positive out-of-sample expectancy anywhere?**
Yes, but in exactly one place with a real sample: **XRP 4h** (+0.367R, 12
out-of-sample trades in the final matrix; it was positive in all three of
today's independent runs: +0.472 original matrix, +0.449 stage 2, +0.367
final matrix). The overall aggregate also turned positive (+0.046R over 83
trades) — but its carrier is almost only XRP, and the rest of it is "not
trading", not a fresh edge.

**The bot's honest state right now:** this program did not make the bot
"profitable"; it turned it from a loss-making signal-producing machine into
an almost silent system that is only willing to give a cautiously positive
opinion (WAIT, not BUY) on one combination (XRP 4h). Three of the four
tested layers were accepted exactly per the pre-announced rules, and the
fourth was rejected; none of the accepted ones built a fresh positive edge
— all three "removed damage". Until the out-of-sample sample of XRP (or
another combination) with this same stack reaches n≥30, the system's
correct decision is NO_TRADE/WAIT, and manual paper trades (ungated, as
as requested by the author) are the only open path to building that sample. The
Telegram bot is off at the end of the program.
