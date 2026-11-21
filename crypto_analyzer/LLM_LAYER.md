# LLM Decision Layer — Design Document

Status: **core + integration complete** (scanner, CLI and bot display —
details in the "Wiring" section below). This layer is entirely additive and
opt-in; without an API key and without the `--llm` flag, the system's
behavior is exactly as before (zero change — proven by the regression
snapshot and the integration test's deep-compare).

Files:

- `analysis/llm_client.py` — a minimal NVIDIA Build client using only the
  standard library's `urllib` (no new dependency; Termux-friendly).
- `analysis/llm_engine.py` — the decision engine: prompt, numeric
  validation, voting, cross-check and fail-safe orchestration.
- `tests/test_llm_layer.py` — 23 offline tests with a fake client (no
  network; including transport-level retry tests with a fake urlopen):
  `python tests/test_llm_layer.py`
- `tests/test_llm_integration.py` — 18 integration checks with the scanner
  (fake client; objective equality of the off output with the baseline, a
  hard floor on the scan path, cap of 3, fail-safe):
  `python tests/test_llm_integration.py`
- `llm_runs/llm_smoke.py` (at the repo root, next to the package) — a live
  end-to-end smoke test with a real key (reads it from the env).

## The layer's place

The order of authority in this system is top to bottom, and the LLM is the
lowest rung:

```
Out-of-sample backtest → EV layer (decide_candidate) → LLM layer (review)
```

The EV layer decides using real backtest statistics; the LLM layer only
works on candidates that have passed the EV gate, and its job is reasoned
review, scenario building and an independent second opinion — not generating
fresh signals.

## Model selection (constants in `llm_engine.py`)

| Role | Model | Constant |
|---|---|---|
| Primary (voting ×3) | `deepseek-ai/deepseek-v4.1-flash` | `LLM_PRIMARY_MODEL` |
| Independent cross-check | `nvidia/nemotron-3-ultra-550b-a55b` | `LLM_CROSSCHECK_MODEL` |
| Tiebreak | `mistralai/mistral-large-4-1t-a52b` | `LLM_TIEBREAK_MODEL` |
| Spare (not in the path for now) | `z-ai/glm-5.3-flash` | `LLM_SPARE_MODEL` |

All four models were verified on 2026-11-07 against the live NVIDIA Build
catalog with a real chat call (correct answer, `finish_reason=stop`).

⚠️ **Live catalog warning:** the NVIDIA Build catalog changes fast, and the
`/v1/models` list alone is not enough:

- `moonshotai/kimi-k2.6` is listed in `/v1/models`, but its chat call returns
  **404** (API access is not enabled for it) — deliberately not used.
- `deepseek-ai/deepseek-v4-pro` has been retired (410) — not referenced
  anywhere.
- Before relying on any ID, check its existence with a real chat call, not
  just the model list. On a 410 the client raises a separate error
  (`LLMRetiredError`) and does not retry, so a model's death is not mistaken
  for service congestion.

## The hard-floor rule (the most important rule)

If the EV layer's decision for a candidate is `NO_TRADE` (non-positive EV,
insufficient sample, missing EV, or an unclear stop/target),
`analyze_candidate` returns that same `NO_TRADE` and **not even one call is
made to the LLM** — this is proven in the tests by counting the fake
client's calls (0 calls). Under no circumstances can the LLM override a
mandatory EV rejection; even if the scanner score is 95.

## Voting and cross-check mechanics

1. The primary model is called 3 times at temperatures 0.2, 0.5 and 0.8 with
   `max_tokens=4096`. (The initial budget was 16384; it came down in the
   live test of 2026-11-08 — the reason is in the "Live-test history and
   transport resilience" section below. The client timeout is still 600
   seconds.)
2. Majority vote on `decision`. A tie → the tiebreak model decides once
   between the tied options.
3. The cross-check model (a different model family, for independence of
   opinion) is called once. The direction mapping is fixed: buy =
   {STRONG_BUY, BUY}, sell = {STRONG_SELL, SELL}, neutral = {WAIT,
   NO_TRADE}. If the cross-check's direction differs from the majority's
   direction, the result is downgraded by exactly **one step toward WAIT**
   (STRONG_BUY→BUY, BUY→WAIT, STRONG_SELL→SELL, SELL→WAIT) and
   `llm_crosscheck_disagreement=True` is recorded.
4. `llm_confidence` is the mean confidence of the valid runs of the primary
   model.

## Programmatic numeric validation (anti number-invention)

The model must return exactly one JSON object with a fixed contract:
`decision` (from the EV layer's vocabulary), `confidence`, the
`entry/stop/target` levels, a list of `scenarios` (with probabilities from
0 to 1), `key_reasons`, `risks`, and `numbers_used` (a restatement of the
key numbers: price, ev_r, win_rate, score). Every output passes through
`parse_and_validate` before being counted in the vote:

- Every `numbers_used` value must match the candidate's/EV's real number
  within a **relative tolerance of 0.5%**; a number whose reference is not
  in the data = invented = the output is void.
- The entry/stop/target levels must be copies of the candidate's own levels
  (within the same tolerance); the model is not allowed to propose a fresh
  price. If the candidate has no level and the model gives a number, the
  output is void.
- Void outputs are discarded and counted (`llm_invalid_outputs`). If fewer
  than 2 valid outputs remain → the result is `WAIT` with the reason
  `llm_validation_failed`; nothing is guessed.
- Any exception at any point → a fail-safe result with `llm_decision=None`
  and an error note; this layer never throws an exception into the scanner.

## Live-test history and transport resilience (2026-11-08)

An honest record of the live test from this very network (VM), with a real
key and the engine's real prompt:

- **The primary model was unstable:** one full pipeline run took 812
  seconds, but all 3 of the primary model's outputs were rejected in
  validation (their content had not been examined), and 6+ direct single
  calls — at both the 2048 and 16384 budgets — died with
  `RemoteDisconnected` or a read timeout; even a simple prompt with a cap
  of 256 tokens timed out at 240 seconds.
- **The other two models were healthy:** the Nemotron Ultra cross-check
  (29 seconds, decision WAIT) and the Mistral Large 4 tiebreak (96 seconds,
  WAIT) answered on that first attempt with the engine's real prompt and
  passed `parse_and_validate`.
- The engine itself (hard floor, parsing, numeric validation) was confirmed
  healthy live; the problem was only the primary model's reliability at our
  request size.

Two resilience fixes that came out of this very record:

1. **Transport-level retry inside `llm_client.chat`:** at most 3 attempts in
   total; on connection-level errors (RemoteDisconnected, connection error,
   socket timeout) and HTTP 5xx and 429, with stepped backoff of 10 and 30
   seconds. There is no retry on a 410 (`LLMRetiredError` on the very first
   attempt), and the other 4xx errors are also thrown without retry. Result:
   another analysis run no longer has to win each of its ~5 calls in a coin
   flip to succeed. The behavior is proven with fake-transport tests
   (`TestClientTransportRetry` in `tests/test_llm_layer.py`: two drops +
   success = exactly 3 attempts; 410 = exactly 1 attempt; a persistent drop
   = `LLMError` after exactly 3 attempts).
2. **A generation budget of 4096 instead of 16384** (`LLM_MAX_TOKENS`): both
   healthy models comfortably fit the full analysis contract inside 4096;
   shorter generation finishes faster and is less likely to be abandoned
   midway.

On the author's own network the primary model may behave better; these two fixes
make the layer resilient in both cases, rather than tying it to a specific
network.

## Wiring (integration)

- **config.py:** the constants `NVIDIA_API_KEY = ""` (empty = disabled),
  `LLM_ENABLED_DEFAULT = False`, `LLM_SCAN_TOP_K = 3` and `LLM_VOTES = 3`.
  The model IDs are not repeated here; they live only in `llm_engine.py`.
- **scanner.py:** the function `attach_llm_analysis(cand,
  report_data_builder=None, client=None)` following the
  `attach_ev_decision` pattern — an additive `"llm"` key on the candidate.
  Two guards before the engine: the EV decision must exist and must not be
  `NO_TRADE` (the second guard sits on top of the engine's own hard floor).
  `report_data_builder` is an optional function that is called with the
  candidate and builds an additional "computational report" for the prompt;
  its errors are silently skipped. Any LLM error → the candidate untouched
  (fail-safe). `scan_market` and `scan_forex_market` have taken an additive
  `llm=False` parameter (existing callers untouched): with `llm=True`, at
  most the first `LLM_SCAN_TOP_K` candidates of the display slice whose
  decision is not `NO_TRADE` are reviewed, and the output gets an additive
  `"llm_summary"` key (`{"active", "analyzed", "decisions"}`; without a
  key: `{"active": False, "reason": "no_api_key", ...}`). There is also a
  helper function `llm_layer_active()`: it is True only when
  `LLM_ENABLED_DEFAULT` is on and a key resolves — the current default is
  False (activation is per-run via the CLI flag, not global).
- **main.py (CLI):** the `--llm` flag for all three flows:
  `python main.py --scan --llm`, `python main.py --scan-forex --llm`, and
  single-symbol analysis `python main.py --symbol XRP --timeframe 4h --llm`
  (forex: `--source forex --llm`). Without the flag, nothing changes. With
  the flag but no key, exactly one line is printed ("the LLM layer is
  disabled: the NVIDIA key is not set…") and the normal output continues.
  For a single-symbol `NO_TRADE` too, a separate line is printed saying the
  review was not performed because of the EV hard floor. Candidates that
  have a review get the "🧠 بازبینی LLM" (LLM review) block (decision,
  confidence, votes, the cross-check disagreement flag and
  `llm_reason_fa`) in the scan table and the rank-1 details.
- **telegram_bot.py:** display only — the `_llm_lines(c)` function is called
  in the single-coin "suggestion" text and the rank-1 details of both scans:
  if the candidate has an `"llm"` key, a compact block (LLM decision,
  confidence, vote breakdown, the cross-check disagreement line if present,
  and `llm_reason_fa`) is added; without that key, the message text is
  byte-for-byte as before. **A toggle button was deliberately not added:**
  the bot itself never calls the layer (each review is up to 5
  multi-minute reasoning-model calls and does not fit the bot's
  button-driven/live-scan flow); so for now it displays a result that the
  CLI path or a programmatic scan has produced, and activation is from the
  CLI only. If it is ever wanted, its connection point is this same
  `_llm_lines` and `attach_llm_analysis`.
- **paper_trading.py:** untouched — the automatic logic has not changed at
  all (it has its own EV gate and the LLM has no role in it), and manual
  opening is ungated, as before.

## Activation (one line)

1. Open the file `crypto_analyzer/config.py` and paste the key value into
   the constant `NVIDIA_API_KEY = ""` (between the quotes) — that's it.
   Second route: set the `NVIDIA_API_KEY` environment variable (the config
   constant takes precedence).
2. Run with the flag: `python main.py --scan-forex --llm` (the current
   priority is forex), or `--scan --llm`, or single-symbol with `--llm`.
3. Keep the key only there; never send it in a chat and never write it in
   another file. The key is never printed in any error message or output.

If neither is set, `analyze_candidate` returns `None`: the layer is
disabled and the system's behavior stays exactly as before.

## Output

An additive dictionary (the candidate is not touched): `llm_decision`,
`llm_confidence`, `llm_votes`, `llm_crosscheck`,
`llm_crosscheck_disagreement`, `llm_validated`, `llm_reason` (machine code),
`llm_reason_fa` (a short Persian summary), `scenarios`, `risks`,
`llm_models`, and the void-output counter.

## Honest limitations

- **The LLM does not create an edge.** No voting or cross-check turns bad
  backtest statistics good; if a combination's EV is negative, the right
  answer is the EV layer's NO_TRADE and this layer is not even called. This
  layer's value is the quality of reasoning, explicit probabilistic
  scenarios, and an independent second reviewer — not profit.
- All three models are on one service and one catalog; the cross-check's
  "independence" is independence of model family, not of infrastructure.
- Numeric validation stops number fabrication, but wrong reasoning with
  correct numbers is still possible; that is why the majority vote, the
  cross-check and the EV cap are stacked on top of each other.
- Cost and time: each full analysis needs up to 5 reasoning-model calls
  (3 primary + 1 cross-check + at most 1 tiebreak) — suitable for a
  manual/button scan, not for an automatic scan every minute. That is why
  the cap per scan is 3 candidates (`LLM_SCAN_TOP_K`), chosen only from the
  display slice (top-N).
- A live end-to-end test was run on 2026-11-08 from this very network and
  its record is in the "Live-test history and transport resilience" section
  above (engine healthy; primary model unstable). The
  `llm_runs/llm_smoke.py` script is also ready, and with
  `NVIDIA_API_KEY=... python3 llm_runs/llm_smoke.py` it passes one XRP 4h
  candidate with a real cached EV (WAIT, +0.3666R) through the whole path.
- Display bug note (outside this layer, found and fixed during wiring): EV
  cache entries with zero forex trades have `expectancy_r=None`, and the
  `:+` formatting in `format_ev_fa` and the ranking scan's `EV:` line were
  crashing on them (TypeError). A None guard was added in `analysis/ev.py`
  and `main.py`; the EV calculations and decisions did not change at all.
- The current EV sample of most combinations is small (fully reliable ≥30
  trades); in this state, the correct output of this layer is WAIT most of
  the time, and that is not a bug.
