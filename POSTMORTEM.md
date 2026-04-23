# Postmortem log — vic-trading-agent (deprecated)

**Status:** deprecated 2026-04-19. Live trading happens in `/Users/gg/vic-lite`
via the AYN TradingView webhook. This log exists for repeat-prevention only
— any new bug in this repo should be handled by porting the learning to
vic-lite, not by patching here.

Classes: `schema-drift` | `silent-failure` | `integration-wiring` | `deploy-config` | `external-api`

---

class: silent-failure
date: 2026-04-04
commit: 6c33c32
symptom: Vic entered a tight loop attempting to close a position that no longer existed on Hyperliquid ("phantom position").
cause: Vic's state had the position open; HL had already liquidated/closed it. Close attempts returned None, which Vic interpreted as "retry".
fix: query HL clearinghouseState before every close; if the position isn't there, flip Vic's state to closed and log.
repeat-guard: never trust local state for position existence — always confirm against the exchange's current state before acting.

---

class: silent-failure
date: 2026-04-03
commit: 35563b2
symptom: ghost trades — Vic logged a filled order that never actually filled on HL; PnL accounting drifted from the exchange.
cause: order-status was being read from HL's "ok" response envelope; actual fill status is inside nested statuses array.
fix: validate the nested HL order statuses after every submit; fix BTC szDecimals.
repeat-guard: HL's response envelope acknowledges receipt, not execution. Fill confirmation comes from the order-status lookup, always.

---

class: external-api
date: 2026-04-03
commit: 40a6d5b
symptom: Hyperliquid leaderboard endpoint stopped returning data; copy engine ran against stale cache.
cause: HL migrated the leaderboard from the `info` endpoint to a separate `stats-data` endpoint.
fix: call `stats-data` directly.
repeat-guard: HL endpoint shape is a moving target — any HL integration must tolerate sudden endpoint deprecations with a fallback path.

---

class: silent-failure
date: 2026-04-08
commit: 19aec19
symptom: TP direction inverted on some trades (SL/TP swapped); AI brain vetoed all trades during macro news windows; strategies ran even when gated off.
cause: three interacting bugs — TP direction computed from entry side wrong; AI brain macro filter was too aggressive; strategy gating used a stale list.
fix: TP direction tied to position.is_long boolean explicitly; macro veto narrowed to a short list; strategy gating reads the live `active_strategies` set.
repeat-guard: SL/TP arithmetic tied to direction always uses a boolean, never inferred from the sign of a delta.

---

class: schema-drift
date: 2026-04-08
commit: 493c77f
symptom: persisted state carried old strategy names after a rename; Vic silently ran a strategy whose code had been deleted.
cause: state file held `active_strategies: ["btc_trend_v1", ...]` but code only exported `STRATEGY_NAMES = ["btc_trend_v2", ...]`.
fix: on load, filter persisted active_strategies to the intersection with `STRATEGY_NAMES`.
repeat-guard: any persisted enum/name set must be validated against the current code's enum on load; extras are dropped, not treated as truth.

---

class: silent-failure
date: 2026-04-07
commit: 4470845
symptom: AI brain refused every trade during market news windows, leaving Vic idle for days.
cause: macro filter's threshold was too strict; nearly any calendar entry within 24h triggered a veto.
fix: narrow the filter to high-impact events only and within 2h windows.
repeat-guard: "veto" logic needs a shipped-day baseline: how many trades would have been allowed in the last 30d? If it's <50% of normal, the filter is too tight.

---

class: deploy-config
date: 2026-04-03
commit: f3d3e5e, 746fa5d
symptom: Docker build failed on Railway — port binding and pandas version conflicts.
cause: Dockerfile hard-coded port 8080; Railway provides PORT at runtime. pandas pinned to an incompatible version.
fix: bind to `$PORT`; unpin pandas.
repeat-guard: Railway Dockerfiles always bind `$PORT`; pin only libraries whose API has broken at a specific version.

---
