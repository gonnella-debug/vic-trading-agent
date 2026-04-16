"""
Hyperliquid Copy-Trading Engine for Vic v7
-------------------------------------------
Replaces the 58 TA strategies with a single approach:
  1. Pull HL leaderboard every 5 min → filter top 15 traders
  2. Poll each trader's positions every 60s
  3. Detect new opens/closes → mirror on Vic's wallet
  4. Size using Option C: SL = 2% of Vic's equity
  5. Drawdown killswitch, fee filter, leverage cap from safety layer

Leaderboard source: https://stats-data.hyperliquid.xyz/Mainnet/leaderboard
Position source: POST https://api.hyperliquid.xyz/info clearinghouseState
"""
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger(__name__)

HL_API = "https://api.hyperliquid.xyz/info"
HL_LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

MAX_TRACKED_TRADERS = 15
MIN_ACCOUNT_VALUE = 500_000
MIN_ALLTIME_PNL = 100_000
MIN_MONTH_ROI = 0.0
LEADERBOARD_REFRESH_SECS = 300
POSITION_POLL_SECS = 30
COINS_ALLOWED = None  # None = trade whatever top traders trade


@dataclass
class TrackedTrader:
    address: str
    name: str
    account_value: float
    alltime_pnl: float
    alltime_roi: float
    month_pnl: float
    month_roi: float
    positions: dict = field(default_factory=dict)  # coin → {side, size, entry, leverage}
    last_polled: float = 0.0


@dataclass
class CopyState:
    traders: list = field(default_factory=list)  # list of TrackedTrader
    last_leaderboard_refresh: float = 0.0
    copy_positions: dict = field(default_factory=dict)  # coin → {source_addr, side, size, entry}
    trades_executed: int = 0
    trades_skipped: int = 0


copy_state = CopyState()


async def fetch_leaderboard() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(HL_LEADERBOARD)
            if r.status_code != 200:
                log.error(f"Leaderboard fetch failed: {r.status_code}")
                return []
            data = r.json()
            return data.get("leaderboardRows", [])
    except Exception as e:
        log.error(f"Leaderboard fetch error: {e}")
        return []


def filter_top_traders(rows: list[dict]) -> list[TrackedTrader]:
    candidates = []
    for r in rows:
        av = float(r.get("accountValue", 0) or 0)
        if av < MIN_ACCOUNT_VALUE:
            continue

        perfs = {wp[0]: wp[1] for wp in r.get("windowPerformances", [])}
        at = perfs.get("allTime", {})
        mo = perfs.get("month", {})
        wk = perfs.get("week", {})

        at_pnl = float(at.get("pnl", 0) or 0)
        at_roi = float(at.get("roi", 0) or 0)
        mo_pnl = float(mo.get("pnl", 0) or 0)
        mo_roi = float(mo.get("roi", 0) or 0)
        wk_pnl = float(wk.get("pnl", 0) or 0)

        if at_pnl < MIN_ALLTIME_PNL:
            continue
        if mo_pnl <= MIN_MONTH_ROI:
            continue
        if wk_pnl < 0:
            continue

        candidates.append(TrackedTrader(
            address=r["ethAddress"],
            name=r.get("displayName") or "anon",
            account_value=av,
            alltime_pnl=at_pnl,
            alltime_roi=at_roi,
            month_pnl=mo_pnl,
            month_roi=mo_roi,
        ))

    # Sort by month PnL (who's winning RIGHT NOW) then take top N
    candidates.sort(key=lambda t: t.month_pnl, reverse=True)
    return candidates[:MAX_TRACKED_TRADERS]


async def fetch_trader_positions(address: str) -> dict:
    """Returns {coin: {side, size, entry, leverage}} for a trader."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(HL_API, json={
                "type": "clearinghouseState",
                "user": address,
            })
            if r.status_code != 200:
                return {}
            data = r.json()
            positions = {}
            for p in data.get("assetPositions", []):
                pos = p.get("position", {})
                coin = pos.get("coin", "")
                szi = float(pos.get("szi", 0) or 0)
                if szi == 0 or not coin:
                    continue
                entry_px = float(pos.get("entryPx", 0) or 0)
                leverage_val = float(pos.get("leverage", {}).get("value", 10) if isinstance(pos.get("leverage"), dict) else pos.get("leverage", 10) or 10)
                positions[coin] = {
                    "side": "long" if szi > 0 else "short",
                    "size": abs(szi),
                    "entry": entry_px,
                    "leverage": leverage_val,
                }
            return positions
    except Exception as e:
        log.error(f"Position fetch for {address[:10]}: {e}")
        return {}


async def refresh_leaderboard():
    now = time.time()
    if now - copy_state.last_leaderboard_refresh < LEADERBOARD_REFRESH_SECS and copy_state.traders:
        return

    rows = await fetch_leaderboard()
    if not rows:
        log.warning("Leaderboard returned 0 rows, keeping existing traders")
        return

    new_traders = filter_top_traders(rows)
    if not new_traders:
        log.warning("No traders passed filter, keeping existing list")
        return

    # Preserve position snapshots for traders we're already tracking
    old_positions = {t.address: t.positions for t in copy_state.traders}
    for t in new_traders:
        if t.address in old_positions:
            t.positions = old_positions[t.address]

    copy_state.traders = new_traders
    copy_state.last_leaderboard_refresh = now
    trader_names = ", ".join(f"{t.name}({t.address[:8]})" for t in new_traders[:5])
    log.info(f"Leaderboard refreshed: {len(new_traders)} traders tracked. Top 5: {trader_names}")


def detect_changes(trader: TrackedTrader, new_positions: dict) -> list[dict]:
    """Compare old vs new positions, return list of {action, coin, side, ...}."""
    changes = []
    old = trader.positions

    # New positions or size increases → OPEN signal
    for coin, new_pos in new_positions.items():
        if COINS_ALLOWED and coin not in COINS_ALLOWED:
            continue
        old_pos = old.get(coin)
        if old_pos is None:
            changes.append({
                "action": "open",
                "coin": coin,
                "side": new_pos["side"],
                "entry": new_pos["entry"],
                "leverage": new_pos["leverage"],
                "trader_size": new_pos["size"],
                "trader": trader.address,
                "trader_name": trader.name,
            })
        elif old_pos["side"] != new_pos["side"]:
            # Flipped direction → close old + open new
            changes.append({"action": "close", "coin": coin, "side": old_pos["side"],
                           "trader": trader.address, "trader_name": trader.name})
            changes.append({
                "action": "open", "coin": coin, "side": new_pos["side"],
                "entry": new_pos["entry"], "leverage": new_pos["leverage"],
                "trader_size": new_pos["size"], "trader": trader.address,
                "trader_name": trader.name,
            })

    # Positions that disappeared → CLOSE signal
    for coin, old_pos in old.items():
        if coin not in new_positions:
            changes.append({"action": "close", "coin": coin, "side": old_pos["side"],
                           "trader": trader.address, "trader_name": trader.name})

    return changes


def calc_copy_size(equity: float, entry: float, leverage: int, sl_pct: float = 2.0) -> float:
    """Option C sizing: SL = 2% of equity.
    risk_dollars = equity * 0.02
    sl_distance = entry * sl_pct / 100
    size = risk_dollars / sl_distance
    """
    risk_dollars = equity * 0.02
    sl_distance = entry * sl_pct / 100.0
    if sl_distance <= 0:
        return 0.0
    raw_size = risk_dollars / sl_distance
    # Floor to 5 decimal places (HL precision)
    return math.floor(raw_size * 100000) / 100000


async def poll_traders_once(equity: float, execute_fn=None, tg_fn=None, fee_filter: float = 12.0,
                             max_leverage: int = 10, check_can_trade=None):
    """Poll all tracked traders, detect changes, execute copies.

    Args:
        equity: current account equity in USD
        execute_fn: async fn(coin, side, size, entry, sl, tp, leverage, strategy_name) → bool
        tg_fn: async fn(text) for Telegram notifications
        fee_filter: minimum expected TP gross in USD (rejects trades below this)
        max_leverage: hard cap
        check_can_trade: fn() → (bool, reason) pre-trade gate
    """
    if not copy_state.traders:
        return

    for trader in copy_state.traders:
        try:
            new_positions = await fetch_trader_positions(trader.address)
        except Exception as e:
            log.error(f"Poll {trader.address[:10]}: {e}")
            continue

        if not trader.positions and new_positions:
            # First poll for this trader — snapshot, don't copy (avoid dumping in mid-trade)
            trader.positions = new_positions
            trader.last_polled = time.time()
            continue

        changes = detect_changes(trader, new_positions)
        trader.positions = new_positions
        trader.last_polled = time.time()

        for ch in changes:
            if ch["action"] == "open" and execute_fn:
                coin = ch["coin"]
                side = ch["side"]
                entry = ch["entry"]
                leverage = min(int(ch.get("leverage", 10)), max_leverage)

                # Option C sizing
                size = calc_copy_size(equity, entry, leverage, sl_pct=2.0)
                if size <= 0:
                    copy_state.trades_skipped += 1
                    continue

                # SL = 2% from entry, TP = 4% (2:1 R:R)
                sl_dist = entry * 0.02
                tp_dist = entry * 0.04
                if side == "long":
                    sl = round(entry - sl_dist, 2)
                    tp = round(entry + tp_dist, 2)
                else:
                    sl = round(entry + sl_dist, 2)
                    tp = round(entry - tp_dist, 2)

                # Fee filter
                expected_tp_gross = size * tp_dist
                if expected_tp_gross < fee_filter:
                    log.info(f"COPY SKIP {coin} {side} — TP gross ${expected_tp_gross:.2f} < fee filter ${fee_filter:.2f}")
                    copy_state.trades_skipped += 1
                    if tg_fn:
                        asyncio.create_task(tg_fn(
                            f"⏩ Copy skip: {trader.name}({trader.address[:8]}) opened {side} {coin} "
                            f"but TP too thin vs fees (${expected_tp_gross:.2f} < ${fee_filter:.2f})"
                        ))
                    continue

                # Pre-trade gate
                if check_can_trade:
                    allowed, reason = check_can_trade(f"copy_{trader.address[:8]}_{coin}", side)
                    if not allowed:
                        copy_state.trades_skipped += 1
                        continue

                strategy_name = f"copy_{trader.name}_{coin}"
                log.info(f"COPY OPEN: {trader.name}({trader.address[:8]}) → {side} {coin} "
                         f"size={size} entry=${entry} SL=${sl} TP=${tp} lev={leverage}x")

                if tg_fn:
                    asyncio.create_task(tg_fn(
                        f"📋 Copy trade: {trader.name} ({trader.address[:8]}...) opened {side.upper()} {coin}\n"
                        f"Entry: ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | {leverage}x\n"
                        f"Mirroring with size {size} ({equity * 0.02:.0f} at risk)"
                    ))

                try:
                    success = await execute_fn(
                        coin=coin, side=side, size=size, entry=entry,
                        sl=sl, tp=tp, leverage=leverage, strategy_name=strategy_name
                    )
                    if success:
                        copy_state.trades_executed += 1
                        copy_state.copy_positions[coin] = {
                            "source_addr": trader.address,
                            "side": side,
                            "size": size,
                            "entry": entry,
                        }
                except Exception as e:
                    log.error(f"COPY EXECUTE FAILED: {e}")
                    if tg_fn:
                        asyncio.create_task(tg_fn(f"❌ Copy trade FAILED: {coin} {side} — {e}"))

            elif ch["action"] == "close":
                coin = ch["coin"]
                if coin in copy_state.copy_positions:
                    log.info(f"COPY CLOSE signal: {trader.name} closed {coin}")
                    if tg_fn:
                        asyncio.create_task(tg_fn(
                            f"📋 Copy close signal: {ch['trader_name']} closed {coin}. "
                            f"Vic's position monitor will handle exit."
                        ))
                    # Actual close is handled by position_monitor_loop via SL/TP
                    # or we can force-close here if position still open
                    del copy_state.copy_positions[coin]


async def copy_monitor_loop(equity_fn, execute_fn=None, tg_fn=None,
                             fee_filter: float = 12.0, max_leverage: int = 10,
                             check_can_trade=None, is_paused=None):
    """Main loop: refresh leaderboard + poll traders continuously."""
    log.info("Copy engine monitor loop started")
    await asyncio.sleep(5)  # Let startup complete

    while True:
        try:
            # Always refresh leaderboard + poll positions (even when paused — keeps data fresh)
            await refresh_leaderboard()
            equity = await equity_fn()

            if is_paused and is_paused():
                # Still poll to track trader positions, but don't execute
                for trader in copy_state.traders:
                    try:
                        new_positions = await fetch_trader_positions(trader.address)
                        trader.positions = new_positions
                        trader.last_polled = time.time()
                    except Exception:
                        pass
            else:
                await poll_traders_once(
                    equity=equity,
                    execute_fn=execute_fn,
                    tg_fn=tg_fn,
                    fee_filter=fee_filter,
                    max_leverage=max_leverage,
                    check_can_trade=check_can_trade,
                )
        except Exception as e:
            log.error(f"Copy monitor error: {e}")

        await asyncio.sleep(POSITION_POLL_SECS)


def get_copy_status() -> dict:
    return {
        "tracked_traders": len(copy_state.traders),
        "top_traders": [
            {
                "address": t.address[:12] + "...",
                "name": t.name,
                "account_value": round(t.account_value),
                "month_pnl": round(t.month_pnl),
                "active_positions": len(t.positions),
            }
            for t in copy_state.traders[:5]
        ],
        "copy_positions": copy_state.copy_positions,
        "trades_executed": copy_state.trades_executed,
        "trades_skipped": copy_state.trades_skipped,
        "last_leaderboard_refresh": datetime.fromtimestamp(
            copy_state.last_leaderboard_refresh, tz=timezone.utc
        ).isoformat() if copy_state.last_leaderboard_refresh else None,
    }


async def backtest_copy_engine(days: int = 90, equity: float = 500.0,
                                 fee_per_trade: float = 3.0) -> dict:
    """Backtest: replay recent top-trader position changes.
    Uses current leaderboard snapshot, pulls each trader's recent fills,
    simulates what Vic would have done following them.
    """
    rows = await fetch_leaderboard()
    traders = filter_top_traders(rows)
    if not traders:
        return {"error": "no traders found"}

    total_pnl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    max_dd = 0.0
    peak_eq = equity
    running_eq = equity

    for trader in traders[:10]:  # Backtest top 10
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                start_ms = int((time.time() - days * 86400) * 1000)
                r = await client.post(HL_API, json={
                    "type": "userFillsByTime",
                    "user": trader.address,
                    "startTime": start_ms,
                })
                if r.status_code != 200:
                    continue
                fills = r.json()
                if not isinstance(fills, list):
                    continue

            # Group fills into round trips (open→close pairs)
            open_fill = None
            for f in fills:
                direction = f.get("dir", "")
                coin = f.get("coin", "")
                px = float(f.get("px", 0) or 0)
                sz = float(f.get("sz", 0) or 0)
                fee = float(f.get("fee", 0) or 0)

                if "Open" in direction:
                    open_fill = {"coin": coin, "px": px, "sz": sz, "side": "long" if "Long" in direction else "short"}
                elif "Close" in direction and open_fill and open_fill["coin"] == coin:
                    close_px = px
                    entry_px = open_fill["px"]
                    is_long = open_fill["side"] == "long"

                    # Calculate what Vic would have made at Option C sizing
                    vic_size = calc_copy_size(running_eq, entry_px, 10, 2.0)
                    if vic_size <= 0:
                        open_fill = None
                        continue

                    if is_long:
                        raw_pnl = vic_size * (close_px - entry_px)
                    else:
                        raw_pnl = vic_size * (entry_px - close_px)

                    net_pnl = raw_pnl - (fee_per_trade * 2)  # Round trip fees
                    running_eq += net_pnl
                    total_pnl += net_pnl
                    total_trades += 1

                    if net_pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    if running_eq > peak_eq:
                        peak_eq = running_eq
                    dd = peak_eq - running_eq
                    if dd > max_dd:
                        max_dd = dd

                    open_fill = None

        except Exception as e:
            log.error(f"Backtest for {trader.address[:10]}: {e}")
            continue

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    return {
        "days": days,
        "starting_equity": 500.0,
        "final_equity": round(running_eq, 2),
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / 500 * 100, 1) if total_trades > 0 else 0,
        "traders_analysed": len(traders[:10]),
        "fee_per_trade": fee_per_trade,
    }
