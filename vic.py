"""
Vic — BTC/USDT Perpetual Futures Scalping Agent
Runs 5 strategies simultaneously on OKX via ccxt.
Paper trading mode by default. Production-ready for Railway.
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum

import ccxt.async_support as ccxt
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request, BackgroundTasks
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("vic")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
OKX_API_KEY = os.getenv("OKX_API_KEY", "")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # paper | live
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
RAILWAY_URL = os.getenv("RAILWAY_URL", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTC/USDT:USDT"
LEVERAGE = 5
CAPITAL_PER_STRATEGY = 200.0  # $200 each, $1000 total
RISK_PER_TRADE = 20.0         # stop-loss dollar risk
TP_PER_TRADE = 60.0           # take-profit dollar target (1:3 R:R)
MAX_DAILY_LOSS = 40.0         # per strategy
STRATEGY_NAMES = [
    "tradingview_webhook",
    "ema_crossover",
    "rsi_divergence",
    "vwap_bounce",
    "bb_squeeze",
]
BB_SQUEEZE_THRESHOLD = 0.02   # bandwidth threshold for BB squeeze

# ---------------------------------------------------------------------------
# App & state
# ---------------------------------------------------------------------------
app = FastAPI(title="Vic Trading Agent", version="1.0.0")

exchange: Optional[ccxt.okx] = None
http_client: Optional[httpx.AsyncClient] = None


class TradingState:
    """Global in-memory state for the bot."""

    def __init__(self):
        self.mode: str = TRADING_MODE  # paper | live
        self.paused: bool = False
        self.last_btc_price: float = 0.0
        self.strategies: dict = {}
        for name in STRATEGY_NAMES:
            self.strategies[name] = {
                "daily_pnl": 0.0,
                "trades_today": 0,
                "paused": False,
                "position": None,  # {side, entry, sl, tp, size, open_time}
            }
        self.trade_history: list = []
        self.startup_time: str = ""

    def reset_daily(self):
        for name in STRATEGY_NAMES:
            self.strategies[name]["daily_pnl"] = 0.0
            self.strategies[name]["trades_today"] = 0
            self.strategies[name]["paused"] = False
        log.info("Daily PnL and trade counts reset.")


state = TradingState()

# ---------------------------------------------------------------------------
# Helpers — Telegram
# ---------------------------------------------------------------------------

async def tg_send(text: str):
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                log.error("Telegram send failed: %s", resp.text)
    except Exception as exc:
        log.error("Telegram error: %s", exc)


# ---------------------------------------------------------------------------
# Helpers — Exchange
# ---------------------------------------------------------------------------

async def init_exchange():
    """Create and configure the OKX ccxt instance."""
    global exchange
    exchange = ccxt.okx({
        "apiKey": OKX_API_KEY,
        "secret": OKX_SECRET_KEY,
        "password": OKX_PASSPHRASE,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",   # perpetual futures
        },
    })
    if state.mode == "paper":
        exchange.set_sandbox_mode(True)
    try:
        await exchange.load_markets()
        log.info("OKX exchange connected. Markets loaded.")
    except Exception as exc:
        log.error("Exchange init error (non-fatal, will retry): %s", exc)


async def close_exchange():
    if exchange:
        await exchange.close()


async def fetch_ohlcv(timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV candles and return a DataFrame."""
    try:
        candles = await exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as exc:
        log.error("fetch_ohlcv error: %s", exc)
        return pd.DataFrame()


async def get_btc_price() -> float:
    """Get the current BTC/USDT mark price."""
    try:
        ticker = await exchange.fetch_ticker(SYMBOL)
        return float(ticker.get("last", 0))
    except Exception as exc:
        log.error("get_btc_price error: %s", exc)
        return state.last_btc_price or 0.0


# ---------------------------------------------------------------------------
# Helpers — Position sizing & order execution
# ---------------------------------------------------------------------------

def calc_position_size(entry: float, sl: float) -> float:
    """Calculate BTC position size so that $RISK_PER_TRADE is risked."""
    distance = abs(entry - sl)
    if distance == 0:
        return 0.0
    size_btc = RISK_PER_TRADE / distance
    return round(size_btc, 6)


async def execute_trade(strategy: str, side: str, entry: float):
    """Open a position (paper or live) for the given strategy."""
    strat = state.strategies[strategy]

    # Guard: already in a position
    if strat["position"] is not None:
        log.info("%s — already in a position, skipping.", strategy)
        return

    # Guard: strategy paused
    if strat["paused"] or state.paused:
        log.info("%s — paused, skipping trade.", strategy)
        return

    # Calculate SL / TP
    if side == "long":
        sl = round(entry - (RISK_PER_TRADE / (CAPITAL_PER_STRATEGY * LEVERAGE / entry)), 2)
        tp = round(entry + (TP_PER_TRADE / (CAPITAL_PER_STRATEGY * LEVERAGE / entry)), 2)
    else:
        sl = round(entry + (RISK_PER_TRADE / (CAPITAL_PER_STRATEGY * LEVERAGE / entry)), 2)
        tp = round(entry - (TP_PER_TRADE / (CAPITAL_PER_STRATEGY * LEVERAGE / entry)), 2)

    size = calc_position_size(entry, sl)
    if size <= 0:
        log.warning("%s — invalid size, skipping.", strategy)
        return

    # Open order
    if state.mode == "live":
        try:
            await exchange.set_leverage(LEVERAGE, SYMBOL)
            order = await exchange.create_order(
                symbol=SYMBOL,
                type="market",
                side="buy" if side == "long" else "sell",
                amount=size,
                params={"tdMode": "cross"},
            )
            log.info("%s LIVE order placed: %s", strategy, order.get("id"))
        except Exception as exc:
            log.error("%s — order error: %s", strategy, exc)
            await tg_send(f"⚠️ <b>{strategy}</b> order FAILED: {exc}")
            return
    else:
        log.info("%s PAPER trade: %s %.6f BTC @ %.2f", strategy, side.upper(), size, entry)

    # Record position
    strat["position"] = {
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "size": size,
        "open_time": datetime.now(timezone.utc).isoformat(),
    }
    strat["trades_today"] += 1

    msg = (
        f"📈 <b>NEW {side.upper()}</b> — {strategy}\n"
        f"Entry: ${entry:,.2f}\n"
        f"SL: ${sl:,.2f}  |  TP: ${tp:,.2f}\n"
        f"Size: {size:.6f} BTC\n"
        f"Mode: {state.mode.upper()}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))


async def close_position(strategy: str, exit_price: float, reason: str):
    """Close a position and book PnL."""
    strat = state.strategies[strategy]
    pos = strat["position"]
    if pos is None:
        return

    if pos["side"] == "long":
        pnl = (exit_price - pos["entry"]) * pos["size"]
    else:
        pnl = (pos["entry"] - exit_price) * pos["size"]

    pnl = round(pnl, 2)
    strat["daily_pnl"] = round(strat["daily_pnl"] + pnl, 2)

    # Close on exchange if live
    if state.mode == "live":
        try:
            close_side = "sell" if pos["side"] == "long" else "buy"
            await exchange.create_order(
                symbol=SYMBOL,
                type="market",
                side=close_side,
                amount=pos["size"],
                params={"tdMode": "cross", "reduceOnly": True},
            )
        except Exception as exc:
            log.error("%s — close order error: %s", strategy, exc)

    state.trade_history.append({
        "strategy": strategy,
        "side": pos["side"],
        "entry": pos["entry"],
        "exit": exit_price,
        "pnl": pnl,
        "reason": reason,
        "time": datetime.now(timezone.utc).isoformat(),
    })

    emoji = "✅" if pnl >= 0 else "❌"
    msg = (
        f"{emoji} <b>CLOSED {pos['side'].upper()}</b> — {strategy}\n"
        f"Entry: ${pos['entry']:,.2f}  →  Exit: ${exit_price:,.2f}\n"
        f"PnL: ${pnl:+,.2f}  |  Reason: {reason}\n"
        f"Daily PnL: ${strat['daily_pnl']:+,.2f}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))

    strat["position"] = None

    # Check daily loss limit
    if strat["daily_pnl"] <= -MAX_DAILY_LOSS:
        strat["paused"] = True
        alert = f"🛑 <b>{strategy}</b> hit daily loss limit (${strat['daily_pnl']:+,.2f}). Paused."
        await tg_send(alert)
        log.warning(alert.replace("<b>", "").replace("</b>", ""))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, bandwidth


def vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP from OHLCV DataFrame."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


# ---------------------------------------------------------------------------
# Strategy 1 — TradingView Webhook (handled via endpoint, not loop)
# ---------------------------------------------------------------------------
# Executed directly from the POST /webhook/tradingview endpoint.


# ---------------------------------------------------------------------------
# Strategy 2 — EMA Crossover Scalp (1m)
# ---------------------------------------------------------------------------

async def strategy_ema_crossover():
    name = "ema_crossover"
    strat = state.strategies[name]
    if strat["paused"] or strat["position"] is not None:
        return

    df = await fetch_ohlcv("1m", 50)
    if df.empty or len(df) < 22:
        return

    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi"] = rsi(df["close"], 14)

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    price = float(curr["close"])
    rsi_val = float(curr["rsi"])

    # Long: 9 EMA crosses above 21 EMA, RSI not overbought
    if prev["ema9"] <= prev["ema21"] and curr["ema9"] > curr["ema21"] and rsi_val < 70:
        await execute_trade(name, "long", price)
    # Short: 9 EMA crosses below 21 EMA, RSI not oversold
    elif prev["ema9"] >= prev["ema21"] and curr["ema9"] < curr["ema21"] and rsi_val > 30:
        await execute_trade(name, "short", price)


# ---------------------------------------------------------------------------
# Strategy 3 — RSI Divergence Scalp (5m)
# ---------------------------------------------------------------------------

async def strategy_rsi_divergence():
    name = "rsi_divergence"
    strat = state.strategies[name]
    if strat["paused"] or strat["position"] is not None:
        return

    df = await fetch_ohlcv("5m", 50)
    if df.empty or len(df) < 30:
        return

    df["rsi"] = rsi(df["close"], 14)

    # Look at last 3 swing points (simplified: use last 20 bars)
    close_vals = df["close"].values[-20:]
    rsi_vals = df["rsi"].values[-20:]

    # Find local lows (simplified: compare triplets)
    price_lows = []
    rsi_lows = []
    for i in range(1, len(close_vals) - 1):
        if close_vals[i] < close_vals[i - 1] and close_vals[i] < close_vals[i + 1]:
            price_lows.append((i, close_vals[i], rsi_vals[i]))

    price_highs = []
    for i in range(1, len(close_vals) - 1):
        if close_vals[i] > close_vals[i - 1] and close_vals[i] > close_vals[i + 1]:
            price_highs.append((i, close_vals[i], rsi_vals[i]))

    price_now = float(df["close"].iloc[-1])

    # Bullish divergence: price lower low, RSI higher low
    if len(price_lows) >= 2:
        prev_low = price_lows[-2]
        curr_low = price_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            await execute_trade(name, "long", price_now)
            return

    # Bearish divergence: price higher high, RSI lower high
    if len(price_highs) >= 2:
        prev_high = price_highs[-2]
        curr_high = price_highs[-1]
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            await execute_trade(name, "short", price_now)


# ---------------------------------------------------------------------------
# Strategy 4 — VWAP Bounce (1m)
# ---------------------------------------------------------------------------

async def strategy_vwap_bounce():
    name = "vwap_bounce"
    strat = state.strategies[name]
    if strat["paused"] or strat["position"] is not None:
        return

    df = await fetch_ohlcv("1m", 100)
    if df.empty or len(df) < 20:
        return

    df["vwap"] = vwap(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(curr["close"])
    vwap_val = float(curr["vwap"])

    if np.isnan(vwap_val):
        return

    distance_pct = abs(price - vwap_val) / vwap_val
    if distance_pct > 0.001:
        return  # price not near VWAP

    is_bullish = curr["close"] > curr["open"]
    is_bearish = curr["close"] < curr["open"]

    # Long: price touched VWAP from above and bounced with bullish candle
    if prev["low"] <= vwap_val and is_bullish and price > vwap_val:
        await execute_trade(name, "long", price)
    # Short: price touched VWAP from below and rejected with bearish candle
    elif prev["high"] >= vwap_val and is_bearish and price < vwap_val:
        await execute_trade(name, "short", price)


# ---------------------------------------------------------------------------
# Strategy 5 — Bollinger Band Squeeze Breakout (5m)
# ---------------------------------------------------------------------------

async def strategy_bb_squeeze():
    name = "bb_squeeze"
    strat = state.strategies[name]
    if strat["paused"] or strat["position"] is not None:
        return

    df = await fetch_ohlcv("5m", 50)
    if df.empty or len(df) < 25:
        return

    mid, upper, lower, bandwidth = bollinger_bands(df["close"], 20, 2.0)
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_bw"] = bandwidth

    prev_bw = float(df["bb_bw"].iloc[-2])
    curr_bw = float(df["bb_bw"].iloc[-1])

    if np.isnan(prev_bw) or np.isnan(curr_bw):
        return

    price = float(df["close"].iloc[-1])
    bb_upper = float(df["bb_upper"].iloc[-1])
    bb_lower = float(df["bb_lower"].iloc[-1])

    # Squeeze: previous bandwidth below threshold, current expanding
    if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
        if price > bb_upper:
            await execute_trade(name, "long", price)
        elif price < bb_lower:
            await execute_trade(name, "short", price)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def strategy_monitor_loop():
    """Check strategies 2-5 every 30 seconds."""
    while True:
        try:
            if not state.paused:
                await strategy_ema_crossover()
                await strategy_rsi_divergence()
                await strategy_vwap_bounce()
                await strategy_bb_squeeze()
        except Exception as exc:
            log.error("Strategy monitor error: %s", exc)
        await asyncio.sleep(30)


async def position_monitor_loop():
    """Check open positions for SL/TP every 10 seconds."""
    while True:
        try:
            price = await get_btc_price()
            if price <= 0:
                await asyncio.sleep(10)
                continue
            state.last_btc_price = price

            for name in STRATEGY_NAMES:
                pos = state.strategies[name]["position"]
                if pos is None:
                    continue

                if pos["side"] == "long":
                    if price <= pos["sl"]:
                        await close_position(name, price, "Stop Loss")
                    elif price >= pos["tp"]:
                        await close_position(name, price, "Take Profit")
                else:  # short
                    if price >= pos["sl"]:
                        await close_position(name, price, "Stop Loss")
                    elif price <= pos["tp"]:
                        await close_position(name, price, "Take Profit")
        except Exception as exc:
            log.error("Position monitor error: %s", exc)
        await asyncio.sleep(10)


async def news_sentiment_monitor():
    """Poll for BTC price moves and basic news sentiment every 5 min."""
    prev_price = 0.0
    while True:
        try:
            price = await get_btc_price()
            if price > 0:
                state.last_btc_price = price
                if prev_price > 0:
                    change_pct = abs(price - prev_price) / prev_price * 100
                    if change_pct >= 2.0:
                        state.paused = True
                        msg = (
                            f"🚨 <b>VOLATILITY ALERT</b>\n"
                            f"BTC moved {change_pct:.1f}% in 5 min "
                            f"(${prev_price:,.0f} → ${price:,.0f})\n"
                            f"ALL trading PAUSED."
                        )
                        await tg_send(msg)
                        log.warning(msg.replace("<b>", "").replace("</b>", ""))
                prev_price = price

            # Simple news check via CryptoCompare
            await _check_crypto_news()
        except Exception as exc:
            log.error("News monitor error: %s", exc)
        await asyncio.sleep(300)


async def _check_crypto_news():
    """Lightweight keyword scan from CryptoCompare news API."""
    danger_keywords = [
        "hack", "exploit", "crash", "ban", "regulation", "sec charges",
        "exchange down", "blackswan", "flash crash", "liquidation cascade",
    ]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
            )
            if resp.status_code != 200:
                return
            data = resp.json()
            articles = data.get("Data", [])[:10]
            for article in articles:
                title = article.get("title", "").lower()
                body = article.get("body", "").lower()
                combined = title + " " + body
                for kw in danger_keywords:
                    if kw in combined:
                        published = article.get("published_on", 0)
                        age_minutes = (time.time() - published) / 60
                        if age_minutes < 10:  # only react to news < 10 min old
                            state.paused = True
                            msg = (
                                f"📰 <b>NEWS ALERT</b>: \"{article.get('title', 'N/A')}\"\n"
                                f"Keyword: {kw}\n"
                                f"ALL trading PAUSED."
                            )
                            await tg_send(msg)
                            log.warning("News pause triggered: %s", kw)
                            return
    except Exception as exc:
        log.debug("News fetch error (non-critical): %s", exc)


async def daily_summary_scheduler():
    """Send daily summary at 17:00 UTC (9 PM Dubai)."""
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        log.info("Daily summary scheduled in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            await send_daily_summary()
        except Exception as exc:
            log.error("Daily summary error: %s", exc)


async def send_daily_summary():
    """Build and send the daily summary to Telegram."""
    total_pnl = 0.0
    total_trades = 0
    best = ("none", -999999.0)
    worst = ("none", 999999.0)
    lines = ["📊 <b>DAILY SUMMARY</b>\n"]

    for name in STRATEGY_NAMES:
        s = state.strategies[name]
        pnl = s["daily_pnl"]
        trades = s["trades_today"]
        status = "PAUSED" if s["paused"] else "Active"
        total_pnl += pnl
        total_trades += trades
        if pnl > best[1]:
            best = (name, pnl)
        if pnl < worst[1]:
            worst = (name, pnl)
        lines.append(f"  {name}: ${pnl:+,.2f} ({trades} trades) [{status}]")

    lines.append(f"\n<b>Total PnL:</b> ${total_pnl:+,.2f}")
    lines.append(f"<b>Total trades:</b> {total_trades}")
    lines.append(f"<b>Best:</b> {best[0]} (${best[1]:+,.2f})")
    lines.append(f"<b>Worst:</b> {worst[0]} (${worst[1]:+,.2f})")
    lines.append(f"<b>Mode:</b> {state.mode.upper()}")
    lines.append(f"<b>BTC Price:</b> ${state.last_btc_price:,.2f}")

    await tg_send("\n".join(lines))


async def daily_reset_scheduler():
    """Reset daily PnL and counters at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (tomorrow - now).total_seconds()
        log.info("Daily reset in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)
        state.reset_daily()
        await tg_send("🔄 Daily reset complete. All strategies resumed.")


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    state.startup_time = datetime.now(timezone.utc).isoformat()
    log.info("Vic starting up — mode: %s", state.mode)

    await init_exchange()

    webhook_url = f"{RAILWAY_URL}/webhook/tradingview?token={WEBHOOK_SECRET}" if RAILWAY_URL else "N/A"
    startup_msg = (
        f"🤖 <b>Vic is online</b>\n"
        f"Mode: {state.mode.upper()} trading\n"
        f"Strategies: 5 active\n"
        f"Pair: BTC/USDT perp\n"
        f"Leverage: {LEVERAGE}x\n"
        f"Capital: $1,000 ($200/strategy)\n"
        f"Webhook: {webhook_url}"
    )
    await tg_send(startup_msg)

    asyncio.create_task(strategy_monitor_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())

    log.info("All background tasks started.")


@app.on_event("shutdown")
async def shutdown():
    log.info("Vic shutting down.")
    await tg_send("⚠️ Vic is shutting down.")
    await close_exchange()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {
        "bot": "Vic",
        "status": "running",
        "mode": state.mode,
        "paused": state.paused,
        "uptime_since": state.startup_time,
        "btc_price": state.last_btc_price,
    }


@app.get("/status")
async def full_status():
    strategies_status = {}
    for name in STRATEGY_NAMES:
        s = state.strategies[name]
        strategies_status[name] = {
            "daily_pnl": s["daily_pnl"],
            "trades_today": s["trades_today"],
            "paused": s["paused"],
            "has_position": s["position"] is not None,
            "position": s["position"],
        }
    return {
        "bot": "Vic",
        "mode": state.mode,
        "global_paused": state.paused,
        "btc_price": state.last_btc_price,
        "strategies": strategies_status,
        "total_trades": sum(s["trades_today"] for s in state.strategies.values()),
        "total_pnl": round(sum(s["daily_pnl"] for s in state.strategies.values()), 2),
        "recent_trades": state.trade_history[-20:],
    }


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request, token: str = Query("")):
    """Receive TradingView webhook alerts for Strategy 1."""
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    action = body.get("action", "").lower()
    price = float(body.get("price", 0))
    strategy_label = body.get("strategy", "ayn_indicator")

    if action not in ("long", "short"):
        return JSONResponse(status_code=400, content={"error": "action must be long or short"})

    if price <= 0:
        price = await get_btc_price()

    log.info("TradingView webhook: %s @ %.2f (%s)", action, price, strategy_label)
    await execute_trade("tradingview_webhook", action, price)

    return {"status": "ok", "action": action, "price": price}


@app.post("/go_live")
async def go_live():
    """Switch from paper to live trading."""
    if state.mode == "live":
        return {"status": "already live"}

    state.mode = "live"
    # Re-initialize exchange without sandbox
    await close_exchange()
    await init_exchange()

    await tg_send("🔴 <b>LIVE TRADING ENABLED</b> — Vic is now trading with real funds.")
    log.warning("Switched to LIVE trading mode.")
    return {"status": "live", "warning": "Real money is now at risk."}


@app.post("/pause")
async def pause_trading():
    """Pause all trading globally."""
    state.paused = True
    await tg_send("⏸️ All trading PAUSED by manual command.")
    return {"status": "paused"}


@app.post("/resume")
async def resume_trading():
    """Resume all trading globally."""
    state.paused = False
    await tg_send("▶️ Trading RESUMED by manual command.")
    return {"status": "resumed"}
