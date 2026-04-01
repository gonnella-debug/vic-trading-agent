"""
Vic — BTC/USDT Perpetual Futures Scalping Agent (v2)
Runs 4 strategies simultaneously on OKX via ccxt.
Paper trading mode by default. Production-ready for Railway.

Strategies:
  1. TradingView Webhook (filtered with regime + bias + confirmations)
  2. RSI Divergence ($300 capital)
  3. BB Squeeze Breakout ($300 capital)
  4. VWAP Bounce ($200 capital)

Regime filter: TRENDING / RANGING / TRANSITIONAL / VOLATILE
1H bias: Strong bullish/bearish or neutral
Risk: $20 per trade, 1:3 R:R, max 4 trades/day, -3R daily cap
"""

import os
import asyncio
import logging
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum

import ccxt.async_support as ccxt
import httpx
import numpy as np
import pandas as pd
import ta
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
RISK_PER_TRADE = 20.0         # stop-loss dollar risk ($1R)
TP_PER_TRADE = 60.0           # take-profit dollar target (1:3 R:R)
MAX_DAILY_LOSS = 60.0         # -3R total across all strategies
MAX_TRADES_PER_DAY = 4
CORRELATION_COOLDOWN_MIN = 15  # minutes cooldown same direction after close
BREAK_EVEN_R_MULTIPLE = 1.5   # move SL to entry at +1.5R
BB_SQUEEZE_THRESHOLD = 0.02   # bandwidth threshold for BB squeeze

# Strategy capital allocations
STRATEGY_CAPITAL = {
    "tv_webhook": 200.0,       # shared reserve
    "rsi_divergence": 300.0,
    "bb_squeeze": 300.0,
    "vwap_bounce": 200.0,
}

# Strategy max hold times in minutes
STRATEGY_MAX_HOLD = {
    "tv_webhook": 120,
    "rsi_divergence": 240,
    "bb_squeeze": 90,
    "vwap_bounce": 60,
}

STRATEGY_NAMES = ["tv_webhook", "rsi_divergence", "bb_squeeze", "vwap_bounce"]

# Paper test: 50 trades OR 2 weeks
PAPER_TEST_MIN_TRADES = 50
PAPER_TEST_MAX_DAYS = 14

# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    TRANSITIONAL = "TRANSITIONAL"
    VOLATILE = "VOLATILE"

# ---------------------------------------------------------------------------
# App & state
# ---------------------------------------------------------------------------
app = FastAPI(title="Vic Trading Agent", version="2.0.0")

exchange: Optional[ccxt.okx] = None


class TradingState:
    """Global in-memory state for the bot."""

    def __init__(self):
        self.mode: str = TRADING_MODE
        self.paused: bool = False
        self.last_btc_price: float = 0.0
        self.regime: Regime = Regime.RANGING
        self.htf_bias: str = "neutral"  # "bullish", "bearish", "neutral"
        self.htf_bias_strength: str = "weak"  # "strong", "weak"

        # Candle-close tracking
        self.last_1m_candle_ts: float = 0.0
        self.last_5m_candle_ts: float = 0.0

        # Trade frequency
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.daily_loss_cap_hit: bool = False

        # Correlation filter
        self.last_trade_direction: Optional[str] = None
        self.last_trade_close_time: Optional[datetime] = None

        # Strategy states
        self.strategies: dict = {}
        for name in STRATEGY_NAMES:
            self.strategies[name] = {
                "daily_pnl": 0.0,
                "trades_today": 0,
                "paused": False,
                "position": None,  # {side, entry, sl, tp, size, open_time, capital, be_moved}
            }

        # Metrics tracking per strategy
        self.metrics: dict = {}
        for name in STRATEGY_NAMES:
            self.metrics[name] = {
                "wins": 0,
                "losses": 0,
                "total_r_achieved": 0.0,  # sum of actual R multiples
                "trade_count": 0,
                "max_drawdown": 0.0,
                "peak_pnl": 0.0,
                "current_pnl": 0.0,
                "current_losing_streak": 0,
                "max_losing_streak": 0,
            }

        self.trade_history: list = []
        self.startup_time: str = ""
        self.total_trade_count: int = 0  # lifetime for paper test

        # Cache for OHLCV data
        self._ohlcv_cache: dict = {}

    def reset_daily(self):
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.daily_loss_cap_hit = False
        self.last_trade_direction = None
        self.last_trade_close_time = None
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
    config = {
        "apiKey": OKX_API_KEY,
        "secret": OKX_SECRET_KEY,
        "password": OKX_PASSPHRASE,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    }
    if state.mode == "paper":
        config["sandbox"] = True
    exchange = ccxt.okx(config)
    try:
        await exchange.load_markets()
        log.info(f"OKX exchange connected ({'sandbox' if state.mode == 'paper' else 'live'}). Markets loaded.")
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
# Indicators
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, bandwidth


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP from OHLCV DataFrame."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX using ta library."""
    indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period)
    return indicator.adx()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using ta library."""
    indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period)
    return indicator.average_true_range()


# ---------------------------------------------------------------------------
# Structure Analysis (Change 1)
# ---------------------------------------------------------------------------

def detect_structure(df: pd.DataFrame) -> str:
    """
    Check last 20 candles for HH/HL (bullish) or LH/LL (bearish) structure.
    Returns: 'bullish', 'bearish', or 'mixed'
    """
    if len(df) < 20:
        return "mixed"

    recent = df.tail(20)
    highs = recent["high"].values
    lows = recent["low"].values

    # Find swing highs and swing lows (local extremes using triplets)
    swing_highs = []
    swing_lows = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "mixed"

    # Check last 2 swing highs and lows
    hh = swing_highs[-1] > swing_highs[-2]  # higher high
    hl = swing_lows[-1] > swing_lows[-2]     # higher low
    lh = swing_highs[-1] < swing_highs[-2]   # lower high
    ll = swing_lows[-1] < swing_lows[-2]     # lower low

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    else:
        return "mixed"


# ---------------------------------------------------------------------------
# Regime Filter (Change 1)
# ---------------------------------------------------------------------------

async def update_regime():
    """
    Recalculate market regime from 5m candles.
    TRENDING = ADX > 25 AND structure aligned
    TRANSITIONAL = ADX > 25 but structure breaking
    RANGING = ADX < 20 AND BB bandwidth < 0.03
    VOLATILE = ATR% > 0.4% OR price moved > 1.5% in 30 mins
    """
    df_5m = await fetch_ohlcv("5m", 100)
    if df_5m.empty or len(df_5m) < 30:
        return

    # Volatility check: price moved > 1.5% in last 30 mins (6 x 5m candles)
    if len(df_5m) >= 6:
        price_now = float(df_5m["close"].iloc[-1])
        price_30m_ago = float(df_5m["close"].iloc[-6])
        move_pct = abs(price_now - price_30m_ago) / price_30m_ago * 100
        if move_pct > 1.5:
            state.regime = Regime.VOLATILE
            return

    # ATR% check
    atr_series = calc_atr(df_5m, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    price = float(df_5m["close"].iloc[-1])
    atr_pct = (atr_val / price * 100) if price > 0 else 0.0
    if atr_pct > 0.4:
        state.regime = Regime.VOLATILE
        return

    # ADX
    adx_series = calc_adx(df_5m, 14)
    adx_val = float(adx_series.iloc[-1]) if not adx_series.empty and not np.isnan(adx_series.iloc[-1]) else 0.0

    # Structure
    structure = detect_structure(df_5m)

    # BB bandwidth
    _, _, _, bw = calc_bollinger_bands(df_5m["close"], 20, 2.0)
    bw_val = float(bw.iloc[-1]) if not bw.empty and not np.isnan(bw.iloc[-1]) else 0.05

    if adx_val > 25:
        if structure in ("bullish", "bearish"):
            state.regime = Regime.TRENDING
        else:
            state.regime = Regime.TRANSITIONAL
    elif adx_val < 20 and bw_val < 0.03:
        state.regime = Regime.RANGING
    else:
        state.regime = Regime.TRANSITIONAL

    log.debug("Regime: %s | ADX: %.1f | Structure: %s | BW: %.4f", state.regime.value, adx_val, structure, bw_val)


# ---------------------------------------------------------------------------
# 1H Bias (Change 3)
# ---------------------------------------------------------------------------

async def update_htf_bias():
    """
    Compute 1H bias using EMA(50) slope + RSI.
    Strong: clear EMA slope + RSI > 55 or < 45
    Weak/neutral: flat slope or RSI 45-55
    """
    df_1h = await fetch_ohlcv("1h", 60)
    if df_1h.empty or len(df_1h) < 55:
        state.htf_bias = "neutral"
        state.htf_bias_strength = "weak"
        return

    ema50 = calc_ema(df_1h["close"], 50)
    rsi_series = calc_rsi(df_1h["close"], 14)

    ema_now = float(ema50.iloc[-1])
    ema_prev = float(ema50.iloc[-5])  # 5 bars ago for slope
    rsi_val = float(rsi_series.iloc[-1])

    if np.isnan(ema_now) or np.isnan(ema_prev) or np.isnan(rsi_val):
        state.htf_bias = "neutral"
        state.htf_bias_strength = "weak"
        return

    ema_slope_pct = (ema_now - ema_prev) / ema_prev * 100
    slope_clear = abs(ema_slope_pct) > 0.05  # at least 0.05% move over 5 bars

    if slope_clear and ema_slope_pct > 0 and rsi_val > 55:
        state.htf_bias = "bullish"
        state.htf_bias_strength = "strong"
    elif slope_clear and ema_slope_pct < 0 and rsi_val < 45:
        state.htf_bias = "bearish"
        state.htf_bias_strength = "strong"
    elif ema_slope_pct > 0 and rsi_val > 50:
        state.htf_bias = "bullish"
        state.htf_bias_strength = "weak"
    elif ema_slope_pct < 0 and rsi_val < 50:
        state.htf_bias = "bearish"
        state.htf_bias_strength = "weak"
    else:
        state.htf_bias = "neutral"
        state.htf_bias_strength = "weak"

    log.debug("HTF Bias: %s (%s) | EMA slope: %.4f%% | RSI: %.1f", state.htf_bias, state.htf_bias_strength, ema_slope_pct, rsi_val)


# ---------------------------------------------------------------------------
# VWAP Distance & Chop Filters (Change 4)
# ---------------------------------------------------------------------------

def vwap_distance_ok(df: pd.DataFrame, vwap_series: pd.Series) -> bool:
    """
    Price must have moved >= 0.2% away from VWAP before returning.
    Returns True if the distance condition is met.
    """
    if len(df) < 20 or len(vwap_series) < 20:
        return False

    recent_closes = df["close"].values[-20:]
    recent_vwap = vwap_series.values[-20:]

    max_distance_pct = 0.0
    for i in range(len(recent_closes)):
        if np.isnan(recent_vwap[i]):
            continue
        dist = abs(recent_closes[i] - recent_vwap[i]) / recent_vwap[i] * 100
        if dist > max_distance_pct:
            max_distance_pct = dist

    return max_distance_pct >= 0.2


def vwap_chop_filter(df: pd.DataFrame, vwap_series: pd.Series) -> bool:
    """
    No trade if VWAP touched > 3 times in last 20 candles.
    Returns True if NOT in chop zone (OK to trade).
    """
    if len(df) < 20 or len(vwap_series) < 20:
        return True

    touch_count = 0
    for i in range(-20, 0):
        if i >= len(df) or np.isnan(vwap_series.iloc[i]):
            continue
        low = df["low"].iloc[i]
        high = df["high"].iloc[i]
        vw = vwap_series.iloc[i]
        if low <= vw <= high:
            touch_count += 1

    return touch_count <= 3


# ---------------------------------------------------------------------------
# Pre-Trade Checklist (Change 15)
# ---------------------------------------------------------------------------

def bias_allows_direction(side: str) -> bool:
    """
    Check if 1H bias allows this trade direction.
    Strong bias: enforce direction only.
    Weak/neutral: allow both directions.
    """
    if state.htf_bias_strength == "weak" or state.htf_bias == "neutral":
        return True
    if state.htf_bias == "bullish" and side == "long":
        return True
    if state.htf_bias == "bearish" and side == "short":
        return True
    # Strong bias opposes this direction
    return False


def regime_allows_strategy(strategy: str) -> bool:
    """
    Check if current regime allows this strategy.
    TRENDING: VWAP pullback + BB Squeeze
    RANGING: RSI Divergence + VWAP Bounce
    TRANSITIONAL: RSI Divergence only
    VOLATILE: nothing
    """
    if state.regime == Regime.VOLATILE:
        return False
    if state.regime == Regime.TRENDING:
        return strategy in ("vwap_bounce", "bb_squeeze", "tv_webhook")
    if state.regime == Regime.RANGING:
        return strategy in ("rsi_divergence", "vwap_bounce", "tv_webhook")
    if state.regime == Regime.TRANSITIONAL:
        return strategy in ("rsi_divergence", "tv_webhook")
    return False


def correlation_filter_ok(side: str) -> bool:
    """
    After any trade closes: no new trade in SAME direction for 15 minutes.
    """
    if state.last_trade_direction is None or state.last_trade_close_time is None:
        return True
    if side != state.last_trade_direction:
        return True
    elapsed = (datetime.now(timezone.utc) - state.last_trade_close_time).total_seconds() / 60.0
    return elapsed >= CORRELATION_COOLDOWN_MIN


def can_execute_trade(strategy: str, side: str) -> tuple[bool, str]:
    """
    Master pre-trade checklist. Returns (allowed, reason).
    """
    # 2. Regime allows this strategy?
    if not regime_allows_strategy(strategy):
        return False, f"Regime {state.regime.value} blocks {strategy}"

    # 3. 1H bias agrees or is neutral?
    if not bias_allows_direction(side):
        return False, f"1H bias ({state.htf_bias} strong) opposes {side}"

    # 7. Trade count < 4 today?
    if state.trades_today >= MAX_TRADES_PER_DAY:
        return False, f"Daily trade cap reached ({state.trades_today}/{MAX_TRADES_PER_DAY})"

    # 8. No recent same-direction trade?
    if not correlation_filter_ok(side):
        return False, f"Correlation cooldown: {side} blocked for {CORRELATION_COOLDOWN_MIN}min"

    # 9. Daily loss cap not hit?
    if state.daily_loss_cap_hit:
        return False, "Daily loss cap hit (-$60)"

    # 10. Strategy not paused?
    strat = state.strategies[strategy]
    if strat["paused"] or state.paused:
        return False, f"{strategy} or global paused"

    # 11. No existing position on this strategy?
    if strat["position"] is not None:
        return False, f"{strategy} already has an open position"

    return True, "OK"


# ---------------------------------------------------------------------------
# Position sizing & order execution
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
    allowed, reason = can_execute_trade(strategy, side)
    if not allowed:
        log.info("%s — trade blocked: %s", strategy, reason)
        return

    capital = STRATEGY_CAPITAL.get(strategy, 200.0)

    # Calculate SL / TP based on risk
    # Size the SL so that $20 is at risk
    # notional = capital * leverage, position_btc = notional / entry
    # For a $20 risk: sl_distance = $20 / position_size_btc
    # But we want consistent $20 risk, so: sl_distance = RISK_PER_TRADE / (capital * LEVERAGE / entry)
    position_notional = capital * LEVERAGE
    position_btc = position_notional / entry
    sl_distance = RISK_PER_TRADE / position_btc
    tp_distance = TP_PER_TRADE / position_btc

    if side == "long":
        sl = round(entry - sl_distance, 2)
        tp = round(entry + tp_distance, 2)
    else:
        sl = round(entry + sl_distance, 2)
        tp = round(entry - tp_distance, 2)

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
    strat = state.strategies[strategy]
    strat["position"] = {
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "size": size,
        "open_time": datetime.now(timezone.utc).isoformat(),
        "capital": capital,
        "be_moved": False,  # break-even not yet moved
    }
    strat["trades_today"] += 1
    state.trades_today += 1
    state.total_trade_count += 1

    strategy_labels = {
        "tv_webhook": "1️⃣ TradingView",
        "rsi_divergence": "2️⃣ RSI Divergence",
        "bb_squeeze": "3️⃣ BB Squeeze",
        "vwap_bounce": "4️⃣ VWAP Bounce",
    }
    label = strategy_labels.get(strategy, strategy)
    arrow = "🟢" if side == "long" else "🔴"
    msg = (
        f"{arrow} <b>{side.upper()}</b> — {label}\n"
        f"Entry ${entry:,.2f} | SL ${sl:,.2f} | TP ${tp:,.2f}\n"
        f"Regime: {state.regime.value} | Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
        f"Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}"
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
    state.daily_pnl = round(state.daily_pnl + pnl, 2)

    # Calculate actual R achieved
    r_achieved = pnl / RISK_PER_TRADE if RISK_PER_TRADE > 0 else 0.0

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

    # Update metrics (Change 17)
    m = state.metrics[strategy]
    m["trade_count"] += 1
    m["total_r_achieved"] += r_achieved
    m["current_pnl"] += pnl
    if pnl >= 0:
        m["wins"] += 1
        m["current_losing_streak"] = 0
    else:
        m["losses"] += 1
        m["current_losing_streak"] += 1
        if m["current_losing_streak"] > m["max_losing_streak"]:
            m["max_losing_streak"] = m["current_losing_streak"]

    if m["current_pnl"] > m["peak_pnl"]:
        m["peak_pnl"] = m["current_pnl"]
    drawdown = m["peak_pnl"] - m["current_pnl"]
    if drawdown > m["max_drawdown"]:
        m["max_drawdown"] = drawdown

    # Record trade history
    state.trade_history.append({
        "strategy": strategy,
        "side": pos["side"],
        "entry": pos["entry"],
        "exit": exit_price,
        "pnl": pnl,
        "r_achieved": round(r_achieved, 2),
        "reason": reason,
        "time": datetime.now(timezone.utc).isoformat(),
    })

    # Correlation filter update
    state.last_trade_direction = pos["side"]
    state.last_trade_close_time = datetime.now(timezone.utc)

    strategy_labels = {
        "tv_webhook": "1️⃣ TradingView",
        "rsi_divergence": "2️⃣ RSI Divergence",
        "bb_squeeze": "3️⃣ BB Squeeze",
        "vwap_bounce": "4️⃣ VWAP Bounce",
    }
    label = strategy_labels.get(strategy, strategy)
    emoji = "💰" if pnl >= 0 else "💸"
    msg = (
        f"{emoji} <b>CLOSED</b> — {label}\n"
        f"${pos['entry']:,.2f} → ${exit_price:,.2f} | <b>${pnl:+,.2f}</b> ({r_achieved:+.1f}R)\n"
        f"Reason: {reason}\n"
        f"Strategy today: ${strat['daily_pnl']:+,.2f} | Total today: ${state.daily_pnl:+,.2f}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))

    strat["position"] = None

    # Check daily loss cap (Change 14: -3R = -$60 total)
    if state.daily_pnl <= -MAX_DAILY_LOSS:
        state.daily_loss_cap_hit = True
        alert = (
            f"🛑 <b>Daily loss limit hit (${state.daily_pnl:+,.2f})</b>\n"
            f"ALL trading stopped until tomorrow."
        )
        await tg_send(alert)
        log.warning(alert.replace("<b>", "").replace("</b>", ""))

    # Paper test milestone check (Change 16)
    if state.mode == "paper" and state.total_trade_count >= PAPER_TEST_MIN_TRADES:
        await send_paper_test_report()


# ---------------------------------------------------------------------------
# Strategy 1 — TradingView Webhook (Change 8)
# ---------------------------------------------------------------------------
# Handled via POST /webhook/tradingview endpoint with regime + bias + confirmations.


async def evaluate_webhook_signal(action: str, price: float) -> tuple[bool, str]:
    """
    Evaluate webhook signal with regime + bias + at least 1 confirmation.
    Returns (execute, reason).
    """
    side = action.lower()

    # Regime check
    if not regime_allows_strategy("tv_webhook"):
        return False, f"Regime {state.regime.value} blocks webhook"

    # Bias check
    if not bias_allows_direction(side):
        return False, f"1H bias ({state.htf_bias}) opposes {side}"

    # Need at least 1 confirmation: RSI agreement, structure alignment, or near VWAP
    confirmations = 0
    reasons = []

    # RSI agreement
    df_5m = await fetch_ohlcv("5m", 30)
    if not df_5m.empty and len(df_5m) >= 14:
        rsi_series = calc_rsi(df_5m["close"], 14)
        rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
        if side == "long" and rsi_val < 65:
            confirmations += 1
            reasons.append(f"RSI={rsi_val:.0f} agrees")
        elif side == "short" and rsi_val > 35:
            confirmations += 1
            reasons.append(f"RSI={rsi_val:.0f} agrees")

    # Structure alignment
    if not df_5m.empty:
        structure = detect_structure(df_5m)
        if (side == "long" and structure == "bullish") or (side == "short" and structure == "bearish"):
            confirmations += 1
            reasons.append(f"Structure={structure}")

    # Near VWAP
    df_1m = await fetch_ohlcv("1m", 100)
    if not df_1m.empty and len(df_1m) >= 20:
        vwap_s = calc_vwap(df_1m)
        vwap_val = float(vwap_s.iloc[-1])
        if not np.isnan(vwap_val) and price > 0:
            dist_pct = abs(price - vwap_val) / vwap_val * 100
            if dist_pct < 0.3:
                confirmations += 1
                reasons.append(f"Near VWAP ({dist_pct:.2f}%)")

    if confirmations == 0:
        return False, "No confirmations (RSI/structure/VWAP)"

    return True, f"Confirmed: {', '.join(reasons)}"


# ---------------------------------------------------------------------------
# Strategy 2 — RSI Divergence (5m) (Change 10)
# ---------------------------------------------------------------------------

async def strategy_rsi_divergence():
    name = "rsi_divergence"

    df = await fetch_ohlcv("5m", 50)
    if df.empty or len(df) < 30:
        return

    df["rsi"] = calc_rsi(df["close"], 14)
    df["vwap"] = calc_vwap(df)

    close_vals = df["close"].values[-20:]
    rsi_vals = df["rsi"].values[-20:]
    high_20 = np.nanmax(close_vals)
    low_20 = np.nanmin(close_vals)

    # Find local lows (triplets)
    price_lows = []
    for i in range(1, len(close_vals) - 1):
        if close_vals[i] < close_vals[i - 1] and close_vals[i] < close_vals[i + 1]:
            price_lows.append((i, close_vals[i], rsi_vals[i]))

    price_highs = []
    for i in range(1, len(close_vals) - 1):
        if close_vals[i] > close_vals[i - 1] and close_vals[i] > close_vals[i + 1]:
            price_highs.append((i, close_vals[i], rsi_vals[i]))

    price_now = float(df["close"].iloc[-1])
    vwap_now = float(df["vwap"].iloc[-1]) if not np.isnan(df["vwap"].iloc[-1]) else price_now

    # Support/resistance context check (Change 10)
    near_support = (price_now - low_20) / low_20 * 100 < 0.3 if low_20 > 0 else False
    near_resistance = (high_20 - price_now) / high_20 * 100 < 0.3 if high_20 > 0 else False
    away_from_vwap = abs(price_now - vwap_now) / vwap_now * 100 > 0.2 if vwap_now > 0 else False
    context_ok = near_support or near_resistance or away_from_vwap

    if not context_ok:
        return

    # Bullish divergence: price lower low, RSI higher low
    if len(price_lows) >= 2:
        prev_low = price_lows[-2]
        curr_low = price_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            allowed, reason = can_execute_trade(name, "long")
            if allowed:
                await execute_trade(name, "long", price_now)
            return

    # Bearish divergence: price higher high, RSI lower high
    if len(price_highs) >= 2:
        prev_high = price_highs[-2]
        curr_high = price_highs[-1]
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            allowed, reason = can_execute_trade(name, "short")
            if allowed:
                await execute_trade(name, "short", price_now)


# ---------------------------------------------------------------------------
# Strategy 3 — BB Squeeze Breakout (5m) (Change 9)
# ---------------------------------------------------------------------------

async def strategy_bb_squeeze():
    name = "bb_squeeze"

    df = await fetch_ohlcv("5m", 50)
    if df.empty or len(df) < 25:
        return

    mid, upper, lower, bandwidth = calc_bollinger_bands(df["close"], 20, 2.0)
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

    if np.isnan(bb_upper) or np.isnan(bb_lower):
        return

    # Squeeze: previous bandwidth below threshold, current expanding
    if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
        # Change 9: candle must CLOSE at least 0.1% beyond the band
        upper_threshold = bb_upper * 1.001
        lower_threshold = bb_lower * 0.999

        if price > upper_threshold:
            allowed, reason = can_execute_trade(name, "long")
            if allowed:
                await execute_trade(name, "long", price)
        elif price < lower_threshold:
            allowed, reason = can_execute_trade(name, "short")
            if allowed:
                await execute_trade(name, "short", price)


# ---------------------------------------------------------------------------
# Strategy 4 — VWAP Bounce (1m) (Changes 4, 11)
# ---------------------------------------------------------------------------

async def strategy_vwap_bounce():
    name = "vwap_bounce"

    df = await fetch_ohlcv("1m", 100)
    if df.empty or len(df) < 20:
        return

    df["vwap"] = calc_vwap(df)
    df["rsi"] = calc_rsi(df["close"], 14)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(curr["close"])
    vwap_val = float(curr["vwap"])

    if np.isnan(vwap_val):
        return

    # VWAP distance filter (Change 4, 11): must have moved >= 0.2% away before returning
    if not vwap_distance_ok(df, df["vwap"]):
        return

    # Chop filter (Change 4, 11): no trade if VWAP touched > 3 times in last 20 candles
    if not vwap_chop_filter(df, df["vwap"]):
        return

    # RSI must be 40-60 (mean reversion zone)
    rsi_val = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
    if rsi_val < 40 or rsi_val > 60:
        return

    # Volume confirmation: volume > 1.3x 20-period average
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    curr_vol = float(curr["volume"])
    if np.isnan(avg_vol) or curr_vol < avg_vol * 1.3:
        return

    # Strong body check
    body = abs(curr["close"] - curr["open"])
    full_range = curr["high"] - curr["low"]
    if full_range == 0 or body / full_range < 0.5:
        return

    is_bullish = curr["close"] > curr["open"]
    is_bearish = curr["close"] < curr["open"]

    # Wick touch check
    wick_touched_vwap = prev["low"] <= vwap_val <= prev["high"]
    if not wick_touched_vwap:
        return

    # Long: price touched VWAP and bounced with bullish candle
    if is_bullish and price > vwap_val:
        allowed, reason = can_execute_trade(name, "long")
        if allowed:
            await execute_trade(name, "long", price)
    # Short: price touched VWAP and rejected with bearish candle
    elif is_bearish and price < vwap_val:
        allowed, reason = can_execute_trade(name, "short")
        if allowed:
            await execute_trade(name, "short", price)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def regime_update_loop():
    """Recalculate regime and HTF bias every 30 seconds."""
    while True:
        try:
            await update_regime()
            await update_htf_bias()
        except Exception as exc:
            log.error("Regime update error: %s", exc)
        await asyncio.sleep(30)


async def strategy_monitor_loop():
    """
    Check strategies only on candle close boundaries (Change 5).
    1m strategies (vwap_bounce): check when new 1m candle closes
    5m strategies (rsi_divergence, bb_squeeze): check when new 5m candle closes
    """
    while True:
        try:
            if state.paused or state.daily_loss_cap_hit:
                await asyncio.sleep(5)
                continue

            now = datetime.now(timezone.utc)
            current_ts = now.timestamp()

            # Check if a new 1m candle has closed
            # 1m candle closes at each full minute
            current_1m_boundary = int(current_ts // 60) * 60
            new_1m_close = current_1m_boundary > state.last_1m_candle_ts

            # Check if a new 5m candle has closed
            current_5m_boundary = int(current_ts // 300) * 300
            new_5m_close = current_5m_boundary > state.last_5m_candle_ts

            if new_1m_close:
                state.last_1m_candle_ts = current_1m_boundary
                # 1m strategies
                try:
                    await strategy_vwap_bounce()
                except Exception as exc:
                    log.error("vwap_bounce error: %s", exc)

            if new_5m_close:
                state.last_5m_candle_ts = current_5m_boundary
                # 5m strategies
                try:
                    await strategy_rsi_divergence()
                except Exception as exc:
                    log.error("rsi_divergence error: %s", exc)
                try:
                    await strategy_bb_squeeze()
                except Exception as exc:
                    log.error("bb_squeeze error: %s", exc)

        except Exception as exc:
            log.error("Strategy monitor error: %s", exc)

        # Check every 5 seconds so we don't miss candle closes
        await asyncio.sleep(5)


async def position_monitor_loop():
    """
    Check open positions for SL/TP, break-even (Change 12),
    and max hold time (Change 13) every 10 seconds.
    """
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

                side = pos["side"]
                entry = pos["entry"]
                sl = pos["sl"]
                tp = pos["tp"]
                size = pos["size"]

                # Calculate unrealized PnL
                if side == "long":
                    unrealized_pnl = (price - entry) * size
                else:
                    unrealized_pnl = (entry - price) * size

                # Break-even check (Change 12): at +1.5R, move SL to entry
                be_threshold = RISK_PER_TRADE * BREAK_EVEN_R_MULTIPLE  # $30
                if not pos["be_moved"] and unrealized_pnl >= be_threshold:
                    pos["sl"] = entry
                    pos["be_moved"] = True
                    sl = entry
                    msg = f"🔄 <b>{name}</b> — SL moved to break-even (entry ${entry:,.2f})"
                    await tg_send(msg)
                    log.info(msg.replace("<b>", "").replace("</b>", ""))

                # Max hold time check (Change 13)
                max_hold_min = STRATEGY_MAX_HOLD.get(name, 120)
                open_time = datetime.fromisoformat(pos["open_time"])
                elapsed_min = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
                if elapsed_min >= max_hold_min:
                    await close_position(name, price, f"Max hold time ({max_hold_min}min)")
                    continue

                # SL/TP check
                if side == "long":
                    if price <= sl:
                        await close_position(name, price, "Stop Loss")
                    elif price >= tp:
                        await close_position(name, price, "Take Profit")
                else:
                    if price >= sl:
                        await close_position(name, price, "Stop Loss")
                    elif price <= tp:
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
                        if age_minutes < 10:
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
    """Build and send the daily summary with metrics (Change 17) to Telegram."""
    total_pnl = 0.0
    total_trades = 0
    lines = ["📊 <b>DAILY SUMMARY</b>\n"]

    for name in STRATEGY_NAMES:
        s = state.strategies[name]
        m = state.metrics[name]
        pnl = s["daily_pnl"]
        trades = s["trades_today"]
        status = "PAUSED" if s["paused"] else "Active"
        total_pnl += pnl
        total_trades += trades

        win_rate = (m["wins"] / m["trade_count"] * 100) if m["trade_count"] > 0 else 0.0
        avg_r = (m["total_r_achieved"] / m["trade_count"]) if m["trade_count"] > 0 else 0.0

        lines.append(
            f"  <b>{name}</b>: ${pnl:+,.2f} ({trades} trades) [{status}]\n"
            f"    WR: {win_rate:.0f}% | Avg R: {avg_r:+.2f} | "
            f"MaxDD: ${m['max_drawdown']:,.2f} | "
            f"Streak: {m['current_losing_streak']}/{m['max_losing_streak']}"
        )

    lines.append(f"\n<b>Total PnL:</b> ${total_pnl:+,.2f}")
    lines.append(f"<b>Total daily PnL:</b> ${state.daily_pnl:+,.2f}")
    lines.append(f"<b>Total trades today:</b> {state.trades_today}/{MAX_TRADES_PER_DAY}")
    lines.append(f"<b>Lifetime trades:</b> {state.total_trade_count}")
    lines.append(f"<b>Regime:</b> {state.regime.value}")
    lines.append(f"<b>1H Bias:</b> {state.htf_bias} ({state.htf_bias_strength})")
    lines.append(f"<b>Mode:</b> {state.mode.upper()}")
    lines.append(f"<b>BTC Price:</b> ${state.last_btc_price:,.2f}")
    lines.append(f"<b>Daily loss cap:</b> {'HIT' if state.daily_loss_cap_hit else 'OK'}")

    await tg_send("\n".join(lines))


async def send_paper_test_report():
    """Send full performance report after 50 paper trades (Change 16)."""
    lines = ["📋 <b>PAPER TEST REPORT — {0} TRADES COMPLETED</b>\n".format(state.total_trade_count)]

    overall_wins = 0
    overall_losses = 0
    overall_pnl = 0.0

    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        if m["trade_count"] == 0:
            lines.append(f"  <b>{name}</b>: No trades")
            continue

        win_rate = m["wins"] / m["trade_count"] * 100
        avg_r = m["total_r_achieved"] / m["trade_count"]
        overall_wins += m["wins"]
        overall_losses += m["losses"]
        overall_pnl += m["current_pnl"]

        lines.append(
            f"  <b>{name}</b>:\n"
            f"    Trades: {m['trade_count']} | Wins: {m['wins']} | Losses: {m['losses']}\n"
            f"    Win Rate: {win_rate:.1f}%\n"
            f"    Avg R Achieved: {avg_r:+.2f}\n"
            f"    Max Drawdown: ${m['max_drawdown']:,.2f}\n"
            f"    Max Losing Streak: {m['max_losing_streak']}\n"
            f"    Net PnL: ${m['current_pnl']:+,.2f}"
        )

    total_trades = overall_wins + overall_losses
    overall_wr = (overall_wins / total_trades * 100) if total_trades > 0 else 0.0
    lines.append(f"\n<b>OVERALL:</b>")
    lines.append(f"  Total: {total_trades} | WR: {overall_wr:.1f}% | PnL: ${overall_pnl:+,.2f}")

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

    # Initialize candle boundary tracking
    now_ts = datetime.now(timezone.utc).timestamp()
    state.last_1m_candle_ts = int(now_ts // 60) * 60
    state.last_5m_candle_ts = int(now_ts // 300) * 300

    await init_exchange()

    webhook_url = f"{RAILWAY_URL}/webhook/tradingview?token={WEBHOOK_SECRET}" if RAILWAY_URL else "N/A"
    startup_msg = (
        f"🤖 <b>Vic is online — {state.mode.upper()} MODE</b>\n"
        f"BTC/USDT perp | {LEVERAGE}x leverage\n\n"
        f"<b>4 Strategies Active:</b>\n"
        f"1️⃣ <b>TradingView Webhook</b> — Filtered: regime + bias + confirmations ($200 shared)\n"
        f"2️⃣ <b>RSI Divergence</b> — Price/RSI divergence at S/R on 5m ($300)\n"
        f"3️⃣ <b>BB Squeeze</b> — Bollinger squeeze breakout on 5m ($300)\n"
        f"4️⃣ <b>VWAP Bounce</b> — VWAP bounce with distance + chop filter on 1m ($200)\n\n"
        f"$20 risk/trade | 1:3 R:R | Max 4 trades/day\n"
        f"Daily loss cap: -$60 (auto-stops all)\n"
        f"Break-even at +1.5R | Candle-close entries only\n\n"
        f"Paper test: {PAPER_TEST_MIN_TRADES} trades or {PAPER_TEST_MAX_DAYS} days\n\n"
        f"Webhook: {webhook_url}"
    )
    await tg_send(startup_msg)

    asyncio.create_task(regime_update_loop())
    asyncio.create_task(strategy_monitor_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())

    log.info("All background tasks started.")


@app.on_event("shutdown")
async def shutdown():
    log.info("Vic shutting down.")
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
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "uptime_since": state.startup_time,
        "btc_price": state.last_btc_price,
        "trades_today": state.trades_today,
        "daily_pnl": state.daily_pnl,
        "daily_loss_cap_hit": state.daily_loss_cap_hit,
    }


@app.get("/status")
async def full_status():
    strategies_status = {}
    for name in STRATEGY_NAMES:
        s = state.strategies[name]
        m = state.metrics[name]
        win_rate = (m["wins"] / m["trade_count"] * 100) if m["trade_count"] > 0 else 0.0
        avg_r = (m["total_r_achieved"] / m["trade_count"]) if m["trade_count"] > 0 else 0.0

        strategies_status[name] = {
            "daily_pnl": s["daily_pnl"],
            "trades_today": s["trades_today"],
            "paused": s["paused"],
            "has_position": s["position"] is not None,
            "position": s["position"],
            "capital": STRATEGY_CAPITAL.get(name, 200.0),
            "max_hold_min": STRATEGY_MAX_HOLD.get(name, 120),
            "metrics": {
                "win_rate": round(win_rate, 1),
                "avg_r_achieved": round(avg_r, 2),
                "max_drawdown": m["max_drawdown"],
                "current_losing_streak": m["current_losing_streak"],
                "max_losing_streak": m["max_losing_streak"],
                "total_trades": m["trade_count"],
            },
        }
    return {
        "bot": "Vic",
        "mode": state.mode,
        "global_paused": state.paused,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "btc_price": state.last_btc_price,
        "strategies": strategies_status,
        "trades_today": state.trades_today,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "daily_pnl": state.daily_pnl,
        "daily_loss_cap_hit": state.daily_loss_cap_hit,
        "total_lifetime_trades": state.total_trade_count,
        "correlation_filter": {
            "last_direction": state.last_trade_direction,
            "last_close_time": state.last_trade_close_time.isoformat() if state.last_trade_close_time else None,
            "cooldown_min": CORRELATION_COOLDOWN_MIN,
        },
        "recent_trades": state.trade_history[-20:],
    }


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request, token: str = Query("")):
    """Receive TradingView webhook alerts for Strategy 1 (Change 8)."""
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

    # Pre-trade checklist
    allowed, reason = can_execute_trade("tv_webhook", action)
    if not allowed:
        msg = f"📨 Signal received: {action.upper()} @ ${price:,.2f}\n⏭️ Blocked: {reason}"
        await tg_send(msg)
        return {"status": "blocked", "reason": reason}

    # Confirmation check (Change 8)
    confirmed, conf_reason = await evaluate_webhook_signal(action, price)
    if not confirmed:
        msg = (
            f"📨 Signal received: {action.upper()} @ ${price:,.2f}\n"
            f"⏭️ Signal received but no confirmation — skipped\n"
            f"Reason: {conf_reason}"
        )
        await tg_send(msg)
        log.info("Webhook signal skipped — no confirmation: %s", conf_reason)
        return {"status": "skipped", "reason": conf_reason}

    await execute_trade("tv_webhook", action, price)
    return {"status": "ok", "action": action, "price": price, "confirmation": conf_reason}


@app.post("/go_live")
async def go_live():
    """Switch from paper to live trading."""
    if state.mode == "live":
        return {"status": "already live"}

    state.mode = "live"
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
