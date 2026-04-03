"""
Vic — BTC/USDT Perpetual Futures Scalping Agent (v2)
Runs 4 strategies simultaneously on Hyperliquid via ccxt.
$500 account, full capital available per trade. Production-ready for Railway.

Strategies:
  1. TradingView Webhook (filtered with regime + bias + 2 confirmations)
  2. RSI Divergence — Price/RSI divergence at S/R on 5m
  3. BB Squeeze Breakout — Bollinger squeeze breakout on 5m
  4. VWAP Bounce — VWAP bounce with distance + chop filter on 1m

Regime filter: TRENDING / RANGING / TRANSITIONAL / VOLATILE
1H bias: Strong bullish/bearish or neutral
Risk: 10% of margin, partial close at 20%, full TP at 30%, max 4 trades/day
Session filter: London open (07-11 UTC) + NY open (13-17 UTC)
AI Market Brain: Claude pre-trade analysis gate
Trade Journal: /data/vic_journal.json (persistent)

Env vars needed:
  HL_WALLET_ADDRESS, HL_PRIVATE_KEY
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  TRADING_MODE (paper|live), WEBHOOK_SECRET, RAILWAY_URL
  CLAUDE_API_KEY (for Telegram chat + AI Market Brain)
"""

import os
import json
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
HL_WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS", "")
HL_PRIVATE_KEY = os.getenv("HL_PRIVATE_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # paper | live
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", os.urandom(16).hex())
RAILWAY_URL = os.getenv("RAILWAY_URL", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTC/USDC:USDC"
LEVERAGE = 5
ACCOUNT_CAPITAL = 500.0       # total wallet balance
RISK_PCT = 0.10               # 10% of margin = max SL risk
PARTIAL_PROFIT_PCT = 0.20     # 20% of margin = partial close trigger
TP_PCT = 0.30                 # 30% of margin = full close trigger
MAX_DAILY_LOSS_PCT = 0.10     # 10% of account = daily loss cap ($50)
MAX_TRADES_PER_DAY = 4
CORRELATION_COOLDOWN_MIN = 15  # minutes cooldown same direction after close
BB_SQUEEZE_THRESHOLD = 0.02   # bandwidth threshold for BB squeeze

# ATR multipliers for SL distance per strategy
STRATEGY_ATR_SL = {
    "tv_webhook": 1.5,         # 1.5 x ATR(14) on 5m
    "rsi_divergence": 1.5,     # 1.5 x ATR(14) on 5m
    "bb_squeeze": 1.0,         # 1.0 x ATR(14) on 5m
    "vwap_bounce": 1.0,        # 1.0 x ATR(14) on 1m
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

# Strategy labels for Telegram messages
STRATEGY_LABELS = {
    "tv_webhook": "1\ufe0f\u20e3 TradingView",
    "rsi_divergence": "2\ufe0f\u20e3 RSI Divergence",
    "bb_squeeze": "3\ufe0f\u20e3 BB Squeeze",
    "vwap_bounce": "4\ufe0f\u20e3 VWAP Bounce",
}

# Session filter — London + NY open only
TRADING_SESSIONS = [
    (7, 11),   # London open: 07:00-11:00 UTC
    (13, 17),  # NY open: 13:00-17:00 UTC
]

# Partial profit settings
PARTIAL_CLOSE_PCT = 0.5       # close 50% of position at PARTIAL_PROFIT_PCT

# Trade journal
JOURNAL_FILE = os.getenv("JOURNAL_FILE", "/data/vic_journal.json")

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

exchange = None


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

        # Signal tracking for periodic status log
        self.signals_checked: int = 0
        self.signals_blocked: int = 0
        self.last_block_reasons: list = []  # keep last 20

        # Polling watchdog
        self.polling_alive: bool = False

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
        self.signals_checked = 0
        self.signals_blocked = 0
        self.last_block_reasons = []
        log.info("Daily PnL and trade counts reset.")


state = TradingState()

# ---------------------------------------------------------------------------
# Helpers — Telegram
# ---------------------------------------------------------------------------

def sanitize_html(text: str) -> str:
    """Escape HTML entities in text so Telegram's HTML parser doesn't choke on Claude's output."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text

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
                if "can't parse entities" in resp.text:
                    log.warning("HTML parse failed, retrying as plain text: %s", resp.text)
                    payload_plain = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
                    resp2 = await client.post(url, json=payload_plain)
                    if resp2.status_code != 200:
                        log.error("Telegram plain text send also failed: %s", resp2.text)
                else:
                    log.error("Telegram send failed: %s", resp.text)
    except Exception as exc:
        log.error("Telegram error: %s", exc)


async def tg_reply(chat_id: str, text: str):
    """Send a reply to a specific Telegram chat_id."""
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                # HTML parse error — retry without parse_mode so the message still gets delivered
                if "can't parse entities" in resp.text:
                    log.warning("HTML parse failed, retrying as plain text: %s", resp.text)
                    payload_plain = {"chat_id": chat_id, "text": text}
                    resp2 = await client.post(url, json=payload_plain)
                    if resp2.status_code != 200:
                        log.error("Telegram plain text reply also failed: %s", resp2.text)
                else:
                    log.error("Telegram reply failed: %s", resp.text)
    except Exception as exc:
        log.error("Telegram reply error: %s", exc)


# ---------------------------------------------------------------------------
# Helpers — Exchange
# ---------------------------------------------------------------------------

async def init_exchange():
    """Create and configure the Hyperliquid ccxt instance."""
    global exchange
    config = {
        "walletAddress": HL_WALLET_ADDRESS,
        "privateKey": HL_PRIVATE_KEY,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    }
    if state.mode == "paper":
        config["sandbox"] = True
    exchange = ccxt.hyperliquid(config)
    try:
        await exchange.load_markets()
        log.info(f"Hyperliquid exchange connected ({'testnet' if state.mode == 'paper' else 'live'}). Markets loaded.")
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
    Skip if price stays within +/-0.1% of VWAP for more than 10 consecutive candles.
    Returns True if NOT in chop zone (OK to trade).
    """
    if len(df) < 12 or len(vwap_series) < 12:
        return True

    consecutive = 0
    max_consecutive = 0
    for i in range(-20, 0):
        if abs(i) > len(df):
            continue
        close_val = float(df["close"].iloc[i])
        vw = float(vwap_series.iloc[i])
        if np.isnan(vw) or vw == 0:
            consecutive = 0
            continue
        dist_pct = abs(close_val - vw) / vw * 100
        if dist_pct < 0.1:
            consecutive += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0

    return max_consecutive <= 10


# ---------------------------------------------------------------------------
# Pre-Trade Checklist (Change 15) — with verbose logging
# ---------------------------------------------------------------------------

def is_trading_session() -> bool:
    """Check if current UTC hour is within London or NY open sessions."""
    hour = datetime.now(timezone.utc).hour
    for start, end in TRADING_SESSIONS:
        if start <= hour < end:
            return True
    return False


def bias_allows_direction(side: str) -> bool:
    """
    Check if 1H bias allows this trade direction.
    RANGING: bias enforces direction at ANY strength.
      - bearish (strong or weak) → shorts only
      - bullish (strong or weak) → longs only
      - neutral → both directions
    TRENDING / other regimes: only strong bias enforces direction.
      - weak bias → both directions allowed
    """
    if state.htf_bias == "neutral":
        return True

    # RANGING: bias filters direction regardless of strength
    if state.regime == Regime.RANGING:
        if state.htf_bias == "bullish" and side == "long":
            return True
        if state.htf_bias == "bearish" and side == "short":
            return True
        return False

    # TRENDING / TRANSITIONAL: only strong bias enforces direction
    if state.htf_bias_strength == "weak":
        return True
    if state.htf_bias == "bullish" and side == "long":
        return True
    if state.htf_bias == "bearish" and side == "short":
        return True
    return False


def regime_allows_strategy(strategy: str) -> bool:
    """
    Check if current regime allows this strategy.
    TRENDING: VWAP pullback + BB Squeeze + TV Webhook
    RANGING: RSI Divergence + VWAP Bounce + TV Webhook
    TRANSITIONAL: RSI Divergence + TV Webhook (+ ALL strategies in paper mode)
    VOLATILE: nothing (in live) / nothing (in paper) — too dangerous
    """
    if state.regime == Regime.VOLATILE:
        return False

    # Paper mode relaxation: TRANSITIONAL allows ALL strategies
    if state.mode == "paper" and state.regime == Regime.TRANSITIONAL:
        return True

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
    Logs EVERY check with pass/fail for debugging.
    """
    state.signals_checked += 1

    # 0. Trading session check
    session_ok = is_trading_session()
    log.info("CHECK %s %s — session_ok: %s (hour=%d UTC)",
             strategy, side.upper(), session_ok, datetime.now(timezone.utc).hour)
    if not session_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — Outside trading session"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, "Outside trading session"

    # 1. Regime allows this strategy?
    regime_ok = regime_allows_strategy(strategy)
    log.info("CHECK %s %s — regime_allows: %s (regime=%s, mode=%s)",
             strategy, side.upper(), regime_ok, state.regime.value, state.mode)
    if not regime_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — regime {state.regime.value} not allowed"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Regime {state.regime.value} blocks {strategy}"

    # 2. 1H bias agrees or is neutral?
    bias_ok = bias_allows_direction(side)
    log.info("CHECK %s %s — bias_allows: %s (bias=%s, strength=%s)",
             strategy, side.upper(), bias_ok, state.htf_bias, state.htf_bias_strength)
    if not bias_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — HTF bias {state.htf_bias} (strong) blocks {side}"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"1H bias ({state.htf_bias} strong) opposes {side}"

    # 3. Trade count < 4 today?
    trades_ok = state.trades_today < MAX_TRADES_PER_DAY
    log.info("CHECK %s %s — trades_ok: %s (%d/%d)",
             strategy, side.upper(), trades_ok, state.trades_today, MAX_TRADES_PER_DAY)
    if not trades_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — daily trade cap {state.trades_today}/{MAX_TRADES_PER_DAY}"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Daily trade cap reached ({state.trades_today}/{MAX_TRADES_PER_DAY})"

    # 4. No recent same-direction trade?
    corr_ok = correlation_filter_ok(side)
    log.info("CHECK %s %s — correlation_ok: %s (last_dir=%s)",
             strategy, side.upper(), corr_ok, state.last_trade_direction)
    if not corr_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — correlation cooldown {CORRELATION_COOLDOWN_MIN}min"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Correlation cooldown: {side} blocked for {CORRELATION_COOLDOWN_MIN}min"

    # 5. Daily loss cap not hit?
    loss_ok = not state.daily_loss_cap_hit
    log.info("CHECK %s %s — loss_cap_ok: %s (daily_pnl=%.2f)",
             strategy, side.upper(), loss_ok, state.daily_pnl)
    if not loss_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} — daily loss cap hit (${state.daily_pnl:.2f})"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Daily loss cap hit (-${ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT:.0f})"

    # 6. Strategy not paused?
    strat = state.strategies[strategy]
    paused = strat["paused"] or state.paused
    log.info("CHECK %s %s — not_paused: %s (strat_paused=%s, global_paused=%s)",
             strategy, side.upper(), not paused, strat["paused"], state.paused)
    if paused:
        reason = f"BLOCKED: {strategy} {side.upper()} — paused (strat={strat['paused']}, global={state.paused})"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"{strategy} or global paused"

    # 7. No existing position on this strategy?
    has_pos = strat["position"] is not None
    log.info("CHECK %s %s — no_position: %s",
             strategy, side.upper(), not has_pos)
    if has_pos:
        reason = f"BLOCKED: {strategy} {side.upper()} — already has open position"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"{strategy} already has an open position"

    log.info("PASSED: %s %s — all checks OK", strategy, side.upper())
    return True, "OK"


# ---------------------------------------------------------------------------
# Position sizing & order execution
# ---------------------------------------------------------------------------

def calc_position_size(entry: float, margin: float) -> float:
    """Calculate BTC position size from margin and leverage."""
    if entry <= 0:
        return 0.0
    notional = margin * LEVERAGE
    return round(notional / entry, 6)


async def execute_trade(strategy: str, side: str, entry: float, atr_value: float = 0.0):
    """Open a position (paper or live) for the given strategy.

    Args:
        strategy: strategy name
        side: 'long' or 'short'
        entry: entry price
        atr_value: ATR value (kept for signal detection, not used for sizing)
    """
    allowed, reason = can_execute_trade(strategy, side)
    if not allowed:
        log.info("%s — trade blocked: %s", strategy, reason)
        return

    # Use full account capital as margin
    margin = ACCOUNT_CAPITAL
    size = calc_position_size(entry, margin)
    if size <= 0:
        log.warning("%s — invalid size, skipping.", strategy)
        return

    # SL: 10% of margin risk
    risk_dollars = margin * RISK_PCT
    sl_distance = risk_dollars / size

    if side == "long":
        sl = round(entry - sl_distance, 2)
    else:
        sl = round(entry + sl_distance, 2)

    # TP: 30% of margin profit
    tp_dollars = margin * TP_PCT
    tp_distance = tp_dollars / size

    if side == "long":
        tp = round(entry + tp_distance, 2)
    else:
        tp = round(entry - tp_distance, 2)

    # Open order
    if state.mode == "live":
        try:
            await exchange.set_leverage(LEVERAGE, SYMBOL)
            order = await exchange.create_order(
                symbol=SYMBOL,
                type="market",
                side="buy" if side == "long" else "sell",
                amount=size,
            )
            log.info("%s LIVE order placed: %s", strategy, order.get("id"))
        except Exception as exc:
            log.error("%s — order error: %s", strategy, exc)
            await tg_send(f"\u26a0\ufe0f <b>{strategy}</b> order FAILED: {exc}")
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
        "margin": margin,
        "open_time": datetime.now(timezone.utc).isoformat(),
        "be_moved": False,  # break-even not yet moved
        "partial_closed": False,  # partial profit not yet taken
        "regime": state.regime.value,
        "bias": f"{state.htf_bias} ({state.htf_bias_strength})",
    }
    strat["trades_today"] += 1
    state.trades_today += 1
    state.total_trade_count += 1

    notional = size * entry
    label = STRATEGY_LABELS.get(strategy, strategy)
    arrow = "\U0001f7e2" if side == "long" else "\U0001f534"
    msg = (
        f"{arrow} <b>{side.upper()}</b> — {label}\n"
        f"Entry ${entry:,.2f} | SL ${sl:,.2f} | TP ${tp:,.2f}\n"
        f"Margin ${margin:,.0f} | Notional ${notional:,.0f} | Risk ${margin * RISK_PCT:,.0f}\n"
        f"Regime: {state.regime.value} | Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
        f"Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))


def _append_journal(entry: dict):
    """Append a trade entry to the JSON journal file."""
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, "r") as f:
                journal = json.load(f)
        else:
            journal = []
        journal.append(entry)
        with open(JOURNAL_FILE, "w") as f:
            json.dump(journal, f, indent=2)
    except Exception as exc:
        log.error("Journal write error: %s", exc)


def _read_journal() -> list:
    """Read the trade journal from file."""
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
    except Exception as exc:
        log.error("Journal read error: %s", exc)
    return []


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

    # Calculate actual return on margin
    margin = pos.get("margin", ACCOUNT_CAPITAL)
    risk_dollars = margin * RISK_PCT
    r_achieved = pnl / risk_dollars if risk_dollars > 0 else 0.0

    # Calculate hold time
    open_time = datetime.fromisoformat(pos["open_time"])
    elapsed_min = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0

    # Close on exchange if live — ABORT if order fails
    if state.mode == "live":
        try:
            close_side = "sell" if pos["side"] == "long" else "buy"
            await exchange.create_order(
                symbol=SYMBOL,
                type="market",
                side=close_side,
                amount=pos["size"],
                params={"reduceOnly": True},
            )
        except Exception as exc:
            log.error("%s — close order error: %s", strategy, exc)
            await tg_send(
                f"🚨 <b>CRITICAL: {strategy} close order FAILED</b>\n\n"
                f"Position is STILL OPEN on Hyperliquid but I tried to close it.\n"
                f"Error: {sanitize_html(str(exc))}\n"
                f"Side: {pos['side']} | Size: {pos['size']:.6f} BTC\n\n"
                f"CHECK HYPERLIQUID MANUALLY."
            )
            return  # Do NOT mark position as closed

    # Book PnL
    strat["daily_pnl"] = round(strat["daily_pnl"] + pnl, 2)
    state.daily_pnl = round(state.daily_pnl + pnl, 2)

    # Update metrics
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
        "hold_time_min": round(elapsed_min, 1),
    })

    # Correlation filter update
    state.last_trade_direction = pos["side"]
    state.last_trade_close_time = datetime.now(timezone.utc)

    label = STRATEGY_LABELS.get(strategy, strategy)
    emoji = "\U0001f4b0" if pnl >= 0 else "\U0001f4b8"
    msg = (
        f"{emoji} <b>CLOSED</b> — {label}\n"
        f"${pos['entry']:,.2f} \u2192 ${exit_price:,.2f} | <b>${pnl:+,.2f}</b> ({r_achieved:+.1f}R)\n"
        f"Reason: {reason} | Hold: {elapsed_min:.0f}min\n"
        f"Strategy today: ${strat['daily_pnl']:+,.2f} | Total today: ${state.daily_pnl:+,.2f}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))

    # Write to trade journal
    now_utc = datetime.now(timezone.utc)
    journal_entry = {
        "id": state.total_trade_count,
        "date": now_utc.strftime("%Y-%m-%d"),
        "time_utc": now_utc.strftime("%H:%M:%S"),
        "strategy": strategy,
        "side": pos["side"],
        "entry_price": pos["entry"],
        "exit_price": exit_price,
        "sl_price": pos["sl"],
        "tp_price": pos["tp"],
        "size_btc": pos["size"],
        "pnl_usd": pnl,
        "r_achieved": round(r_achieved, 2),
        "exit_reason": reason,
        "regime_at_entry": pos.get("regime", "unknown"),
        "bias_at_entry": pos.get("bias", "unknown"),
        "hold_time_min": round(elapsed_min, 1),
        "be_moved": pos["be_moved"],
        "session": "london" if 7 <= now_utc.hour < 11 else "ny" if 13 <= now_utc.hour < 17 else "off-hours",
        "cumulative_pnl": round(state.daily_pnl, 2),
    }
    _append_journal(journal_entry)

    strat["position"] = None

    # Check daily loss cap (10% of account)
    max_daily_loss = ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT
    if state.daily_pnl <= -max_daily_loss:
        state.daily_loss_cap_hit = True
        alert = (
            f"\U0001f6d1 <b>Daily loss limit hit (${state.daily_pnl:+,.2f})</b>\n"
            f"All trading stopped until tomorrow."
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

    if confirmations < 2:
        return False, f"Need 2 confirmations, got {confirmations} (RSI/structure/VWAP)"

    return True, f"Confirmed: {', '.join(reasons)}"


# ---------------------------------------------------------------------------
# Strategy 2 — RSI Divergence (5m) (Change 10)
# ---------------------------------------------------------------------------

async def strategy_rsi_divergence():
    name = "rsi_divergence"

    df = await fetch_ohlcv("5m", 60)
    if df.empty or len(df) < 50:
        return

    df["rsi"] = calc_rsi(df["close"], 14)
    df["vwap"] = calc_vwap(df)
    atr_series = calc_atr(df, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0

    close_vals = df["close"].values[-20:]
    rsi_vals = df["rsi"].values[-20:]

    rsi_now = float(df["rsi"].iloc[-1]) if not np.isnan(df["rsi"].iloc[-1]) else 50.0
    log.info("Checking rsi_divergence... RSI=%.1f, ATR=%.2f", rsi_now, atr_val)

    # 50-candle lookback for swing high/low S/R levels
    lookback_50 = df["close"].values[-50:]
    high_50 = float(np.nanmax(df["high"].values[-50:]))
    low_50 = float(np.nanmin(df["low"].values[-50:]))

    # Find local lows (triplets) in last 20 candles for divergence detection
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

    # Support/resistance context check: within 0.25% of 50-candle swing high/low
    near_support = (price_now - low_50) / low_50 * 100 < 0.25 if low_50 > 0 else False
    near_resistance = (high_50 - price_now) / high_50 * 100 < 0.25 if high_50 > 0 else False
    away_from_vwap = abs(price_now - vwap_now) / vwap_now * 100 > 0.2 if vwap_now > 0 else False
    context_ok = near_support or near_resistance or away_from_vwap

    if not context_ok:
        log.info("rsi_divergence — no S/R context (near_sup=%s, near_res=%s, away_vwap=%s)",
                 near_support, near_resistance, away_from_vwap)
        return

    # Bullish divergence: price lower low, RSI higher low
    if len(price_lows) >= 2:
        prev_low = price_lows[-2]
        curr_low = price_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            log.info("rsi_divergence — BULLISH divergence detected: price LL (%.2f<%.2f), RSI HL (%.1f>%.1f)",
                     curr_low[1], prev_low[1], curr_low[2], prev_low[2])
            allowed, reason = can_execute_trade(name, "long")
            if allowed:
                ai_ok, ai_reason = await ai_market_analysis(name, "long", price_now)
                if ai_ok:
                    await execute_trade(name, "long", price_now, atr_value=atr_val)
                else:
                    log.info("rsi_divergence LONG blocked by AI: %s", ai_reason)
            return

    # Bearish divergence: price higher high, RSI lower high
    if len(price_highs) >= 2:
        prev_high = price_highs[-2]
        curr_high = price_highs[-1]
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            log.info("rsi_divergence — BEARISH divergence detected: price HH (%.2f>%.2f), RSI LH (%.1f<%.1f)",
                     curr_high[1], prev_high[1], curr_high[2], prev_high[2])
            allowed, reason = can_execute_trade(name, "short")
            if allowed:
                ai_ok, ai_reason = await ai_market_analysis(name, "short", price_now)
                if ai_ok:
                    await execute_trade(name, "short", price_now, atr_value=atr_val)
                else:
                    log.info("rsi_divergence SHORT blocked by AI: %s", ai_reason)


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
    atr_series = calc_atr(df, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0

    prev_bw = float(df["bb_bw"].iloc[-2])
    curr_bw = float(df["bb_bw"].iloc[-1])

    if np.isnan(prev_bw) or np.isnan(curr_bw):
        return

    price = float(df["close"].iloc[-1])
    bb_upper = float(df["bb_upper"].iloc[-1])
    bb_lower = float(df["bb_lower"].iloc[-1])

    log.info("Checking bb_squeeze... BW_prev=%.4f, BW_curr=%.4f, price=%.2f, upper=%.2f, lower=%.2f, ATR=%.2f",
             prev_bw, curr_bw, price, bb_upper, bb_lower, atr_val)

    if np.isnan(bb_upper) or np.isnan(bb_lower):
        return

    # Squeeze: previous bandwidth below threshold, current expanding
    if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
        # Candle must CLOSE at least 0.1% beyond the band
        upper_threshold = bb_upper * 1.001
        lower_threshold = bb_lower * 0.999

        if price > upper_threshold or price < lower_threshold:
            # Body filter: candle body must be >= 60% of full range
            body = abs(float(df["close"].iloc[-1]) - float(df["open"].iloc[-1]))
            full_range = float(df["high"].iloc[-1]) - float(df["low"].iloc[-1])
            if full_range == 0 or body / full_range < 0.6:
                log.info("bb_squeeze — weak candle body (%.1f%%), skipping", (body / full_range * 100) if full_range > 0 else 0)
                return

            if price > upper_threshold:
                allowed, reason = can_execute_trade(name, "long")
                if allowed:
                    ai_ok, ai_reason = await ai_market_analysis(name, "long", price)
                    if ai_ok:
                        await execute_trade(name, "long", price, atr_value=atr_val)
                    else:
                        log.info("bb_squeeze LONG blocked by AI: %s", ai_reason)
            elif price < lower_threshold:
                allowed, reason = can_execute_trade(name, "short")
                if allowed:
                    ai_ok, ai_reason = await ai_market_analysis(name, "short", price)
                    if ai_ok:
                        await execute_trade(name, "short", price, atr_value=atr_val)
                    else:
                        log.info("bb_squeeze SHORT blocked by AI: %s", ai_reason)


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
    atr_series = calc_atr(df, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(curr["close"])
    vwap_val = float(curr["vwap"])

    if np.isnan(vwap_val):
        return

    rsi_val = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
    log.info("Checking vwap_bounce... price=%.2f, VWAP=%.2f, RSI=%.1f", price, vwap_val, rsi_val)

    # VWAP distance filter (Change 4, 11): must have moved >= 0.2% away before returning
    if not vwap_distance_ok(df, df["vwap"]):
        return

    # Chop filter (Change 4, 11): no trade if VWAP touched > 3 times in last 20 candles
    if not vwap_chop_filter(df, df["vwap"]):
        return

    # RSI must be 40-60 (mean reversion zone)
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
            ai_ok, ai_reason = await ai_market_analysis(name, "long", price)
            if ai_ok:
                await execute_trade(name, "long", price, atr_value=atr_val)
            else:
                log.info("vwap_bounce LONG blocked by AI: %s", ai_reason)
    # Short: price touched VWAP and rejected with bearish candle
    elif is_bearish and price < vwap_val:
        allowed, reason = can_execute_trade(name, "short")
        if allowed:
            ai_ok, ai_reason = await ai_market_analysis(name, "short", price)
            if ai_ok:
                await execute_trade(name, "short", price, atr_value=atr_val)
            else:
                log.info("vwap_bounce SHORT blocked by AI: %s", ai_reason)



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

                # 1. Stop Loss check (highest priority)
                if side == "long" and price <= sl:
                    await close_position(name, price, "Stop Loss")
                    continue
                if side == "short" and price >= sl:
                    await close_position(name, price, "Stop Loss")
                    continue

                # 2. Take Profit check
                if side == "long" and price >= tp:
                    await close_position(name, price, "Take Profit")
                    continue
                if side == "short" and price <= tp:
                    await close_position(name, price, "Take Profit")
                    continue

                # 3. Break-even move: at +10% of margin (same as risk), move SL to entry
                margin = pos.get("margin", ACCOUNT_CAPITAL)
                be_threshold = margin * RISK_PCT
                if not pos["be_moved"] and unrealized_pnl >= be_threshold:
                    pos["sl"] = entry
                    pos["be_moved"] = True
                    msg = f"\U0001f504 <b>{name}</b> — SL moved to break-even (entry ${entry:,.2f})"
                    await tg_send(msg)
                    log.info(msg.replace("<b>", "").replace("</b>", ""))

                # 4. Partial profit at 20% of margin
                if not pos.get("partial_closed", False):
                    partial_threshold = margin * PARTIAL_PROFIT_PCT
                    if unrealized_pnl >= partial_threshold:
                        old_size = pos["size"]
                        new_size = round(old_size * (1 - PARTIAL_CLOSE_PCT), 6)
                        if state.mode == "live":
                            try:
                                close_side = "sell" if side == "long" else "buy"
                                partial_amount = round(old_size * PARTIAL_CLOSE_PCT, 6)
                                await exchange.create_order(
                                    symbol=SYMBOL,
                                    type="market",
                                    side=close_side,
                                    amount=partial_amount,
                                    params={"reduceOnly": True},
                                )
                            except Exception as exc:
                                log.error("%s — partial close error: %s", name, exc)
                                await tg_send(
                                    f"🚨 <b>{name} partial close FAILED</b>\n"
                                    f"Error: {sanitize_html(str(exc))}\n"
                                    f"Position size unchanged. CHECK HYPERLIQUID."
                                )
                                continue  # Do NOT update size if order failed
                        pos["size"] = new_size
                        pos["partial_closed"] = True
                        partial_pnl = round(unrealized_pnl * PARTIAL_CLOSE_PCT, 2)
                        msg = (
                            f"✂️ <b>{name}</b> — Partial close at +{PARTIAL_PROFIT_PCT*100:.0f}% margin profit\n"
                            f"Closed {PARTIAL_CLOSE_PCT*100:.0f}% ({old_size:.6f} → {new_size:.6f} BTC)\n"
                            f"Locked ~${partial_pnl:+,.2f}"
                        )
                        await tg_send(msg)
                        log.info(msg.replace("<b>", "").replace("</b>", ""))

                # 5. Max hold time check (lowest priority)
                max_hold_min = STRATEGY_MAX_HOLD.get(name, 120)
                open_time = datetime.fromisoformat(pos["open_time"])
                elapsed_min = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
                if elapsed_min >= max_hold_min:
                    await close_position(name, price, f"Max hold time ({max_hold_min}min)")
                    continue

        except Exception as exc:
            log.error("Position monitor error: %s", exc)
        await asyncio.sleep(10)


async def periodic_status_log():
    """Log a periodic status summary every 5 minutes for debugging."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        try:
            open_positions = []
            for name in STRATEGY_NAMES:
                pos = state.strategies[name]["position"]
                if pos is not None:
                    open_positions.append(f"{name}({pos['side']}@{pos['entry']:.0f})")

            recent_blocks = state.last_block_reasons[-5:] if state.last_block_reasons else ["none"]

            log.info(
                "=== PERIODIC STATUS ===\n"
                "  Regime: %s | HTF Bias: %s (%s) | BTC: $%.2f | Mode: %s\n"
                "  Signals checked: %d | Signals blocked: %d\n"
                "  Trades today: %d/%d | Daily PnL: $%.2f | Loss cap: %s\n"
                "  Open positions: %s\n"
                "  Recent blocks: %s",
                state.regime.value, state.htf_bias, state.htf_bias_strength,
                state.last_btc_price, state.mode,
                state.signals_checked, state.signals_blocked,
                state.trades_today, MAX_TRADES_PER_DAY,
                state.daily_pnl, state.daily_loss_cap_hit,
                ", ".join(open_positions) if open_positions else "none",
                " | ".join(recent_blocks),
            )
        except Exception as exc:
            log.error("Periodic status error: %s", exc)


async def news_sentiment_monitor():
    """Poll for BTC price moves and basic news sentiment every 5 min.
    Also runs a deeper macro scan via web search every 30 min."""
    prev_price = 0.0
    _macro_scan_counter = 0  # runs deep scan every 6 cycles (30 min)
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
                            f"\U0001f6a8 <b>VOLATILITY ALERT</b>\n"
                            f"BTC moved {change_pct:.1f}% in 5 min "
                            f"(${prev_price:,.0f} \u2192 ${price:,.0f})\n"
                            f"ALL trading PAUSED."
                        )
                        await tg_send(msg)
                        log.warning(msg.replace("<b>", "").replace("</b>", ""))
                prev_price = price

            await _check_crypto_news()

            # Deep macro scan every 30 minutes via web search
            _macro_scan_counter += 1
            if _macro_scan_counter >= 6:
                _macro_scan_counter = 0
                await _run_macro_scan()

        except Exception as exc:
            log.error("News monitor error: %s", exc)
        await asyncio.sleep(300)


# Cached macro intelligence — updated every 30 min by _run_macro_scan
_macro_intel_cache = {"text": "", "fetched_at": 0}


async def _run_macro_scan():
    """Web search for macro events that affect BTC trading decisions."""
    if not CLAUDE_API_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1024,
                    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                    "messages": [{"role": "user", "content": (
                        "Search for the latest macro events affecting Bitcoin and crypto markets RIGHT NOW. "
                        "Check: 1) Fed/FOMC decisions or speeches today, 2) US CPI/jobs data releases, "
                        "3) Tariff or trade war developments, 4) SEC/regulatory actions on crypto, "
                        "5) Major exchange issues or hacks, 6) Geopolitical events affecting risk assets. "
                        "Give a concise factual summary. Flag anything that should STOP a trader from "
                        "entering new BTC positions right now. End with: RISK_LEVEL: LOW/MEDIUM/HIGH/CRITICAL"
                    )}],
                },
            )
            if resp.status_code != 200:
                log.warning("Macro scan API error: %d", resp.status_code)
                return

            data = resp.json()
            texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            result = "\n".join(texts) if texts else ""

            if result:
                _macro_intel_cache["text"] = result
                _macro_intel_cache["fetched_at"] = time.time()
                log.info("Macro scan complete")

                # Auto-pause on CRITICAL risk level
                if "RISK_LEVEL: CRITICAL" in result.upper():
                    if not state.paused:
                        state.paused = True
                        await tg_send(
                            f"\U0001f6a8 <b>MACRO RISK — CRITICAL</b>\n\n"
                            f"Auto-scan detected critical macro conditions.\n"
                            f"ALL trading PAUSED.\n\n"
                            f"Details:\n{result[:500]}"
                        )
                        log.warning("Trading paused by macro scan — CRITICAL risk")
                elif "RISK_LEVEL: HIGH" in result.upper():
                    await tg_send(
                        f"\u26a0\ufe0f <b>MACRO RISK — HIGH</b>\n\n"
                        f"{result[:400]}\n\n"
                        f"Trading continues but AI Market Brain will factor this in."
                    )
    except Exception as exc:
        log.error("Macro scan error: %s", exc)


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
                                f"\U0001f4f0 <b>NEWS ALERT</b>: \"{article.get('title', 'N/A')}\"\n"
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
    lines = ["\U0001f4ca <b>DAILY SUMMARY</b>\n"]

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

        expectancy = (m["total_r_achieved"] / m["trade_count"]) if m["trade_count"] > 0 else 0.0

        lines.append(
            f"  <b>{name}</b>: ${pnl:+,.2f} ({trades} trades) [{status}]\n"
            f"    WR: {win_rate:.0f}% | Avg R: {avg_r:+.2f} | Expectancy: {expectancy:+.2f}R/trade\n"
            f"    MaxDD: ${m['max_drawdown']:,.2f} | "
            f"Streak: {m['current_losing_streak']}/{m['max_losing_streak']}"
        )

    # Overall expectancy
    total_r = sum(state.metrics[n]["total_r_achieved"] for n in STRATEGY_NAMES)
    total_tc = sum(state.metrics[n]["trade_count"] for n in STRATEGY_NAMES)
    overall_expectancy = total_r / total_tc if total_tc > 0 else 0.0

    lines.append(f"\n<b>Total PnL:</b> ${total_pnl:+,.2f}")
    lines.append(f"<b>Expectancy:</b> {overall_expectancy:+.2f}R per trade")
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
    lines = ["\U0001f4cb <b>PAPER TEST REPORT \u2014 {0} TRADES COMPLETED</b>\n".format(state.total_trade_count)]

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
    total_r = sum(state.metrics[n]["total_r_achieved"] for n in STRATEGY_NAMES)
    overall_expectancy = total_r / total_trades if total_trades > 0 else 0.0
    lines.append(f"\n<b>OVERALL:</b>")
    lines.append(f"  Total: {total_trades} | WR: {overall_wr:.1f}% | PnL: ${overall_pnl:+,.2f}")
    lines.append(f"  Expectancy: {overall_expectancy:+.2f}R per trade")

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
        await tg_send("\U0001f504 Daily reset complete. All strategies resumed.")


# ---------------------------------------------------------------------------
# AI Market Brain — Pre-trade analysis
# ---------------------------------------------------------------------------

async def _fetch_recent_news_headlines() -> list[str]:
    """Fetch recent BTC news headlines from CryptoCompare."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            articles = data.get("Data", [])[:5]
            return [a.get("title", "") for a in articles if a.get("title")]
    except Exception:
        return []


async def ai_market_analysis(strategy: str, side: str, price: float) -> tuple[bool, str]:
    """
    AI Market Brain: ask Claude Haiku whether this trade should be taken.
    Called after all mechanical checks pass, before execute_trade.
    Returns (should_trade, reason).
    Uses cached macro intel + web search for real-time awareness.
    Defaults to YES on API errors.
    """
    if not CLAUDE_API_KEY:
        return True, "No API key — defaulting to YES"

    try:
        headlines = await _fetch_recent_news_headlines()
        news_text = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news available"

        # Inject cached macro intelligence from 30-min scans
        macro_text = ""
        if _macro_intel_cache["text"]:
            age_min = int((time.time() - _macro_intel_cache["fetched_at"]) / 60)
            macro_text = f"\n\nMACRO INTELLIGENCE (updated {age_min}min ago):\n{_macro_intel_cache['text']}"

        prompt = (
            f"You are a BTC futures trading risk analyst. Evaluate this proposed trade:\n\n"
            f"Strategy: {strategy}\n"
            f"Direction: {side.upper()}\n"
            f"Entry price: ${price:,.2f}\n"
            f"Current regime: {state.regime.value}\n"
            f"1H bias: {state.htf_bias} ({state.htf_bias_strength})\n"
            f"Daily PnL: ${state.daily_pnl:+,.2f}\n"
            f"Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}\n\n"
            f"Recent BTC news:\n{news_text}"
            f"{macro_text}\n\n"
            f"You have web_search — if the macro intel above is older than 30 minutes or mentions "
            f"a developing situation (Fed speaking, data release pending, breaking event), "
            f"SEARCH for the latest update before deciding.\n\n"
            f"Should this trade be taken? Reply YES or NO with a 1-sentence reason."
        )

        payload = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            if resp.status_code != 200:
                log.warning("AI Market Brain API error %d — defaulting YES", resp.status_code)
                return True, "API error — defaulting to YES"

            data = resp.json()
            # Extract text from potentially mixed content blocks
            text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            content = " ".join(text_blocks).strip()
            log.info("AI Market Brain response for %s %s: %s", strategy, side, content)

            if content.upper().startswith("NO") or "NO —" in content.upper() or "NO." in content.upper():
                return False, content
            return True, content

    except Exception as exc:
        log.warning("AI Market Brain error: %s — defaulting YES", exc)
        return True, f"Error: {exc} — defaulting to YES"


# ---------------------------------------------------------------------------
# Telegram Chat — Claude-powered market Q&A
# ---------------------------------------------------------------------------

async def telegram_polling_loop():
    """
    Poll Telegram for incoming text messages and respond using Claude API.
    Does NOT conflict with webhook functionality (webhooks use FastAPI routes).
    """
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set — Telegram polling DEAD.")
        state.polling_alive = False
        return
    if not CLAUDE_API_KEY:
        log.error("CLAUDE_API_KEY not set — Telegram polling DEAD.")
        state.polling_alive = False
        await tg_send("⚠️ <b>Vic polling failed to start</b>\n\nCLAUDE_API_KEY not set. I can't respond to messages.")
        return

    state.polling_alive = True

    last_update_id = 0
    log.info("Telegram polling loop started.")

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"offset": last_update_id + 1, "timeout": 30, "allowed_updates": ["message"]}

            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    log.error("Telegram getUpdates failed: %s", resp.text)
                    await asyncio.sleep(5)
                    continue

                data = resp.json()
                results = data.get("result", [])

                for update in results:
                    update_id = update.get("update_id", 0)
                    if update_id > last_update_id:
                        last_update_id = update_id

                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = str(message.get("chat", {}).get("id", ""))

                    if not text or not chat_id:
                        continue

                    # Skip bot commands that aren't questions
                    if text.startswith("/start"):
                        await tg_reply(chat_id, "Hey! I'm Vic, your BTC trading agent. Ask me anything about the market.\n\nCommands: /journal /metrics /regime /news")
                        continue

                    # Handle special commands before sending to Claude
                    if text.strip().lower() == "/journal":
                        journal = _read_journal()
                        last_5 = journal[-5:] if journal else []
                        if not last_5:
                            await tg_reply(chat_id, "No trades in journal yet.")
                        else:
                            lines = ["📒 <b>Last 5 Trades</b>\n"]
                            for t in reversed(last_5):
                                emoji = "✅" if t.get("pnl_usd", 0) >= 0 else "❌"
                                lines.append(
                                    f"{emoji} #{t.get('id','-')} {t.get('strategy','-')} {t.get('side','').upper()}\n"
                                    f"  ${t.get('entry_price',0):,.2f} → ${t.get('exit_price',0):,.2f} | "
                                    f"<b>${t.get('pnl_usd',0):+,.2f}</b> ({t.get('r_achieved',0):+.1f}R)\n"
                                    f"  {t.get('exit_reason','-')} | {t.get('hold_time_min',0):.0f}min | {t.get('session','-')}"
                                )
                            await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if text.strip().lower() == "/metrics":
                        lines = ["📊 <b>Per-Strategy Metrics</b>\n"]
                        for n in STRATEGY_NAMES:
                            m = state.metrics[n]
                            tc = m["trade_count"]
                            wr = (m["wins"] / tc * 100) if tc > 0 else 0
                            exp = (m["total_r_achieved"] / tc) if tc > 0 else 0
                            lines.append(
                                f"<b>{n}</b>: {tc} trades | WR {wr:.0f}% | Exp {exp:+.2f}R\n"
                                f"  PnL ${m['current_pnl']:+,.2f} | MaxDD ${m['max_drawdown']:,.2f} | "
                                f"Streak {m['current_losing_streak']}/{m['max_losing_streak']}"
                            )
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if text.strip().lower() == "/regime":
                        msg = (
                            f"🔍 <b>Current Regime: {state.regime.value}</b>\n\n"
                            f"Math:\n"
                            f"- TRENDING: ADX > 25 AND structure aligned (HH/HL or LH/LL)\n"
                            f"- TRANSITIONAL: ADX > 25 but structure breaking\n"
                            f"- RANGING: ADX < 20 AND BB bandwidth < 0.03\n"
                            f"- VOLATILE: ATR% > 0.4% OR >1.5% move in 30min\n\n"
                            f"1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
                            f"BTC: ${state.last_btc_price:,.2f}"
                        )
                        await tg_reply(chat_id, msg)
                        continue

                    if text.strip().lower() == "/news":
                        headlines = await _fetch_recent_news_headlines()
                        if headlines:
                            lines = ["📰 <b>Recent BTC News</b>\n"]
                            for h in headlines:
                                lines.append(f"• {sanitize_html(h)}")
                            await tg_reply(chat_id, "\n".join(lines))
                        else:
                            await tg_reply(chat_id, "No recent news available.")
                        continue

                    # Process the question with Claude
                    try:
                        answer = await ask_claude_market_question(text)
                        await tg_reply(chat_id, sanitize_html(answer))
                    except Exception as exc:
                        log.error("Claude chat error: %s", exc)
                        await tg_reply(chat_id, f"Sorry, I hit an error: {sanitize_html(str(exc))}")

        except Exception as exc:
            log.error("Telegram polling error: %s", exc)
            await asyncio.sleep(5)

    # If we exit the while True loop somehow, mark as dead
    state.polling_alive = False


async def telegram_polling_watchdog():
    """Watchdog that monitors the polling loop and restarts it if it dies."""
    restart_count = 0
    max_restarts = 5  # prevent infinite restart loop

    while True:
        task = asyncio.create_task(telegram_polling_loop())
        try:
            await task
        except Exception as exc:
            log.error("Polling task crashed: %s", exc)

        state.polling_alive = False

        if restart_count >= max_restarts:
            log.error("Polling loop hit max restarts (%d). Giving up.", max_restarts)
            await tg_send(
                "🚨 <b>Vic polling is DEAD</b>\n\n"
                f"Crashed {max_restarts} times. I cannot respond to messages.\n"
                "Redeploy me on Railway to fix this."
            )
            return

        restart_count += 1
        log.warning("Polling loop died — restarting (attempt %d/%d)...", restart_count, max_restarts)
        await tg_send(
            f"⚠️ <b>Vic polling crashed — restarting</b> (attempt {restart_count}/{max_restarts})\n\n"
            "If you see this repeatedly, something is broken."
        )
        await asyncio.sleep(3)


async def ask_claude_market_question(question: str) -> str:
    """Send a market question to Claude API with full trading state context."""
    # Build position summary with unrealized PnL
    open_positions = []
    for name in STRATEGY_NAMES:
        pos = state.strategies[name]["position"]
        if pos is not None:
            pnl_label = ""
            if state.last_btc_price > 0:
                if pos["side"] == "long":
                    unrealized = (state.last_btc_price - pos["entry"]) * pos["size"]
                else:
                    unrealized = (pos["entry"] - state.last_btc_price) * pos["size"]
                pnl_label = f", unrealized ${unrealized:+.2f}"
            open_positions.append(
                f"{name}: {pos['side'].upper()} @ ${pos['entry']:,.2f} "
                f"(SL ${pos['sl']:,.2f}, TP ${pos['tp']:,.2f}{pnl_label}, BE={'yes' if pos['be_moved'] else 'no'})"
            )
    positions_text = "\n".join(open_positions) if open_positions else "No open positions"

    # Per-strategy metrics
    metrics_lines = []
    for n in STRATEGY_NAMES:
        m = state.metrics[n]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        exp = (m["total_r_achieved"] / tc) if tc > 0 else 0
        metrics_lines.append(
            f"  {n}: {tc} trades, WR {wr:.0f}%, expectancy {exp:+.2f}R, "
            f"PnL ${m['current_pnl']:+,.2f}, MaxDD ${m['max_drawdown']:,.2f}, "
            f"losing streak {m['current_losing_streak']}/{m['max_losing_streak']}"
        )
    metrics_text = "\n".join(metrics_lines)

    # Recent trade history from journal
    journal = _read_journal()
    recent_trades = journal[-10:] if journal else []
    trades_text = ""
    if recent_trades:
        trade_lines = []
        for t in recent_trades:
            trade_lines.append(
                f"  #{t.get('id','-')} {t.get('strategy','-')} {t.get('side','').upper()} "
                f"${t.get('entry_price',0):,.2f}->${t.get('exit_price',0):,.2f} "
                f"PnL ${t.get('pnl_usd',0):+,.2f} ({t.get('r_achieved',0):+.1f}R) "
                f"[{t.get('exit_reason','-')}, {t.get('hold_time_min',0):.0f}min]"
            )
        trades_text = "\n".join(trade_lines)
    else:
        trades_text = "  No trades yet"

    # Recent news
    headlines = await _fetch_recent_news_headlines()
    news_text = "\n".join(f"  - {h}" for h in headlines) if headlines else "  No recent news"

    system_prompt = (
        f"You are Vic, an AI crypto trading agent monitoring BTC/USDC perpetual futures on Hyperliquid. "
        f"You run 4 strategies. $500 account, full capital available per trade, percentage-based risk.\n\n"
        f"=== STRATEGIES ===\n"
        f"1. TradingView Webhook: External alerts filtered by regime + bias + 2 confirmations (RSI/structure/VWAP)\n"
        f"2. RSI Divergence: Price/RSI divergence at support/resistance on 5m (50-candle lookback, 0.25% S/R threshold)\n"
        f"3. BB Squeeze: Bollinger squeeze breakout on 5m with 60% body filter and 0.1% band penetration\n"
        f"4. VWAP Bounce: VWAP bounce on 1m with distance, chop filter, volume + body confirmation\n\n"
        f"=== REGIME DEFINITIONS ===\n"
        f"TRENDING: ADX > 25 AND structure aligned (HH/HL or LH/LL)\n"
        f"TRANSITIONAL: ADX > 25 but structure breaking down\n"
        f"RANGING: ADX < 20 AND BB bandwidth < 0.03\n"
        f"VOLATILE: ATR% > 0.4% OR >1.5% move in 30min\n\n"
        f"=== CURRENT STATE ===\n"
        f"- BTC Price: ${state.last_btc_price:,.2f}\n"
        f"- Regime: {state.regime.value}\n"
        f"- 1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
        f"- Mode: {state.mode.upper()}\n"
        f"- Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}\n"
        f"- Daily PnL: ${state.daily_pnl:+,.2f}\n"
        f"- Lifetime trades: {state.total_trade_count}\n"
        f"- Daily loss cap: {'HIT' if state.daily_loss_cap_hit else 'OK'}\n"
        f"- Paused: {state.paused}\n"
        f"- Session: {'active' if is_trading_session() else 'inactive'}\n"
        f"- Signals checked today: {state.signals_checked}, blocked: {state.signals_blocked}\n\n"
        f"=== OPEN POSITIONS ===\n{positions_text}\n\n"
        f"=== PER-STRATEGY METRICS ===\n{metrics_text}\n\n"
        f"=== RECENT TRADES (last 10) ===\n{trades_text}\n\n"
        f"=== RECENT NEWS ===\n{news_text}\n\n"
        f"$500 account, full capital per trade. 10% risk, 20% partial close, 30% full TP.\n"
        f"Max 4 trades/day, 10% daily loss cap (${ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT:.0f}).\n"
        f"Sessions: London 07-11 UTC, NY 13-17 UTC. AI pre-trade analysis enabled.\n\n"
        f"Answer the user's question about the market concisely. "
        f"Be direct, use numbers, and keep it under 300 words.\n\n"
        f"WEB SEARCH: You have access to web_search. USE IT when the user asks about macro events, "
        f"Fed decisions, tariffs, regulations, breaking news, on-chain data, or anything you can't "
        f"answer from the trading state above. Search first, then answer with real data."
    )

    # Detect if question needs live web data
    q_lower = question.lower()
    search_keywords = [
        "news", "macro", "fed", "cpi", "fomc", "inflation", "tariff", "regulation",
        "sec", "etf", "halving", "on-chain", "whale", "liquidat", "funding rate",
        "what's happening", "why is btc", "why did btc", "crash", "pump", "dump",
        "sentiment", "fear", "greed", "dominance", "altcoin", "eth", "sol",
        "geopolit", "war", "sanction", "trump", "policy", "rate cut", "rate hike",
        "latest", "today", "this week", "current", "search", "look up", "find out",
    ]
    needs_search = any(kw in q_lower for kw in search_keywords)

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2048 if needs_search else 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": question}],
    }
    if needs_search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]

    async with httpx.AsyncClient(timeout=60 if needs_search else 30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        if resp.status_code != 200:
            log.error("Claude API error: %s", resp.text)
            return f"Claude API error ({resp.status_code}). Try again in a moment."

        data = resp.json()
        content_blocks = data.get("content", [])
        # Extract text blocks (response may contain mixed search result + text blocks)
        text_blocks = [b.get("text", "") for b in content_blocks if b.get("type") == "text" and b.get("text", "").strip()]
        if text_blocks:
            return "\n".join(text_blocks)
        return "No response from Claude."


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    _required = ["HL_WALLET_ADDRESS", "HL_PRIVATE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "CLAUDE_API_KEY"]
    _missing = [v for v in _required if not os.getenv(v)]
    if _missing:
        log.error(f"MISSING REQUIRED ENV VARS: {', '.join(_missing)} — Vic cannot start properly")

    # Ensure journal directory exists
    journal_dir = os.path.dirname(JOURNAL_FILE)
    if journal_dir:
        os.makedirs(journal_dir, exist_ok=True)

    state.startup_time = datetime.now(timezone.utc).isoformat()
    log.info("Vic starting up — mode: %s", state.mode)

    # Initialize candle boundary tracking
    now_ts = datetime.now(timezone.utc).timestamp()
    state.last_1m_candle_ts = int(now_ts // 60) * 60
    state.last_5m_candle_ts = int(now_ts // 300) * 300

    await init_exchange()

    startup_msg = (
        f"\U0001f916 <b>Vic is online \u2014 {state.mode.upper()} MODE</b>\n"
        f"BTC/USDC perp | {LEVERAGE}x leverage | ${ACCOUNT_CAPITAL:.0f} account\n\n"
        f"1\ufe0f\u20e3 TradingView Webhook | 2\ufe0f\u20e3 RSI Divergence\n"
        f"3\ufe0f\u20e3 BB Squeeze | 4\ufe0f\u20e3 VWAP Bounce\n"
        f"10% risk | 20% partial | 30% TP | Max 4/day | -${ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT:.0f} cap\n\n"
        f"Session: London 07-11 + NY 13-17 UTC\n"
        f"AI Market Brain: enabled"
    )
    await tg_send(startup_msg)

    asyncio.create_task(regime_update_loop())
    asyncio.create_task(strategy_monitor_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())
    asyncio.create_task(periodic_status_log())
    asyncio.create_task(telegram_polling_watchdog())

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
        "status": "running" if state.polling_alive else "DEGRADED — polling dead",
        "telegram_polling": "alive" if state.polling_alive else "DEAD",
        "mode": state.mode,
        "paused": state.paused,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "uptime_since": state.startup_time,
        "btc_price": state.last_btc_price,
        "trades_today": state.trades_today,
        "daily_pnl": state.daily_pnl,
        "loss_cap_hit": state.daily_loss_cap_hit,
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
        "signals_checked": state.signals_checked,
        "signals_blocked": state.signals_blocked,
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
    strategy_label = body.get("strategy", "tv_indicator")

    if action not in ("long", "short"):
        return JSONResponse(status_code=400, content={"error": "action must be long or short"})

    if price <= 0:
        price = await get_btc_price()

    log.info("TradingView webhook: %s @ %.2f (%s)", action, price, strategy_label)

    # Pre-trade checklist
    allowed, reason = can_execute_trade("tv_webhook", action)
    if not allowed:
        msg = f"\U0001f4e8 Signal received: {action.upper()} @ ${price:,.2f}\n\u23ed\ufe0f Blocked: {reason}"
        await tg_send(msg)
        return {"status": "blocked", "reason": reason}

    # Confirmation check (Change 8)
    confirmed, conf_reason = await evaluate_webhook_signal(action, price)
    if not confirmed:
        msg = (
            f"\U0001f4e8 Signal received: {action.upper()} @ ${price:,.2f}\n"
            f"\u23ed\ufe0f Signal received but no confirmation \u2014 skipped\n"
            f"Reason: {conf_reason}"
        )
        await tg_send(msg)
        log.info("Webhook signal skipped — no confirmation: %s", conf_reason)
        return {"status": "skipped", "reason": conf_reason}

    # Fetch ATR for SL computation
    df_5m = await fetch_ohlcv("5m", 30)
    atr_val = 0.0
    if not df_5m.empty and len(df_5m) >= 14:
        atr_s = calc_atr(df_5m, 14)
        atr_val = float(atr_s.iloc[-1]) if not atr_s.empty and not np.isnan(atr_s.iloc[-1]) else 0.0

    # AI Market Brain check
    ai_ok, ai_reason = await ai_market_analysis("tv_webhook", action, price)
    if not ai_ok:
        log.info("Webhook blocked by AI: %s", ai_reason)
        return {"status": "ai_blocked", "reason": ai_reason}

    await execute_trade("tv_webhook", action, price, atr_value=atr_val)
    return {"status": "ok", "action": action, "price": price, "confirmation": conf_reason}


@app.post("/go_live")
async def go_live(token: str = Query("")):
    """Switch from paper to live trading."""
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    if state.mode == "live":
        return {"status": "already live"}

    state.mode = "live"
    await close_exchange()
    await init_exchange()

    await tg_send("\U0001f534 <b>LIVE TRADING ENABLED</b> \u2014 Vic is now trading with real funds.")
    log.warning("Switched to LIVE trading mode.")
    return {"status": "live", "warning": "Real money is now at risk."}


@app.post("/pause")
async def pause_trading(token: str = Query("")):
    """Pause all trading globally."""
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    state.paused = True
    await tg_send("\u23f8\ufe0f All trading PAUSED by manual command.")
    return {"status": "paused"}


@app.post("/resume")
async def resume_trading(token: str = Query("")):
    """Resume all trading globally."""
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    state.paused = False
    await tg_send("\u25b6\ufe0f Trading RESUMED by manual command.")
    return {"status": "resumed"}


@app.get("/journal")
async def get_journal():
    """Return the trade journal."""
    journal = _read_journal()
    return {"trades": journal, "count": len(journal)}
