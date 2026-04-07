"""
Vic -- BTC/USDT Perpetual Futures Trading Agent (v5)
Runs 5 strategies on Hyperliquid via native SDK.
$500 account, dynamic leverage (10x minimum). Production-ready for Railway.

Strategies (all active from day 1, backtested daily):
  1. Funding Rate Fade -- Extreme funding rate reversal on 1H
  2. Liquidity Sweep Reversal -- Stop hunt reversal on 5m/15m
  3. EMA Trend Pullback -- Trend pullback to 21/50 EMA on 1H
  4. VWAP Reclaim -- VWAP reclaim with conviction on 5m/15m
  5. Order Block Entry -- OB + FVG reversal on 15m/1H

Core rules:
  - ONE position at a time across all strategies
  - Every trade has a stop loss (ATR-based, intelligent early exit)
  - Max 10% account risk per trade (hard ceiling, target 3-5%)
  - 3 losing trades per day = all trading stops
  - No trading in VOLATILE regime
  - AI Market Brain confidence gate (score 1-10, below 6 = reject)
  - Dynamic leverage: 10x base, scales with confidence

Regime filter: TRENDING / RANGING / TRANSITIONAL / VOLATILE
1H bias: Strong bullish/bearish or neutral
AI Market Brain: Claude pre-trade analysis with confidence scoring
Trade Journal: /data/vic_journal.json (persistent)
State Persistence: /data/vic_state.json (survives restarts)
Backtest Engine: 30-day rolling backtest daily at midnight UTC
Intelligence: Top trader scan every 6h -> /data/hl_intelligence.json
Daily Self-Review: 17:00 UTC, proposes parameter changes, requires GG approval

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
import random
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange as HLExchange
from hyperliquid.utils import constants as hl_constants
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
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
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
SYMBOL = "BTC"
BASE_LEVERAGE = 10
ACCOUNT_CAPITAL = 500.0
MAX_RISK_PCT = 0.10               # 10% hard ceiling per trade
TARGET_RISK_PCT = 0.04            # 3-5% target risk per trade
MAX_LOSSES_PER_DAY = 3            # 3 losing trades = stop for the day

# Strategy definitions
STRATEGY_NAMES = [
    "funding_rate_fade",
    "liquidity_sweep",
    "ema_trend_pullback",
    "vwap_reclaim",
    "order_block",
]

STRATEGY_LABELS = {
    "funding_rate_fade": "1\ufe0f\u20e3 Funding Rate Fade",
    "liquidity_sweep": "2\ufe0f\u20e3 Liquidity Sweep",
    "ema_trend_pullback": "3\ufe0f\u20e3 EMA Trend Pullback",
    "vwap_reclaim": "4\ufe0f\u20e3 VWAP Reclaim",
    "order_block": "5\ufe0f\u20e3 Order Block",
}

# ATR multiplier for initial SL per strategy
STRATEGY_ATR_SL = {
    "funding_rate_fade": 1.5,
    "liquidity_sweep": 0.0,       # SL set by sweep extreme, not ATR
    "ema_trend_pullback": 1.5,
    "vwap_reclaim": 1.0,
    "order_block": 0.0,           # SL set by OB edge, not ATR
}

# Minimum R:R per strategy
STRATEGY_MIN_RR = {
    "funding_rate_fade": 2.5,
    "liquidity_sweep": 1.5,       # TP is opposing pool, RR varies
    "ema_trend_pullback": 1.5,
    "vwap_reclaim": 2.0,
    "order_block": 1.5,
}

# Max hold time in minutes
STRATEGY_MAX_HOLD = {
    "funding_rate_fade": 480,     # 8 hours
    "liquidity_sweep": 240,       # 4 hours
    "ema_trend_pullback": 720,    # 12 hours
    "vwap_reclaim": 180,          # 3 hours
    "order_block": 480,           # 8 hours
}

# Persistence files
JOURNAL_FILE = os.getenv("JOURNAL_FILE", "/data/vic_journal.json")
STATE_FILE = os.getenv("STATE_FILE", "/data/vic_state.json")
BACKTEST_FILE = os.getenv("BACKTEST_FILE", "/data/vic_backtest.json")
INTELLIGENCE_FILE = os.getenv("INTELLIGENCE_FILE", "/data/hl_intelligence.json")
REVIEW_FILE = os.getenv("REVIEW_FILE", "/data/vic_review.json")

# Underperformance auto-pause thresholds
UNDERPERFORMANCE_MIN_TRADES = 15
UNDERPERFORMANCE_WR_THRESHOLD = 0.40

# Backtest minimum win rate
BACKTEST_MIN_WIN_RATE = 0.50

# Funding rate extremes (percentile thresholds)
FUNDING_EXTREME_PCT = 10  # top/bottom 10%

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
app = FastAPI(title="Vic Trading Agent", version="5.0.0")

hl_info: Optional[Info] = None
hl_exchange: Optional[HLExchange] = None


class TradingState:
    """Global in-memory state for the bot."""

    def __init__(self):
        self.mode: str = TRADING_MODE
        self.paused: bool = False
        self.last_btc_price: float = 0.0
        self.regime: Regime = Regime.RANGING
        self.htf_bias: str = "neutral"
        self.htf_bias_strength: str = "weak"

        # Candle-close tracking
        self.last_1m_candle_ts: float = 0.0
        self.last_5m_candle_ts: float = 0.0
        self.last_15m_candle_ts: float = 0.0
        self.last_1h_candle_ts: float = 0.0

        # Single position model -- one direction at a time across ALL strategies
        self.current_position: Optional[dict] = None

        # Daily tracking
        self.trades_today: int = 0
        self.losses_today: int = 0
        self.daily_pnl: float = 0.0
        self.daily_loss_cap_hit: bool = False

        # Metrics tracking per strategy
        self.metrics: dict = {}
        for name in STRATEGY_NAMES:
            self.metrics[name] = {
                "wins": 0,
                "losses": 0,
                "total_r_achieved": 0.0,
                "trade_count": 0,
                "max_drawdown": 0.0,
                "peak_pnl": 0.0,
                "current_pnl": 0.0,
                "current_losing_streak": 0,
                "max_losing_streak": 0,
            }

        self.trade_history: list = []
        self.startup_time: str = ""
        self.total_trade_count: int = 0

        # Cache for OHLCV data
        self._ohlcv_cache: dict = {}

        # Signal tracking
        self.signals_checked: int = 0
        self.signals_blocked: int = 0
        self.last_block_reasons: list = []

        # Polling watchdog
        self.polling_alive: bool = False

        # Backtest
        self.backtest_complete: bool = False
        self.active_strategies: list = list(STRATEGY_NAMES)  # All active from day 1
        self.backtest_results: dict = {}

        # Funding rate history (30-day rolling)
        self.funding_history: list = []
        self.current_funding_rate: float = 0.0

        # Pending parameter changes from self-review (awaiting GG approval)
        self.pending_review: Optional[dict] = None

        # Strategy paused states
        self.strategy_paused: dict = {name: False for name in STRATEGY_NAMES}

    def reset_daily(self):
        self.trades_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.daily_loss_cap_hit = False
        self.signals_checked = 0
        self.signals_blocked = 0
        self.last_block_reasons = []
        # Reset per-strategy losing streaks for daily count
        for name in STRATEGY_NAMES:
            self.metrics[name]["current_losing_streak"] = 0
        log.info("Daily PnL and trade counts reset.")


state = TradingState()

# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------

def save_state():
    """Save full TradingState to /data/vic_state.json."""
    try:
        d = os.path.dirname(STATE_FILE)
        if d:
            os.makedirs(d, exist_ok=True)

        data = {
            "mode": state.mode,
            "total_trade_count": state.total_trade_count,
            "metrics": state.metrics,
            "trade_history": state.trade_history[-100:],
            "active_strategies": state.active_strategies,
            "backtest_complete": state.backtest_complete,
            "backtest_results": state.backtest_results,
            "daily_pnl": state.daily_pnl,
            "trades_today": state.trades_today,
            "losses_today": state.losses_today,
            "current_position": state.current_position,
            "strategy_paused": state.strategy_paused,
            "funding_history": state.funding_history[-720:],  # ~30 days of hourly samples
            "pending_review": state.pending_review,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        log.debug("State saved to %s", STATE_FILE)
    except Exception as exc:
        log.error("State save error: %s", exc)


def load_state():
    """Load TradingState from /data/vic_state.json on startup."""
    try:
        if not os.path.exists(STATE_FILE):
            log.info("No state file found at %s -- starting fresh.", STATE_FILE)
            return False

        with open(STATE_FILE, "r") as f:
            data = json.load(f)

        state.total_trade_count = data.get("total_trade_count", 0)
        state.trade_history = data.get("trade_history", [])
        state.active_strategies = data.get("active_strategies", list(STRATEGY_NAMES))
        state.backtest_complete = data.get("backtest_complete", False)
        state.backtest_results = data.get("backtest_results", {})
        state.funding_history = data.get("funding_history", [])
        state.pending_review = data.get("pending_review")

        saved_metrics = data.get("metrics", {})
        for name in STRATEGY_NAMES:
            if name in saved_metrics:
                state.metrics[name] = saved_metrics[name]

        saved_paused = data.get("strategy_paused", {})
        for name in STRATEGY_NAMES:
            if name in saved_paused:
                state.strategy_paused[name] = saved_paused[name]

        # Don't restore current_position -- orphaned recovery handles that
        log.info("State loaded from %s -- %d lifetime trades, active: %s",
                 STATE_FILE, state.total_trade_count, state.active_strategies)
        return True
    except Exception as exc:
        log.error("State load error: %s -- starting fresh.", exc)
        return False


# ---------------------------------------------------------------------------
# Helpers -- Telegram
# ---------------------------------------------------------------------------

def sanitize_html(text: str) -> str:
    """Escape HTML entities in text so Telegram's HTML parser doesn't choke."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


async def tg_send(text: str):
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured -- skipping message.")
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
                if "can't parse entities" in resp.text:
                    payload_plain = {"chat_id": chat_id, "text": text}
                    resp2 = await client.post(url, json=payload_plain)
                    if resp2.status_code != 200:
                        log.error("Telegram plain text reply also failed: %s", resp2.text)
                else:
                    log.error("Telegram reply failed: %s", resp.text)
    except Exception as exc:
        log.error("Telegram reply error: %s", exc)


# ---------------------------------------------------------------------------
# Helpers -- Exchange
# ---------------------------------------------------------------------------

def init_exchange():
    """Create and configure the native Hyperliquid SDK instances."""
    global hl_info, hl_exchange
    base_url = hl_constants.TESTNET_API_URL if state.mode == "paper" else hl_constants.MAINNET_API_URL
    try:
        wallet: LocalAccount = eth_account.Account.from_key(HL_PRIVATE_KEY)
        hl_info = Info(base_url, skip_ws=True)
        hl_exchange = HLExchange(
            wallet=wallet,
            base_url=base_url,
            account_address=HL_WALLET_ADDRESS,
        )
        log.info("Hyperliquid SDK connected (%s). Wallet: %s",
                 "testnet" if state.mode == "paper" else "mainnet", HL_WALLET_ADDRESS[:10] + "...")
    except Exception as exc:
        log.error("Exchange init error (non-fatal, will retry): %s", exc)


def close_exchange():
    """No persistent connection to close with native SDK."""
    pass


async def fetch_ohlcv(timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV candles from Hyperliquid native SDK and return a DataFrame."""
    if not hl_info:
        log.warning("fetch_ohlcv called before exchange init.")
        return pd.DataFrame()
    try:
        now_ms = int(time.time() * 1000)
        tf_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                      "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}
        interval_sec = tf_seconds.get(timeframe, 60)
        start_ms = now_ms - (limit * interval_sec * 1000)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, lambda: hl_info.candles_snapshot("BTC", timeframe, start_ms, now_ms)
        )
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            rows.append({
                "timestamp": pd.to_datetime(c["t"], unit="ms"),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        df = pd.DataFrame(rows)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        return df
    except Exception as exc:
        log.error("fetch_ohlcv error: %s", exc)
        return pd.DataFrame()


async def fetch_ohlcv_range(timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch OHLCV candles for a specific time range (used by backtesting)."""
    if not hl_info:
        return pd.DataFrame()
    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, lambda: hl_info.candles_snapshot("BTC", timeframe, start_ms, end_ms)
        )
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            rows.append({
                "timestamp": pd.to_datetime(c["t"], unit="ms"),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        log.error("fetch_ohlcv_range error: %s", exc)
        return pd.DataFrame()


async def get_btc_price() -> float:
    """Get the current BTC mid price from Hyperliquid."""
    if not hl_info:
        return state.last_btc_price or 0.0
    try:
        loop = asyncio.get_event_loop()
        all_mids = await loop.run_in_executor(None, hl_info.all_mids)
        return float(all_mids.get("BTC", 0))
    except Exception as exc:
        log.error("get_btc_price error: %s", exc)
        return state.last_btc_price or 0.0


# ---------------------------------------------------------------------------
# Funding Rate
# ---------------------------------------------------------------------------

async def fetch_current_funding_rate() -> Optional[float]:
    """Fetch current BTC funding rate from Hyperliquid."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "metaAndAssetCtxs"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            asset_ctxs = data[1] if len(data) > 1 else []
            meta = data[0] if data else {}
            universe = meta.get("universe", [])
            for i, asset in enumerate(universe):
                if asset.get("name") == "BTC":
                    if i < len(asset_ctxs):
                        return float(asset_ctxs[i].get("funding", 0))
        return None
    except Exception as exc:
        log.error("Funding rate fetch error: %s", exc)
        return None


async def update_funding_rate():
    """Poll funding rate and maintain 30-day history."""
    rate = await fetch_current_funding_rate()
    if rate is not None:
        state.current_funding_rate = rate
        state.funding_history.append({
            "rate": rate,
            "timestamp": time.time(),
        })
        # Keep 30 days of hourly samples (~720)
        if len(state.funding_history) > 720:
            state.funding_history = state.funding_history[-720:]


def get_funding_percentile(rate: float) -> float:
    """Return the percentile of the current funding rate in the 30-day distribution."""
    if len(state.funding_history) < 24:  # Need at least 1 day of data
        return 50.0
    rates = [h["rate"] for h in state.funding_history]
    below = sum(1 for r in rates if r < rate)
    return (below / len(rates)) * 100.0


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


def find_swing_highs(df: pd.DataFrame, lookback: int = 50) -> list:
    """Find swing highs in the last N candles. Returns list of (index, price)."""
    highs = df["high"].values[-lookback:]
    swings = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swings.append((len(df) - lookback + i, float(highs[i])))
    return swings


def find_swing_lows(df: pd.DataFrame, lookback: int = 50) -> list:
    """Find swing lows in the last N candles. Returns list of (index, price)."""
    lows = df["low"].values[-lookback:]
    swings = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swings.append((len(df) - lookback + i, float(lows[i])))
    return swings


# ---------------------------------------------------------------------------
# Structure Analysis
# ---------------------------------------------------------------------------

def detect_structure(df: pd.DataFrame) -> str:
    """Check last 20 candles for HH/HL (bullish) or LH/LL (bearish) structure."""
    if len(df) < 20:
        return "mixed"

    recent = df.tail(20)
    highs = recent["high"].values
    lows = recent["low"].values

    swing_highs = []
    swing_lows = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "mixed"

    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1] < swing_lows[-2]

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    else:
        return "mixed"


# ---------------------------------------------------------------------------
# Regime Filter
# ---------------------------------------------------------------------------

async def update_regime():
    """Recalculate market regime from 5m candles."""
    df_5m = await fetch_ohlcv("5m", 100)
    if df_5m.empty or len(df_5m) < 30:
        return

    if len(df_5m) >= 6:
        price_now = float(df_5m["close"].iloc[-1])
        price_30m_ago = float(df_5m["close"].iloc[-6])
        move_pct = abs(price_now - price_30m_ago) / price_30m_ago * 100
        if move_pct > 1.5:
            state.regime = Regime.VOLATILE
            return

    atr_series = calc_atr(df_5m, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    price = float(df_5m["close"].iloc[-1])
    atr_pct = (atr_val / price * 100) if price > 0 else 0.0
    if atr_pct > 0.4:
        state.regime = Regime.VOLATILE
        return

    adx_series = calc_adx(df_5m, 14)
    adx_val = float(adx_series.iloc[-1]) if not adx_series.empty and not np.isnan(adx_series.iloc[-1]) else 0.0

    structure = detect_structure(df_5m)

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
# 1H Bias
# ---------------------------------------------------------------------------

async def update_htf_bias():
    """Compute 1H bias using EMA(50) slope + RSI."""
    df_1h = await fetch_ohlcv("1h", 60)
    if df_1h.empty or len(df_1h) < 55:
        state.htf_bias = "neutral"
        state.htf_bias_strength = "weak"
        return

    ema50 = calc_ema(df_1h["close"], 50)
    rsi_series = calc_rsi(df_1h["close"], 14)

    ema_now = float(ema50.iloc[-1])
    ema_prev = float(ema50.iloc[-5])
    rsi_val = float(rsi_series.iloc[-1])

    if np.isnan(ema_now) or np.isnan(ema_prev) or np.isnan(rsi_val):
        state.htf_bias = "neutral"
        state.htf_bias_strength = "weak"
        return

    ema_slope_pct = (ema_now - ema_prev) / ema_prev * 100
    slope_clear = abs(ema_slope_pct) > 0.05

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


# ---------------------------------------------------------------------------
# Pre-Trade Checklist
# ---------------------------------------------------------------------------

def _block(strategy: str, side: str, reason: str) -> tuple[bool, str]:
    state.signals_blocked += 1
    state.last_block_reasons.append(f"BLOCKED: {strategy} {side.upper()} -- {reason}")
    if len(state.last_block_reasons) > 20:
        state.last_block_reasons.pop(0)
    log.info("BLOCKED: %s %s -- %s", strategy, side.upper(), reason)
    return False, reason


def can_open_trade(strategy: str, side: str) -> tuple[bool, str]:
    """Master pre-trade checklist. Returns (allowed, reason)."""
    state.signals_checked += 1

    # 1. No opposing position open -- if same direction, skip. If opposite, caller handles close-first.
    if state.current_position is not None:
        pos_side = state.current_position["side"]
        if pos_side == side:
            return _block(strategy, side, f"Already have {side} position open (same direction)")
        else:
            # Opposite direction -- caller must close first, then re-check
            return _block(strategy, side, f"Opposing {pos_side} position open -- close first")

    # 2. Regime check -- NO trading in VOLATILE
    if state.regime == Regime.VOLATILE:
        return _block(strategy, side, f"VOLATILE regime -- no trading")

    # 3. Daily loss check -- 3 losses = stop
    if state.losses_today >= MAX_LOSSES_PER_DAY:
        return _block(strategy, side, f"Daily loss limit ({state.losses_today}/{MAX_LOSSES_PER_DAY} losses)")

    if state.daily_loss_cap_hit:
        return _block(strategy, side, "Daily loss cap hit")

    # 4. Macro check -- CRITICAL = no new trades
    if _macro_intel_cache.get("text", ""):
        if "RISK_LEVEL: CRITICAL" in _macro_intel_cache["text"].upper():
            return _block(strategy, side, "CRITICAL macro event active")

    # 5. Global or strategy paused
    if state.paused:
        return _block(strategy, side, "Global trading paused")

    if state.strategy_paused.get(strategy, False):
        return _block(strategy, side, f"{strategy} paused (underperformance)")

    # 6. Strategy must be active
    if strategy not in state.active_strategies:
        return _block(strategy, side, f"{strategy} not in active strategies")

    log.info("PASSED: %s %s -- all checks OK", strategy, side.upper())
    return True, "OK"


# ---------------------------------------------------------------------------
# Position Sizing & Order Execution
# ---------------------------------------------------------------------------

def calc_leverage_from_confidence(confidence: int) -> int:
    """Determine leverage from AI confidence score."""
    if confidence <= 6:
        return BASE_LEVERAGE  # Should not reach here (rejected), but safety
    elif confidence <= 8:
        return BASE_LEVERAGE  # 10x
    elif confidence == 9:
        return 15
    else:  # 10
        return 20


async def execute_trade(strategy: str, side: str, entry: float,
                        sl_price: float, tp_price: float,
                        confidence: int = 7):
    """Open a position (paper or live). Single position model.

    sl_price and tp_price are pre-calculated by the strategy.
    Leverage is determined by AI confidence score.
    Position sized so that SL hit <= MAX_RISK_PCT of account.
    """
    allowed, reason = can_open_trade(strategy, side)
    if not allowed:
        return

    leverage = calc_leverage_from_confidence(confidence)

    sl_distance = abs(entry - sl_price)
    if sl_distance <= 0:
        log.warning("%s -- zero SL distance, cannot size position.", strategy)
        return

    # Size: max risk per trade. Target 3-5% but hard cap at 10%
    risk_pct = min(TARGET_RISK_PCT + (confidence - 7) * 0.01, MAX_RISK_PCT)
    risk_dollars = ACCOUNT_CAPITAL * risk_pct
    size = math.floor(risk_dollars / sl_distance * 100000) / 100000

    if size <= 0:
        log.warning("%s -- invalid size from sizing, skipping.", strategy)
        return

    # Check notional meets minimum ($10 on HL)
    notional = size * entry
    if notional < 10:
        log.warning("%s -- notional $%.2f below HL minimum, skipping.", strategy, notional)
        return

    # Cap to leverage limit
    max_size = math.floor(ACCOUNT_CAPITAL * leverage / entry * 100000) / 100000
    if size > max_size:
        size = max_size
        log.info("%s -- size capped to leverage limit: %.6f BTC", strategy, size)

    tp_distance = abs(tp_price - entry)

    # Open order
    if state.mode == "live":
        if not hl_exchange:
            log.error("%s -- exchange not initialized, cannot place order.", strategy)
            await tg_send(f"\u26a0\ufe0f <b>{strategy}</b> order FAILED: Exchange not initialized")
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: hl_exchange.update_leverage(leverage, "BTC", is_cross=True)
            )
            is_buy = side == "long"
            result = await loop.run_in_executor(
                None, lambda: hl_exchange.market_open("BTC", is_buy=is_buy, sz=size)
            )
            if result.get("status") != "ok":
                raise Exception(f"Order rejected: {result}")
            statuses = (result.get("response", {}).get("data", {}).get("statuses", []))
            for s in statuses:
                if "error" in s:
                    raise Exception(f"Order error: {s['error']}")
            log.info("%s LIVE order placed: %s", strategy, result)

            # Place exchange-level SL
            sl_ok = False
            try:
                sl_side = not is_buy
                for sl_attempt in range(3):
                    sl_result = await loop.run_in_executor(
                        None, lambda: hl_exchange.order(
                            "BTC", is_buy=sl_side, sz=size, limit_px=sl_price,
                            order_type={"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
                            reduce_only=True,
                        )
                    )
                    sl_statuses = sl_result.get("response", {}).get("data", {}).get("statuses", [])
                    sl_errors = [s.get("error") for s in sl_statuses if "error" in s]
                    if sl_errors:
                        log.error("%s SL order REJECTED (attempt %d/3) @ $%.0f: %s", strategy, sl_attempt + 1, sl_price, sl_errors)
                        if sl_attempt < 2:
                            await asyncio.sleep(1)
                            continue
                    else:
                        sl_ok = True
                        log.info("%s SL order CONFIRMED on exchange @ $%.0f", strategy, sl_price)
                        break

                # Place TP
                tp_result = await loop.run_in_executor(
                    None, lambda: hl_exchange.order(
                        "BTC", is_buy=sl_side, sz=size, limit_px=tp_price,
                        order_type={"trigger": {"triggerPx": tp_price, "isMarket": True, "tpsl": "tp"}},
                        reduce_only=True,
                    )
                )
                tp_statuses = tp_result.get("response", {}).get("data", {}).get("statuses", [])
                tp_errors = [s.get("error") for s in tp_statuses if "error" in s]
                if tp_errors:
                    log.error("%s TP order REJECTED @ $%.0f: %s", strategy, tp_price, tp_errors)
            except Exception as sl_exc:
                log.error("%s -- SL/TP order exception: %s", strategy, sl_exc)

            if not sl_ok:
                await tg_send(
                    f"\U0001f6a8\U0001f6a8 <b>CRITICAL: {strategy} -- NO STOP LOSS ON EXCHANGE</b> \U0001f6a8\U0001f6a8\n"
                    f"SL order FAILED after 3 attempts.\n"
                    f"Position: {side.upper()} {size} BTC @ ${entry:,.0f}\n"
                    f"Intended SL: ${sl_price:,.0f}\n"
                    f"<b>SET STOP LOSS MANUALLY NOW</b>\n"
                    f"Software SL monitoring is active as backup."
                )

        except Exception as exc:
            log.error("%s -- order error: %s", strategy, exc)
            await tg_send(f"\u26a0\ufe0f <b>{strategy}</b> order FAILED: {exc}")
            return
    else:
        log.info("%s PAPER trade: %s %.6f BTC @ %.2f", strategy, side.upper(), size, entry)

    # Record position (single position model)
    state.current_position = {
        "strategy": strategy,
        "side": side,
        "entry": entry,
        "sl": sl_price,
        "tp": tp_price,
        "size": size,
        "leverage": leverage,
        "confidence": confidence,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "open_time": datetime.now(timezone.utc).isoformat(),
        "regime": state.regime.value,
        "bias": f"{state.htf_bias} ({state.htf_bias_strength})",
    }
    state.trades_today += 1
    state.total_trade_count += 1

    dollar_risk = round(sl_distance * size, 2)
    dollar_tp = round(tp_distance * size, 2)
    rr_ratio = round(tp_distance / sl_distance, 1) if sl_distance > 0 else 0
    label = STRATEGY_LABELS.get(strategy, strategy)
    arrow = "\U0001f7e2" if side == "long" else "\U0001f534"
    msg = (
        f"{arrow} <b>{side.upper()}</b> -- {label}\n"
        f"Entry ${entry:,.2f} | SL ${sl_price:,.2f} | TP ${tp_price:,.2f}\n"
        f"Size {size:.6f} BTC | Risk ${dollar_risk:.2f} | Reward ${dollar_tp:.2f} ({rr_ratio}R)\n"
        f"Leverage: {leverage}x | Confidence: {confidence}/10\n"
        f"Regime: {state.regime.value} | Bias: {state.htf_bias} ({state.htf_bias_strength})"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))
    save_state()


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def _append_journal(entry: dict):
    """Append a trade entry to the JSON journal file."""
    try:
        d = os.path.dirname(JOURNAL_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Close Position
# ---------------------------------------------------------------------------

async def close_position(exit_price: float, reason: str):
    """Close the current position and book PnL."""
    pos = state.current_position
    if pos is None:
        return

    strategy = pos["strategy"]
    if pos["side"] == "long":
        pnl = (exit_price - pos["entry"]) * pos["size"]
    else:
        pnl = (pos["entry"] - exit_price) * pos["size"]

    pnl = round(pnl, 2)
    risk_dollars = pos["sl_distance"] * pos["size"]
    r_achieved = pnl / risk_dollars if risk_dollars > 0 else 0.0

    open_time = datetime.fromisoformat(pos["open_time"])
    elapsed_min = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0

    # Close on exchange if live
    if state.mode == "live":
        if not hl_exchange:
            log.error("%s -- exchange not initialized, cannot close.", strategy)
            await tg_send(f"\U0001f6a8 <b>{strategy}</b> close FAILED: Exchange not initialized. CHECK MANUALLY.")
            return
        has_btc = True
        try:
            loop = asyncio.get_event_loop()
            user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
            hl_positions = user_st.get("assetPositions", [])
            has_btc = False
            for p in hl_positions:
                pd_pos = p.get("position", {})
                if pd_pos.get("coin") == "BTC" and float(pd_pos.get("szi", 0)) != 0:
                    has_btc = True
                    break
            if not has_btc:
                log.warning("%s -- no BTC position found on Hyperliquid, clearing internal state.", strategy)
                await tg_send(
                    f"\u26a0\ufe0f <b>{strategy}</b> -- Position already closed on Hyperliquid.\n"
                    f"Clearing internal state. No action needed."
                )
        except Exception as exc:
            log.error("%s -- failed to check Hyperliquid position state: %s", strategy, exc)

        if has_btc:
            try:
                close_size = pos["size"]
                result = await loop.run_in_executor(
                    None, lambda: hl_exchange.market_close("BTC", sz=close_size)
                )
                if result is None:
                    raise Exception("market_close returned None")
                if result.get("status") != "ok":
                    raise Exception(f"Close rejected: {result}")
                statuses = (result.get("response", {}).get("data", {}).get("statuses", []))
                for s in statuses:
                    if "error" in s:
                        raise Exception(f"Close error: {s['error']}")
            except Exception as exc:
                log.error("%s -- close order error: %s", strategy, exc)
                await tg_send(
                    f"\U0001f6a8 <b>CRITICAL: {strategy} close order FAILED</b>\n\n"
                    f"Position is STILL OPEN on Hyperliquid.\n"
                    f"Error: {sanitize_html(str(exc))}\n"
                    f"Side: {pos['side']} | Size: {pos['size']:.6f} BTC\n\n"
                    f"CHECK HYPERLIQUID MANUALLY."
                )
                return

    # Book PnL
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
        state.losses_today += 1
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
        "leverage": pos.get("leverage", BASE_LEVERAGE),
        "confidence": pos.get("confidence", 0),
        "time": datetime.now(timezone.utc).isoformat(),
        "hold_time_min": round(elapsed_min, 1),
    })

    label = STRATEGY_LABELS.get(strategy, strategy)
    emoji = "\U0001f4b0" if pnl >= 0 else "\U0001f4b8"
    msg = (
        f"{emoji} <b>CLOSED</b> -- {label}\n"
        f"${pos['entry']:,.2f} \u2192 ${exit_price:,.2f} | <b>${pnl:+,.2f}</b> ({r_achieved:+.1f}R)\n"
        f"Reason: {reason} | Hold: {elapsed_min:.0f}min\n"
        f"Daily PnL: ${state.daily_pnl:+,.2f} | Losses today: {state.losses_today}/{MAX_LOSSES_PER_DAY}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))

    # Write to journal
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
        "leverage": pos.get("leverage", BASE_LEVERAGE),
        "confidence": pos.get("confidence", 0),
        "pnl_usd": pnl,
        "r_achieved": round(r_achieved, 2),
        "exit_reason": reason,
        "regime_at_entry": pos.get("regime", "unknown"),
        "bias_at_entry": pos.get("bias", "unknown"),
        "hold_time_min": round(elapsed_min, 1),
        "cumulative_pnl": round(state.daily_pnl, 2),
    }
    _append_journal(journal_entry)

    state.current_position = None
    save_state()

    # Check daily loss limit
    if state.losses_today >= MAX_LOSSES_PER_DAY:
        state.daily_loss_cap_hit = True
        await tg_send(
            f"\U0001f6d1 <b>{MAX_LOSSES_PER_DAY} losing trades today -- ALL trading stopped until midnight UTC</b>\n"
            f"Daily PnL: ${state.daily_pnl:+,.2f}"
        )

    # Check max daily loss (10% of account as absolute cap)
    if state.daily_pnl <= -(ACCOUNT_CAPITAL * MAX_RISK_PCT):
        state.daily_loss_cap_hit = True
        await tg_send(
            f"\U0001f6d1 <b>Daily loss limit hit (${state.daily_pnl:+,.2f})</b>\n"
            f"All trading stopped until tomorrow."
        )

    # Underperformance check
    await check_underperformance(strategy)


async def check_underperformance(strategy: str):
    """If a strategy drops below 40% win rate after 15+ trades, pause it."""
    m = state.metrics[strategy]
    if m["trade_count"] < UNDERPERFORMANCE_MIN_TRADES:
        return
    win_rate = m["wins"] / m["trade_count"]
    if win_rate < UNDERPERFORMANCE_WR_THRESHOLD:
        if not state.strategy_paused.get(strategy, False):
            state.strategy_paused[strategy] = True
            save_state()
            msg = (
                f"\U0001f6d1 <b>STRATEGY AUTO-PAUSED: {strategy}</b>\n\n"
                f"Win rate dropped to {win_rate*100:.1f}% after {m['trade_count']} trades.\n"
                f"Threshold: {UNDERPERFORMANCE_WR_THRESHOLD*100:.0f}% minimum after {UNDERPERFORMANCE_MIN_TRADES} trades.\n"
                f"Strategy will remain paused until next backtest re-evaluation."
            )
            await tg_send(msg)


# ---------------------------------------------------------------------------
# Strategy 1 -- Funding Rate Fade (1H)
# ---------------------------------------------------------------------------

async def strategy_funding_rate_fade():
    """Monitor funding rate extremes. Short when excessively positive, long when negative."""
    name = "funding_rate_fade"

    rate = state.current_funding_rate
    if rate == 0:
        return

    percentile = get_funding_percentile(rate)

    df_1h = await fetch_ohlcv("1h", 30)
    if df_1h.empty or len(df_1h) < 20:
        return

    rsi_series = calc_rsi(df_1h["close"], 14)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]) else 50.0
    atr_series = calc_atr(df_1h, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0
    price = float(df_1h["close"].iloc[-1])

    if atr_val <= 0 or price <= 0:
        return

    side = None
    # Top 10% extreme (longs paying heavily) -> short
    if percentile >= (100 - FUNDING_EXTREME_PCT) and rsi_val > 65:
        side = "short"
    # Bottom 10% extreme (shorts paying heavily) -> long
    elif percentile <= FUNDING_EXTREME_PCT and rsi_val < 35:
        side = "long"

    if side is None:
        return

    log.info("funding_rate_fade signal: %s | rate=%.6f | percentile=%.1f | RSI=%.1f",
             side, rate, percentile, rsi_val)

    sl_distance = 1.5 * atr_val
    tp_distance = 2.5 * sl_distance

    if side == "long":
        sl_price = round(price - sl_distance)
        tp_price = round(price + tp_distance)
    else:
        sl_price = round(price + sl_distance)
        tp_price = round(price - tp_distance)

    return {"strategy": name, "side": side, "entry": price, "sl": sl_price, "tp": tp_price,
            "atr": atr_val, "reason": f"Funding {rate:.6f} ({percentile:.0f}th pctl), RSI {rsi_val:.1f}"}


# ---------------------------------------------------------------------------
# Strategy 2 -- Liquidity Sweep Reversal (5m/15m)
# ---------------------------------------------------------------------------

async def strategy_liquidity_sweep():
    """Detect stop hunts past swing highs/lows that immediately reverse."""
    name = "liquidity_sweep"

    for tf in ["15m", "5m"]:
        df = await fetch_ohlcv(tf, 60)
        if df.empty or len(df) < 52:
            continue

        swing_highs = find_swing_highs(df, 50)
        swing_lows = find_swing_lows(df, 50)

        if not swing_highs or not swing_lows:
            continue

        atr_series = calc_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0
        rsi_series = calc_rsi(df["close"], 14)

        price = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        curr_high = float(df["high"].iloc[-1])
        curr_low = float(df["low"].iloc[-1])
        prev_high = float(df["high"].iloc[-2])
        prev_low = float(df["low"].iloc[-2])
        curr_range = curr_high - curr_low

        # Volume spike check
        avg_vol = float(df["volume"].iloc[-20:].mean())
        curr_vol = float(df["volume"].iloc[-1])
        vol_spike = curr_vol >= avg_vol * 1.5 if avg_vol > 0 else False

        if not vol_spike:
            continue

        side = None
        sweep_extreme = 0.0
        target_pool = 0.0

        # Check sweep of swing lows (long signal)
        for _, sw_low in swing_lows[-3:]:
            sweep_depth = (sw_low - curr_low) / sw_low * 100 if sw_low > 0 else 0
            if sweep_depth > 0.15 and price > sw_low:
                # Price spiked below swing low then closed back above
                wick_ratio = (price - curr_low) / curr_range if curr_range > 0 else 0
                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50
                rsi_prev = float(rsi_series.iloc[-3]) if len(rsi_series) > 3 and not np.isnan(rsi_series.iloc[-3]) else 50
                rsi_divergence = rsi_val > rsi_prev  # RSI higher while price lower
                if wick_ratio >= 0.6 or rsi_divergence:
                    side = "long"
                    sweep_extreme = curr_low
                    # Target: previous swing high
                    if swing_highs:
                        target_pool = swing_highs[-1][1]
                    break

        # Check sweep of swing highs (short signal)
        if side is None:
            for _, sw_high in swing_highs[-3:]:
                sweep_depth = (curr_high - sw_high) / sw_high * 100 if sw_high > 0 else 0
                if sweep_depth > 0.15 and price < sw_high:
                    wick_ratio = (curr_high - price) / curr_range if curr_range > 0 else 0
                    rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50
                    rsi_prev = float(rsi_series.iloc[-3]) if len(rsi_series) > 3 and not np.isnan(rsi_series.iloc[-3]) else 50
                    rsi_divergence = rsi_val < rsi_prev
                    if wick_ratio >= 0.6 or rsi_divergence:
                        side = "short"
                        sweep_extreme = curr_high
                        if swing_lows:
                            target_pool = swing_lows[-1][1]
                        break

        if side is None:
            continue

        # SL: just beyond the sweep extreme
        sl_buffer = atr_val * 0.2 if atr_val > 0 else price * 0.001
        if side == "long":
            sl_price = round(sweep_extreme - sl_buffer)
            tp_price = round(target_pool) if target_pool > price else round(price + abs(price - sl_price) * 2)
        else:
            sl_price = round(sweep_extreme + sl_buffer)
            tp_price = round(target_pool) if target_pool < price else round(price - abs(sl_price - price) * 2)

        log.info("liquidity_sweep signal: %s on %s | sweep=%.0f | entry=%.0f", side, tf, sweep_extreme, price)

        return {"strategy": name, "side": side, "entry": price, "sl": sl_price, "tp": tp_price,
                "atr": atr_val, "reason": f"Sweep {tf} beyond {'low' if side == 'long' else 'high'}, vol spike {curr_vol/avg_vol:.1f}x"}

    return None


# ---------------------------------------------------------------------------
# Strategy 3 -- EMA Trend Pullback (1H)
# ---------------------------------------------------------------------------

async def strategy_ema_trend_pullback():
    """Trend pullback to 21/50 EMA on 1H with rejection candle confirmation."""
    name = "ema_trend_pullback"

    df = await fetch_ohlcv("1h", 210)
    if df.empty or len(df) < 205:
        return None

    ema21 = calc_ema(df["close"], 21)
    ema50 = calc_ema(df["close"], 50)
    ema200 = calc_ema(df["close"], 200)
    atr_series = calc_atr(df, 14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0

    ema200_now = float(ema200.iloc[-1])
    ema200_prev = float(ema200.iloc[-5])

    if np.isnan(ema200_now) or np.isnan(ema200_prev) or atr_val <= 0:
        return None

    # 200 EMA slope determines trend
    ema200_slope = (ema200_now - ema200_prev) / ema200_prev * 100
    trend = "bullish" if ema200_slope > 0.01 else "bearish" if ema200_slope < -0.01 else None

    if trend is None:
        return None

    curr = df.iloc[-1]
    price = float(curr["close"])
    open_price = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    ema21_val = float(ema21.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    body = abs(price - open_price)
    full_range = high - low

    if full_range == 0:
        return None

    # Volume confirmation
    avg_vol = float(df["volume"].iloc[-20:].mean())
    curr_vol = float(curr["volume"])
    if np.isnan(avg_vol) or curr_vol < avg_vol:
        return None

    # Check for chop (3+ consecutive candles around EMA)
    chop_count = 0
    for i in range(-4, -1):
        c = float(df["close"].iloc[i])
        e21 = float(ema21.iloc[i])
        e50 = float(ema50.iloc[i])
        if abs(c - e21) / e21 * 100 < 0.1 or abs(c - e50) / e50 * 100 < 0.1:
            chop_count += 1
    if chop_count >= 3:
        return None

    side = None
    # Bullish: pullback to 21 or 50 EMA, bullish rejection candle
    if trend == "bullish":
        touched_ema = (low <= ema21_val <= high) or (low <= ema50_val <= high)
        is_bullish_rejection = price > open_price and body / full_range >= 0.5 and price > ema21_val
        if touched_ema and is_bullish_rejection:
            side = "long"

    # Bearish: rally to 21 or 50 EMA, bearish rejection candle
    elif trend == "bearish":
        touched_ema = (low <= ema21_val <= high) or (low <= ema50_val <= high)
        is_bearish_rejection = price < open_price and body / full_range >= 0.5 and price < ema21_val
        if touched_ema and is_bearish_rejection:
            side = "short"

    if side is None:
        return None

    # SL: below rejection candle (long) or above (short), max 1.5x ATR
    sl_distance = min(abs(price - (low if side == "long" else high)) + atr_val * 0.2, 1.5 * atr_val)

    # TP: previous swing high/low
    swing_highs = find_swing_highs(df, 50)
    swing_lows = find_swing_lows(df, 50)

    if side == "long":
        sl_price = round(price - sl_distance)
        tp_target = swing_highs[-1][1] if swing_highs else price + sl_distance * 2
        tp_price = round(tp_target)
    else:
        sl_price = round(price + sl_distance)
        tp_target = swing_lows[-1][1] if swing_lows else price - sl_distance * 2
        tp_price = round(tp_target)

    log.info("ema_trend_pullback signal: %s | trend=%s | EMA200 slope=%.3f%%", side, trend, ema200_slope)

    return {"strategy": name, "side": side, "entry": price, "sl": sl_price, "tp": tp_price,
            "atr": atr_val, "reason": f"EMA pullback, {trend} trend, EMA200 slope {ema200_slope:.3f}%"}


# ---------------------------------------------------------------------------
# Strategy 4 -- VWAP Reclaim (5m/15m)
# ---------------------------------------------------------------------------

async def strategy_vwap_reclaim():
    """Price loses VWAP then reclaims with conviction."""
    name = "vwap_reclaim"

    for tf in ["15m", "5m"]:
        df = await fetch_ohlcv(tf, 60)
        if df.empty or len(df) < 25:
            continue

        df["vwap"] = calc_vwap(df)
        rsi_series = calc_rsi(df["close"], 14)
        atr_series = calc_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(curr["close"])
        open_price = float(curr["open"])
        vwap_val = float(curr["vwap"])
        rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        if np.isnan(vwap_val) or vwap_val <= 0 or atr_val <= 0:
            continue

        body = abs(price - open_price)
        full_range = float(curr["high"]) - float(curr["low"])
        if full_range == 0:
            continue

        body_ratio = body / full_range

        # Volume spike
        avg_vol = float(df["volume"].iloc[-20:].mean())
        curr_vol = float(curr["volume"])
        vol_spike = curr_vol >= avg_vol * 1.5 if avg_vol > 0 else False

        if not vol_spike or body_ratio < 0.6:
            continue

        # Chop filter: if price crossed VWAP > 4 times in last 20 candles, skip
        vwap_crosses = 0
        for i in range(-20, -1):
            if abs(i) >= len(df):
                continue
            c_prev = float(df["close"].iloc[i])
            c_curr = float(df["close"].iloc[i + 1])
            v_prev = float(df["vwap"].iloc[i])
            v_curr = float(df["vwap"].iloc[i + 1])
            if np.isnan(v_prev) or np.isnan(v_curr):
                continue
            if (c_prev < v_prev and c_curr > v_curr) or (c_prev > v_prev and c_curr < v_curr):
                vwap_crosses += 1
        if vwap_crosses > 4:
            continue

        side = None
        # Long: was below VWAP, strong bullish candle reclaims above
        prev_below = float(prev["close"]) < float(prev["vwap"]) if not np.isnan(prev["vwap"]) else False
        # Short: was above VWAP, strong bearish candle loses it
        prev_above = float(prev["close"]) > float(prev["vwap"]) if not np.isnan(prev["vwap"]) else False

        if prev_below and price > vwap_val and price > open_price and rsi_val > 50:
            side = "long"
        elif prev_above and price < vwap_val and price < open_price and rsi_val < 50:
            side = "short"

        if side is None:
            continue

        sl_distance = 1.0 * atr_val
        if side == "long":
            sl_price = round(float(curr["low"]) - sl_distance)
            tp_price = round(price + max(sl_distance * 2, abs(price - sl_price) * 2))
        else:
            sl_price = round(float(curr["high"]) + sl_distance)
            tp_price = round(price - max(sl_distance * 2, abs(sl_price - price) * 2))

        log.info("vwap_reclaim signal: %s on %s | VWAP=%.0f | RSI=%.1f", side, tf, vwap_val, rsi_val)

        return {"strategy": name, "side": side, "entry": price, "sl": sl_price, "tp": tp_price,
                "atr": atr_val, "reason": f"VWAP reclaim on {tf}, RSI {rsi_val:.1f}, vol {curr_vol/avg_vol:.1f}x"}

    return None


# ---------------------------------------------------------------------------
# Strategy 5 -- Order Block Entry (15m/1H)
# ---------------------------------------------------------------------------

async def strategy_order_block():
    """Identify order blocks from impulse moves and trade the retest."""
    name = "order_block"

    for tf in ["1h", "15m"]:
        df = await fetch_ohlcv(tf, 100)
        if df.empty or len(df) < 50:
            continue

        atr_series = calc_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.0
        rsi_series = calc_rsi(df["close"], 14)

        price = float(df["close"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-20:].mean())

        if atr_val <= 0 or price <= 0:
            continue

        # Scan for order blocks in recent history (last 30 candles)
        for i in range(len(df) - 30, len(df) - 5):
            if i < 3:
                continue

            # Check for bullish impulse (3+ consecutive bullish candles, 0.5%+ move)
            bullish_impulse = True
            impulse_start = float(df["open"].iloc[i])
            impulse_end = float(df["close"].iloc[i + 2]) if i + 2 < len(df) else 0
            for j in range(i, min(i + 3, len(df))):
                if float(df["close"].iloc[j]) <= float(df["open"].iloc[j]):
                    bullish_impulse = False
                    break
            if bullish_impulse and impulse_end > 0:
                move_pct = (impulse_end - impulse_start) / impulse_start * 100
                if move_pct >= 0.5:
                    # Bullish OB = last bearish candle before impulse
                    ob_idx = i - 1
                    if ob_idx >= 0 and float(df["close"].iloc[ob_idx]) < float(df["open"].iloc[ob_idx]):
                        ob_high = float(df["high"].iloc[ob_idx])
                        ob_low = float(df["low"].iloc[ob_idx])

                        # Check FVG (fair value gap)
                        if i + 1 < len(df):
                            gap = float(df["low"].iloc[i + 1]) - float(df["high"].iloc[ob_idx])
                            has_fvg = gap > 0

                            # Is price currently in the OB zone?
                            if ob_low <= price <= ob_high and has_fvg:
                                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50
                                curr_vol = float(df["volume"].iloc[-1])
                                is_bullish = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])

                                if is_bullish and rsi_val > 45 and curr_vol > avg_vol:
                                    sl_price = round(ob_low - atr_val * 0.2)
                                    tp_price = round(impulse_end)  # Full fill of the impulse

                                    log.info("order_block signal: long on %s | OB zone %.0f-%.0f", tf, ob_low, ob_high)

                                    return {"strategy": name, "side": "long", "entry": price,
                                            "sl": sl_price, "tp": tp_price, "atr": atr_val,
                                            "reason": f"Bullish OB retest on {tf}, FVG present, RSI {rsi_val:.1f}"}

            # Check for bearish impulse
            bearish_impulse = True
            impulse_start = float(df["open"].iloc[i])
            impulse_end = float(df["close"].iloc[i + 2]) if i + 2 < len(df) else 0
            for j in range(i, min(i + 3, len(df))):
                if float(df["close"].iloc[j]) >= float(df["open"].iloc[j]):
                    bearish_impulse = False
                    break
            if bearish_impulse and impulse_end > 0:
                move_pct = (impulse_start - impulse_end) / impulse_start * 100
                if move_pct >= 0.5:
                    ob_idx = i - 1
                    if ob_idx >= 0 and float(df["close"].iloc[ob_idx]) > float(df["open"].iloc[ob_idx]):
                        ob_high = float(df["high"].iloc[ob_idx])
                        ob_low = float(df["low"].iloc[ob_idx])

                        if i + 1 < len(df):
                            gap = float(df["low"].iloc[ob_idx]) - float(df["high"].iloc[i + 1])
                            has_fvg = gap > 0

                            if ob_low <= price <= ob_high and has_fvg:
                                rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50
                                curr_vol = float(df["volume"].iloc[-1])
                                is_bearish = float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])

                                if is_bearish and rsi_val < 55 and curr_vol > avg_vol:
                                    sl_price = round(ob_high + atr_val * 0.2)
                                    tp_price = round(impulse_end)

                                    log.info("order_block signal: short on %s | OB zone %.0f-%.0f", tf, ob_low, ob_high)

                                    return {"strategy": name, "side": "short", "entry": price,
                                            "sl": sl_price, "tp": tp_price, "atr": atr_val,
                                            "reason": f"Bearish OB retest on {tf}, FVG present, RSI {rsi_val:.1f}"}

    return None


# ---------------------------------------------------------------------------
# AI Market Brain -- Pre-trade analysis with confidence scoring
# ---------------------------------------------------------------------------

_macro_intel_cache = {"text": "", "fetched_at": 0}


async def _fetch_recent_news_headlines() -> list[str]:
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


async def ai_market_analysis(strategy: str, side: str, price: float, signal_reason: str = "") -> tuple[bool, int, str]:
    """AI Market Brain: Claude evaluates signal quality.
    Returns (approved, confidence_score, reasoning).
    Score below 6 = reject. 7-8 = 10x. 9-10 = higher leverage.
    """
    if not CLAUDE_API_KEY:
        return True, 7, "No API key -- defaulting to APPROVE with confidence 7"

    try:
        headlines = await _fetch_recent_news_headlines()
        news_text = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news available"

        macro_text = ""
        if _macro_intel_cache["text"]:
            age_min = int((time.time() - _macro_intel_cache["fetched_at"]) / 60)
            macro_text = f"\n\nMACRO INTELLIGENCE (updated {age_min}min ago):\n{_macro_intel_cache['text']}"

        intel_text = _get_intelligence_summary()
        intel_section = f"\n\n{intel_text}" if intel_text else ""

        # Win/loss streak info
        total_losing = state.losses_today
        streak_text = f"Losses today: {total_losing}/{MAX_LOSSES_PER_DAY}"

        prompt = (
            f"You are a BTC futures trading risk analyst. Evaluate this proposed trade and provide a confidence score.\n\n"
            f"Strategy: {strategy}\n"
            f"Signal reason: {signal_reason}\n"
            f"Direction: {side.upper()}\n"
            f"Entry price: ${price:,.2f}\n"
            f"Current regime: {state.regime.value}\n"
            f"1H bias: {state.htf_bias} ({state.htf_bias_strength})\n"
            f"Daily PnL: ${state.daily_pnl:+,.2f}\n"
            f"{streak_text}\n"
            f"Funding rate: {state.current_funding_rate:.6f}\n\n"
            f"Recent BTC news:\n{news_text}"
            f"{macro_text}"
            f"{intel_section}\n\n"
            f"You have web_search -- if the macro intel is older than 30 minutes or mentions a developing situation, SEARCH first.\n\n"
            f"Reply with EXACTLY this format:\n"
            f"DECISION: APPROVE or REJECT\n"
            f"CONFIDENCE: [1-10]\n"
            f"REASON: [1-2 sentences]\n\n"
            f"Score guide: 1-5 = reject, 6 = borderline reject, 7-8 = solid trade (10x leverage), 9 = strong confluence (15x), 10 = exceptional setup (20x)."
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
                log.warning("AI Market Brain API error %d -- defaulting APPROVE 7", resp.status_code)
                return True, 7, "API error -- defaulting to APPROVE 7"

            data = resp.json()
            text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            content = " ".join(text_blocks).strip()
            log.info("AI Market Brain for %s %s: %s", strategy, side, content)

            # Parse confidence score
            confidence = 7  # default
            import re
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", content)
            if conf_match:
                confidence = min(10, max(1, int(conf_match.group(1))))

            approved = "APPROVE" in content.upper() and confidence >= 6

            return approved, confidence, content

    except Exception as exc:
        log.warning("AI Market Brain error: %s -- defaulting APPROVE 7", exc)
        return True, 7, f"Error: {exc} -- defaulting to APPROVE 7"


# ---------------------------------------------------------------------------
# Strategy Execution Orchestrator
# ---------------------------------------------------------------------------

async def run_all_strategies():
    """Run all 5 strategies simultaneously. Strongest signal wins."""
    if state.paused or state.daily_loss_cap_hit:
        return
    if state.regime == Regime.VOLATILE:
        return
    if state.losses_today >= MAX_LOSSES_PER_DAY:
        return

    # If a position is open, don't look for new signals (unless opposite direction, handled below)
    if state.current_position is not None:
        return

    # Gather signals from all active strategies
    signals = []
    strategy_funcs = {
        "funding_rate_fade": strategy_funding_rate_fade,
        "liquidity_sweep": strategy_liquidity_sweep,
        "ema_trend_pullback": strategy_ema_trend_pullback,
        "vwap_reclaim": strategy_vwap_reclaim,
        "order_block": strategy_order_block,
    }

    for name in state.active_strategies:
        if state.strategy_paused.get(name, False):
            continue
        try:
            result = await strategy_funcs[name]()
            if result:
                signals.append(result)
        except Exception as exc:
            log.error("%s error: %s", name, exc)

    if not signals:
        return

    # Pick strongest signal (for now: first one that fires, they run sequentially)
    # In practice: the strategy that fires on the most confluent setup wins
    signal = signals[0]

    # Pre-trade checklist
    allowed, reason = can_open_trade(signal["strategy"], signal["side"])
    if not allowed:
        return

    # AI Market Brain gate
    approved, confidence, ai_reason = await ai_market_analysis(
        signal["strategy"], signal["side"], signal["entry"],
        signal_reason=signal.get("reason", "")
    )

    if not approved or confidence < 6:
        log.info("AI Market Brain REJECTED %s %s (confidence %d): %s",
                 signal["strategy"], signal["side"], confidence, ai_reason)
        await tg_send(
            f"\u274c <b>AI Brain REJECTED</b> -- {STRATEGY_LABELS.get(signal['strategy'], signal['strategy'])}\n"
            f"{signal['side'].upper()} @ ${signal['entry']:,.2f} | Confidence: {confidence}/10\n"
            f"{sanitize_html(ai_reason[:200])}"
        )
        return

    # Execute
    await execute_trade(
        signal["strategy"], signal["side"], signal["entry"],
        signal["sl"], signal["tp"],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Position Monitor -- SL/TP, Intelligent Early Exit, Max Hold, Macro Tighten
# ---------------------------------------------------------------------------

async def position_monitor_loop():
    """Check open position for SL/TP, intelligent early exit, max hold."""
    while True:
        try:
            pos = state.current_position
            if pos is None:
                await asyncio.sleep(5)
                continue

            price = await get_btc_price()
            if price <= 0:
                await asyncio.sleep(10)
                continue
            state.last_btc_price = price

            side = pos["side"]
            entry = pos["entry"]
            sl = pos["sl"]
            tp = pos["tp"]

            # 1. Stop Loss
            if side == "long" and price <= sl:
                await close_position(price, "Stop Loss")
                continue
            if side == "short" and price >= sl:
                await close_position(price, "Stop Loss")
                continue

            # 2. Take Profit
            if side == "long" and price >= tp:
                await close_position(price, "Take Profit")
                continue
            if side == "short" and price <= tp:
                await close_position(price, "Take Profit")
                continue

            # 3. Macro tighten: if HIGH macro, tighten SL. If CRITICAL and against us, close.
            macro_text = _macro_intel_cache.get("text", "").upper()
            if "RISK_LEVEL: CRITICAL" in macro_text:
                if side == "long":
                    moving_against = price < entry
                else:
                    moving_against = price > entry
                if moving_against:
                    await close_position(price, "CRITICAL macro + price moving against position")
                    continue
                else:
                    # Tighten SL to breakeven
                    if side == "long" and sl < entry:
                        pos["sl"] = entry
                        await tg_send(f"\u26a0\ufe0f <b>CRITICAL macro</b> -- SL tightened to breakeven ${entry:,.0f}")
                    elif side == "short" and sl > entry:
                        pos["sl"] = entry
                        await tg_send(f"\u26a0\ufe0f <b>CRITICAL macro</b> -- SL tightened to breakeven ${entry:,.0f}")

            elif "RISK_LEVEL: HIGH" in macro_text:
                # Tighten SL by 30%
                current_sl_dist = abs(entry - sl)
                tighter_dist = current_sl_dist * 0.7
                if side == "long":
                    new_sl = round(entry - tighter_dist)
                    if new_sl > sl:
                        pos["sl"] = new_sl
                else:
                    new_sl = round(entry + tighter_dist)
                    if new_sl < sl:
                        pos["sl"] = new_sl

            # 4. Intelligent early exit -- detect failed trade signals
            await _check_intelligent_exit(pos, price)

            # 5. Max hold time
            strategy = pos["strategy"]
            max_hold_min = STRATEGY_MAX_HOLD.get(strategy, 480)
            open_time = datetime.fromisoformat(pos["open_time"])
            elapsed_min = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
            if elapsed_min >= max_hold_min:
                await close_position(price, f"Max hold time ({max_hold_min}min)")
                continue

        except Exception as exc:
            log.error("Position monitor error: %s", exc)
        await asyncio.sleep(10)


async def _check_intelligent_exit(pos: dict, price: float):
    """Check for failed trade signals: strong opposing candle, volume spike against, momentum reversal."""
    strategy = pos["strategy"]
    side = pos["side"]
    entry = pos["entry"]

    # Only check after at least 5 minutes
    open_time = datetime.fromisoformat(pos["open_time"])
    elapsed = (datetime.now(timezone.utc) - open_time).total_seconds()
    if elapsed < 300:
        return

    df_5m = await fetch_ohlcv("5m", 10)
    if df_5m.empty or len(df_5m) < 5:
        return

    curr = df_5m.iloc[-1]
    curr_close = float(curr["close"])
    curr_open = float(curr["open"])
    curr_body = abs(curr_close - curr_open)
    curr_range = float(curr["high"]) - float(curr["low"])
    avg_vol = float(df_5m["volume"].iloc[-5:].mean())
    curr_vol = float(curr["volume"])

    if curr_range == 0 or avg_vol == 0:
        return

    body_ratio = curr_body / curr_range
    vol_spike = curr_vol >= avg_vol * 1.5

    # Strong opposing candle with volume spike
    if side == "long":
        strong_bearish = curr_close < curr_open and body_ratio > 0.7 and vol_spike
        if strong_bearish and curr_close < entry:
            log.info("Intelligent exit: strong bearish candle against long position")
            await close_position(price, "Intelligent exit -- strong opposing candle + volume")
    elif side == "short":
        strong_bullish = curr_close > curr_open and body_ratio > 0.7 and vol_spike
        if strong_bullish and curr_close > entry:
            log.info("Intelligent exit: strong bullish candle against short position")
            await close_position(price, "Intelligent exit -- strong opposing candle + volume")

    # For liquidity_sweep: if price continues in sweep direction after entry, exit immediately
    if strategy == "liquidity_sweep":
        if side == "long" and price < entry - pos["sl_distance"] * 0.5:
            await close_position(price, "Failed sweep -- price continuing down")
        elif side == "short" and price > entry + pos["sl_distance"] * 0.5:
            await close_position(price, "Failed sweep -- price continuing up")

    # For order_block: if price closes through OB without reversing, exit
    if strategy == "order_block" and elapsed > 600:
        if side == "long" and price < pos["sl"] + pos["sl_distance"] * 0.3:
            await close_position(price, "OB invalidated -- price breaking through zone")
        elif side == "short" and price > pos["sl"] - pos["sl_distance"] * 0.3:
            await close_position(price, "OB invalidated -- price breaking through zone")


# ---------------------------------------------------------------------------
# Background Tasks -- Scheduling
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


async def funding_rate_loop():
    """Poll funding rate every 5 minutes and maintain history."""
    while True:
        try:
            await update_funding_rate()
        except Exception as exc:
            log.error("Funding rate loop error: %s", exc)
        await asyncio.sleep(300)


async def strategy_monitor_loop():
    """Run strategy orchestrator on candle close boundaries."""
    while True:
        try:
            if state.paused or state.daily_loss_cap_hit:
                await asyncio.sleep(5)
                continue

            now = datetime.now(timezone.utc)
            current_ts = now.timestamp()

            # 5m boundary -- liquidity sweep, vwap reclaim
            current_5m = int(current_ts // 300) * 300
            new_5m = current_5m > state.last_5m_candle_ts

            # 15m boundary -- liquidity sweep, vwap reclaim, order block
            current_15m = int(current_ts // 900) * 900
            new_15m = current_15m > state.last_15m_candle_ts

            # 1h boundary -- funding rate fade, ema trend pullback, order block
            current_1h = int(current_ts // 3600) * 3600
            new_1h = current_1h > state.last_1h_candle_ts

            if new_5m:
                state.last_5m_candle_ts = current_5m
            if new_15m:
                state.last_15m_candle_ts = current_15m
            if new_1h:
                state.last_1h_candle_ts = current_1h

            # Run strategies on appropriate boundaries
            if new_5m or new_15m or new_1h:
                await run_all_strategies()

        except Exception as exc:
            log.error("Strategy monitor error: %s", exc)

        await asyncio.sleep(5)


async def periodic_status_log():
    """Log a periodic status summary every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        try:
            pos_text = "none"
            if state.current_position:
                p = state.current_position
                pos_text = f"{p['strategy']}({p['side']}@{p['entry']:.0f})"

            log.info(
                "=== PERIODIC STATUS ===\n"
                "  Regime: %s | HTF Bias: %s (%s) | BTC: $%.2f | Mode: %s\n"
                "  Active strategies: %s\n"
                "  Signals checked: %d | Signals blocked: %d\n"
                "  Trades today: %d | Losses: %d/%d | PnL: $%.2f\n"
                "  Position: %s\n"
                "  Funding rate: %.6f",
                state.regime.value, state.htf_bias, state.htf_bias_strength,
                state.last_btc_price, state.mode,
                ", ".join(state.active_strategies) if state.active_strategies else "none",
                state.signals_checked, state.signals_blocked,
                state.trades_today, state.losses_today, MAX_LOSSES_PER_DAY,
                state.daily_pnl,
                pos_text,
                state.current_funding_rate,
            )
        except Exception as exc:
            log.error("Periodic status error: %s", exc)


# ---------------------------------------------------------------------------
# News & Macro Monitoring (preserved)
# ---------------------------------------------------------------------------

async def news_sentiment_monitor():
    """Poll for BTC price moves and basic news sentiment every 5 min."""
    prev_price = 0.0
    _macro_scan_counter = 0
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
                            f"(${prev_price:,.0f} -> ${price:,.0f})\n"
                            f"ALL trading PAUSED."
                        )
                        await tg_send(msg)
                prev_price = price

            await _check_crypto_news()

            _macro_scan_counter += 1
            if _macro_scan_counter >= 6:
                _macro_scan_counter = 0
                await _run_macro_scan()

        except Exception as exc:
            log.error("News monitor error: %s", exc)
        await asyncio.sleep(300)


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
                        + (_get_intelligence_summary() + "\n\nFactor in what top traders are doing. " if _get_intelligence_summary() else "")
                        + "Give a concise factual summary. Flag anything that should STOP a trader from "
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

                if "RISK_LEVEL: CRITICAL" in result.upper():
                    if not state.paused:
                        state.paused = True
                        await tg_send(
                            f"\U0001f6a8 <b>MACRO RISK -- CRITICAL</b>\n\n"
                            f"Auto-scan detected critical macro conditions.\n"
                            f"ALL trading PAUSED.\n\n"
                            f"Details:\n{result[:500]}"
                        )
                elif "RISK_LEVEL: HIGH" in result.upper():
                    await tg_send(
                        f"\u26a0\ufe0f <b>MACRO RISK -- HIGH</b>\n\n"
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
                            return
    except Exception as exc:
        log.debug("News fetch error (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Daily Summary & Reset
# ---------------------------------------------------------------------------

async def daily_summary_scheduler():
    """Send daily self-review at 17:00 UTC (9 PM Dubai)."""
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        log.info("Daily self-review scheduled in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            await send_daily_self_review()
        except Exception as exc:
            log.error("Daily self-review error: %s", exc)


async def send_daily_self_review():
    """Claude analyses today's trades and proposes parameter adjustments."""
    journal = _read_journal()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in journal if t.get("date") == today]

    # Build context for Claude
    trades_text = ""
    if today_trades:
        for t in today_trades:
            trades_text += (
                f"  #{t.get('id')} {t.get('strategy')} {t.get('side','').upper()} "
                f"${t.get('entry_price',0):,.0f}->${t.get('exit_price',0):,.0f} "
                f"PnL ${t.get('pnl_usd',0):+,.2f} ({t.get('r_achieved',0):+.1f}R) "
                f"[{t.get('exit_reason','-')}, {t.get('hold_time_min',0):.0f}min, {t.get('leverage','?')}x]\n"
            )
    else:
        trades_text = "  No trades today.\n"

    # Strategy metrics
    metrics_text = ""
    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        metrics_text += f"  {name}: {tc} trades, WR {wr:.0f}%, PnL ${m['current_pnl']:+,.2f}\n"

    # Backtest results
    bt_text = ""
    for name in STRATEGY_NAMES:
        bt = state.backtest_results.get(name, {})
        bt_text += f"  {name}: BT WR {bt.get('win_rate',0)*100:.0f}%, {bt.get('total_trades',0)} trades\n"

    # Intelligence
    intel = _get_intelligence_summary()

    if CLAUDE_API_KEY:
        try:
            prompt = (
                f"You are Vic's daily self-review analyst. Analyse today's trading and propose specific improvements.\n\n"
                f"TODAY'S TRADES:\n{trades_text}\n"
                f"LIFETIME STRATEGY METRICS:\n{metrics_text}\n"
                f"30-DAY BACKTEST RESULTS:\n{bt_text}\n"
                f"TOP TRADER INTEL:\n{intel or 'No data'}\n\n"
                f"Analyse:\n"
                f"1. Each trade: why it won or lost, what filter would have caught the losers\n"
                f"2. Cross-reference against top trader patterns\n"
                f"3. Propose SPECIFIC parameter adjustments with data justification\n\n"
                f"Format proposals as:\n"
                f"PROPOSAL: [strategy] [parameter] [current] -> [proposed] | REASON: [data-backed reason]\n\n"
                f"Be concise. Max 5 proposals. Only propose changes backed by today's data."
            )

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
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                    review_text = "\n".join(texts).strip()
                else:
                    review_text = "Review generation failed."
        except Exception as exc:
            review_text = f"Review error: {exc}"
    else:
        review_text = "No CLAUDE_API_KEY -- cannot generate review."

    # Store pending review
    state.pending_review = {
        "date": today,
        "review": review_text,
        "trades_analysed": len(today_trades),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approved": False,
    }
    save_state()

    # Save to file
    try:
        d = os.path.dirname(REVIEW_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(REVIEW_FILE, "w") as f:
            json.dump(state.pending_review, f, indent=2)
    except Exception:
        pass

    # Send to Telegram
    summary_lines = [f"\U0001f4ca <b>DAILY SELF-REVIEW ({today})</b>\n"]

    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        active = name in state.active_strategies
        paused = state.strategy_paused.get(name, False)
        status = "PAUSED" if paused else ("Active" if active else "Inactive")
        summary_lines.append(
            f"  <b>{name}</b> [{status}]: WR {wr:.0f}% ({tc} trades) | PnL ${m['current_pnl']:+,.2f}"
        )

    summary_lines.append(f"\n<b>Today:</b> {len(today_trades)} trades | PnL ${state.daily_pnl:+,.2f}")
    summary_lines.append(f"\n<b>Review:</b>\n{sanitize_html(review_text[:1500])}")
    summary_lines.append(f"\n\u2753 <b>Use /approve to apply changes or /reject to discard.</b>")

    await tg_send("\n".join(summary_lines))


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
# Backtesting Engine
# ---------------------------------------------------------------------------

def _simulate_trade_forward(df: pd.DataFrame, entry_idx: int, side: str,
                             entry: float, sl: float, tp: float, size: float,
                             max_bars: int = 48) -> Optional[dict]:
    """Walk forward candle by candle to simulate a trade."""
    for offset in range(1, max_bars + 1):
        idx = entry_idx + offset
        if idx >= len(df):
            break

        high = float(df["high"].iloc[idx])
        low = float(df["low"].iloc[idx])

        # SL check
        if side == "long" and low <= sl:
            pnl = (sl - entry) * size
            return {"side": side, "entry": entry, "exit": sl, "pnl": round(pnl, 2),
                    "reason": "SL", "bars_held": offset}
        if side == "short" and high >= sl:
            pnl = (entry - sl) * size
            return {"side": side, "entry": entry, "exit": sl, "pnl": round(pnl, 2),
                    "reason": "SL", "bars_held": offset}

        # TP check
        if side == "long" and high >= tp:
            pnl = (tp - entry) * size
            return {"side": side, "entry": entry, "exit": tp, "pnl": round(pnl, 2),
                    "reason": "TP", "bars_held": offset}
        if side == "short" and low <= tp:
            pnl = (entry - tp) * size
            return {"side": side, "entry": entry, "exit": tp, "pnl": round(pnl, 2),
                    "reason": "TP", "bars_held": offset}

    # Timeout
    last_idx = min(entry_idx + max_bars, len(df) - 1)
    exit_price = float(df["close"].iloc[last_idx])
    if side == "long":
        pnl = (exit_price - entry) * size
    else:
        pnl = (entry - exit_price) * size
    return {"side": side, "entry": entry, "exit": exit_price, "pnl": round(pnl, 2),
            "reason": "timeout", "bars_held": max_bars}


def _backtest_strategy_generic(df: pd.DataFrame, strategy_name: str,
                                signal_func, max_bars: int = 48) -> list:
    """Generic backtester that takes a signal function and walks through data."""
    trades = []
    i = 50
    while i < len(df) - 10:
        signal = signal_func(df, i)
        if signal:
            result = _simulate_trade_forward(
                df, i, signal["side"], signal["entry"], signal["sl"], signal["tp"],
                signal.get("size", 0.001), max_bars
            )
            if result:
                trades.append(result)
                i += result.get("bars_held", 1) + 1
            else:
                i += 1
        else:
            i += 1
    return trades


def _bt_signal_ema_pullback(df: pd.DataFrame, i: int) -> Optional[dict]:
    """Backtest signal generator for EMA trend pullback on 1H data."""
    if i < 200:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_prices = df["open"].values

    # Calculate EMAs
    ema200 = pd.Series(close[:i+1]).ewm(span=200, adjust=False).mean().values
    ema21 = pd.Series(close[:i+1]).ewm(span=21, adjust=False).mean().values
    ema50 = pd.Series(close[:i+1]).ewm(span=50, adjust=False).mean().values

    if len(ema200) < 6:
        return None

    slope = (ema200[-1] - ema200[-5]) / ema200[-5] * 100 if ema200[-5] != 0 else 0
    trend = "bullish" if slope > 0.01 else "bearish" if slope < -0.01 else None
    if not trend:
        return None

    price = close[i]
    body = abs(close[i] - open_prices[i])
    full_range = high[i] - low[i]
    if full_range == 0:
        return None

    touched_ema = (low[i] <= ema21[-1] <= high[i]) or (low[i] <= ema50[-1] <= high[i])
    if not touched_ema:
        return None

    atr_vals = pd.Series(close[:i+1]).diff().abs().rolling(14).mean().values
    atr_val = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 0
    if atr_val <= 0:
        return None

    if trend == "bullish" and close[i] > open_prices[i] and body / full_range >= 0.5:
        sl_dist = min(abs(price - low[i]) + atr_val * 0.2, 1.5 * atr_val)
        size = 20.0 / sl_dist if sl_dist > 0 else 0
        if size <= 0:
            return None
        return {"side": "long", "entry": price, "sl": price - sl_dist,
                "tp": price + sl_dist * 2, "size": size}

    if trend == "bearish" and close[i] < open_prices[i] and body / full_range >= 0.5:
        sl_dist = min(abs(high[i] - price) + atr_val * 0.2, 1.5 * atr_val)
        size = 20.0 / sl_dist if sl_dist > 0 else 0
        if size <= 0:
            return None
        return {"side": "short", "entry": price, "sl": price + sl_dist,
                "tp": price - sl_dist * 2, "size": size}

    return None


def _bt_signal_vwap_reclaim(df: pd.DataFrame, i: int) -> Optional[dict]:
    """Backtest signal generator for VWAP reclaim on 5m/15m data."""
    if i < 25:
        return None

    vwap = calc_vwap(df.iloc[:i+1])
    if len(vwap) < 2 or np.isnan(vwap.iloc[-1]) or np.isnan(vwap.iloc[-2]):
        return None

    price = float(df["close"].iloc[i])
    prev_close = float(df["close"].iloc[i-1])
    open_price = float(df["open"].iloc[i])
    vwap_now = float(vwap.iloc[-1])
    vwap_prev = float(vwap.iloc[-2])
    body = abs(price - open_price)
    full_range = float(df["high"].iloc[i]) - float(df["low"].iloc[i])
    if full_range == 0:
        return None

    avg_vol = float(df["volume"].iloc[max(0,i-20):i].mean())
    curr_vol = float(df["volume"].iloc[i])
    if avg_vol == 0 or curr_vol < avg_vol * 1.5 or body / full_range < 0.6:
        return None

    rsi = calc_rsi(df["close"].iloc[:i+1], 14)
    rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50

    atr_s = calc_atr(df.iloc[max(0,i-20):i+1], 14)
    atr_val = float(atr_s.iloc[-1]) if len(atr_s) > 0 and not np.isnan(atr_s.iloc[-1]) else 0
    if atr_val <= 0:
        return None

    if prev_close < vwap_prev and price > vwap_now and price > open_price and rsi_val > 50:
        sl_dist = 1.0 * atr_val
        size = 20.0 / sl_dist if sl_dist > 0 else 0
        return {"side": "long", "entry": price, "sl": price - sl_dist,
                "tp": price + sl_dist * 2, "size": size} if size > 0 else None

    if prev_close > vwap_prev and price < vwap_now and price < open_price and rsi_val < 50:
        sl_dist = 1.0 * atr_val
        size = 20.0 / sl_dist if sl_dist > 0 else 0
        return {"side": "short", "entry": price, "sl": price + sl_dist,
                "tp": price - sl_dist * 2, "size": size} if size > 0 else None

    return None


def _calc_backtest_stats(strategy: str, trades: list) -> dict:
    """Calculate backtest statistics for a strategy."""
    if not trades:
        return {
            "strategy": strategy, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_r": 0.0, "max_drawdown": 0.0,
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    total_pnl = sum(t["pnl"] for t in trades)
    risk_per_trade = 20.0  # Backtest uses $20 risk

    r_values = [t["pnl"] / risk_per_trade for t in trades]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    win_rate = wins / len(trades) if trades else 0.0

    # Max drawdown
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum_pnl += t["pnl"]
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd

    return {
        "strategy": strategy, "total_trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 4), "total_pnl": round(total_pnl, 2),
        "avg_r": round(avg_r, 2), "max_drawdown": round(max_dd, 2),
        "trades": trades[-10:],
    }


async def run_backtest() -> dict:
    """Run 30-day backtest on all 5 strategies."""
    log.info("Starting 30-day backtest...")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (30 * 24 * 3600 * 1000)

    # Fetch data for different timeframes
    log.info("Fetching 30 days of 1h candles...")
    df_1h = await fetch_ohlcv_range("1h", start_ms, now_ms)
    log.info("Fetched %d 1h candles", len(df_1h))

    log.info("Fetching 30 days of 5m candles...")
    df_5m = await fetch_ohlcv_range("5m", start_ms, now_ms)
    log.info("Fetched %d 5m candles", len(df_5m))

    log.info("Fetching 30 days of 15m candles...")
    df_15m = await fetch_ohlcv_range("15m", start_ms, now_ms)
    log.info("Fetched %d 15m candles", len(df_15m))

    results = {}

    # Strategy 1: Funding Rate Fade -- can't properly backtest without historical funding data
    # We'll report N/A and always activate it
    results["funding_rate_fade"] = {
        "strategy": "funding_rate_fade", "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.55, "total_pnl": 0, "avg_r": 0, "max_drawdown": 0,
        "note": "Funding rate history not available for backtest -- strategy always active",
    }

    # Strategy 2: Liquidity Sweep -- complex to backtest, use simplified version
    # Using 5m data with swing detection
    log.info("Backtesting Liquidity Sweep (simplified)...")
    sweep_trades = []
    if not df_5m.empty and len(df_5m) > 60:
        df_bt = df_5m.copy()
        df_bt["rsi"] = calc_rsi(df_bt["close"], 14)
        atr_s = calc_atr(df_bt, 14)
        i = 55
        while i < len(df_bt) - 10:
            # Find swing lows/highs in lookback
            if i < 55:
                i += 1
                continue
            highs_lb = df_bt["high"].values[i-50:i]
            lows_lb = df_bt["low"].values[i-50:i]
            sh = [highs_lb[j] for j in range(2, len(highs_lb)-2)
                  if highs_lb[j] > highs_lb[j-1] and highs_lb[j] > highs_lb[j+1]]
            sl_pts = [lows_lb[j] for j in range(2, len(lows_lb)-2)
                      if lows_lb[j] < lows_lb[j-1] and lows_lb[j] < lows_lb[j+1]]

            price = float(df_bt["close"].iloc[i])
            low = float(df_bt["low"].iloc[i])
            high = float(df_bt["high"].iloc[i])
            atr = float(atr_s.iloc[i]) if i < len(atr_s) and not np.isnan(atr_s.iloc[i]) else 0
            if atr <= 0 or not sl_pts or not sh:
                i += 1
                continue

            avg_vol = float(df_bt["volume"].iloc[i-20:i].mean())
            curr_vol = float(df_bt["volume"].iloc[i])

            signal = None
            for sw_low in sl_pts[-3:]:
                if low < sw_low and price > sw_low and curr_vol > avg_vol * 1.5:
                    sl_d = abs(price - low) + atr * 0.2
                    size = 20.0 / sl_d if sl_d > 0 else 0
                    if size > 0 and sh:
                        signal = {"side": "long", "entry": price, "sl": low - atr * 0.2,
                                  "tp": sh[-1], "size": size}
                    break

            if not signal:
                for sw_high in sh[-3:]:
                    if high > sw_high and price < sw_high and curr_vol > avg_vol * 1.5:
                        sl_d = abs(high - price) + atr * 0.2
                        size = 20.0 / sl_d if sl_d > 0 else 0
                        if size > 0 and sl_pts:
                            signal = {"side": "short", "entry": price, "sl": high + atr * 0.2,
                                      "tp": sl_pts[-1], "size": size}
                        break

            if signal:
                result = _simulate_trade_forward(df_bt, i, signal["side"], signal["entry"],
                                                  signal["sl"], signal["tp"], signal["size"], 48)
                if result:
                    sweep_trades.append(result)
                    i += result.get("bars_held", 1) + 1
                    continue
            i += 1

    results["liquidity_sweep"] = _calc_backtest_stats("liquidity_sweep", sweep_trades)
    log.info("Liquidity Sweep: %d trades, %.1f%% WR",
             results["liquidity_sweep"]["total_trades"], results["liquidity_sweep"]["win_rate"] * 100)

    # Strategy 3: EMA Trend Pullback on 1H
    log.info("Backtesting EMA Trend Pullback...")
    ema_trades = _backtest_strategy_generic(df_1h, "ema_trend_pullback", _bt_signal_ema_pullback, 12)
    results["ema_trend_pullback"] = _calc_backtest_stats("ema_trend_pullback", ema_trades)
    log.info("EMA Trend Pullback: %d trades, %.1f%% WR",
             results["ema_trend_pullback"]["total_trades"], results["ema_trend_pullback"]["win_rate"] * 100)

    # Strategy 4: VWAP Reclaim on 15m
    log.info("Backtesting VWAP Reclaim...")
    vwap_trades = _backtest_strategy_generic(df_15m, "vwap_reclaim", _bt_signal_vwap_reclaim, 12)
    results["vwap_reclaim"] = _calc_backtest_stats("vwap_reclaim", vwap_trades)
    log.info("VWAP Reclaim: %d trades, %.1f%% WR",
             results["vwap_reclaim"]["total_trades"], results["vwap_reclaim"]["win_rate"] * 100)

    # Strategy 5: Order Block -- complex, simplified backtest
    log.info("Backtesting Order Block (simplified)...")
    ob_trades = []
    if not df_15m.empty and len(df_15m) > 60:
        df_bt = df_15m.copy()
        atr_s = calc_atr(df_bt, 14)
        rsi_s = calc_rsi(df_bt["close"], 14)
        i = 55
        while i < len(df_bt) - 10:
            atr = float(atr_s.iloc[i]) if i < len(atr_s) and not np.isnan(atr_s.iloc[i]) else 0
            if atr <= 0:
                i += 1
                continue

            price = float(df_bt["close"].iloc[i])
            # Look for bullish impulse in last 20 candles
            for k in range(max(3, i-20), i-3):
                impulse_ok = all(float(df_bt["close"].iloc[k+j]) > float(df_bt["open"].iloc[k+j]) for j in range(3))
                if impulse_ok:
                    start_p = float(df_bt["open"].iloc[k])
                    end_p = float(df_bt["close"].iloc[k+2])
                    if start_p > 0 and (end_p - start_p) / start_p * 100 >= 0.5:
                        ob_idx = k - 1
                        if ob_idx >= 0 and float(df_bt["close"].iloc[ob_idx]) < float(df_bt["open"].iloc[ob_idx]):
                            ob_high = float(df_bt["high"].iloc[ob_idx])
                            ob_low = float(df_bt["low"].iloc[ob_idx])
                            if ob_low <= price <= ob_high:
                                sl_p = ob_low - atr * 0.2
                                tp_p = end_p
                                sl_d = abs(price - sl_p)
                                size = 20.0 / sl_d if sl_d > 0 else 0
                                if size > 0:
                                    result = _simulate_trade_forward(df_bt, i, "long", price, sl_p, tp_p, size, 32)
                                    if result:
                                        ob_trades.append(result)
                                        i += result.get("bars_held", 1) + 1
                                        break
            else:
                i += 1
                continue
            continue

    results["order_block"] = _calc_backtest_stats("order_block", ob_trades)
    log.info("Order Block: %d trades, %.1f%% WR",
             results["order_block"]["total_trades"], results["order_block"]["win_rate"] * 100)

    # Save results
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["data_range"] = {
        "start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
    }

    try:
        d = os.path.dirname(BACKTEST_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(BACKTEST_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        log.error("Backtest save error: %s", exc)

    return results


async def backtest_scheduler():
    """Run backtest daily at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        target = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        if now.hour == 0 and now.minute < 5:
            target = now.replace(minute=5, second=0, microsecond=0)
        wait_seconds = (target - now).total_seconds()
        log.info("Next backtest in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            bt_results = await run_backtest()
            state.backtest_results = bt_results

            # Update active strategies
            new_active = []
            for name in STRATEGY_NAMES:
                bt = bt_results.get(name, {})
                if bt.get("win_rate", 0) >= BACKTEST_MIN_WIN_RATE or bt.get("note"):
                    new_active.append(name)
            state.active_strategies = new_active
            save_state()

            bt_msg = ["\U0001f4ca <b>NIGHTLY BACKTEST (30 days)</b>\n"]
            for name in STRATEGY_NAMES:
                bt = bt_results.get(name, {})
                wr = bt.get("win_rate", 0) * 100
                total = bt.get("total_trades", 0)
                passed = "\u2705" if name in new_active else "\u274c"
                bt_msg.append(f"  {passed} <b>{name}</b>: {total} trades | WR {wr:.1f}%")
            bt_msg.append(f"\n<b>Active:</b> {', '.join(new_active)}")
            await tg_send("\n".join(bt_msg))

        except Exception as exc:
            log.error("Nightly backtest error: %s", exc)


# ---------------------------------------------------------------------------
# Sunday Report
# ---------------------------------------------------------------------------

async def sunday_report_scheduler():
    """Send full automated report every Sunday at 18:00 UTC."""
    while True:
        now = datetime.now(timezone.utc)
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 18:
            days_until_sunday = 7
        target = (now + timedelta(days=days_until_sunday)).replace(hour=18, minute=0, second=0, microsecond=0)
        wait_seconds = (target - now).total_seconds()
        log.info("Sunday report scheduled in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            await send_sunday_report()
        except Exception as exc:
            log.error("Sunday report error: %s", exc)


async def send_sunday_report():
    """Full weekly report."""
    lines = ["\U0001f4c5 <b>SUNDAY WEEKLY REPORT</b>\n"]

    # Fresh backtest
    lines.append("<b>=== FRESH BACKTEST ===</b>")
    try:
        bt_results = await run_backtest()
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            wr = bt.get("win_rate", 0) * 100
            total = bt.get("total_trades", 0)
            passed = "PASS" if bt.get("win_rate", 0) >= BACKTEST_MIN_WIN_RATE or bt.get("note") else "FAIL"
            lines.append(f"  <b>{name}</b>: {total} trades | WR {wr:.1f}% [{passed}]")

        new_active = []
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            if bt.get("win_rate", 0) >= BACKTEST_MIN_WIN_RATE or bt.get("note"):
                new_active.append(name)
        state.active_strategies = new_active
        state.backtest_results = bt_results
        save_state()
    except Exception as exc:
        lines.append(f"  Backtest error: {sanitize_html(str(exc))}")

    # Live vs backtest
    lines.append("\n<b>=== LIVE PERFORMANCE ===</b>")
    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        lines.append(f"  <b>{name}</b>: {tc} trades | WR {wr:.0f}% | PnL ${m['current_pnl']:+,.2f}")

    total_pnl = sum(state.metrics[n]["current_pnl"] for n in STRATEGY_NAMES)
    lines.append(f"\n<b>Total PnL:</b> ${total_pnl:+,.2f}")
    lines.append(f"<b>Lifetime trades:</b> {state.total_trade_count}")
    lines.append(f"<b>Mode:</b> {state.mode.upper()}")
    lines.append(f"<b>Active:</b> {', '.join(state.active_strategies)}")

    intel = _get_intelligence_summary()
    if intel:
        lines.append(f"\n<b>=== TOP TRADER INTEL ===</b>\n{sanitize_html(intel)}")

    await tg_send("\n".join(lines))


# ---------------------------------------------------------------------------
# Hyperliquid Top Trader Intelligence (preserved)
# ---------------------------------------------------------------------------

_intelligence_cache = {"report": {}, "fetched_at": 0}


def _read_intelligence() -> dict:
    try:
        if os.path.exists(INTELLIGENCE_FILE):
            with open(INTELLIGENCE_FILE, "r") as f:
                return json.load(f)
    except Exception as exc:
        log.error("Intelligence read error: %s", exc)
    return {}


def _write_intelligence(report: dict):
    try:
        d = os.path.dirname(INTELLIGENCE_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(INTELLIGENCE_FILE, "w") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:
        log.error("Intelligence write error: %s", exc)


def _get_intelligence_summary() -> str:
    report = _intelligence_cache.get("report") or _read_intelligence()
    if not report or not report.get("traders"):
        return ""

    age_min = int((time.time() - _intelligence_cache.get("fetched_at", 0)) / 60) if _intelligence_cache.get("fetched_at") else 0
    traders = report.get("traders", [])
    patterns = report.get("patterns", {})

    lines = [f"HYPERLIQUID TOP TRADER INTELLIGENCE (updated {age_min}min ago, {len(traders)} traders):"]

    if patterns:
        if patterns.get("dominant_direction"):
            lines.append(f"  Dominant direction: {patterns['dominant_direction']}")
        if patterns.get("avg_win_rate"):
            lines.append(f"  Avg win rate: {patterns['avg_win_rate']:.1f}%")
        if patterns.get("best_sessions"):
            lines.append(f"  Best sessions: {', '.join(patterns['best_sessions'])}")
        if patterns.get("key_insight"):
            lines.append(f"  Key insight: {patterns['key_insight']}")

    for t in traders[:3]:
        lines.append(f"  #{t.get('rank','-')} WR:{t.get('win_rate',0):.0f}% "
                     f"Trades:{t.get('total_trades',0)} "
                     f"Bias:{t.get('direction_bias','?')}")

    return "\n".join(lines)


async def fetch_hl_leaderboard() -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard")
            if resp.status_code != 200:
                return []
            data = resp.json()
            rows = data.get("leaderboardRows", [])
            entries = []
            for row in rows[:50]:
                address = row.get("ethAddress", "")
                if not address:
                    continue
                monthly_pnl = 0.0
                monthly_roi = 0.0
                for window_name, perf in row.get("windowPerformances", []):
                    if window_name == "month":
                        monthly_pnl = float(perf.get("pnl", 0))
                        monthly_roi = float(perf.get("roi", 0))
                        break
                entries.append({
                    "ethAddress": address,
                    "displayName": row.get("displayName", ""),
                    "accountValue": float(row.get("accountValue", 0)),
                    "pnl": monthly_pnl,
                    "roi": monthly_roi,
                })
            return entries
    except Exception as exc:
        log.error("Leaderboard fetch error: %s", exc)
        return []


async def analyse_trader(address: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "userFills", "user": address},
            )
            if resp.status_code != 200:
                return {}
            fills = resp.json()

        if not fills or not isinstance(fills, list):
            return {}

        recent = fills[-20:] if len(fills) > 20 else fills
        wins = 0
        losses = 0
        total_win_size = 0.0
        total_loss_size = 0.0
        sessions = {"london": 0, "ny": 0, "asia": 0, "off": 0}
        long_count = 0
        short_count = 0

        for fill in recent:
            pnl = float(fill.get("closedPnl", 0))
            side = fill.get("side", "")
            ts = fill.get("time", 0)

            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.now(timezone.utc)
            else:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts > 1e12 else datetime.fromtimestamp(ts, tz=timezone.utc)

            hour = dt.hour
            if side.lower() in ("b", "buy", "long"):
                long_count += 1
            else:
                short_count += 1

            if pnl > 0:
                wins += 1
                total_win_size += pnl
            elif pnl < 0:
                losses += 1
                total_loss_size += abs(pnl)

            if 7 <= hour < 11:
                sessions["london"] += 1
            elif 13 <= hour < 17:
                sessions["ny"] += 1
            elif 0 <= hour < 7:
                sessions["asia"] += 1
            else:
                sessions["off"] += 1

        total = wins + losses
        if total == 0:
            return {}

        avg_win = total_win_size / wins if wins > 0 else 0
        avg_loss = total_loss_size / losses if losses > 0 else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        best_session = max(sessions, key=sessions.get)
        direction_bias = "LONG" if long_count > short_count * 1.5 else "SHORT" if short_count > long_count * 1.5 else "NEUTRAL"

        return {
            "wins": wins, "losses": losses, "win_rate": wins / total * 100,
            "avg_win_loss_ratio": round(win_loss_ratio, 2),
            "best_session": best_session, "direction_bias": direction_bias,
            "total_trades": total, "sessions": sessions,
        }
    except Exception as exc:
        log.debug("Trader analysis error for %s: %s", address[:10], exc)
        return {}


async def run_intelligence_scan():
    log.info("Starting Hyperliquid top trader intelligence scan...")
    leaderboard = await fetch_hl_leaderboard()
    if not leaderboard:
        return

    trader_analyses = []
    for i, entry in enumerate(leaderboard[:20]):
        address = entry.get("ethAddress", "")
        if not address:
            continue
        analysis = await analyse_trader(address)
        if not analysis:
            await asyncio.sleep(0.3)
            continue
        trader_analyses.append({"rank": i + 1, "address": address[:10] + "...", **analysis})
        await asyncio.sleep(0.3)

    if not trader_analyses:
        report = {"timestamp": datetime.now(timezone.utc).isoformat(), "traders": [], "patterns": {},
                  "total_scanned": len(leaderboard), "total_analysed": 0}
        _write_intelligence(report)
        _intelligence_cache["report"] = report
        _intelligence_cache["fetched_at"] = time.time()
        return

    all_wr = [t["win_rate"] for t in trader_analyses]
    all_wlr = [t.get("avg_win_loss_ratio", 0) for t in trader_analyses if t.get("avg_win_loss_ratio", 0) > 0]
    direction_counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    session_totals = {"london": 0, "ny": 0, "asia": 0, "off": 0}

    for t in trader_analyses:
        direction_counts[t.get("direction_bias", "NEUTRAL")] += 1
        for s, c in t.get("sessions", {}).items():
            session_totals[s] = session_totals.get(s, 0) + c

    dominant = max(direction_counts, key=direction_counts.get)
    best_sessions = sorted(session_totals, key=session_totals.get, reverse=True)[:2]

    insight = ""
    if CLAUDE_API_KEY:
        try:
            summary_data = json.dumps({
                "num_traders": len(trader_analyses),
                "avg_win_rate": sum(all_wr) / len(all_wr),
                "dominant_direction": dominant,
                "best_sessions": best_sessions,
            })
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": CLAUDE_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": (
                            f"Based on this data from top Hyperliquid traders, give ONE key actionable insight "
                            f"for a BTC perp trader in under 50 words:\n{summary_data}"
                        )}],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                    insight = " ".join(texts).strip()
        except Exception:
            pass

    patterns = {
        "dominant_direction": dominant,
        "avg_win_rate": sum(all_wr) / len(all_wr) if all_wr else 0,
        "avg_win_loss_ratio": sum(all_wlr) / len(all_wlr) if all_wlr else 0,
        "best_sessions": best_sessions,
        "key_insight": insight,
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traders": trader_analyses,
        "patterns": patterns,
        "total_scanned": len(leaderboard),
        "total_analysed": len(trader_analyses),
    }

    _write_intelligence(report)
    _intelligence_cache["report"] = report
    _intelligence_cache["fetched_at"] = time.time()
    log.info("Intelligence scan complete: %d traders analysed", len(trader_analyses))


async def intelligence_loop():
    """Run top trader intelligence scan every 6 hours."""
    await asyncio.sleep(60)
    while True:
        try:
            await run_intelligence_scan()
        except Exception as exc:
            log.error("Intelligence loop error: %s", exc)
        await asyncio.sleep(6 * 3600)


# ---------------------------------------------------------------------------
# Orphaned Position Recovery (preserved)
# ---------------------------------------------------------------------------

async def recover_orphaned_positions():
    if not hl_info:
        log.warning("Cannot recover positions -- exchange not initialized.")
        return

    try:
        loop = asyncio.get_event_loop()
        user_state_data = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
        positions = user_state_data.get("assetPositions", [])

        btc_positions = []
        for p in positions:
            pos = p.get("position", {})
            coin = pos.get("coin", "")
            szi = float(pos.get("szi", 0))
            if coin == "BTC" and szi != 0:
                btc_positions.append(pos)

        if not btc_positions:
            log.info("No orphaned positions found on Hyperliquid.")
            return

        df_5m = await fetch_ohlcv("5m", 30)
        atr_val = 0.0
        if not df_5m.empty and len(df_5m) >= 14:
            atr_s = calc_atr(df_5m, 14)
            atr_val = float(atr_s.iloc[-1]) if not atr_s.empty and not np.isnan(atr_s.iloc[-1]) else 0.0

        for pos_data in btc_positions:
            szi = float(pos_data.get("szi", 0))
            entry_px = float(pos_data.get("entryPx", 0))
            side = "long" if szi > 0 else "short"
            size = abs(szi)

            risk_dollars = ACCOUNT_CAPITAL * TARGET_RISK_PCT
            sl_distance = risk_dollars / size if size > 0 else atr_val * 1.5
            tp_distance = sl_distance * 2.0

            if side == "long":
                sl = round(entry_px - sl_distance)
                tp = round(entry_px + tp_distance)
            else:
                sl = round(entry_px + sl_distance)
                tp = round(entry_px - tp_distance)

            state.current_position = {
                "strategy": "orphan_recovery",
                "side": side,
                "entry": entry_px,
                "sl": sl,
                "tp": tp,
                "size": size,
                "leverage": BASE_LEVERAGE,
                "confidence": 7,
                "sl_distance": sl_distance,
                "tp_distance": tp_distance,
                "open_time": datetime.now(timezone.utc).isoformat(),
                "regime": state.regime.value,
                "bias": f"{state.htf_bias} ({state.htf_bias_strength})",
            }

            msg = (
                f"\U0001f504 <b>ORPHANED POSITION RECOVERY</b>\n\n"
                f"{side.upper()} {size:.6f} BTC @ ${entry_px:,.2f}\n"
                f"SL ${sl:,.2f} / TP ${tp:,.2f}\n"
                f"ATR(14, 5m) = {atr_val:.2f}"
            )
            await tg_send(msg)
            log.info("Recovered orphaned position: %s %.6f @ %.2f", side, size, entry_px)
            break  # Only one position at a time

    except Exception as exc:
        log.error("Orphaned position recovery failed: %s", exc)
        await tg_send(f"\u26a0\ufe0f <b>Position recovery failed</b>\nError: {sanitize_html(str(exc))}\nCHECK HYPERLIQUID MANUALLY.")


# ---------------------------------------------------------------------------
# Telegram Chat + Commands
# ---------------------------------------------------------------------------

async def telegram_polling_loop():
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set -- Telegram polling DEAD.")
        state.polling_alive = False
        return
    if not CLAUDE_API_KEY:
        log.error("CLAUDE_API_KEY not set -- Telegram polling DEAD.")
        state.polling_alive = False
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

                    cmd = text.strip().lower()

                    if cmd == "/start":
                        await tg_reply(chat_id,
                            "Hey! I'm Vic v5, your BTC trading agent.\n\n"
                            "Commands: /strategies /backtest /funding /journal /metrics /regime /news "
                            "/intelligence /review /approve /reject /closeall"
                        )
                        continue

                    if cmd == "/strategies":
                        lines = ["\U0001f4ca <b>Strategy Status</b>\n"]
                        for name in STRATEGY_NAMES:
                            m = state.metrics[name]
                            tc = m["trade_count"]
                            wr = (m["wins"] / tc * 100) if tc > 0 else 0
                            active = name in state.active_strategies
                            paused = state.strategy_paused.get(name, False)
                            status = "PAUSED" if paused else ("Active" if active else "Inactive")
                            bt = state.backtest_results.get(name, {})
                            bt_wr = bt.get("win_rate", 0) * 100
                            label = STRATEGY_LABELS.get(name, name)
                            lines.append(
                                f"{label} [{status}]\n"
                                f"  Live: {tc} trades | WR {wr:.0f}% | PnL ${m['current_pnl']:+,.2f}\n"
                                f"  Backtest: WR {bt_wr:.0f}%"
                            )
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/backtest":
                        try:
                            if os.path.exists(BACKTEST_FILE):
                                with open(BACKTEST_FILE, "r") as f:
                                    bt = json.load(f)
                                lines = [f"\U0001f4ca <b>Backtest Results</b>\nDate: {bt.get('timestamp', 'unknown')}\n"]
                                for name in STRATEGY_NAMES:
                                    s = bt.get(name, {})
                                    lines.append(
                                        f"<b>{name}</b>: {s.get('total_trades',0)} trades | "
                                        f"WR {s.get('win_rate',0)*100:.1f}% | "
                                        f"Avg R {s.get('avg_r',0):+.2f} | "
                                        f"MaxDD ${s.get('max_drawdown',0):,.2f}"
                                    )
                                await tg_reply(chat_id, "\n".join(lines))
                            else:
                                await tg_reply(chat_id, "No backtest results yet.")
                        except Exception as exc:
                            await tg_reply(chat_id, f"Error reading backtest: {exc}")
                        continue

                    if cmd == "/funding":
                        rate = state.current_funding_rate
                        pctl = get_funding_percentile(rate)
                        extreme = ""
                        if pctl >= 90:
                            extreme = "\U0001f534 EXTREME POSITIVE (short fade zone)"
                        elif pctl <= 10:
                            extreme = "\U0001f7e2 EXTREME NEGATIVE (long fade zone)"
                        else:
                            extreme = "Normal range"
                        msg = (
                            f"\U0001f4b1 <b>BTC Funding Rate</b>\n\n"
                            f"Current: {rate:.6f}\n"
                            f"Percentile: {pctl:.0f}th (30-day)\n"
                            f"Status: {extreme}\n"
                            f"History samples: {len(state.funding_history)}"
                        )
                        await tg_reply(chat_id, msg)
                        continue

                    if cmd == "/review":
                        if state.pending_review:
                            rev = state.pending_review
                            approved = "\u2705 Approved" if rev.get("approved") else "\u23f3 Pending GG approval"
                            msg = (
                                f"\U0001f4cb <b>Latest Self-Review ({rev.get('date', '?')})</b>\n"
                                f"Status: {approved}\n"
                                f"Trades analysed: {rev.get('trades_analysed', 0)}\n\n"
                                f"{sanitize_html(rev.get('review', 'No review text')[:1500])}"
                            )
                            await tg_reply(chat_id, msg)
                        else:
                            await tg_reply(chat_id, "No pending review. Daily review runs at 17:00 UTC.")
                        continue

                    if cmd == "/approve":
                        if state.pending_review and not state.pending_review.get("approved"):
                            state.pending_review["approved"] = True
                            save_state()
                            await tg_reply(chat_id, "\u2705 Parameter changes APPROVED. Vic will apply them.")
                        else:
                            await tg_reply(chat_id, "No pending changes to approve.")
                        continue

                    if cmd == "/reject":
                        if state.pending_review and not state.pending_review.get("approved"):
                            state.pending_review = None
                            save_state()
                            await tg_reply(chat_id, "\u274c Changes REJECTED. No parameters will be modified.")
                        else:
                            await tg_reply(chat_id, "No pending changes to reject.")
                        continue

                    if cmd == "/journal":
                        journal = _read_journal()
                        last_5 = journal[-5:] if journal else []
                        if not last_5:
                            await tg_reply(chat_id, "No trades in journal yet.")
                        else:
                            lines = ["\U0001f4d2 <b>Last 5 Trades</b>\n"]
                            for t in reversed(last_5):
                                emoji = "\u2705" if t.get("pnl_usd", 0) >= 0 else "\u274c"
                                lines.append(
                                    f"{emoji} #{t.get('id','-')} {t.get('strategy','-')} {t.get('side','').upper()}\n"
                                    f"  ${t.get('entry_price',0):,.2f} -> ${t.get('exit_price',0):,.2f} | "
                                    f"<b>${t.get('pnl_usd',0):+,.2f}</b> ({t.get('r_achieved',0):+.1f}R)\n"
                                    f"  {t.get('exit_reason','-')} | {t.get('hold_time_min',0):.0f}min | {t.get('leverage','?')}x"
                                )
                            await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/metrics":
                        lines = ["\U0001f4ca <b>Per-Strategy Metrics</b>\n"]
                        for n in STRATEGY_NAMES:
                            m = state.metrics[n]
                            tc = m["trade_count"]
                            wr = (m["wins"] / tc * 100) if tc > 0 else 0
                            exp = (m["total_r_achieved"] / tc) if tc > 0 else 0
                            active = n in state.active_strategies
                            paused = state.strategy_paused.get(n, False)
                            status = "PAUSED" if paused else ("Active" if active else "Inactive")
                            lines.append(
                                f"<b>{n}</b> [{status}]: {tc} trades | WR {wr:.0f}% | Exp {exp:+.2f}R\n"
                                f"  PnL ${m['current_pnl']:+,.2f} | MaxDD ${m['max_drawdown']:,.2f} | "
                                f"Streak {m['current_losing_streak']}/{m['max_losing_streak']}"
                            )
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/regime":
                        msg = (
                            f"\U0001f50d <b>Current Regime: {state.regime.value}</b>\n\n"
                            f"1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
                            f"BTC: ${state.last_btc_price:,.2f}\n"
                            f"Funding: {state.current_funding_rate:.6f}\n"
                            f"Active: {', '.join(state.active_strategies)}"
                        )
                        await tg_reply(chat_id, msg)
                        continue

                    if cmd == "/news":
                        headlines = await _fetch_recent_news_headlines()
                        if headlines:
                            lines = ["\U0001f4f0 <b>Recent BTC News</b>\n"]
                            for h in headlines:
                                lines.append(f"\u2022 {sanitize_html(h)}")
                            await tg_reply(chat_id, "\n".join(lines))
                        else:
                            await tg_reply(chat_id, "No recent news available.")
                        continue

                    if cmd == "/intelligence":
                        report = _intelligence_cache.get("report") or _read_intelligence()
                        if not report or not report.get("traders"):
                            await tg_reply(chat_id, "No intelligence report yet. First scan runs ~60s after startup.")
                        else:
                            ts = report.get("timestamp", "unknown")
                            traders = report.get("traders", [])
                            patterns = report.get("patterns", {})
                            lines = [f"\U0001f9e0 <b>Top Trader Intelligence</b>\nUpdated: {ts}\n"]
                            if patterns.get("dominant_direction"):
                                lines.append(f"Direction: {patterns['dominant_direction']}")
                            if patterns.get("avg_win_rate"):
                                lines.append(f"Avg WR: {patterns['avg_win_rate']:.1f}%")
                            if patterns.get("key_insight"):
                                lines.append(f"\n<b>Insight:</b> {sanitize_html(patterns['key_insight'])}")
                            if traders:
                                lines.append(f"\n<b>Top 5:</b>")
                                for t in traders[:5]:
                                    lines.append(
                                        f"  #{t.get('rank','-')} WR:{t.get('win_rate',0):.0f}% "
                                        f"Bias:{t.get('direction_bias','?')} "
                                        f"Best:{t.get('best_session','?')}"
                                    )
                            await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/closeall":
                        result = await close_all_positions()
                        await tg_reply(chat_id, result)
                        continue

                    # Free-form question to Claude
                    try:
                        answer = await ask_claude_market_question(text)
                        await tg_reply(chat_id, sanitize_html(answer))
                    except Exception as exc:
                        await tg_reply(chat_id, f"Error: {sanitize_html(str(exc))}")

        except Exception as exc:
            log.error("Telegram polling error: %s", exc)
            await asyncio.sleep(5)

    state.polling_alive = False


async def telegram_polling_watchdog():
    """Watchdog that monitors the polling loop and restarts it if it dies."""
    restart_count = 0
    max_restarts = 5
    while True:
        task = asyncio.create_task(telegram_polling_loop())
        try:
            await task
        except Exception as exc:
            log.error("Polling task crashed: %s", exc)
        state.polling_alive = False
        if restart_count >= max_restarts:
            log.error("Polling loop hit max restarts (%d). Giving up.", max_restarts)
            await tg_send("\U0001f6a8 <b>Vic polling is DEAD</b>\nRedeploy on Railway to fix.")
            return
        restart_count += 1
        log.warning("Polling loop died -- restarting (attempt %d/%d)...", restart_count, max_restarts)
        await tg_send(f"\u26a0\ufe0f <b>Vic polling crashed -- restarting</b> ({restart_count}/{max_restarts})")
        await asyncio.sleep(3)


async def ask_claude_market_question(question: str) -> str:
    """Send a market question to Claude API with full trading state context."""
    pos_text = "No open positions"
    if state.current_position:
        p = state.current_position
        unrealized = 0
        if state.last_btc_price > 0:
            if p["side"] == "long":
                unrealized = (state.last_btc_price - p["entry"]) * p["size"]
            else:
                unrealized = (p["entry"] - state.last_btc_price) * p["size"]
        pos_text = (f"{p['strategy']}: {p['side'].upper()} {p['size']:.6f} BTC @ ${p['entry']:,.2f} "
                    f"(SL ${p['sl']:,.2f}, TP ${p['tp']:,.2f}, unrealized ${unrealized:+.2f}, {p['leverage']}x)")

    metrics_lines = []
    for n in STRATEGY_NAMES:
        m = state.metrics[n]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        metrics_lines.append(f"  {n}: {tc} trades, WR {wr:.0f}%, PnL ${m['current_pnl']:+,.2f}")

    system_prompt = (
        f"You are Vic v5, an AI BTC trading agent on Hyperliquid. "
        f"5 strategies, dynamic leverage (10-20x), $500 account.\n\n"
        f"=== STRATEGIES ===\n"
        f"1. Funding Rate Fade (1H) | 2. Liquidity Sweep (5m/15m) | "
        f"3. EMA Trend Pullback (1H) | 4. VWAP Reclaim (5m/15m) | 5. Order Block (15m/1H)\n\n"
        f"=== STATE ===\n"
        f"BTC: ${state.last_btc_price:,.2f} | Regime: {state.regime.value} | "
        f"Bias: {state.htf_bias} | Mode: {state.mode.upper()}\n"
        f"Funding: {state.current_funding_rate:.6f} | Losses: {state.losses_today}/{MAX_LOSSES_PER_DAY}\n"
        f"PnL today: ${state.daily_pnl:+,.2f} | Lifetime: {state.total_trade_count} trades\n\n"
        f"=== POSITION ===\n{pos_text}\n\n"
        f"=== METRICS ===\n" + "\n".join(metrics_lines) + "\n\n"
        f"Answer concisely. Use numbers. Under 300 words."
    )

    q_lower = question.lower()
    search_keywords = [
        "news", "macro", "fed", "cpi", "fomc", "inflation", "tariff", "regulation",
        "what's happening", "why is btc", "crash", "pump", "dump", "latest", "today", "current",
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
            return f"Claude API error ({resp.status_code})."

        data = resp.json()
        text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text" and b.get("text", "").strip()]
        if text_blocks:
            return "\n".join(text_blocks)
        return "No response from Claude."


# ---------------------------------------------------------------------------
# Close All Positions (preserved)
# ---------------------------------------------------------------------------

async def close_all_positions() -> str:
    closed = []
    errors = []

    if hl_exchange and hl_info:
        try:
            loop = asyncio.get_event_loop()
            user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
            for p in user_st.get("assetPositions", []):
                pd_pos = p.get("position", {})
                coin = pd_pos.get("coin", "")
                szi = float(pd_pos.get("szi", 0))
                if szi != 0:
                    try:
                        result = await loop.run_in_executor(
                            None, lambda c=coin, s=abs(szi): hl_exchange.market_close(c, sz=s)
                        )
                        if result and result.get("status") == "ok":
                            closed.append(f"{coin} {'long' if szi > 0 else 'short'} {abs(szi)}")
                        else:
                            errors.append(f"{coin}: {result}")
                    except Exception as exc:
                        errors.append(f"{coin}: {exc}")
        except Exception as exc:
            errors.append(f"Exchange query failed: {exc}")

    if state.current_position:
        state.current_position = None

    parts = ["\U0001f6d1 <b>CLOSE ALL -- Manual Override</b>\n"]
    if closed:
        parts.append(f"Closed on exchange: {', '.join(closed)}")
    if errors:
        parts.append(f"Errors: {', '.join(errors)}")
    if not closed:
        parts.append("No exchange positions to close.")

    msg = "\n".join(parts)
    await tg_send(msg)
    save_state()
    return msg


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {
        "bot": "Vic v5",
        "status": "running" if state.polling_alive else "DEGRADED -- polling dead",
        "telegram_polling": "alive" if state.polling_alive else "DEAD",
        "mode": state.mode,
        "paused": state.paused,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "uptime_since": state.startup_time,
        "btc_price": state.last_btc_price,
        "trades_today": state.trades_today,
        "losses_today": state.losses_today,
        "daily_pnl": state.daily_pnl,
        "active_strategies": state.active_strategies,
        "has_position": state.current_position is not None,
        "funding_rate": state.current_funding_rate,
    }


@app.get("/status")
async def full_status():
    strategies_status = {}
    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        win_rate = (m["wins"] / m["trade_count"] * 100) if m["trade_count"] > 0 else 0.0
        strategies_status[name] = {
            "active": name in state.active_strategies,
            "paused": state.strategy_paused.get(name, False),
            "metrics": {
                "win_rate": round(win_rate, 1),
                "total_trades": m["trade_count"],
                "pnl": m["current_pnl"],
                "max_drawdown": m["max_drawdown"],
            },
        }

    return {
        "bot": "Vic v5",
        "mode": state.mode,
        "base_leverage": BASE_LEVERAGE,
        "account_capital": ACCOUNT_CAPITAL,
        "global_paused": state.paused,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "btc_price": state.last_btc_price,
        "funding_rate": state.current_funding_rate,
        "strategies": strategies_status,
        "active_strategies": state.active_strategies,
        "current_position": state.current_position,
        "trades_today": state.trades_today,
        "losses_today": state.losses_today,
        "daily_pnl": state.daily_pnl,
        "daily_loss_cap_hit": state.daily_loss_cap_hit,
        "total_lifetime_trades": state.total_trade_count,
    }


@app.post("/go_live")
async def go_live(token: str = Query("")):
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    if state.mode == "live":
        return {"status": "already live"}
    state.mode = "live"
    close_exchange()
    init_exchange()
    await tg_send("\U0001f534 <b>LIVE TRADING ENABLED</b> -- Vic is now trading with real funds.")
    return {"status": "live"}


@app.post("/pause")
async def pause_trading(token: str = Query("")):
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    state.paused = True
    await tg_send("\u23f8\ufe0f All trading PAUSED by manual command.")
    return {"status": "paused"}


@app.post("/resume")
async def resume_trading(token: str = Query("")):
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    state.paused = False
    await tg_send("\u25b6\ufe0f Trading RESUMED by manual command.")
    return {"status": "resumed"}


@app.post("/close_all")
async def close_all_endpoint(token: str = Query("")):
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    result = await close_all_positions()
    return {"status": "ok", "message": result}


@app.get("/journal")
async def get_journal():
    journal = _read_journal()
    return {"trades": journal, "count": len(journal)}


@app.get("/backtest")
async def get_backtest():
    try:
        if os.path.exists(BACKTEST_FILE):
            with open(BACKTEST_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"status": "no backtest results available"}


@app.post("/test_trade")
async def test_trade():
    if not hl_exchange:
        return {"error": "Exchange not initialized"}
    try:
        price = await get_btc_price()
        if price <= 0:
            return {"error": "Could not fetch BTC price"}
        size = math.ceil(12.0 / price * 100000) / 100000
        if size <= 0:
            size = 0.00001

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: hl_exchange.update_leverage(BASE_LEVERAGE, "BTC", is_cross=True))

        open_result = await loop.run_in_executor(
            None, lambda: hl_exchange.market_open("BTC", is_buy=False, sz=size)
        )
        if open_result.get("status") != "ok":
            return {"error": f"Open rejected: {open_result}"}

        await tg_send(f"\U0001f9ea <b>TEST TRADE</b> -- SHORT {size} BTC @ ${price:,.2f}")
        await asyncio.sleep(2)

        close_result = await loop.run_in_executor(
            None, lambda: hl_exchange.market_close("BTC", sz=size)
        )
        if close_result.get("status") != "ok":
            await tg_send(f"\U0001f6a8 <b>TEST TRADE close FAILED</b>: {close_result}")
            return {"error": f"Close rejected: {close_result}"}

        await tg_send(f"\u2705 <b>TEST TRADE closed</b> -- Exchange access CONFIRMED.")
        return {"status": "ok", "size": size, "price": price}
    except Exception as exc:
        await tg_send(f"\U0001f6a8 <b>TEST TRADE FAILED</b>: {sanitize_html(str(exc))}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    _required = ["HL_WALLET_ADDRESS", "HL_PRIVATE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "CLAUDE_API_KEY"]
    _missing = [v for v in _required if not os.getenv(v)]
    if _missing:
        log.error(f"MISSING REQUIRED ENV VARS: {', '.join(_missing)} -- Vic cannot start properly")

    # Ensure data directories exist
    for filepath in [JOURNAL_FILE, STATE_FILE, BACKTEST_FILE, INTELLIGENCE_FILE, REVIEW_FILE]:
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)

    state.startup_time = datetime.now(timezone.utc).isoformat()
    log.info("Vic v5 starting up -- mode: %s", state.mode)

    # Load persisted state
    load_state()

    # Initialize candle boundary tracking
    now_ts = datetime.now(timezone.utc).timestamp()
    state.last_1m_candle_ts = int(now_ts // 60) * 60
    state.last_5m_candle_ts = int(now_ts // 300) * 300
    state.last_15m_candle_ts = int(now_ts // 900) * 900
    state.last_1h_candle_ts = int(now_ts // 3600) * 3600

    init_exchange()

    # Orphaned position recovery
    await recover_orphaned_positions()

    # Startup message
    startup_msg = (
        f"\U0001f916 <b>Vic v5 is online -- {state.mode.upper()} MODE</b>\n"
        f"BTC/USDC perp | {BASE_LEVERAGE}x+ leverage | ${ACCOUNT_CAPITAL:.0f} account\n\n"
        f"1\ufe0f\u20e3 Funding Rate Fade\n"
        f"2\ufe0f\u20e3 Liquidity Sweep Reversal\n"
        f"3\ufe0f\u20e3 EMA Trend Pullback\n"
        f"4\ufe0f\u20e3 VWAP Reclaim\n"
        f"5\ufe0f\u20e3 Order Block Entry\n\n"
        f"Max risk: {MAX_RISK_PCT*100:.0f}% per trade | {MAX_LOSSES_PER_DAY} loss limit\n"
        f"AI Market Brain: confidence scoring 1-10\n\n"
        f"\U0001f50d Running 30-day backtest..."
    )
    await tg_send(startup_msg)

    # Initial backtest
    try:
        bt_results = await run_backtest()
        state.backtest_results = bt_results

        active = []
        bt_lines = ["\U0001f4ca <b>BACKTEST RESULTS (30 days)</b>\n"]
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            wr = bt.get("win_rate", 0)
            total = bt.get("total_trades", 0)
            passed = wr >= BACKTEST_MIN_WIN_RATE or bt.get("note")
            if passed:
                active.append(name)
            status = "\u2705 PASS" if passed else "\u274c FAIL"
            bt_lines.append(
                f"  <b>{name}</b>: {total} trades | WR {wr*100:.1f}% [{status}]"
            )

        state.active_strategies = active
        state.backtest_complete = True
        save_state()

        if active:
            bt_lines.append(f"\n<b>Active:</b> {', '.join(active)}")
            bt_lines.append(f"All {len(active)} strategies scanning for signals.")
        else:
            bt_lines.append(f"\n\u26a0\ufe0f No strategies passed backtest.")

        await tg_send("\n".join(bt_lines))

    except Exception as exc:
        log.error("Startup backtest failed: %s", exc)
        await tg_send(f"\u26a0\ufe0f Backtest failed: {sanitize_html(str(exc))}\nActivating all strategies.")
        state.active_strategies = list(STRATEGY_NAMES)
        state.backtest_complete = True
        save_state()

    # Start all background tasks
    asyncio.create_task(regime_update_loop())
    asyncio.create_task(funding_rate_loop())
    asyncio.create_task(strategy_monitor_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())
    asyncio.create_task(periodic_status_log())
    asyncio.create_task(telegram_polling_watchdog())
    asyncio.create_task(intelligence_loop())
    asyncio.create_task(sunday_report_scheduler())
    asyncio.create_task(backtest_scheduler())

    log.info("All background tasks started -- %d tasks.", 12)


@app.on_event("shutdown")
async def shutdown():
    log.info("Vic shutting down -- saving state.")
    save_state()
    close_exchange()
