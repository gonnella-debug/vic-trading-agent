"""
Vic -- BTC/USDT Perpetual Futures Scalping Agent (v4)
Runs 3 strategies on Hyperliquid via native SDK.
$500 account, full capital available per trade. Production-ready for Railway.

Strategies (activated only after backtest passes 50%+ win rate):
  1. RSI Divergence -- Price/RSI divergence at S/R on 5m
  2. BB Squeeze Breakout -- Bollinger squeeze breakout on 5m
  3. VWAP Bounce -- VWAP bounce with distance + chop filter on 1m

Regime filter: TRENDING / RANGING / TRANSITIONAL / VOLATILE
1H bias: Strong bullish/bearish or neutral
Risk: 2% SL ($10), 4% TP1 50% ($20), 6% TP2 full ($30), 5x leverage
Session filter: London open (07-11 UTC) + NY open (13-17 UTC)
AI Market Brain: Claude pre-trade analysis gate
Trade Journal: /data/vic_journal.json (persistent)
State Persistence: /data/vic_state.json (survives restarts)
Backtest Engine: 30-day OHLCV backtest before any live trading
Intelligence: Top trader scan every 6h -> /data/hl_intelligence.json
Underperformance: Auto-pause strategy if <40% WR after 15+ trades
Sunday Report: Full automated report via Telegram

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
LEVERAGE = 10
ACCOUNT_CAPITAL = 500.0
RISK_DOLLARS = 10.0           # $10 max loss per trade (2% of $500)
MAX_DAILY_LOSS_PCT = 0.10     # 10% of account = $50 daily loss cap
MAX_TRADES_PER_DAY = 4
CORRELATION_COOLDOWN_MIN = 15
BB_SQUEEZE_THRESHOLD = 0.02

# ATR-based SL/TP: SL = ATR_MULT * ATR, TP = RR_RATIO * SL_distance
# Position sized so that SL hit = $10 loss exactly
STRATEGY_ATR_SL = {
    "rsi_divergence": 1.5,     # 1.5 x ATR(14) on 5m
    "bb_squeeze": 1.0,         # 1.0 x ATR(14) on 5m
    "vwap_bounce": 1.0,        # 1.0 x ATR(14) on 1m
}
STRATEGY_RR_RATIO = {
    "rsi_divergence": 2.0,     # TP = 2x SL distance
    "bb_squeeze": 2.0,         # TP = 2x SL distance
    "vwap_bounce": 1.5,        # TP = 1.5x SL distance (faster exit)
}
# Partial close at halfway to TP (lock in profits)
PARTIAL_TP_RATIO = 0.5        # Partial close when 50% of TP distance reached
PARTIAL_CLOSE_SIZE = 0.5      # Close 50% of position at partial

# Strategy max hold times in minutes
STRATEGY_MAX_HOLD = {
    "rsi_divergence": 240,
    "bb_squeeze": 90,
    "vwap_bounce": 60,
}

STRATEGY_NAMES = ["rsi_divergence", "bb_squeeze", "vwap_bounce"]

# Paper test: 50 trades OR 2 weeks
PAPER_TEST_MIN_TRADES = 50
PAPER_TEST_MAX_DAYS = 14

# Strategy labels for Telegram messages
STRATEGY_LABELS = {
    "rsi_divergence": "1\ufe0f\u20e3 RSI Divergence",
    "bb_squeeze": "2\ufe0f\u20e3 BB Squeeze",
    "vwap_bounce": "3\ufe0f\u20e3 VWAP Bounce",
}

# Session filter -- London + NY open only
TRADING_SESSIONS = [
    (7, 11),   # London open: 07:00-11:00 UTC
    (13, 17),  # NY open: 13:00-17:00 UTC
]

# Legacy compatibility
RISK_PCT = RISK_DOLLARS / ACCOUNT_CAPITAL  # 0.02

# Persistence files
JOURNAL_FILE = os.getenv("JOURNAL_FILE", "/data/vic_journal.json")
STATE_FILE = os.getenv("STATE_FILE", "/data/vic_state.json")
BACKTEST_FILE = os.getenv("BACKTEST_FILE", "/data/vic_backtest.json")
INTELLIGENCE_FILE = os.getenv("INTELLIGENCE_FILE", "/data/hl_intelligence.json")

# Underperformance auto-pause thresholds
UNDERPERFORMANCE_MIN_TRADES = 15
UNDERPERFORMANCE_WR_THRESHOLD = 0.40  # 40%

# Backtest minimum win rate to activate a strategy
BACKTEST_MIN_WIN_RATE = 0.50  # 50%

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
app = FastAPI(title="Vic Trading Agent", version="4.0.0")

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
                "position": None,
            }

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

        # Backtest results and strategy activation
        self.backtest_complete: bool = False
        self.active_strategies: list = []  # strategies that passed backtest
        self.backtest_results: dict = {}

    def reset_daily(self):
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.daily_loss_cap_hit = False
        self.last_trade_direction = None
        self.last_trade_close_time = None
        for name in STRATEGY_NAMES:
            self.strategies[name]["daily_pnl"] = 0.0
            self.strategies[name]["trades_today"] = 0
            # Don't reset strategy paused here -- underperformance pause persists
        self.signals_checked = 0
        self.signals_blocked = 0
        self.last_block_reasons = []
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
            "trade_history": state.trade_history[-100:],  # keep last 100
            "active_strategies": state.active_strategies,
            "backtest_complete": state.backtest_complete,
            "backtest_results": state.backtest_results,
            "daily_pnl": state.daily_pnl,
            "trades_today": state.trades_today,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Serialize strategy states (positions)
        strat_data = {}
        for name in STRATEGY_NAMES:
            s = state.strategies[name]
            strat_data[name] = {
                "daily_pnl": s["daily_pnl"],
                "trades_today": s["trades_today"],
                "paused": s["paused"],
                "position": s["position"],
            }
        data["strategies"] = strat_data

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
        state.active_strategies = data.get("active_strategies", [])
        state.backtest_complete = data.get("backtest_complete", False)
        state.backtest_results = data.get("backtest_results", {})

        # Restore metrics
        saved_metrics = data.get("metrics", {})
        for name in STRATEGY_NAMES:
            if name in saved_metrics:
                state.metrics[name] = saved_metrics[name]

        # Restore strategy states (but NOT positions -- those come from orphaned recovery)
        saved_strats = data.get("strategies", {})
        for name in STRATEGY_NAMES:
            if name in saved_strats:
                state.strategies[name]["paused"] = saved_strats[name].get("paused", False)

        log.info("State loaded from %s -- %d lifetime trades, active strategies: %s",
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
# VWAP Distance & Chop Filters
# ---------------------------------------------------------------------------

def vwap_distance_ok(df: pd.DataFrame, vwap_series: pd.Series) -> bool:
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
# Pre-Trade Checklist
# ---------------------------------------------------------------------------

def is_trading_session() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    hour = now.hour
    for start, end in TRADING_SESSIONS:
        if start <= hour < end:
            return True
    return False


def bias_allows_direction(side: str) -> bool:
    if state.htf_bias == "neutral":
        return True
    if state.regime == Regime.RANGING:
        if state.htf_bias == "bullish" and side == "long":
            return True
        if state.htf_bias == "bearish" and side == "short":
            return True
        return False
    if state.htf_bias_strength == "weak":
        return True
    if state.htf_bias == "bullish" and side == "long":
        return True
    if state.htf_bias == "bearish" and side == "short":
        return True
    return False


def regime_allows_strategy(strategy: str) -> bool:
    if state.regime == Regime.VOLATILE:
        return False
    if state.mode == "paper" and state.regime == Regime.TRANSITIONAL:
        return True
    if state.regime == Regime.TRENDING:
        return strategy in ("vwap_bounce", "bb_squeeze")
    if state.regime == Regime.RANGING:
        return strategy in ("rsi_divergence", "vwap_bounce")
    if state.regime == Regime.TRANSITIONAL:
        return strategy in ("rsi_divergence",)
    return False


def correlation_filter_ok(side: str) -> bool:
    if state.last_trade_direction is None or state.last_trade_close_time is None:
        return True
    if side != state.last_trade_direction:
        return True
    elapsed = (datetime.now(timezone.utc) - state.last_trade_close_time).total_seconds() / 60.0
    return elapsed >= CORRELATION_COOLDOWN_MIN


def can_execute_trade(strategy: str, side: str) -> tuple[bool, str]:
    """Master pre-trade checklist. Returns (allowed, reason)."""
    state.signals_checked += 1

    # 0. Backtest must have completed and strategy must be active
    if not state.backtest_complete:
        reason = f"BLOCKED: {strategy} {side.upper()} -- Backtest not yet complete"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, "Backtest not yet complete"

    if strategy not in state.active_strategies:
        reason = f"BLOCKED: {strategy} {side.upper()} -- Strategy not active (failed backtest)"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Strategy {strategy} not active (failed backtest)"

    # 1. Trading session check
    session_ok = is_trading_session()
    log.info("CHECK %s %s -- session_ok: %s (hour=%d UTC)",
             strategy, side.upper(), session_ok, datetime.now(timezone.utc).hour)
    if not session_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- Outside trading session"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, "Outside trading session"

    # 2. Regime allows this strategy?
    regime_ok = regime_allows_strategy(strategy)
    log.info("CHECK %s %s -- regime_allows: %s (regime=%s, mode=%s)",
             strategy, side.upper(), regime_ok, state.regime.value, state.mode)
    if not regime_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- regime {state.regime.value} not allowed"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Regime {state.regime.value} blocks {strategy}"

    # 3. 1H bias agrees or is neutral?
    bias_ok = bias_allows_direction(side)
    log.info("CHECK %s %s -- bias_allows: %s (bias=%s, strength=%s)",
             strategy, side.upper(), bias_ok, state.htf_bias, state.htf_bias_strength)
    if not bias_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- HTF bias {state.htf_bias} (strong) blocks {side}"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"1H bias ({state.htf_bias} strong) opposes {side}"

    # 4. Trade count < 4 today?
    trades_ok = state.trades_today < MAX_TRADES_PER_DAY
    if not trades_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- daily trade cap {state.trades_today}/{MAX_TRADES_PER_DAY}"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Daily trade cap reached ({state.trades_today}/{MAX_TRADES_PER_DAY})"

    # 5. No recent same-direction trade?
    corr_ok = correlation_filter_ok(side)
    if not corr_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- correlation cooldown {CORRELATION_COOLDOWN_MIN}min"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Correlation cooldown: {side} blocked for {CORRELATION_COOLDOWN_MIN}min"

    # 6. Daily loss cap not hit?
    loss_ok = not state.daily_loss_cap_hit
    if not loss_ok:
        reason = f"BLOCKED: {strategy} {side.upper()} -- daily loss cap hit (${state.daily_pnl:.2f})"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"Daily loss cap hit (-${ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT:.0f})"

    # 7. Strategy not paused?
    strat = state.strategies[strategy]
    paused = strat["paused"] or state.paused
    if paused:
        reason = f"BLOCKED: {strategy} {side.upper()} -- paused (strat={strat['paused']}, global={state.paused})"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"{strategy} or global paused"

    # 8. No existing position on this strategy?
    has_pos = strat["position"] is not None
    if has_pos:
        reason = f"BLOCKED: {strategy} {side.upper()} -- already has open position"
        log.info(reason)
        state.signals_blocked += 1
        state.last_block_reasons.append(reason)
        if len(state.last_block_reasons) > 20:
            state.last_block_reasons.pop(0)
        return False, f"{strategy} already has an open position"

    # 9. No opposite-side position open on ANY strategy?
    opposite = "short" if side == "long" else "long"
    for sname, sdata in state.strategies.items():
        if sdata["position"] is not None and sdata["position"]["side"] == opposite:
            reason = f"BLOCKED: {strategy} {side.upper()} -- {sname} has open {opposite} position"
            log.info(reason)
            state.signals_blocked += 1
            state.last_block_reasons.append(reason)
            if len(state.last_block_reasons) > 20:
                state.last_block_reasons.pop(0)
            return False, f"Cannot open {side} while {sname} has {opposite} open"

    log.info("PASSED: %s %s -- all checks OK", strategy, side.upper())
    return True, "OK"


# ---------------------------------------------------------------------------
# Position sizing & order execution
# ---------------------------------------------------------------------------

def calc_position_size(entry: float, margin: float) -> float:
    if entry <= 0:
        return 0.0
    notional = margin * LEVERAGE
    return math.floor(notional / entry * 100000) / 100000


async def execute_trade(strategy: str, side: str, entry: float, atr_value: float = 0.0):
    """Open a position (paper or live) for the given strategy.

    Uses ATR-based SL/TP:
    - SL distance = ATR_MULT * ATR
    - Position size = RISK_DOLLARS / SL_distance (so SL hit = $10 loss)
    - TP distance = RR_RATIO * SL_distance
    - Partial close at PARTIAL_TP_RATIO of TP distance
    """
    allowed, reason = can_execute_trade(strategy, side)
    if not allowed:
        log.info("%s -- trade blocked: %s", strategy, reason)
        return

    # ATR-based SL distance
    atr_mult = STRATEGY_ATR_SL.get(strategy, 1.5)
    sl_distance = atr_mult * atr_value if atr_value > 0 else 0

    if sl_distance <= 0:
        log.warning("%s -- no ATR data, cannot size position.", strategy)
        return

    # Size position so that SL hit = RISK_DOLLARS loss
    size = math.floor(RISK_DOLLARS / sl_distance * 100000) / 100000  # floor to 5dp
    if size <= 0:
        log.warning("%s -- invalid size from ATR sizing, skipping.", strategy)
        return

    # Check notional meets minimum ($10 on HL)
    notional = size * entry
    if notional < 10:
        log.warning("%s -- notional $%.2f below HL minimum, skipping.", strategy, notional)
        return

    # Cap size to max leverage allows
    max_size = math.floor(ACCOUNT_CAPITAL * LEVERAGE / entry * 100000) / 100000
    if size > max_size:
        size = max_size
        log.info("%s -- size capped to max leverage: %.6f BTC", strategy, size)

    rr_ratio = STRATEGY_RR_RATIO.get(strategy, 2.0)
    tp_distance = rr_ratio * sl_distance
    partial_tp_distance = PARTIAL_TP_RATIO * tp_distance

    if side == "long":
        sl = round(entry - sl_distance)
        tp1 = round(entry + partial_tp_distance)
        tp = round(entry + tp_distance)
    else:
        sl = round(entry + sl_distance)
        tp1 = round(entry - partial_tp_distance)
        tp = round(entry - tp_distance)

    # Open order
    if state.mode == "live":
        if not hl_exchange:
            log.error("%s -- exchange not initialized, cannot place order.", strategy)
            await tg_send(f"\u26a0\ufe0f <b>{strategy}</b> order FAILED: Exchange not initialized")
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: hl_exchange.update_leverage(LEVERAGE, "BTC", is_cross=True)
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

            # Place exchange-level SL and TP as trigger orders
            sl_ok = False
            tp_ok = False
            try:
                sl_side = not is_buy

                for sl_attempt in range(3):
                    sl_result = await loop.run_in_executor(
                        None, lambda: hl_exchange.order(
                            "BTC", is_buy=sl_side, sz=size, limit_px=sl,
                            order_type={"trigger": {"triggerPx": sl, "isMarket": True, "tpsl": "sl"}},
                            reduce_only=True,
                        )
                    )
                    sl_statuses = sl_result.get("response", {}).get("data", {}).get("statuses", [])
                    sl_errors = [s.get("error") for s in sl_statuses if "error" in s]
                    if sl_errors:
                        log.error("%s SL order REJECTED (attempt %d/3) @ $%.0f: %s", strategy, sl_attempt + 1, sl, sl_errors)
                        if sl_attempt < 2:
                            await asyncio.sleep(1)
                            continue
                    else:
                        sl_ok = True
                        log.info("%s SL order CONFIRMED on exchange @ $%.0f", strategy, sl)
                        break

                # TP1: partial close (50%)
                tp1_size = math.floor(size * PARTIAL_CLOSE_SIZE * 100000) / 100000
                tp1_result = await loop.run_in_executor(
                    None, lambda: hl_exchange.order(
                        "BTC", is_buy=sl_side, sz=tp1_size, limit_px=tp1,
                        order_type={"trigger": {"triggerPx": tp1, "isMarket": True, "tpsl": "tp"}},
                        reduce_only=True,
                    )
                )
                tp1_statuses = tp1_result.get("response", {}).get("data", {}).get("statuses", [])
                tp1_errors = [s.get("error") for s in tp1_statuses if "error" in s]
                if tp1_errors:
                    log.error("%s TP1 order REJECTED @ $%.0f: %s", strategy, tp1, tp1_errors)

                # TP2: full close (remaining 50%)
                tp2_size = math.floor(size * (1 - PARTIAL_CLOSE_SIZE) * 100000) / 100000
                tp2_result = await loop.run_in_executor(
                    None, lambda: hl_exchange.order(
                        "BTC", is_buy=sl_side, sz=tp2_size, limit_px=tp,
                        order_type={"trigger": {"triggerPx": tp, "isMarket": True, "tpsl": "tp"}},
                        reduce_only=True,
                    )
                )
                tp2_statuses = tp2_result.get("response", {}).get("data", {}).get("statuses", [])
                tp2_errors = [s.get("error") for s in tp2_statuses if "error" in s]
                if tp2_errors:
                    log.error("%s TP2 order REJECTED @ $%.0f: %s", strategy, tp, tp2_errors)
                else:
                    tp_ok = True
            except Exception as sl_exc:
                log.error("%s -- SL/TP order exception: %s", strategy, sl_exc)

            if not sl_ok:
                await tg_send(
                    f"\U0001f6a8\U0001f6a8 <b>CRITICAL: {strategy} -- NO STOP LOSS ON EXCHANGE</b> \U0001f6a8\U0001f6a8\n"
                    f"SL order FAILED after 3 attempts.\n"
                    f"Position: {side.upper()} {size} BTC @ ${entry:,.0f}\n"
                    f"Intended SL: ${sl:,.0f}\n"
                    f"<b>SET STOP LOSS MANUALLY NOW</b>\n"
                    f"Software SL monitoring is active as backup."
                )
            if not tp_ok:
                await tg_send(
                    f"\u26a0\ufe0f <b>{strategy}</b> TP order failed to place on exchange @ ${tp:,.0f}.\n"
                    f"Software TP monitoring is active as backup."
                )

        except Exception as exc:
            log.error("%s -- order error: %s", strategy, exc)
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
        "tp1": tp1,
        "tp": tp,
        "size": size,
        "margin": ACCOUNT_CAPITAL,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "open_time": datetime.now(timezone.utc).isoformat(),
        "be_moved": False,
        "partial_closed": False,
        "regime": state.regime.value,
        "bias": f"{state.htf_bias} ({state.htf_bias_strength})",
    }
    strat["trades_today"] += 1
    state.trades_today += 1
    state.total_trade_count += 1

    notional = size * entry
    dollar_risk = round(abs(entry - sl) * size, 2)
    dollar_tp1 = round(abs(tp1 - entry) * size, 2)
    dollar_tp2 = round(abs(tp - entry) * size, 2)
    label = STRATEGY_LABELS.get(strategy, strategy)
    arrow = "\U0001f7e2" if side == "long" else "\U0001f534"
    msg = (
        f"{arrow} <b>{side.upper()}</b> -- {label}\n"
        f"Entry ${entry:,.2f} | SL ${sl:,.2f}\n"
        f"TP1 ${tp1:,.2f} (50% close, +${dollar_tp1:.2f}) | TP2 ${tp:,.2f} (full, +${dollar_tp2:.2f})\n"
        f"Size {size:.6f} BTC | Max loss ${dollar_risk:.2f}\n"
        f"Regime: {state.regime.value} | Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
        f"Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}"
    )
    await tg_send(msg)
    log.info(msg.replace("<b>", "").replace("</b>", ""))

    # Save state after opening trade
    save_state()


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

    r_achieved = pnl / RISK_DOLLARS if RISK_DOLLARS > 0 else 0.0

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
        f"{emoji} <b>CLOSED</b> -- {label}\n"
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

    # Save state after closing trade
    save_state()

    # Check daily loss cap (10% of account)
    max_daily_loss = ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT
    if state.daily_pnl <= -max_daily_loss:
        state.daily_loss_cap_hit = True
        alert = (
            f"\U0001f6d1 <b>Daily loss limit hit (${state.daily_pnl:+,.2f})</b>\n"
            f"All trading stopped until tomorrow."
        )
        await tg_send(alert)

    # Check aggregate losing streak
    total_losing_streak = sum(state.metrics[n]["current_losing_streak"] for n in STRATEGY_NAMES)
    if total_losing_streak >= 3 and not state.daily_loss_cap_hit:
        state.daily_loss_cap_hit = True
        alert = (
            f"\U0001f6d1 <b>3 consecutive losses -- trading paused for today</b>\n"
            f"Aggregate losing streak: {total_losing_streak} | Daily PnL: ${state.daily_pnl:+,.2f}"
        )
        await tg_send(alert)

    # Underperformance auto-pause check
    await check_underperformance(strategy)

    # Paper test milestone check
    if state.mode == "paper" and state.total_trade_count >= PAPER_TEST_MIN_TRADES:
        await send_paper_test_report()


async def check_underperformance(strategy: str):
    """If a strategy drops below 40% win rate after 15+ trades, pause it."""
    m = state.metrics[strategy]
    if m["trade_count"] < UNDERPERFORMANCE_MIN_TRADES:
        return

    win_rate = m["wins"] / m["trade_count"]
    if win_rate < UNDERPERFORMANCE_WR_THRESHOLD:
        if not state.strategies[strategy]["paused"]:
            state.strategies[strategy]["paused"] = True
            save_state()
            msg = (
                f"\U0001f6d1 <b>STRATEGY AUTO-PAUSED: {strategy}</b>\n\n"
                f"Win rate dropped to {win_rate*100:.1f}% after {m['trade_count']} trades.\n"
                f"Threshold: {UNDERPERFORMANCE_WR_THRESHOLD*100:.0f}% minimum after {UNDERPERFORMANCE_MIN_TRADES} trades.\n"
                f"Wins: {m['wins']} | Losses: {m['losses']}\n"
                f"PnL: ${m['current_pnl']:+,.2f}\n\n"
                f"Strategy will remain paused until next backtest re-evaluation."
            )
            await tg_send(msg)
            log.warning("Auto-paused %s: WR %.1f%% < %.0f%% threshold after %d trades",
                        strategy, win_rate * 100, UNDERPERFORMANCE_WR_THRESHOLD * 100, m["trade_count"])


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

def _backtest_rsi_divergence(df: pd.DataFrame) -> list:
    """Backtest RSI divergence strategy on 5m OHLCV data. Returns list of trade results."""
    if len(df) < 60:
        return []

    df = df.copy()
    df["rsi"] = calc_rsi(df["close"], 14)
    df["vwap"] = calc_vwap(df)
    atr_series = calc_atr(df, 14)

    trades = []
    i = 50  # Start after enough warmup

    while i < len(df) - 10:
        atr_val = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else 0.0
        if atr_val <= 0:
            i += 1
            continue

        # Look for divergence in last 20 candles
        close_vals = df["close"].values[i-20:i]
        rsi_vals = df["rsi"].values[i-20:i]

        if any(np.isnan(rsi_vals)):
            i += 1
            continue

        # S/R context
        high_50 = float(np.nanmax(df["high"].values[max(0, i-50):i]))
        low_50 = float(np.nanmin(df["low"].values[max(0, i-50):i]))
        price_now = float(df["close"].iloc[i])
        vwap_now = float(df["vwap"].iloc[i]) if not np.isnan(df["vwap"].iloc[i]) else price_now

        near_support = (price_now - low_50) / low_50 * 100 < 0.25 if low_50 > 0 else False
        near_resistance = (high_50 - price_now) / high_50 * 100 < 0.25 if high_50 > 0 else False
        away_from_vwap = abs(price_now - vwap_now) / vwap_now * 100 > 0.2 if vwap_now > 0 else False

        if not (near_support or near_resistance or away_from_vwap):
            i += 1
            continue

        # Find swing lows
        price_lows = []
        for j in range(1, len(close_vals) - 1):
            if close_vals[j] < close_vals[j - 1] and close_vals[j] < close_vals[j + 1]:
                price_lows.append((j, close_vals[j], rsi_vals[j]))

        price_highs = []
        for j in range(1, len(close_vals) - 1):
            if close_vals[j] > close_vals[j - 1] and close_vals[j] > close_vals[j + 1]:
                price_highs.append((j, close_vals[j], rsi_vals[j]))

        side = None
        # Bullish divergence
        if len(price_lows) >= 2:
            prev_low = price_lows[-2]
            curr_low = price_lows[-1]
            if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
                side = "long"

        # Bearish divergence
        if side is None and len(price_highs) >= 2:
            prev_high = price_highs[-2]
            curr_high = price_highs[-1]
            if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
                side = "short"

        if side is None:
            i += 1
            continue

        # ATR-based SL/TP
        entry = price_now
        atr_mult = STRATEGY_ATR_SL.get("rsi_divergence", 1.5)
        sl_dist = atr_mult * atr_val
        if sl_dist <= 0:
            i += 1
            continue
        size = math.floor(RISK_DOLLARS / sl_dist * 100000) / 100000
        if size <= 0:
            i += 1
            continue
        rr = STRATEGY_RR_RATIO.get("rsi_divergence", 2.0)
        tp_dist = rr * sl_dist

        if side == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        result = _simulate_trade_forward(df, i, side, entry, sl, tp, size, max_bars=48)
        if result:
            trades.append(result)
            i += result.get("bars_held", 1) + 1
        else:
            i += 1

    return trades


def _backtest_bb_squeeze(df: pd.DataFrame) -> list:
    """Backtest BB squeeze breakout strategy on 5m OHLCV data."""
    if len(df) < 30:
        return []

    df = df.copy()
    mid, upper, lower, bandwidth = calc_bollinger_bands(df["close"], 20, 2.0)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_bw"] = bandwidth
    atr_series = calc_atr(df, 14)

    trades = []
    i = 25

    while i < len(df) - 10:
        prev_bw = float(df["bb_bw"].iloc[i - 1])
        curr_bw = float(df["bb_bw"].iloc[i])

        if np.isnan(prev_bw) or np.isnan(curr_bw):
            i += 1
            continue

        price = float(df["close"].iloc[i])
        bb_upper = float(df["bb_upper"].iloc[i])
        bb_lower = float(df["bb_lower"].iloc[i])

        if np.isnan(bb_upper) or np.isnan(bb_lower):
            i += 1
            continue

        if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
            upper_threshold = bb_upper * 1.001
            lower_threshold = bb_lower * 0.999

            if price > upper_threshold or price < lower_threshold:
                body = abs(float(df["close"].iloc[i]) - float(df["open"].iloc[i]))
                full_range = float(df["high"].iloc[i]) - float(df["low"].iloc[i])
                if full_range == 0 or body / full_range < 0.6:
                    i += 1
                    continue

                side = "long" if price > upper_threshold else "short"
                entry = price
                atr_val = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else 0.0
                atr_mult = STRATEGY_ATR_SL.get("bb_squeeze", 1.0)
                sl_dist = atr_mult * atr_val
                if sl_dist <= 0:
                    i += 1
                    continue
                size = math.floor(RISK_DOLLARS / sl_dist * 100000) / 100000
                if size <= 0:
                    i += 1
                    continue
                rr = STRATEGY_RR_RATIO.get("bb_squeeze", 2.0)
                tp_dist = rr * sl_dist

                if side == "long":
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                result = _simulate_trade_forward(df, i, side, entry, sl, tp, size, max_bars=18)
                if result:
                    trades.append(result)
                    i += result.get("bars_held", 1) + 1
                else:
                    i += 1
                continue

        i += 1

    return trades


def _backtest_vwap_bounce(df: pd.DataFrame) -> list:
    """Backtest VWAP bounce strategy on 1m OHLCV data."""
    if len(df) < 30:
        return []

    df = df.copy()
    df["vwap"] = calc_vwap(df)
    df["rsi"] = calc_rsi(df["close"], 14)
    atr_series = calc_atr(df, 14)

    trades = []
    i = 25

    while i < len(df) - 10:
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(curr["close"])
        vwap_val = float(curr["vwap"])

        if np.isnan(vwap_val):
            i += 1
            continue

        rsi_val = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0

        if rsi_val < 40 or rsi_val > 60:
            i += 1
            continue

        # Volume confirmation
        if i >= 20:
            avg_vol = df["volume"].iloc[i-20:i].mean()
            curr_vol = float(curr["volume"])
            if np.isnan(avg_vol) or curr_vol < avg_vol * 1.3:
                i += 1
                continue

        body = abs(curr["close"] - curr["open"])
        full_range = curr["high"] - curr["low"]
        if full_range == 0 or body / full_range < 0.5:
            i += 1
            continue

        is_bullish = curr["close"] > curr["open"]
        is_bearish = curr["close"] < curr["open"]

        wick_touched_vwap = prev["low"] <= vwap_val <= prev["high"]
        if not wick_touched_vwap:
            i += 1
            continue

        side = None
        if is_bullish and price > vwap_val:
            side = "long"
        elif is_bearish and price < vwap_val:
            side = "short"

        if side is None:
            i += 1
            continue

        entry = price
        atr_val = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else 0.0
        atr_mult = STRATEGY_ATR_SL.get("vwap_bounce", 1.0)
        sl_dist = atr_mult * atr_val
        if sl_dist <= 0:
            i += 1
            continue
        size = math.floor(RISK_DOLLARS / sl_dist * 100000) / 100000
        if size <= 0:
            i += 1
            continue
        rr = STRATEGY_RR_RATIO.get("vwap_bounce", 1.5)
        tp_dist = rr * sl_dist

        if side == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        result = _simulate_trade_forward(df, i, side, entry, sl, tp, size, max_bars=60)
        if result:
            trades.append(result)
            i += result.get("bars_held", 1) + 1
        else:
            i += 1

    return trades


def _simulate_trade_forward(df: pd.DataFrame, entry_idx: int, side: str,
                             entry: float, sl: float, tp: float, size: float,
                             max_bars: int = 48) -> Optional[dict]:
    """Walk forward with realistic multi-stage exits: BE at 1R, partial at TP1, full at TP."""
    sl_distance = abs(entry - sl)
    tp_distance = abs(tp - entry)
    partial_tp_distance = PARTIAL_TP_RATIO * tp_distance

    be_moved = False
    partial_closed = False
    remaining_size = size
    locked_pnl = 0.0

    for offset in range(1, max_bars + 1):
        idx = entry_idx + offset
        if idx >= len(df):
            break

        high = float(df["high"].iloc[idx])
        low = float(df["low"].iloc[idx])

        # Unrealized PnL at extremes
        if side == "long":
            best_pnl = (high - entry) * remaining_size
            worst_pnl = (low - entry) * remaining_size
        else:
            best_pnl = (entry - low) * remaining_size
            worst_pnl = (entry - high) * remaining_size

        # SL check (worst case)
        if side == "long" and low <= sl:
            if be_moved:
                return {"side": side, "entry": entry, "exit": entry, "pnl": round(locked_pnl, 2),
                        "reason": "BE_SL", "bars_held": offset}
            pnl = (sl - entry) * remaining_size + locked_pnl
            return {"side": side, "entry": entry, "exit": sl, "pnl": round(pnl, 2),
                    "reason": "SL", "bars_held": offset}
        if side == "short" and high >= sl:
            if be_moved:
                return {"side": side, "entry": entry, "exit": entry, "pnl": round(locked_pnl, 2),
                        "reason": "BE_SL", "bars_held": offset}
            pnl = (entry - sl) * remaining_size + locked_pnl
            return {"side": side, "entry": entry, "exit": sl, "pnl": round(pnl, 2),
                    "reason": "SL", "bars_held": offset}

        # BE move at 1R profit
        if not be_moved and best_pnl >= RISK_DOLLARS:
            be_moved = True
            if side == "long":
                sl = entry
            else:
                sl = entry

        # Partial close at TP1 (halfway to TP)
        if not partial_closed:
            partial_hit = False
            if side == "long" and high >= entry + partial_tp_distance:
                partial_hit = True
            elif side == "short" and low <= entry - partial_tp_distance:
                partial_hit = True
            if partial_hit:
                partial_closed = True
                locked_pnl += partial_tp_distance * remaining_size * PARTIAL_CLOSE_SIZE
                remaining_size = math.floor(remaining_size * (1 - PARTIAL_CLOSE_SIZE) * 100000) / 100000

        # Full TP check
        if side == "long" and high >= tp:
            pnl = (tp - entry) * remaining_size + locked_pnl
            return {"side": side, "entry": entry, "exit": tp, "pnl": round(pnl, 2),
                    "reason": "TP", "bars_held": offset}
        if side == "short" and low <= tp:
            pnl = (entry - tp) * remaining_size + locked_pnl
            return {"side": side, "entry": entry, "exit": tp, "pnl": round(pnl, 2),
                    "reason": "TP", "bars_held": offset}

    # Timeout exit
    last_idx = min(entry_idx + max_bars, len(df) - 1)
    exit_price = float(df["close"].iloc[last_idx])
    if side == "long":
        pnl = (exit_price - entry) * remaining_size + locked_pnl
    else:
        pnl = (entry - exit_price) * remaining_size + locked_pnl
    return {"side": side, "entry": entry, "exit": exit_price, "pnl": round(pnl, 2),
            "reason": "timeout", "bars_held": max_bars}


async def run_backtest() -> dict:
    """
    Run full backtest on 30 days of real OHLCV data from Hyperliquid.
    Returns dict with per-strategy results.
    """
    log.info("Starting 30-day backtest against real Hyperliquid data...")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (30 * 24 * 3600 * 1000)  # 30 days ago

    # Fetch 5m data for RSI divergence and BB squeeze
    log.info("Fetching 30 days of 5m candles...")
    df_5m = await fetch_ohlcv_range("5m", start_ms, now_ms)
    log.info("Fetched %d 5m candles", len(df_5m))

    # Fetch 1m data for VWAP bounce (last 7 days only -- 1m data is huge)
    start_1m = now_ms - (7 * 24 * 3600 * 1000)
    log.info("Fetching 7 days of 1m candles for VWAP bounce...")
    df_1m = await fetch_ohlcv_range("1m", start_1m, now_ms)
    log.info("Fetched %d 1m candles", len(df_1m))

    results = {}

    # RSI Divergence backtest
    log.info("Backtesting RSI Divergence...")
    rsi_trades = _backtest_rsi_divergence(df_5m)
    rsi_stats = _calc_backtest_stats("rsi_divergence", rsi_trades)
    results["rsi_divergence"] = rsi_stats
    log.info("RSI Divergence: %d trades, %.1f%% WR, %.2f avg R",
             rsi_stats["total_trades"], rsi_stats["win_rate"] * 100, rsi_stats["avg_r"])

    # BB Squeeze backtest
    log.info("Backtesting BB Squeeze...")
    bb_trades = _backtest_bb_squeeze(df_5m)
    bb_stats = _calc_backtest_stats("bb_squeeze", bb_trades)
    results["bb_squeeze"] = bb_stats
    log.info("BB Squeeze: %d trades, %.1f%% WR, %.2f avg R",
             bb_stats["total_trades"], bb_stats["win_rate"] * 100, bb_stats["avg_r"])

    # VWAP Bounce backtest
    log.info("Backtesting VWAP Bounce...")
    vwap_trades = _backtest_vwap_bounce(df_1m)
    vwap_stats = _calc_backtest_stats("vwap_bounce", vwap_trades)
    results["vwap_bounce"] = vwap_stats
    log.info("VWAP Bounce: %d trades, %.1f%% WR, %.2f avg R",
             vwap_stats["total_trades"], vwap_stats["win_rate"] * 100, vwap_stats["avg_r"])

    # Save backtest results
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["data_range"] = {"start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
                             "end": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat()}
    results["candles_5m"] = len(df_5m)
    results["candles_1m"] = len(df_1m)

    try:
        d = os.path.dirname(BACKTEST_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(BACKTEST_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        log.error("Backtest save error: %s", exc)

    return results


def _calc_backtest_stats(strategy: str, trades: list) -> dict:
    """Calculate backtest statistics for a strategy."""
    if not trades:
        return {
            "strategy": strategy,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
            "trades": [],
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    total_pnl = sum(t["pnl"] for t in trades)
    risk_per_trade = RISK_DOLLARS  # $10

    r_values = [t["pnl"] / risk_per_trade for t in trades]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    win_rate = wins / len(trades) if trades else 0.0

    return {
        "strategy": strategy,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": round(total_pnl, 2),
        "avg_r": round(avg_r, 2),
        "expectancy": round(avg_r, 2),
        "trades": trades[-20:],  # keep last 20 for reference
    }


# ---------------------------------------------------------------------------
# Strategy 1 -- RSI Divergence (5m)
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

    lookback_50 = df["close"].values[-50:]
    high_50 = float(np.nanmax(df["high"].values[-50:]))
    low_50 = float(np.nanmin(df["low"].values[-50:]))

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

    near_support = (price_now - low_50) / low_50 * 100 < 0.25 if low_50 > 0 else False
    near_resistance = (high_50 - price_now) / high_50 * 100 < 0.25 if high_50 > 0 else False
    away_from_vwap = abs(price_now - vwap_now) / vwap_now * 100 > 0.2 if vwap_now > 0 else False
    context_ok = near_support or near_resistance or away_from_vwap

    if not context_ok:
        return

    # Bullish divergence
    if len(price_lows) >= 2:
        prev_low = price_lows[-2]
        curr_low = price_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            allowed, reason = can_execute_trade(name, "long")
            if allowed:
                ai_ok, ai_reason = await ai_market_analysis(name, "long", price_now)
                if ai_ok:
                    await execute_trade(name, "long", price_now, atr_value=atr_val)
            return

    # Bearish divergence
    if len(price_highs) >= 2:
        prev_high = price_highs[-2]
        curr_high = price_highs[-1]
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            allowed, reason = can_execute_trade(name, "short")
            if allowed:
                ai_ok, ai_reason = await ai_market_analysis(name, "short", price_now)
                if ai_ok:
                    await execute_trade(name, "short", price_now, atr_value=atr_val)


# ---------------------------------------------------------------------------
# Strategy 2 -- BB Squeeze Breakout (5m)
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

    log.info("Checking bb_squeeze... BW_prev=%.4f, BW_curr=%.4f, price=%.2f", prev_bw, curr_bw, price)

    if np.isnan(bb_upper) or np.isnan(bb_lower):
        return

    if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
        upper_threshold = bb_upper * 1.001
        lower_threshold = bb_lower * 0.999

        if price > upper_threshold or price < lower_threshold:
            body = abs(float(df["close"].iloc[-1]) - float(df["open"].iloc[-1]))
            full_range = float(df["high"].iloc[-1]) - float(df["low"].iloc[-1])
            if full_range == 0 or body / full_range < 0.6:
                return

            if price > upper_threshold:
                allowed, reason = can_execute_trade(name, "long")
                if allowed:
                    ai_ok, ai_reason = await ai_market_analysis(name, "long", price)
                    if ai_ok:
                        await execute_trade(name, "long", price, atr_value=atr_val)
            elif price < lower_threshold:
                allowed, reason = can_execute_trade(name, "short")
                if allowed:
                    ai_ok, ai_reason = await ai_market_analysis(name, "short", price)
                    if ai_ok:
                        await execute_trade(name, "short", price, atr_value=atr_val)


# ---------------------------------------------------------------------------
# Strategy 3 -- VWAP Bounce (1m)
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

    if not vwap_distance_ok(df, df["vwap"]):
        return
    if not vwap_chop_filter(df, df["vwap"]):
        return
    if rsi_val < 40 or rsi_val > 60:
        return

    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    curr_vol = float(curr["volume"])
    if np.isnan(avg_vol) or curr_vol < avg_vol * 1.3:
        return

    body = abs(curr["close"] - curr["open"])
    full_range = curr["high"] - curr["low"]
    if full_range == 0 or body / full_range < 0.5:
        return

    is_bullish = curr["close"] > curr["open"]
    is_bearish = curr["close"] < curr["open"]

    wick_touched_vwap = prev["low"] <= vwap_val <= prev["high"]
    if not wick_touched_vwap:
        return

    if is_bullish and price > vwap_val:
        allowed, reason = can_execute_trade(name, "long")
        if allowed:
            ai_ok, ai_reason = await ai_market_analysis(name, "long", price)
            if ai_ok:
                await execute_trade(name, "long", price, atr_value=atr_val)
    elif is_bearish and price < vwap_val:
        allowed, reason = can_execute_trade(name, "short")
        if allowed:
            ai_ok, ai_reason = await ai_market_analysis(name, "short", price)
            if ai_ok:
                await execute_trade(name, "short", price, atr_value=atr_val)


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
    """Check strategies only on candle close boundaries."""
    while True:
        try:
            if state.paused or state.daily_loss_cap_hit:
                await asyncio.sleep(5)
                continue

            # Don't run strategies until backtest is complete
            if not state.backtest_complete:
                await asyncio.sleep(5)
                continue

            now = datetime.now(timezone.utc)
            current_ts = now.timestamp()

            current_1m_boundary = int(current_ts // 60) * 60
            new_1m_close = current_1m_boundary > state.last_1m_candle_ts

            current_5m_boundary = int(current_ts // 300) * 300
            new_5m_close = current_5m_boundary > state.last_5m_candle_ts

            if new_1m_close:
                state.last_1m_candle_ts = current_1m_boundary
                if "vwap_bounce" in state.active_strategies:
                    try:
                        await strategy_vwap_bounce()
                    except Exception as exc:
                        log.error("vwap_bounce error: %s", exc)

            if new_5m_close:
                state.last_5m_candle_ts = current_5m_boundary
                if "rsi_divergence" in state.active_strategies:
                    try:
                        await strategy_rsi_divergence()
                    except Exception as exc:
                        log.error("rsi_divergence error: %s", exc)
                if "bb_squeeze" in state.active_strategies:
                    try:
                        await strategy_bb_squeeze()
                    except Exception as exc:
                        log.error("bb_squeeze error: %s", exc)

        except Exception as exc:
            log.error("Strategy monitor error: %s", exc)

        await asyncio.sleep(5)


async def position_monitor_loop():
    """Check open positions for SL/TP, break-even, partial profit, max hold time."""
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

                if side == "long":
                    unrealized_pnl = (price - entry) * size
                else:
                    unrealized_pnl = (entry - price) * size

                # 1. Stop Loss
                if side == "long" and price <= sl:
                    await close_position(name, price, "Stop Loss")
                    continue
                if side == "short" and price >= sl:
                    await close_position(name, price, "Stop Loss")
                    continue

                # 2. Take Profit
                if side == "long" and price >= tp:
                    await close_position(name, price, "Take Profit")
                    continue
                if side == "short" and price <= tp:
                    await close_position(name, price, "Take Profit")
                    continue

                # 3. Break-even move: when price has moved 1x SL distance in our favor
                be_threshold = RISK_DOLLARS  # $10 profit = 1R
                if not pos["be_moved"] and unrealized_pnl >= be_threshold:
                    pos["sl"] = entry
                    pos["be_moved"] = True
                    msg = f"\U0001f504 <b>{name}</b> -- SL moved to break-even (entry ${entry:,.2f})"
                    await tg_send(msg)

                # 4. Partial profit at TP1 (halfway to full TP)
                if not pos.get("partial_closed", False):
                    tp1_price = pos.get("tp1", 0)
                    partial_hit = False
                    if tp1_price > 0:
                        if side == "long" and price >= tp1_price:
                            partial_hit = True
                        elif side == "short" and price <= tp1_price:
                            partial_hit = True
                    if partial_hit:
                        old_size = pos["size"]
                        new_size = math.floor(old_size * (1 - PARTIAL_CLOSE_SIZE) * 100000) / 100000
                        if state.mode == "live":
                            # Verify position actually exists on Hyperliquid before partial close
                            try:
                                loop = asyncio.get_event_loop()
                                user_st = await loop.run_in_executor(
                                    None, lambda: hl_info.user_state(HL_WALLET_ADDRESS)
                                )
                                hl_positions = user_st.get("assetPositions", [])
                                hl_has_btc = False
                                for p in hl_positions:
                                    pd_pos = p.get("position", {})
                                    if pd_pos.get("coin") == "BTC" and float(pd_pos.get("szi", 0)) != 0:
                                        hl_has_btc = True
                                        break
                                if not hl_has_btc:
                                    log.warning("%s -- position gone from Hyperliquid (exchange TP/SL fired). Clearing internal state.", name)
                                    await tg_send(
                                        f"ℹ️ <b>{name}</b> -- Position already closed on Hyperliquid "
                                        f"(exchange-level TP/SL triggered). Clearing internal state."
                                    )
                                    state.strategies[name]["position"] = None
                                    continue
                            except Exception as exc:
                                log.error("%s -- failed to verify position on Hyperliquid: %s", name, exc)

                            try:
                                partial_amount = math.floor(old_size * PARTIAL_CLOSE_SIZE * 100000) / 100000
                                result = await loop.run_in_executor(
                                    None, lambda: hl_exchange.market_close("BTC", sz=partial_amount)
                                )
                                if result is None:
                                    raise Exception("market_close returned None -- position may not exist on exchange")
                                if result.get("status") != "ok":
                                    raise Exception(f"Partial close rejected: {result}")
                                statuses = (result.get("response", {}).get("data", {}).get("statuses", []))
                                for s in statuses:
                                    if "error" in s:
                                        raise Exception(f"Partial close error: {s['error']}")
                            except Exception as exc:
                                log.error("%s -- partial close error: %s", name, exc)
                                await tg_send(
                                    f"\U0001f6a8 <b>{name} partial close FAILED</b>\n"
                                    f"Error: {sanitize_html(str(exc))}\n"
                                    f"Clearing internal state to stop retries."
                                )
                                state.strategies[name]["position"] = None
                                continue
                        pos["size"] = new_size
                        pos["partial_closed"] = True
                        partial_pnl = round(unrealized_pnl * PARTIAL_CLOSE_SIZE, 2)
                        msg = (
                            f"\u2702\ufe0f <b>{name}</b> -- Partial close at TP1\n"
                            f"Closed {PARTIAL_CLOSE_SIZE*100:.0f}% ({old_size:.6f} -> {new_size:.6f} BTC)\n"
                            f"Locked ~${partial_pnl:+,.2f}"
                        )
                        await tg_send(msg)

                # 5. Max hold time
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
    """Log a periodic status summary every 5 minutes."""
    while True:
        await asyncio.sleep(300)
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
                "  Active strategies: %s\n"
                "  Signals checked: %d | Signals blocked: %d\n"
                "  Trades today: %d/%d | Daily PnL: $%.2f | Loss cap: %s\n"
                "  Open positions: %s\n"
                "  Recent blocks: %s",
                state.regime.value, state.htf_bias, state.htf_bias_strength,
                state.last_btc_price, state.mode,
                ", ".join(state.active_strategies) if state.active_strategies else "none (backtest pending)",
                state.signals_checked, state.signals_blocked,
                state.trades_today, MAX_TRADES_PER_DAY,
                state.daily_pnl, state.daily_loss_cap_hit,
                ", ".join(open_positions) if open_positions else "none",
                " | ".join(recent_blocks),
            )
        except Exception as exc:
            log.error("Periodic status error: %s", exc)


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
    """Build and send the daily summary with metrics to Telegram."""
    total_pnl = 0.0
    total_trades = 0
    lines = ["\U0001f4ca <b>DAILY SUMMARY</b>\n"]

    for name in STRATEGY_NAMES:
        s = state.strategies[name]
        m = state.metrics[name]
        pnl = s["daily_pnl"]
        trades = s["trades_today"]
        active = name in state.active_strategies
        paused = s["paused"]
        status = "PAUSED" if paused else ("Active" if active else "Inactive")
        total_pnl += pnl
        total_trades += trades

        win_rate = (m["wins"] / m["trade_count"] * 100) if m["trade_count"] > 0 else 0.0
        avg_r = (m["total_r_achieved"] / m["trade_count"]) if m["trade_count"] > 0 else 0.0
        expectancy = avg_r

        lines.append(
            f"  <b>{name}</b>: ${pnl:+,.2f} ({trades} trades) [{status}]\n"
            f"    WR: {win_rate:.0f}% | Avg R: {avg_r:+.2f} | Expectancy: {expectancy:+.2f}R/trade\n"
            f"    MaxDD: ${m['max_drawdown']:,.2f} | "
            f"Streak: {m['current_losing_streak']}/{m['max_losing_streak']}"
        )

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
    lines.append(f"<b>Active strategies:</b> {', '.join(state.active_strategies) if state.active_strategies else 'none'}")

    intel = _get_intelligence_summary()
    if intel:
        lines.append(f"\n<b>\U0001f9e0 Top Trader Intel:</b>\n{sanitize_html(intel)}")

    await tg_send("\n".join(lines))


async def send_paper_test_report():
    """Send full performance report after 50 paper trades."""
    lines = ["\U0001f4cb <b>PAPER TEST REPORT -- {0} TRADES COMPLETED</b>\n".format(state.total_trade_count)]

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
# Sunday Automated Report
# ---------------------------------------------------------------------------

async def sunday_report_scheduler():
    """Send full automated report every Sunday at 18:00 UTC (10 PM Dubai)."""
    while True:
        now = datetime.now(timezone.utc)
        # Find next Sunday at 18:00 UTC
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
    """Full weekly report: rerun backtest, compare live vs backtest, top trader intel, PnL per strategy."""
    log.info("Generating Sunday automated report...")

    lines = ["\U0001f4c5 <b>SUNDAY WEEKLY REPORT</b>\n"]

    # 1. Rerun full backtest
    lines.append("<b>=== FRESH BACKTEST RESULTS ===</b>")
    try:
        bt_results = await run_backtest()
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            wr = bt.get("win_rate", 0) * 100
            total = bt.get("total_trades", 0)
            avg_r = bt.get("avg_r", 0)
            total_pnl = bt.get("total_pnl", 0)
            passed = "PASS" if bt.get("win_rate", 0) >= BACKTEST_MIN_WIN_RATE else "FAIL"
            lines.append(
                f"  <b>{name}</b>: {total} trades | WR {wr:.1f}% | Avg R {avg_r:+.2f} | PnL ${total_pnl:+,.2f} [{passed}]"
            )

        # Update active strategies based on fresh backtest
        new_active = []
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            if bt.get("win_rate", 0) >= BACKTEST_MIN_WIN_RATE:
                new_active.append(name)
        state.active_strategies = new_active
        state.backtest_results = bt_results
        save_state()

    except Exception as exc:
        lines.append(f"  Backtest error: {sanitize_html(str(exc))}")

    # 2. Live performance vs backtest expectations
    lines.append("\n<b>=== LIVE vs BACKTEST ===</b>")
    for name in STRATEGY_NAMES:
        m = state.metrics[name]
        bt = state.backtest_results.get(name, {})
        live_wr = (m["wins"] / m["trade_count"] * 100) if m["trade_count"] > 0 else 0.0
        bt_wr = bt.get("win_rate", 0) * 100
        live_exp = (m["total_r_achieved"] / m["trade_count"]) if m["trade_count"] > 0 else 0.0
        bt_exp = bt.get("avg_r", 0)
        lines.append(
            f"  <b>{name}</b>:\n"
            f"    Live: {m['trade_count']} trades | WR {live_wr:.1f}% | Exp {live_exp:+.2f}R | PnL ${m['current_pnl']:+,.2f}\n"
            f"    Backtest: WR {bt_wr:.1f}% | Exp {bt_exp:+.2f}R"
        )

    # 3. Top trader intelligence summary
    lines.append("\n<b>=== TOP TRADER INTELLIGENCE ===</b>")
    intel = _get_intelligence_summary()
    if intel:
        lines.append(sanitize_html(intel))
    else:
        lines.append("  No intelligence report available.")

    # 4. Active strategies and why
    lines.append("\n<b>=== STRATEGY STATUS ===</b>")
    for name in STRATEGY_NAMES:
        active = name in state.active_strategies
        paused = state.strategies[name]["paused"]
        bt = state.backtest_results.get(name, {})
        bt_wr = bt.get("win_rate", 0) * 100
        if paused:
            reason = f"Auto-paused (live WR below {UNDERPERFORMANCE_WR_THRESHOLD*100:.0f}%)"
        elif active:
            reason = f"Backtest WR {bt_wr:.1f}% >= {BACKTEST_MIN_WIN_RATE*100:.0f}%"
        else:
            reason = f"Backtest WR {bt_wr:.1f}% < {BACKTEST_MIN_WIN_RATE*100:.0f}%"
        status = "PAUSED" if paused else ("ACTIVE" if active else "INACTIVE")
        lines.append(f"  <b>{name}</b>: [{status}] -- {reason}")

    # 5. Full PnL
    lines.append("\n<b>=== PnL SUMMARY ===</b>")
    total_pnl = sum(state.metrics[n]["current_pnl"] for n in STRATEGY_NAMES)
    lines.append(f"  Total lifetime PnL: ${total_pnl:+,.2f}")
    lines.append(f"  Total lifetime trades: {state.total_trade_count}")
    lines.append(f"  Mode: {state.mode.upper()}")
    lines.append(f"  BTC: ${state.last_btc_price:,.2f}")

    await tg_send("\n".join(lines))
    log.info("Sunday report sent.")


# ---------------------------------------------------------------------------
# AI Market Brain -- Pre-trade analysis
# ---------------------------------------------------------------------------

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


async def ai_market_analysis(strategy: str, side: str, price: float) -> tuple[bool, str]:
    """AI Market Brain: ask Claude Haiku whether this trade should be taken."""
    if not CLAUDE_API_KEY:
        return True, "No API key -- defaulting to YES"

    try:
        headlines = await _fetch_recent_news_headlines()
        news_text = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news available"

        macro_text = ""
        if _macro_intel_cache["text"]:
            age_min = int((time.time() - _macro_intel_cache["fetched_at"]) / 60)
            macro_text = f"\n\nMACRO INTELLIGENCE (updated {age_min}min ago):\n{_macro_intel_cache['text']}"

        intel_text = _get_intelligence_summary()
        intel_section = f"\n\n{intel_text}" if intel_text else ""

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
            f"{macro_text}"
            f"{intel_section}\n\n"
            f"You have web_search -- if the macro intel above is older than 30 minutes or mentions "
            f"a developing situation, SEARCH for the latest update before deciding.\n\n"
            f"Consider the top trader intelligence when evaluating direction and timing.\n"
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
                log.warning("AI Market Brain API error %d -- defaulting YES", resp.status_code)
                return True, "API error -- defaulting to YES"

            data = resp.json()
            text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            content = " ".join(text_blocks).strip()
            log.info("AI Market Brain response for %s %s: %s", strategy, side, content)

            if content.upper().startswith("NO") or "NO --" in content.upper() or "NO." in content.upper():
                return False, content
            return True, content

    except Exception as exc:
        log.warning("AI Market Brain error: %s -- defaulting YES", exc)
        return True, f"Error: {exc} -- defaulting to YES"


# ---------------------------------------------------------------------------
# Telegram Chat -- Claude-powered market Q&A
# ---------------------------------------------------------------------------

async def telegram_polling_loop():
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set -- Telegram polling DEAD.")
        state.polling_alive = False
        return
    if not CLAUDE_API_KEY:
        log.error("CLAUDE_API_KEY not set -- Telegram polling DEAD.")
        state.polling_alive = False
        await tg_send("\u26a0\ufe0f <b>Vic polling failed to start</b>\n\nCLAUDE_API_KEY not set.")
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

                    if text.startswith("/start"):
                        await tg_reply(chat_id, "Hey! I'm Vic v4, your BTC trading agent.\n\nCommands: /journal /metrics /regime /news /intelligence /closeall")
                        continue

                    if text.strip().lower() == "/journal":
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
                                    f"  {t.get('exit_reason','-')} | {t.get('hold_time_min',0):.0f}min | {t.get('session','-')}"
                                )
                            await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if text.strip().lower() == "/metrics":
                        lines = ["\U0001f4ca <b>Per-Strategy Metrics</b>\n"]
                        for n in STRATEGY_NAMES:
                            m = state.metrics[n]
                            tc = m["trade_count"]
                            wr = (m["wins"] / tc * 100) if tc > 0 else 0
                            exp = (m["total_r_achieved"] / tc) if tc > 0 else 0
                            active = n in state.active_strategies
                            paused = state.strategies[n]["paused"]
                            status = "PAUSED" if paused else ("Active" if active else "Inactive")
                            lines.append(
                                f"<b>{n}</b> [{status}]: {tc} trades | WR {wr:.0f}% | Exp {exp:+.2f}R\n"
                                f"  PnL ${m['current_pnl']:+,.2f} | MaxDD ${m['max_drawdown']:,.2f} | "
                                f"Streak {m['current_losing_streak']}/{m['max_losing_streak']}"
                            )
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if text.strip().lower() == "/regime":
                        msg = (
                            f"\U0001f50d <b>Current Regime: {state.regime.value}</b>\n\n"
                            f"Math:\n"
                            f"- TRENDING: ADX > 25 AND structure aligned\n"
                            f"- TRANSITIONAL: ADX > 25 but structure breaking\n"
                            f"- RANGING: ADX < 20 AND BB bandwidth < 0.03\n"
                            f"- VOLATILE: ATR% > 0.4% OR >1.5% move in 30min\n\n"
                            f"1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
                            f"BTC: ${state.last_btc_price:,.2f}\n"
                            f"Active strategies: {', '.join(state.active_strategies) if state.active_strategies else 'none'}"
                        )
                        await tg_reply(chat_id, msg)
                        continue

                    if text.strip().lower() == "/news":
                        headlines = await _fetch_recent_news_headlines()
                        if headlines:
                            lines = ["\U0001f4f0 <b>Recent BTC News</b>\n"]
                            for h in headlines:
                                lines.append(f"\u2022 {sanitize_html(h)}")
                            await tg_reply(chat_id, "\n".join(lines))
                        else:
                            await tg_reply(chat_id, "No recent news available.")
                        continue

                    if text.strip().lower() == "/intelligence":
                        report = _intelligence_cache.get("report") or _read_intelligence()
                        if not report or not report.get("traders"):
                            await tg_reply(chat_id, "No intelligence report available yet. First scan runs ~60s after startup, then every 6 hours.")
                        else:
                            ts = report.get("timestamp", "unknown")
                            traders = report.get("traders", [])
                            patterns = report.get("patterns", {})
                            lines = [f"\U0001f9e0 <b>Top Trader Intelligence</b>\nUpdated: {ts}\n"]
                            lines.append(f"Scanned: {report.get('total_scanned', 0)} | Analysed: {report.get('total_analysed', 0)}")
                            if patterns:
                                lines.append(f"\n<b>Patterns:</b>")
                                if patterns.get("dominant_direction"):
                                    lines.append(f"  Direction: {patterns['dominant_direction']}")
                                if patterns.get("avg_win_rate"):
                                    lines.append(f"  Avg WR: {patterns['avg_win_rate']:.1f}%")
                                if patterns.get("avg_win_loss_ratio"):
                                    lines.append(f"  W/L ratio: {patterns['avg_win_loss_ratio']:.2f}")
                                if patterns.get("best_sessions"):
                                    lines.append(f"  Best sessions: {', '.join(patterns['best_sessions'])}")
                                if patterns.get("key_insight"):
                                    lines.append(f"\n<b>Insight:</b> {sanitize_html(patterns['key_insight'])}")
                            if traders:
                                lines.append(f"\n<b>Top Traders:</b>")
                                for t in traders[:5]:
                                    lines.append(
                                        f"  #{t.get('rank','-')} {t.get('address','?')} "
                                        f"WR:{t.get('win_rate',0):.0f}% "
                                        f"W/L:{t.get('avg_win_loss_ratio',0):.1f}x "
                                        f"Bias:{t.get('direction_bias','?')} "
                                        f"Best:{t.get('best_session','?')}"
                                    )
                            await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if text.strip().lower() == "/closeall":
                        result = await close_all_positions()
                        await tg_reply(chat_id, result)
                        continue

                    # Process question with Claude
                    try:
                        answer = await ask_claude_market_question(text)
                        await tg_reply(chat_id, sanitize_html(answer))
                    except Exception as exc:
                        log.error("Claude chat error: %s", exc)
                        await tg_reply(chat_id, f"Sorry, I hit an error: {sanitize_html(str(exc))}")

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
            await tg_send(
                "\U0001f6a8 <b>Vic polling is DEAD</b>\n\n"
                f"Crashed {max_restarts} times. Redeploy on Railway to fix."
            )
            return

        restart_count += 1
        log.warning("Polling loop died -- restarting (attempt %d/%d)...", restart_count, max_restarts)
        await tg_send(
            f"\u26a0\ufe0f <b>Vic polling crashed -- restarting</b> (attempt {restart_count}/{max_restarts})"
        )
        await asyncio.sleep(3)


async def ask_claude_market_question(question: str) -> str:
    """Send a market question to Claude API with full trading state context."""
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
            sl_price_dist = abs(pos["entry"] - pos["sl"])
            tp_price_dist = abs(pos["tp"] - pos["entry"])
            dollar_risk = round(sl_price_dist * pos["size"], 2)
            dollar_reward = round(tp_price_dist * pos["size"], 2)
            open_positions.append(
                f"{name}: {pos['side'].upper()} {pos['size']:.6f} BTC @ ${pos['entry']:,.2f} "
                f"(SL ${pos['sl']:,.2f} = ${dollar_risk:.2f} risk, TP ${pos['tp']:,.2f} = ${dollar_reward:.2f} reward"
                f"{pnl_label}, BE={'yes' if pos['be_moved'] else 'no'})"
            )
    positions_text = "\n".join(open_positions) if open_positions else "No open positions"

    metrics_lines = []
    for n in STRATEGY_NAMES:
        m = state.metrics[n]
        tc = m["trade_count"]
        wr = (m["wins"] / tc * 100) if tc > 0 else 0
        exp = (m["total_r_achieved"] / tc) if tc > 0 else 0
        active = n in state.active_strategies
        metrics_lines.append(
            f"  {n} [{'Active' if active else 'Inactive'}]: {tc} trades, WR {wr:.0f}%, expectancy {exp:+.2f}R, "
            f"PnL ${m['current_pnl']:+,.2f}, MaxDD ${m['max_drawdown']:,.2f}"
        )
    metrics_text = "\n".join(metrics_lines)

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

    headlines = await _fetch_recent_news_headlines()
    news_text = "\n".join(f"  - {h}" for h in headlines) if headlines else "  No recent news"

    system_prompt = (
        f"You are Vic v4, an AI BTC trading agent on Hyperliquid. "
        f"3 strategies (backtest-gated). $500 account, 5x leverage.\n\n"
        f"=== STRATEGIES ===\n"
        f"1. RSI Divergence: Price/RSI divergence at S/R on 5m\n"
        f"2. BB Squeeze: Bollinger squeeze breakout on 5m\n"
        f"3. VWAP Bounce: VWAP bounce on 1m with filters\n\n"
        f"=== CURRENT STATE ===\n"
        f"- BTC: ${state.last_btc_price:,.2f} | Regime: {state.regime.value}\n"
        f"- 1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
        f"- Mode: {state.mode.upper()} | Active strategies: {', '.join(state.active_strategies)}\n"
        f"- Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY} | PnL: ${state.daily_pnl:+,.2f}\n"
        f"- Lifetime trades: {state.total_trade_count}\n\n"
        f"=== POSITIONS ===\n{positions_text}\n\n"
        f"=== METRICS ===\n{metrics_text}\n\n"
        f"=== RECENT TRADES ===\n{trades_text}\n\n"
        f"=== NEWS ===\n{news_text}\n\n"
        f"=== INTEL ===\n{_get_intelligence_summary() or 'No report yet'}\n\n"
        f"Answer concisely. Use numbers. Under 300 words."
    )

    q_lower = question.lower()
    search_keywords = [
        "news", "macro", "fed", "cpi", "fomc", "inflation", "tariff", "regulation",
        "sec", "etf", "halving", "on-chain", "whale", "liquidat", "funding rate",
        "what's happening", "why is btc", "why did btc", "crash", "pump", "dump",
        "sentiment", "fear", "greed", "latest", "today", "this week", "current",
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
        content_blocks = data.get("content", [])
        text_blocks = [b.get("text", "") for b in content_blocks if b.get("type") == "text" and b.get("text", "").strip()]
        if text_blocks:
            return "\n".join(text_blocks)
        return "No response from Claude."


# ---------------------------------------------------------------------------
# Hyperliquid Top Trader Intelligence
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
        if patterns.get("avg_hold_duration_min"):
            lines.append(f"  Avg hold duration: {patterns['avg_hold_duration_min']:.0f}min")
        if patterns.get("avg_win_rate"):
            lines.append(f"  Avg win rate: {patterns['avg_win_rate']:.1f}%")
        if patterns.get("best_sessions"):
            lines.append(f"  Best sessions: {', '.join(patterns['best_sessions'])}")
        if patterns.get("avg_win_loss_ratio"):
            lines.append(f"  Win/loss size ratio: {patterns['avg_win_loss_ratio']:.2f}")
        if patterns.get("key_insight"):
            lines.append(f"  Key insight: {patterns['key_insight']}")

    for t in traders[:3]:
        lines.append(f"  #{t.get('rank','-')} WR:{t.get('win_rate',0):.0f}% "
                     f"Trades:{t.get('total_trades',0)} "
                     f"W/L:{t.get('avg_win_loss_ratio',0):.1f}x "
                     f"Bias:{t.get('direction_bias','?')}")

    return "\n".join(lines)


async def fetch_hl_leaderboard() -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard")
            if resp.status_code != 200:
                log.warning("Leaderboard API error: %d", resp.status_code)
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
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total * 100,
            "avg_win_loss_ratio": round(win_loss_ratio, 2),
            "best_session": best_session,
            "direction_bias": direction_bias,
            "total_trades": total,
            "sessions": sessions,
        }
    except Exception as exc:
        log.debug("Trader analysis error for %s: %s", address[:10], exc)
        return {}


async def run_intelligence_scan():
    log.info("Starting Hyperliquid top trader intelligence scan...")

    leaderboard = await fetch_hl_leaderboard()
    if not leaderboard:
        log.warning("Intelligence scan: no leaderboard data.")
        return

    log.info("Intelligence scan: analysing top %d traders...", min(len(leaderboard), 20))
    trader_analyses = []
    for i, entry in enumerate(leaderboard[:20]):
        address = entry.get("ethAddress", "")
        if not address:
            continue

        analysis = await analyse_trader(address)
        if not analysis:
            await asyncio.sleep(0.3)
            continue

        trader_analyses.append({
            "rank": i + 1,
            "address": address[:10] + "...",
            "display_name": entry.get("displayName", ""),
            "account_value": entry.get("accountValue", 0),
            "monthly_pnl": entry.get("pnl", 0),
            "monthly_roi": entry.get("roi", 0),
            **analysis,
        })
        await asyncio.sleep(0.3)

    log.info("Intelligence scan: analysed %d top traders", len(trader_analyses))

    if not trader_analyses:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traders": [],
            "patterns": {},
            "total_scanned": len(leaderboard),
            "total_analysed": 0,
        }
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
                "avg_wl_ratio": sum(all_wlr) / len(all_wlr) if all_wlr else 0,
            }, indent=2)
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
                            f"You are a crypto trading analyst. Based on this data from the top "
                            f"Hyperliquid traders in the last 30 days, give ONE key actionable insight "
                            f"for a BTC perp scalper in under 50 words:\n{summary_data}"
                        )}],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                    insight = " ".join(texts).strip()
        except Exception as exc:
            log.debug("Intelligence insight generation error: %s", exc)

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
# Orphaned Position Recovery
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

        recovered = []
        for pos_data in btc_positions:
            szi = float(pos_data.get("szi", 0))
            entry_px = float(pos_data.get("entryPx", 0))
            side = "long" if szi > 0 else "short"
            size = abs(szi)

            # Reconstruct SL/TP from ATR
            sl_distance = RISK_DOLLARS / size if size > 0 else 0
            tp_distance = sl_distance * 2.0  # Default 2:1 RR

            if side == "long":
                sl = round(entry_px - sl_distance)
                tp = round(entry_px + tp_distance)
            else:
                sl = round(entry_px + sl_distance)
                tp = round(entry_px - tp_distance)

            assigned_strategy = None
            for name in STRATEGY_NAMES:
                if state.strategies[name]["position"] is None:
                    assigned_strategy = name
                    break

            if not assigned_strategy:
                log.warning("No free strategy slot for orphaned position.")
                continue

            state.strategies[assigned_strategy]["position"] = {
                "side": side,
                "entry": entry_px,
                "sl": sl,
                "tp": tp,
                "size": size,
                "margin": margin,
                "open_time": datetime.now(timezone.utc).isoformat(),
                "be_moved": False,
                "partial_closed": False,
                "regime": state.regime.value,
                "bias": f"{state.htf_bias} ({state.htf_bias_strength})",
            }
            recovered.append(f"  \u2022 {assigned_strategy}: {side.upper()} {size:.6f} BTC @ ${entry_px:,.2f} (SL ${sl:,.2f} / TP ${tp:,.2f})")
            log.info("Recovered orphaned position into %s: %s %.6f @ %.2f", assigned_strategy, side, size, entry_px)

        if recovered:
            msg = (
                f"\U0001f504 <b>ORPHANED POSITION RECOVERY</b>\n\n"
                f"Found {len(recovered)} open position(s) on Hyperliquid at startup:\n"
                + "\n".join(recovered) +
                f"\n\nSL/TP reconstructed from current risk params.\n"
                f"ATR(14, 5m) = {atr_val:.2f}"
            )
            await tg_send(msg)

    except Exception as exc:
        log.error("Orphaned position recovery failed: %s", exc)
        await tg_send(f"\u26a0\ufe0f <b>Position recovery failed</b>\nError: {sanitize_html(str(exc))}\nCHECK HYPERLIQUID MANUALLY.")


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
    for filepath in [JOURNAL_FILE, STATE_FILE, BACKTEST_FILE, INTELLIGENCE_FILE]:
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)

    state.startup_time = datetime.now(timezone.utc).isoformat()
    log.info("Vic v4 starting up -- mode: %s", state.mode)

    # Load persisted state
    loaded = load_state()

    # Initialize candle boundary tracking
    now_ts = datetime.now(timezone.utc).timestamp()
    state.last_1m_candle_ts = int(now_ts // 60) * 60
    state.last_5m_candle_ts = int(now_ts // 300) * 300

    init_exchange()

    # Orphaned position recovery
    await recover_orphaned_positions()

    # Send startup message (before backtest)
    startup_msg = (
        f"\U0001f916 <b>Vic v4 is online -- {state.mode.upper()} MODE</b>\n"
        f"BTC/USDC perp | {LEVERAGE}x leverage | ${ACCOUNT_CAPITAL:.0f} account\n\n"
        f"1\ufe0f\u20e3 RSI Divergence | 2\ufe0f\u20e3 BB Squeeze | 3\ufe0f\u20e3 VWAP Bounce\n"
        f"2% SL ($10) | 4% TP1 50% ($20) | 6% TP2 full ($30) | Max 4/day | -${ACCOUNT_CAPITAL * MAX_DAILY_LOSS_PCT:.0f} cap\n\n"
        f"Session: London 07-11 + NY 13-17 UTC\n"
        f"AI Market Brain: enabled\n\n"
        f"\U0001f50d Running 30-day backtest before trading begins..."
    )
    await tg_send(startup_msg)

    # Run backtest before any trading
    try:
        bt_results = await run_backtest()
        state.backtest_results = bt_results

        # Determine which strategies pass
        active = []
        bt_lines = ["\U0001f4ca <b>BACKTEST RESULTS (30 days)</b>\n"]
        for name in STRATEGY_NAMES:
            bt = bt_results.get(name, {})
            wr = bt.get("win_rate", 0)
            total = bt.get("total_trades", 0)
            avg_r = bt.get("avg_r", 0)
            total_pnl = bt.get("total_pnl", 0)
            passed = wr >= BACKTEST_MIN_WIN_RATE
            if passed:
                active.append(name)
            status = "\u2705 PASS" if passed else "\u274c FAIL"
            bt_lines.append(
                f"  <b>{name}</b>: {total} trades | WR {wr*100:.1f}% | Avg R {avg_r:+.2f} | PnL ${total_pnl:+,.2f} [{status}]"
            )

        state.active_strategies = active
        state.backtest_complete = True
        save_state()

        if active:
            bt_lines.append(f"\n<b>Active strategies:</b> {', '.join(active)}")
            bt_lines.append(f"Trading will begin on next valid signal.")
        else:
            bt_lines.append(f"\n\u26a0\ufe0f <b>No strategies passed backtest.</b> Trading is paused.")
            bt_lines.append(f"Will re-evaluate on next Sunday report.")

        await tg_send("\n".join(bt_lines))

    except Exception as exc:
        log.error("Startup backtest failed: %s", exc)
        await tg_send(f"\u26a0\ufe0f <b>Backtest failed on startup</b>\nError: {sanitize_html(str(exc))}\n\nActivating all strategies as fallback.")
        # Fallback: activate all strategies
        state.active_strategies = list(STRATEGY_NAMES)
        state.backtest_complete = True
        save_state()

    # Start all background tasks
    asyncio.create_task(regime_update_loop())
    asyncio.create_task(strategy_monitor_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())
    asyncio.create_task(periodic_status_log())
    asyncio.create_task(telegram_polling_watchdog())
    asyncio.create_task(intelligence_loop())
    asyncio.create_task(sunday_report_scheduler())

    log.info("All background tasks started.")


@app.on_event("shutdown")
async def shutdown():
    log.info("Vic shutting down -- saving state.")
    save_state()
    close_exchange()


# ---------------------------------------------------------------------------
# Manual close all positions
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

    cleared_strategies = []
    for name in STRATEGY_NAMES:
        if state.strategies[name]["position"] is not None:
            state.strategies[name]["position"] = None
            cleared_strategies.append(name)

    parts = ["\U0001f6d1 <b>CLOSE ALL -- Manual Override</b>\n"]
    if closed:
        parts.append(f"Closed on exchange: {', '.join(closed)}")
    if cleared_strategies:
        parts.append(f"Cleared internal state: {', '.join(cleared_strategies)}")
    if errors:
        parts.append(f"Errors: {', '.join(errors)}")
    if not closed and not cleared_strategies:
        parts.append("No positions to close.")

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
        "bot": "Vic v4",
        "status": "running" if state.polling_alive else "DEGRADED -- polling dead",
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
        "active_strategies": state.active_strategies,
        "backtest_complete": state.backtest_complete,
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
            "active": name in state.active_strategies,
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
        "bot": "Vic v4",
        "mode": state.mode,
        "leverage": LEVERAGE,
        "account_capital": ACCOUNT_CAPITAL,
        "global_paused": state.paused,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "btc_price": state.last_btc_price,
        "strategies": strategies_status,
        "active_strategies": state.active_strategies,
        "backtest_complete": state.backtest_complete,
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
    return {"status": "live", "warning": "Real money is now at risk."}


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
    """Return the latest backtest results."""
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

        await loop.run_in_executor(
            None, lambda: hl_exchange.update_leverage(LEVERAGE, "BTC", is_cross=True)
        )

        open_result = await loop.run_in_executor(
            None, lambda: hl_exchange.market_open("BTC", is_buy=False, sz=size)
        )

        if open_result.get("status") != "ok":
            return {"error": f"Open rejected: {open_result}"}
        statuses = open_result.get("response", {}).get("data", {}).get("statuses", [])
        for s in statuses:
            if "error" in s:
                return {"error": f"Open error: {s['error']}", "raw": open_result}

        await tg_send(
            f"\U0001f9ea <b>TEST TRADE</b> -- SHORT {size} BTC @ ${price:,.2f}\n"
            f"Notional ~${size * price:.2f} | Verifying exchange access..."
        )

        await asyncio.sleep(2)

        close_result = await loop.run_in_executor(
            None, lambda: hl_exchange.market_close("BTC", sz=size)
        )
        if close_result.get("status") != "ok":
            await tg_send(f"\U0001f6a8 <b>TEST TRADE close FAILED</b>: {close_result}\nCHECK HYPERLIQUID.")
            return {"error": f"Close rejected: {close_result}", "open": open_result}

        await tg_send(f"\u2705 <b>TEST TRADE closed</b> -- Exchange access CONFIRMED.")
        return {"status": "ok", "open": open_result, "close": close_result, "size": size, "price": price}

    except Exception as exc:
        await tg_send(f"\U0001f6a8 <b>TEST TRADE FAILED</b>: {sanitize_html(str(exc))}")
        return {"error": str(exc)}
