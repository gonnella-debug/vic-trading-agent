"""
Vic -- BTC/USDT Perpetual Futures Trading Agent (v7)
Runs 58 strategies (50 original + 5 perp-native + 3 retested on 15m) on Hyperliquid via native SDK.
$370 account, dynamic leverage (10x minimum). Production-ready for Railway.

Research-first approach:
  - Starts PAUSED in research phase
  - Downloads 90 days of data (1m, 5m, 15m, 1h)
  - Backtests all 50 strategies with strict PASS criteria
  - Only activates passing strategies
  - Re-evaluates nightly

Core rules:
  - ONE position at a time across all strategies
  - Every trade has a stop loss (ATR-based, intelligent early exit)
  - Max 4 trades per day
  - 2% account risk per trade
  - Sessions only: London (07:00-11:00 UTC) and NY (13:00-17:00 UTC)
  - No weekends (Saturday/Sunday)
  - AI Market Brain confidence gate (score 1-10, 6+ = informational only)
  - Dynamic leverage: 10x base, scales with confidence

Regime filter: TRENDING / RANGING / TRANSITIONAL / VOLATILE
1H bias: Strong bullish/bearish or neutral
AI Market Brain: Claude pre-trade analysis with confidence scoring
Trade Journal: /data/vic_journal.json (persistent)
State Persistence: /data/vic_state.json (survives restarts)
Backtest Engine: 90-day rolling backtest daily at midnight UTC
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
import re
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
MAX_LEVERAGE_HARD_CAP = 10        # GG standard: never exceed 10x regardless of AI confidence
ACCOUNT_CAPITAL_FALLBACK = 500.0  # Only used if Hyperliquid balance fetch fails
ACCOUNT_CAPITAL = ACCOUNT_CAPITAL_FALLBACK  # Legacy alias (display/backtest only — runtime uses fetch_live_equity)
MAX_RISK_PCT = 0.02               # 2% hard cap per trade — no confidence scaling
TARGET_RISK_PCT = 0.02            # 2% target risk per trade
MIN_SL_PCT = 2.0                  # 2% minimum stop distance (was 0.5%)
MAX_TRADES_PER_DAY = 4            # Max 4 trades per day
MAX_LOSSES_PER_DAY = 3            # 3 losing trades = stop for the day
FEE_PER_TRADE_USD = 3.0           # Hyperliquid fee per entry/exit (round trip ~ $6)
MIN_NET_PROFIT_MULT = 2.0         # TP must clear >= 2× round-trip fees net
EQUITY_DRAWDOWN_KILLSWITCH = 0.20  # Pause bot if equity drops 20% from peak

# Persistence files
JOURNAL_FILE = os.getenv("JOURNAL_FILE", "/data/vic_journal.json")
STATE_FILE = os.getenv("STATE_FILE", "/data/vic_state.json")
BACKTEST_FILE = os.getenv("BACKTEST_FILE", "/data/vic_backtest.json")
FAILED_STRATEGIES_FILE = os.getenv("FAILED_STRATEGIES_FILE", "/data/failed_strategies.json")
INTELLIGENCE_FILE = os.getenv("INTELLIGENCE_FILE", "/data/hl_intelligence.json")
REVIEW_FILE = os.getenv("REVIEW_FILE", "/data/vic_review.json")

# Underperformance auto-pause thresholds
UNDERPERFORMANCE_MIN_TRADES = 20
UNDERPERFORMANCE_WR_THRESHOLD = 0.40

# Backtest minimum win rate
BACKTEST_MIN_WIN_RATE = 0.40

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
app = FastAPI(title="Vic Trading Agent", version="7.0.0")

hl_info: Optional[Info] = None
hl_exchange: Optional[HLExchange] = None


class TradingState:
    """Global in-memory state for the bot."""

    def __init__(self):
        self.mode: str = TRADING_MODE
        self.paused: bool = True  # Start paused for research phase
        self.research_complete: bool = False
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

        # Live equity + drawdown killswitch
        self.live_equity: float = 0.0           # Last fetched Hyperliquid account equity
        self.peak_equity: float = 0.0           # Highest equity ever seen (for drawdown calc)
        self.drawdown_killswitch_hit: bool = False

        # Metrics tracking per strategy (dynamic -- populated after backtest)
        self.metrics: dict = {}

        self.trade_history: list = []
        self.startup_time: str = ""
        self.total_trade_count: int = 0

        # Cache for OHLCV data
        self._ohlcv_cache: dict = {}

        # Signal tracking
        self.signals_checked: int = 0  # Candidates that reached can_open_trade gate
        self.signals_blocked: int = 0
        self.strategy_evals: int = 0   # Total sig_* function invocations (all returned signals + Nones)
        self.last_block_reasons: list = []

        # Polling watchdog
        self.polling_alive: bool = False

        # Backtest
        self.backtest_complete: bool = False
        self.active_strategies: list = []
        self.backtest_results: dict = {}

        # Funding rate history (30-day rolling)
        self.funding_history: list = []
        self.current_funding_rate: float = 0.0

        # Pending parameter changes from self-review (awaiting GG approval)
        self.pending_review: Optional[dict] = None

        # Strategy paused states (dynamic)
        self.strategy_paused: dict = {}

    def _ensure_strategy_metrics(self, name: str):
        """Ensure metrics exist for a strategy name."""
        if name not in self.metrics:
            self.metrics[name] = {
                "wins": 0, "losses": 0, "total_r_achieved": 0.0,
                "trade_count": 0, "max_drawdown": 0.0, "peak_pnl": 0.0,
                "current_pnl": 0.0, "current_losing_streak": 0, "max_losing_streak": 0,
            }

    def reset_daily(self):
        self.trades_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.daily_loss_cap_hit = False
        self.signals_checked = 0
        self.signals_blocked = 0
        self.strategy_evals = 0
        self.last_block_reasons = []
        for name in list(self.metrics.keys()):
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
            "research_complete": state.research_complete,
            "daily_loss_cap_hit": state.daily_loss_cap_hit,
            "live_equity": state.live_equity,
            "peak_equity": state.peak_equity,
            "drawdown_killswitch_hit": state.drawdown_killswitch_hit,
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
        state.active_strategies = data.get("active_strategies", [])
        state.backtest_complete = data.get("backtest_complete", False)
        state.backtest_results = data.get("backtest_results", {})
        state.funding_history = data.get("funding_history", [])
        state.pending_review = data.get("pending_review")
        state.research_complete = data.get("research_complete", False)
        state.live_equity = float(data.get("live_equity", 0.0) or 0.0)
        state.peak_equity = float(data.get("peak_equity", 0.0) or 0.0)
        state.drawdown_killswitch_hit = bool(data.get("drawdown_killswitch_hit", False))

        saved_metrics = data.get("metrics", {})
        for name, m in saved_metrics.items():
            state.metrics[name] = m

        saved_paused = data.get("strategy_paused", {})
        for name, p in saved_paused.items():
            state.strategy_paused[name] = p

        # Restore daily counters if saved state is from today
        saved_at = data.get("saved_at", "")
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if saved_at and saved_at[:10] == today_str:
            state.losses_today = data.get("losses_today", 0)
            state.trades_today = data.get("trades_today", 0)
            state.daily_pnl = data.get("daily_pnl", 0.0)
            state.daily_loss_cap_hit = data.get("daily_loss_cap_hit", False)
            log.info("Daily counters restored (same day): losses=%d, trades=%d, pnl=$%.2f, cap_hit=%s",
                     state.losses_today, state.trades_today, state.daily_pnl, state.daily_loss_cap_hit)
        else:
            log.info("State from different day (%s) -- daily counters start fresh", saved_at[:10] if saved_at else "unknown")

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


async def fetch_funding_rate_history(days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates from Hyperliquid for backtesting."""
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 3600 * 1000)
        all_rows = []
        cursor = start_ms

        while cursor < now_ms:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.hyperliquid.xyz/info",
                    json={"type": "fundingHistory", "coin": "BTC", "startTime": cursor},
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                if not data:
                    break
                for entry in data:
                    ts = entry.get("time", 0)
                    rate = float(entry.get("fundingRate", 0))
                    all_rows.append({"timestamp": pd.to_datetime(ts, unit="ms"), "funding_rate": rate})
                # Move cursor past last entry
                last_ts = data[-1].get("time", 0)
                if last_ts <= cursor:
                    break
                cursor = last_ts + 1
            await asyncio.sleep(0.2)

        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        log.info("Downloaded %d funding rate entries (%d days)", len(df), days)
        return df
    except Exception as exc:
        log.error("Funding history download error: %s", exc)
        return pd.DataFrame()


def get_funding_percentile(rate: float) -> float:
    """Return the percentile of the current funding rate in the 30-day distribution."""
    if len(state.funding_history) < 24:  # Need at least 1 day of data
        return 50.0
    rates = [h["rate"] for h in state.funding_history]
    below = sum(1 for r in rates if r < rate)
    return (below / len(rates)) * 100.0


# ---------------------------------------------------------------------------
# Indicators -- Base
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
# Extended Indicators
# ---------------------------------------------------------------------------

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Calculate Supertrend indicator."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = calc_atr(df, period)
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1 if df["close"].iloc[i] > upper_band.iloc[i] else 1
            continue
        prev_st = supertrend.iloc[i - 1]
        if np.isnan(prev_st):
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
            continue
        if direction.iloc[i - 1] == -1:  # Was bullish
            new_lower = max(lower_band.iloc[i], prev_st) if not np.isnan(lower_band.iloc[i]) else prev_st
            if df["close"].iloc[i] < new_lower:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = new_lower
                direction.iloc[i] = -1
        else:  # Was bearish
            new_upper = min(upper_band.iloc[i], prev_st) if not np.isnan(upper_band.iloc[i]) else prev_st
            if df["close"].iloc[i] > new_upper:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = new_upper
                direction.iloc[i] = 1
    return supertrend


def calc_parabolic_sar(df: pd.DataFrame) -> pd.Series:
    """Calculate Parabolic SAR using ta library."""
    indicator = ta.trend.PSARIndicator(df["high"], df["low"], df["close"])
    return indicator.psar()


def calc_ichimoku(df: pd.DataFrame) -> dict:
    """Returns dict with tenkan, kijun, senkou_a, senkou_b, chikou."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou}


def calc_hull_ma(series: pd.Series, period: int = 9) -> pd.Series:
    """Calculate Hull Moving Average."""
    half_period = max(period // 2, 1)
    sqrt_period = max(int(math.sqrt(period)), 1)
    wma_half = series.rolling(half_period).mean()
    wma_full = series.rolling(period).mean()
    hull_input = 2 * wma_half - wma_full
    return hull_input.rolling(sqrt_period).mean()


def calc_stoch_rsi(series: pd.Series, period: int = 14):
    """Returns (k, d) Series."""
    rsi = calc_rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10) * 100
    k = stoch_rsi.rolling(3).mean()
    d = k.rolling(3).mean()
    return k, d


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def calc_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    return -100 * (highest - df["close"]) / (highest - lowest + 1e-10)


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume."""
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


# ---------------------------------------------------------------------------
# Precompute all indicators for backtesting efficiency
# ---------------------------------------------------------------------------

def precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ALL indicator columns to a DataFrame for backtesting."""
    if len(df) < 55:
        return df
    df = df.copy()
    c = df["close"]
    # EMAs
    for p in [9, 14, 20, 21, 50, 55, 200]:
        df[f"ema_{p}"] = calc_ema(c, p)
    # RSI
    df["rsi"] = calc_rsi(c, 14)
    # Bollinger Bands
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_bw"] = calc_bollinger_bands(c, 20, 2.0)
    # VWAP
    df["vwap"] = calc_vwap(df)
    # ADX
    df["adx"] = calc_adx(df, 14)
    # ATR
    df["atr"] = calc_atr(df, 14)
    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(c)
    # Supertrend
    df["supertrend"] = calc_supertrend(df, 10, 3.0)
    # Parabolic SAR
    try:
        df["psar"] = calc_parabolic_sar(df)
    except Exception:
        df["psar"] = np.nan
    # Hull MA
    df["hull_9"] = calc_hull_ma(c, 9)
    # Stoch RSI
    df["stoch_k"], df["stoch_d"] = calc_stoch_rsi(c, 14)
    # CCI
    df["cci"] = calc_cci(df, 20)
    # Williams %R
    df["williams_r"] = calc_williams_r(df, 14)
    # OBV
    df["obv"] = calc_obv(df)
    # Volume MA
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


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
# Signal Helper
# ---------------------------------------------------------------------------

def _make_signal(side: str, price: float, atr: float, sl_mult: float = 1.5, tp_mult: float = 3.0) -> Optional[dict]:
    """Build a signal dict with correct SL/TP directions.
    LONG:  SL = price - sl_mult*atr,  TP = price + tp_mult*sl_mult*atr  (TP ABOVE entry)
    SHORT: SL = price + sl_mult*atr,  TP = price - tp_mult*sl_mult*atr  (TP BELOW entry)
    """
    if atr <= 0 or price <= 0:
        return None
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * sl_dist
    if side == "long":
        sl = round(price - sl_dist)
        tp = round(price + tp_dist)
        if tp <= price:
            return None
    else:
        sl = round(price + sl_dist)
        tp = round(price - tp_dist)
        if tp >= price:
            return None
    return {"side": side, "entry": price, "sl": sl, "tp": tp}


def _make_signal_pct(side: str, price: float, sl_pct: float, tp_pct: float) -> Optional[dict]:
    """Build a signal with percentage-based SL/TP distances.
    sl_pct/tp_pct are percentages (e.g., 0.5 = 0.5% from entry).
    Enforces minimum SL of 0.5%.
    """
    if price <= 0:
        return None
    sl_pct = max(sl_pct, MIN_SL_PCT)  # Enforce 2% minimum SL (GG standard)
    sl_dist = price * sl_pct / 100
    tp_dist = price * tp_pct / 100
    if side == "long":
        sl = round(price - sl_dist)
        tp = round(price + tp_dist)
        if tp <= price:
            return None
    else:
        sl = round(price + sl_dist)
        tp = round(price - tp_dist)
        if tp >= price:
            return None
    return {"side": side, "entry": price, "sl": sl, "tp": tp}


def _safe_val(series, idx):
    """Safely get a float value from a pandas Series at index idx."""
    if idx < 0 or idx >= len(series):
        return np.nan
    v = series.iloc[idx]
    return float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan


def _nan(v):
    """Check if value is nan."""
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# 50 Strategy Signal Functions
# Each takes (df, i, extras=None) where df has precomputed indicators.
# Returns dict with {side, entry, sl, tp} or None.
# ---------------------------------------------------------------------------

# --- MOMENTUM ---

def sig_ema_cross_9_21(df, i, extras=None):
    """EMA 9/21 crossover on 5m."""
    if i < 2:
        return None
    e9 = _safe_val(df["ema_9"], i)
    e21 = _safe_val(df["ema_21"], i)
    e9p = _safe_val(df["ema_9"], i - 1)
    e21p = _safe_val(df["ema_21"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(e9) or _nan(e21) or _nan(e9p) or _nan(e21p) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if e9p <= e21p and e9 > e21:
        return _make_signal("long", price, atr)
    if e9p >= e21p and e9 < e21:
        return _make_signal("short", price, atr)
    return None


def sig_ema_cross_21_55(df, i, extras=None):
    """EMA 21/55 crossover on 15m."""
    if i < 2:
        return None
    e21 = _safe_val(df["ema_21"], i)
    e55 = _safe_val(df["ema_55"], i)
    e21p = _safe_val(df["ema_21"], i - 1)
    e55p = _safe_val(df["ema_55"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(e21) or _nan(e55) or _nan(e21p) or _nan(e55p) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if e21p <= e55p and e21 > e55:
        return _make_signal("long", price, atr, 1.5, 2.5)
    if e21p >= e55p and e21 < e55:
        return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_macd_cross_5m(df, i, extras=None):
    """MACD crossover on 5m."""
    if i < 2:
        return None
    m = _safe_val(df["macd"], i)
    s = _safe_val(df["macd_signal"], i)
    mp = _safe_val(df["macd"], i - 1)
    sp = _safe_val(df["macd_signal"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(m) or _nan(s) or _nan(mp) or _nan(sp) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if mp <= sp and m > s:
        return _make_signal("long", price, atr)
    if mp >= sp and m < s:
        return _make_signal("short", price, atr)
    return None


def sig_macd_cross_15m(df, i, extras=None):
    """MACD crossover on 15m."""
    return sig_macd_cross_5m(df, i, extras)  # Same logic, different timeframe data


def sig_adx_ema(df, i, extras=None):
    """ADX > 25 with EMA direction on 5m."""
    adx = _safe_val(df["adx"], i)
    e9 = _safe_val(df["ema_9"], i)
    e21 = _safe_val(df["ema_21"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(adx) or _nan(e9) or _nan(e21) or _nan(atr) or atr <= 0:
        return None
    if adx < 25:
        return None
    price = float(df["close"].iloc[i])
    if e9 > e21 and price > e9:
        return _make_signal("long", price, atr, 1.5, 2.5)
    if e9 < e21 and price < e9:
        return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_supertrend_5m(df, i, extras=None):
    """Supertrend crossover on 5m."""
    if i < 2:
        return None
    st = _safe_val(df["supertrend"], i)
    stp = _safe_val(df["supertrend"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(st) or _nan(stp) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    prev_price = float(df["close"].iloc[i - 1])
    if prev_price <= stp and price > st:
        return _make_signal("long", price, atr)
    if prev_price >= stp and price < st:
        return _make_signal("short", price, atr)
    return None


def sig_supertrend_15m(df, i, extras=None):
    """Supertrend crossover on 15m."""
    return sig_supertrend_5m(df, i, extras)


def sig_parabolic_sar(df, i, extras=None):
    """SAR direction change on 5m."""
    if i < 2 or "psar" not in df.columns:
        return None
    sar = _safe_val(df["psar"], i)
    sarp = _safe_val(df["psar"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(sar) or _nan(sarp) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    prev_price = float(df["close"].iloc[i - 1])
    if prev_price < sarp and price > sar:
        return _make_signal("long", price, atr)
    if prev_price > sarp and price < sar:
        return _make_signal("short", price, atr)
    return None


def sig_ichimoku_breakout(df, i, extras=None):
    """Cloud breakout on 1h."""
    if i < 2:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Compute ichimoku inline since we can't store dict in precompute
    high = df["high"]
    low = df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    sa = _safe_val(senkou_a, i)
    sb = _safe_val(senkou_b, i)
    sap = _safe_val(senkou_a, i - 1)
    sbp = _safe_val(senkou_b, i - 1)
    if _nan(sa) or _nan(sb) or _nan(sap) or _nan(sbp):
        return None
    price = float(df["close"].iloc[i])
    prev_price = float(df["close"].iloc[i - 1])
    cloud_top = max(sa, sb)
    cloud_bottom = min(sa, sb)
    prev_cloud_top = max(sap, sbp)
    if prev_price <= prev_cloud_top and price > cloud_top:
        return _make_signal("long", price, atr, 2.0, 3.0)
    prev_cloud_bottom = min(sap, sbp)
    if prev_price >= prev_cloud_bottom and price < cloud_bottom:
        return _make_signal("short", price, atr, 2.0, 3.0)
    return None


def sig_hull_ma_cross(df, i, extras=None):
    """Hull MA crossover on 5m."""
    if i < 2 or "hull_9" not in df.columns:
        return None
    hull = _safe_val(df["hull_9"], i)
    hullp = _safe_val(df["hull_9"], i - 1)
    e21 = _safe_val(df["ema_21"], i)
    e21p = _safe_val(df["ema_21"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(hull) or _nan(hullp) or _nan(e21) or _nan(e21p) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if hullp <= e21p and hull > e21:
        return _make_signal("long", price, atr)
    if hullp >= e21p and hull < e21:
        return _make_signal("short", price, atr)
    return None


# --- MEAN REVERSION ---

def sig_rsi_extreme(df, i, extras=None):
    """RSI < 30 long, > 70 short on 5m."""
    rsi = _safe_val(df["rsi"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(rsi) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if rsi < 30:
        return _make_signal("long", price, atr, 1.0, 2.0)
    if rsi > 70:
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_rsi_divergence(df, i, extras=None):
    """Price/RSI divergence on 5m."""
    if i < 15:
        return None
    rsi = _safe_val(df["rsi"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(rsi) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Bullish divergence: price makes lower low, RSI makes higher low
    price_low = float(df["low"].iloc[i])
    price_prev_low = float(df["low"].iloc[i - 10:i].min())
    rsi_now = rsi
    rsi_prev_low = float(df["rsi"].iloc[i - 10:i].min())
    if price_low <= price_prev_low and rsi_now > rsi_prev_low and rsi < 40:
        return _make_signal("long", price, atr, 1.5, 2.5)
    # Bearish divergence: price makes higher high, RSI makes lower high
    price_high = float(df["high"].iloc[i])
    price_prev_high = float(df["high"].iloc[i - 10:i].max())
    rsi_prev_high = float(df["rsi"].iloc[i - 10:i].max())
    if price_high >= price_prev_high and rsi_now < rsi_prev_high and rsi > 60:
        return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_bb_touch_reversal(df, i, extras=None):
    """BB band touch + reversal candle on 5m."""
    if i < 2:
        return None
    bb_upper = _safe_val(df["bb_upper"], i)
    bb_lower = _safe_val(df["bb_lower"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(bb_upper) or _nan(bb_lower) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    low = float(df["low"].iloc[i])
    high = float(df["high"].iloc[i])
    if low <= bb_lower and price > open_p:  # Touched lower band, bullish reversal
        return _make_signal("long", price, atr, 1.0, 2.0)
    if high >= bb_upper and price < open_p:  # Touched upper band, bearish reversal
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_bb_squeeze_breakout(df, i, extras=None):
    """BB squeeze then breakout on 5m."""
    if i < 5:
        return None
    bw = _safe_val(df["bb_bw"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(bw) or _nan(atr) or atr <= 0:
        return None
    # Check for squeeze: bandwidth was below 0.02 in last 5 candles
    squeeze = False
    for j in range(i - 5, i):
        v = _safe_val(df["bb_bw"], j)
        if not _nan(v) and v < 0.02:
            squeeze = True
            break
    if not squeeze or bw < 0.02:
        return None
    price = float(df["close"].iloc[i])
    bb_upper = _safe_val(df["bb_upper"], i)
    bb_lower = _safe_val(df["bb_lower"], i)
    if _nan(bb_upper) or _nan(bb_lower):
        return None
    if price > bb_upper:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < bb_lower:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_stoch_rsi_cross(df, i, extras=None):
    """Stoch RSI crossover in extreme zones on 5m."""
    if i < 2:
        return None
    k = _safe_val(df["stoch_k"], i)
    d = _safe_val(df["stoch_d"], i)
    kp = _safe_val(df["stoch_k"], i - 1)
    dp = _safe_val(df["stoch_d"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(k) or _nan(d) or _nan(kp) or _nan(dp) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if kp <= dp and k > d and k < 20:
        return _make_signal("long", price, atr, 1.0, 2.5)
    if kp >= dp and k < d and k > 80:
        return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_cci_extreme(df, i, extras=None):
    """CCI < -100 long, > 100 short on 5m."""
    cci = _safe_val(df["cci"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(cci) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if cci < -100:
        return _make_signal("long", price, atr, 1.0, 2.0)
    if cci > 100:
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_williams_r_extreme(df, i, extras=None):
    """Williams %R extremes on 5m."""
    wr = _safe_val(df["williams_r"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(wr) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if wr < -80:
        return _make_signal("long", price, atr, 1.0, 2.0)
    if wr > -20:
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_mean_reversion_ema(df, i, extras=None):
    """Deviation from 20 EMA on 5m."""
    e20 = _safe_val(df["ema_20"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(e20) or _nan(atr) or atr <= 0 or e20 <= 0:
        return None
    price = float(df["close"].iloc[i])
    dev = (price - e20) / e20 * 100
    if dev < -0.3:  # >0.3% below EMA
        return _make_signal("long", price, atr, 1.0, 2.0)
    if dev > 0.3:  # >0.3% above EMA
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


# --- VOLUME ---

def sig_vwap_bounce(df, i, extras=None):
    """VWAP bounce with volume on 1m."""
    if i < 2:
        return None
    vwap = _safe_val(df["vwap"], i)
    atr = _safe_val(df["atr"], i)
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(vwap) or _nan(atr) or _nan(vol_ma) or atr <= 0 or vol_ma <= 0:
        return None
    price = float(df["close"].iloc[i])
    vol = float(df["volume"].iloc[i])
    low = float(df["low"].iloc[i])
    high = float(df["high"].iloc[i])
    open_p = float(df["open"].iloc[i])
    if vol < vol_ma * 1.5:
        return None
    if low <= vwap and price > vwap and price > open_p:
        return _make_signal("long", price, atr, 1.0, 2.0)
    if high >= vwap and price < vwap and price < open_p:
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_vwap_reclaim(df, i, extras=None):
    """VWAP reclaim after breakdown on 5m."""
    if i < 2:
        return None
    vwap = _safe_val(df["vwap"], i)
    vwap_p = _safe_val(df["vwap"], i - 1)
    atr = _safe_val(df["atr"], i)
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(vwap) or _nan(vwap_p) or _nan(atr) or _nan(vol_ma) or atr <= 0 or vol_ma <= 0:
        return None
    price = float(df["close"].iloc[i])
    prev_close = float(df["close"].iloc[i - 1])
    open_p = float(df["open"].iloc[i])
    vol = float(df["volume"].iloc[i])
    body = abs(price - open_p)
    full_range = float(df["high"].iloc[i]) - float(df["low"].iloc[i])
    if full_range == 0 or body / full_range < 0.6 or vol < vol_ma * 1.5:
        return None
    rsi = _safe_val(df["rsi"], i)
    if _nan(rsi):
        rsi = 50.0
    if prev_close < vwap_p and price > vwap and price > open_p and rsi > 50:
        return _make_signal("long", price, atr, 1.0, 2.0)
    if prev_close > vwap_p and price < vwap and price < open_p and rsi < 50:
        return _make_signal("short", price, atr, 1.0, 2.0)
    return None


def sig_volume_spike_reversal(df, i, extras=None):
    """3x volume with opposing candle on 5m."""
    vol_ma = _safe_val(df["vol_ma"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(vol_ma) or _nan(atr) or atr <= 0 or vol_ma <= 0:
        return None
    vol = float(df["volume"].iloc[i])
    if vol < vol_ma * 3.0:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    # After a down move, bullish reversal candle
    if i >= 3:
        prev_trend = float(df["close"].iloc[i - 3]) - float(df["close"].iloc[i - 1])
        if prev_trend < 0 and price > open_p:  # Was going down, now bullish
            return _make_signal("long", price, atr, 1.0, 2.5)
        if prev_trend > 0 and price < open_p:  # Was going up, now bearish
            return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_obv_divergence(df, i, extras=None):
    """OBV divergence from price on 15m."""
    if i < 15:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    obv_now = _safe_val(df["obv"], i)
    obv_prev = _safe_val(df["obv"], i - 10)
    price_prev = float(df["close"].iloc[i - 10])
    if _nan(obv_now) or _nan(obv_prev):
        return None
    # Bullish: price lower, OBV higher
    if price < price_prev and obv_now > obv_prev:
        return _make_signal("long", price, atr, 1.5, 2.5)
    # Bearish: price higher, OBV lower
    if price > price_prev and obv_now < obv_prev:
        return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_funding_rate_fade(df, i, extras=None):
    """Extreme funding rate fade (needs extras={funding_rate, funding_pctl})."""
    if not extras:
        return None
    rate = extras.get("funding_rate", 0)
    pctl = extras.get("funding_pctl", 50)
    rsi = _safe_val(df["rsi"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(rsi) or _nan(atr) or atr <= 0 or rate == 0:
        return None
    price = float(df["close"].iloc[i])
    if pctl >= 90 and rsi > 65:
        return _make_signal("short", price, atr, 1.5, 2.5)
    if pctl <= 10 and rsi < 35:
        return _make_signal("long", price, atr, 1.5, 2.5)
    return None


def sig_liquidation_fade(df, i, extras=None):
    """After large ATR candle + reversal, fade the move."""
    if i < 2:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    prev_range = float(df["high"].iloc[i - 1]) - float(df["low"].iloc[i - 1])
    if prev_range < atr * 3:  # Previous candle was huge
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    prev_close = float(df["close"].iloc[i - 1])
    prev_open = float(df["open"].iloc[i - 1])
    # Previous candle was bearish, current is bullish (reversal)
    if prev_close < prev_open and price > open_p:
        return _make_signal("long", price, atr, 1.5, 2.0)
    if prev_close > prev_open and price < open_p:
        return _make_signal("short", price, atr, 1.5, 2.0)
    return None


def sig_delta_divergence(df, i, extras=None):
    """Volume vs price divergence on 5m."""
    if i < 10:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Approximate delta as: bullish candle vol = positive, bearish = negative
    delta_sum = 0.0
    for j in range(i - 5, i + 1):
        if j < 0 or j >= len(df):
            continue
        c = float(df["close"].iloc[j])
        o = float(df["open"].iloc[j])
        v = float(df["volume"].iloc[j])
        delta_sum += v if c > o else -v
    price = float(df["close"].iloc[i])
    price_5ago = float(df["close"].iloc[i - 5])
    # Price up but delta negative -> bearish divergence
    if price > price_5ago and delta_sum < 0:
        return _make_signal("short", price, atr, 1.5, 2.0)
    if price < price_5ago and delta_sum > 0:
        return _make_signal("long", price, atr, 1.5, 2.0)
    return None


# --- PRICE ACTION ---

def sig_order_block(df, i, extras=None):
    """OB retest on 5m."""
    if i < 20:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Look for bullish impulse (3 consecutive bullish candles) in lookback
    for k in range(max(3, i - 20), i - 3):
        impulse_ok = all(float(df["close"].iloc[k + j]) > float(df["open"].iloc[k + j]) for j in range(3))
        if impulse_ok:
            start_p = float(df["open"].iloc[k])
            end_p = float(df["close"].iloc[k + 2])
            if start_p > 0 and (end_p - start_p) / start_p * 100 >= 0.5:
                ob_idx = k - 1
                if ob_idx >= 0 and float(df["close"].iloc[ob_idx]) < float(df["open"].iloc[ob_idx]):
                    ob_high = float(df["high"].iloc[ob_idx])
                    ob_low = float(df["low"].iloc[ob_idx])
                    if ob_low <= price <= ob_high and float(df["close"].iloc[i]) > float(df["open"].iloc[i]):
                        sl = round(ob_low - atr * 0.2)
                        tp = round(max(end_p, price + abs(price - sl) * 2))
                        if tp > price:
                            return {"side": "long", "entry": price, "sl": sl, "tp": tp}
    # Bearish
    for k in range(max(3, i - 20), i - 3):
        impulse_ok = all(float(df["close"].iloc[k + j]) < float(df["open"].iloc[k + j]) for j in range(3))
        if impulse_ok:
            start_p = float(df["open"].iloc[k])
            end_p = float(df["close"].iloc[k + 2])
            if start_p > 0 and (start_p - end_p) / start_p * 100 >= 0.5:
                ob_idx = k - 1
                if ob_idx >= 0 and float(df["close"].iloc[ob_idx]) > float(df["open"].iloc[ob_idx]):
                    ob_high = float(df["high"].iloc[ob_idx])
                    ob_low = float(df["low"].iloc[ob_idx])
                    if ob_low <= price <= ob_high and float(df["close"].iloc[i]) < float(df["open"].iloc[i]):
                        sl = round(ob_high + atr * 0.2)
                        tp = round(min(end_p, price - abs(sl - price) * 2))
                        if tp < price:
                            return {"side": "short", "entry": price, "sl": sl, "tp": tp}
    return None


def sig_liquidity_sweep(df, i, extras=None):
    """Sweep of swing high/low on 5m."""
    if i < 20:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    high = float(df["high"].iloc[i])
    low = float(df["low"].iloc[i])
    vol_ma = _safe_val(df["vol_ma"], i)
    vol = float(df["volume"].iloc[i])
    if _nan(vol_ma) or vol_ma <= 0 or vol < vol_ma * 2:
        return None
    # Find recent swing lows
    lows = df["low"].values[max(0, i - 30):i]
    for j in range(2, len(lows) - 2):
        if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
            sw_low = lows[j]
            if low < sw_low and price > sw_low and price > open_p:
                return _make_signal("long", price, atr, 1.5, 2.5)
    # Find recent swing highs
    highs = df["high"].values[max(0, i - 30):i]
    for j in range(2, len(highs) - 2):
        if highs[j] > highs[j - 1] and highs[j] > highs[j + 1]:
            sw_high = highs[j]
            if high > sw_high and price < sw_high and price < open_p:
                return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_fvg_fill(df, i, extras=None):
    """Fair value gap fill on 5m."""
    if i < 10:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Look for bullish FVG (gap up)
    for k in range(max(0, i - 10), i - 2):
        gap_low = float(df["low"].iloc[k + 2])
        gap_high = float(df["high"].iloc[k])
        if gap_low > gap_high:  # Bullish FVG
            if float(df["low"].iloc[i]) <= gap_low and price > gap_high:
                return _make_signal("long", price, atr, 1.0, 2.5)
    # Bearish FVG
    for k in range(max(0, i - 10), i - 2):
        gap_high = float(df["high"].iloc[k + 2])
        gap_low = float(df["low"].iloc[k])
        if gap_high < gap_low:  # Bearish FVG
            if float(df["high"].iloc[i]) >= gap_high and price < gap_low:
                return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_bos_pullback(df, i, extras=None):
    """Break of structure + pullback on 15m."""
    if i < 20:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Detect bullish BOS: new high above recent swing high, then pullback
    highs = df["high"].values[max(0, i - 20):i]
    lows = df["low"].values[max(0, i - 20):i]
    if len(highs) < 10:
        return None
    recent_high = float(np.max(highs[:-3]))
    recent_low = float(np.min(lows[:-3]))
    e21 = _safe_val(df["ema_21"], i)
    if _nan(e21):
        return None
    # Bullish BOS + pullback to EMA
    if float(np.max(highs[-5:])) > recent_high:
        if abs(price - e21) / e21 * 100 < 0.1 and price > e21:
            return _make_signal("long", price, atr, 1.5, 3.0)
    # Bearish BOS + pullback
    if float(np.min(lows[-5:])) < recent_low:
        if abs(price - e21) / e21 * 100 < 0.1 and price < e21:
            return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_inside_bar(df, i, extras=None):
    """Inside bar breakout on 5m."""
    if i < 2:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Check if previous candle was inside bar (contained within candle before it)
    prev_high = float(df["high"].iloc[i - 1])
    prev_low = float(df["low"].iloc[i - 1])
    mother_high = float(df["high"].iloc[i - 2])
    mother_low = float(df["low"].iloc[i - 2])
    if prev_high > mother_high or prev_low < mother_low:
        return None  # Not an inside bar
    price = float(df["close"].iloc[i])
    if price > mother_high:
        return _make_signal("long", price, atr, 1.0, 2.5)
    if price < mother_low:
        return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_engulfing(df, i, extras=None):
    """Engulfing candle at key level on 5m."""
    if i < 2:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    prev_close = float(df["close"].iloc[i - 1])
    prev_open = float(df["open"].iloc[i - 1])
    # Bullish engulfing
    if prev_close < prev_open and price > open_p:
        if price > prev_open and open_p < prev_close:
            return _make_signal("long", price, atr, 1.0, 2.5)
    # Bearish engulfing
    if prev_close > prev_open and price < open_p:
        if price < prev_open and open_p > prev_close:
            return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_pin_bar(df, i, extras=None):
    """Pin bar reversal on 5m."""
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    high = float(df["high"].iloc[i])
    low = float(df["low"].iloc[i])
    body = abs(price - open_p)
    full_range = high - low
    if full_range == 0 or body == 0:
        return None
    upper_wick = high - max(price, open_p)
    lower_wick = min(price, open_p) - low
    # Bullish pin bar: long lower wick, small body
    if lower_wick >= body * 2 and upper_wick < body:
        return _make_signal("long", price, atr, 1.0, 2.5)
    # Bearish pin bar: long upper wick
    if upper_wick >= body * 2 and lower_wick < body:
        return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_three_candle_reversal(df, i, extras=None):
    """Three candle reversal pattern on 5m."""
    if i < 3:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Three consecutive bearish then one bullish = bullish reversal
    all_bear = all(float(df["close"].iloc[i - j]) < float(df["open"].iloc[i - j]) for j in range(1, 4))
    all_bull = all(float(df["close"].iloc[i - j]) > float(df["open"].iloc[i - j]) for j in range(1, 4))
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    if all_bear and price > open_p:
        return _make_signal("long", price, atr, 1.0, 2.5)
    if all_bull and price < open_p:
        return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_hh_hl_entry(df, i, extras=None):
    """HH/HL or LH/LL confirmation on 15m."""
    if i < 20:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    structure = detect_structure(df.iloc[max(0, i - 20):i + 1])
    if structure == "bullish" and price > float(df["open"].iloc[i]):
        return _make_signal("long", price, atr, 1.5, 3.0)
    if structure == "bearish" and price < float(df["open"].iloc[i]):
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


# --- SESSION ---

def sig_london_breakout(df, i, extras=None):
    """First 30 min of London (07:00-07:30 UTC)."""
    if i < 10 or "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        h, m = ts.hour, ts.minute
    else:
        return None
    if not (h == 7 and m < 30):
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Get Asia range (00:00-07:00)
    asia_high = float(df["high"].iloc[max(0, i - 84):i].max())  # ~7h of 5m candles
    asia_low = float(df["low"].iloc[max(0, i - 84):i].min())
    if price > asia_high:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < asia_low:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_ny_breakout(df, i, extras=None):
    """First 30 min of NY (13:00-13:30 UTC)."""
    if i < 10 or "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        h, m = ts.hour, ts.minute
    else:
        return None
    if not (h == 13 and m < 30):
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    # Get London range
    london_high = float(df["high"].iloc[max(0, i - 72):i].max())  # ~6h of 5m candles
    london_low = float(df["low"].iloc[max(0, i - 72):i].min())
    if price > london_high:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < london_low:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_asia_range_fade(df, i, extras=None):
    """Fade Asia range exceeded by 20%."""
    if i < 84 or "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        h = ts.hour
    else:
        return None
    if not (7 <= h < 11):
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    asia_high = float(df["high"].iloc[max(0, i - 84):i].max())
    asia_low = float(df["low"].iloc[max(0, i - 84):i].min())
    asia_range = asia_high - asia_low
    if asia_range <= 0:
        return None
    if price > asia_high + asia_range * 0.2:
        return _make_signal("short", price, atr, 1.5, 2.0)  # Fade the overextension
    if price < asia_low - asia_range * 0.2:
        return _make_signal("long", price, atr, 1.5, 2.0)
    return None


def sig_session_end_mr(df, i, extras=None):
    """Mean reversion last 30 min of session."""
    if i < 5 or "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        h, m = ts.hour, ts.minute
    else:
        return None
    # End of London: 10:30-11:00 or end of NY: 16:30-17:00
    is_end = (h == 10 and m >= 30) or (h == 16 and m >= 30)
    if not is_end:
        return None
    rsi = _safe_val(df["rsi"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(rsi) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if rsi < 35:
        return _make_signal("long", price, atr, 1.0, 1.5)
    if rsi > 65:
        return _make_signal("short", price, atr, 1.0, 1.5)
    return None


def sig_monday_range(df, i, extras=None):
    """Monday range breakout."""
    if i < 2 or "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "weekday"):
        if ts.weekday() != 1:  # Only trade on Tuesday
            return None
    else:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Look back to find Monday candles
    monday_high = -np.inf
    monday_low = np.inf
    for j in range(max(0, i - 288), i):  # 288 = 24h of 5m candles
        t = df["timestamp"].iloc[j]
        if hasattr(t, "weekday") and t.weekday() == 0:
            monday_high = max(monday_high, float(df["high"].iloc[j]))
            monday_low = min(monday_low, float(df["low"].iloc[j]))
    if monday_high == -np.inf:
        return None
    price = float(df["close"].iloc[i])
    if price > monday_high:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < monday_low:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


# --- HL SPECIFIC ---

def sig_top_trader_bias(df, i, extras=None):
    """Only trade in top trader direction (needs extras)."""
    if not extras or "top_trader_bias" not in extras:
        return None
    bias = extras["top_trader_bias"]
    atr = _safe_val(df["atr"], i)
    rsi = _safe_val(df["rsi"], i)
    if _nan(atr) or _nan(rsi) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    e21 = _safe_val(df["ema_21"], i)
    if _nan(e21):
        return None
    if bias == "LONG" and price > e21 and rsi > 50:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if bias == "SHORT" and price < e21 and rsi < 50:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_oi_spike(df, i, extras=None):
    """OI spike approximation (using volume as proxy)."""
    if i < 10:
        return None
    atr = _safe_val(df["atr"], i)
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(atr) or _nan(vol_ma) or atr <= 0 or vol_ma <= 0:
        return None
    vol = float(df["volume"].iloc[i])
    if vol < vol_ma * 4:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    if price > open_p:
        return _make_signal("long", price, atr, 1.5, 2.5)
    if price < open_p:
        return _make_signal("short", price, atr, 1.5, 2.5)
    return None


def sig_funding_extreme(df, i, extras=None):
    """More aggressive funding rate fade."""
    if not extras:
        return None
    rate = extras.get("funding_rate", 0)
    pctl = extras.get("funding_pctl", 50)
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0 or rate == 0:
        return None
    price = float(df["close"].iloc[i])
    if pctl >= 95:
        return _make_signal("short", price, atr, 1.0, 2.0)
    if pctl <= 5:
        return _make_signal("long", price, atr, 1.0, 2.0)
    return None


def sig_perp_premium(df, i, extras=None):
    """Perp premium/discount convergence."""
    if not extras or "perp_premium" not in extras:
        return None
    premium = extras["perp_premium"]
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if premium > 0.5:  # Perp trading at premium -> short
        return _make_signal("short", price, atr, 1.0, 2.0)
    if premium < -0.5:  # Perp at discount -> long
        return _make_signal("long", price, atr, 1.0, 2.0)
    return None


# --- PERP-NATIVE STRATEGIES ---

def sig_funding_extreme_fade_15m(df, i, extras=None):
    """STRATEGY 1: Funding rate extreme fade on 15m.
    Funding > +0.05% → SHORT. Funding < -0.05% → LONG.
    SL 0.5%, TP 1.5% (1:3 R:R). Only if 1h bias agrees with fade direction.
    """
    if not extras:
        return None
    rate = extras.get("funding_rate", 0)
    bias = extras.get("htf_bias", "neutral")
    if rate == 0:
        return None
    price = float(df["close"].iloc[i])
    if rate > 0.0005:  # +0.05% = longs overcrowded → SHORT
        if bias in ("bearish", "neutral"):
            return _make_signal_pct("short", price, 0.5, 1.5)
    elif rate < -0.0005:  # -0.05% = shorts overcrowded → LONG
        if bias in ("bullish", "neutral"):
            return _make_signal_pct("long", price, 0.5, 1.5)
    return None


def sig_liquidation_cascade_fade_15m(df, i, extras=None):
    """STRATEGY 2: Liquidation cascade fade on 15m.
    Detect liquidation via sudden OI drop (volume spike) + price spike.
    Fade the spike within 2 candles. SL beyond spike + 0.2%. TP 1.5x SL.
    Filter: liquidation must be 3x average volume.
    """
    if i < 3:
        return None
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(vol_ma) or vol_ma <= 0:
        return None

    # Check previous 2 candles for liquidation cascade
    for lookback in range(1, 3):
        idx = i - lookback
        if idx < 0:
            continue
        prev_vol = float(df["volume"].iloc[idx])
        if prev_vol < vol_ma * 3:  # Must be 3x average volume
            continue
        prev_high = float(df["high"].iloc[idx])
        prev_low = float(df["low"].iloc[idx])
        prev_close = float(df["close"].iloc[idx])
        prev_open = float(df["open"].iloc[idx])
        prev_range = prev_high - prev_low
        price = float(df["close"].iloc[i])

        if prev_close > prev_open and prev_range > 0:
            # Bullish liquidation spike (shorts liquidated) → fade SHORT
            spike_high = prev_high
            sl_price = round(spike_high * 1.002)  # Beyond spike + 0.2%
            sl_dist = abs(price - sl_price)
            tp_dist = sl_dist * 1.5
            tp_price = round(price - tp_dist)
            if tp_price < price and sl_price > price:
                return {"side": "short", "entry": price, "sl": sl_price, "tp": tp_price}

        elif prev_close < prev_open and prev_range > 0:
            # Bearish liquidation spike (longs liquidated) → fade LONG
            spike_low = prev_low
            sl_price = round(spike_low * 0.998)  # Beyond spike - 0.2%
            sl_dist = abs(sl_price - price)
            tp_dist = sl_dist * 1.5
            tp_price = round(price + tp_dist)
            if tp_price > price and sl_price < price:
                return {"side": "long", "entry": price, "sl": sl_price, "tp": tp_price}
    return None


def sig_oi_divergence_15m(df, i, extras=None):
    """STRATEGY 3: Open interest divergence on 15m.
    Price new high + OI/volume dropping → fake breakout → SHORT.
    Price new low + OI/volume dropping → shorts covering → LONG.
    Divergence must persist 2+ candles. SL 0.4%, TP 1.2%.
    """
    if i < 15:
        return None
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(vol_ma) or vol_ma <= 0:
        return None

    price = float(df["close"].iloc[i])
    lookback = min(i, 10)

    # Check for new high
    recent_high = float(df["high"].iloc[i - lookback:i].max())
    current_high = float(df["high"].iloc[i])

    # Check volume declining for last 2 candles (OI proxy)
    vol_declining = True
    for j in range(1, 3):
        if i - j < 0:
            vol_declining = False
            break
        if float(df["volume"].iloc[i - j + 1]) >= float(df["volume"].iloc[i - j]) * 1.1:
            vol_declining = False
            break

    if not vol_declining:
        return None

    # Use volume below average as OI decline proxy
    curr_vol = float(df["volume"].iloc[i])
    if curr_vol >= vol_ma:
        return None

    if current_high >= recent_high:
        # Price new high but OI/volume dropping → SHORT
        return _make_signal_pct("short", price, 0.4, 1.2)

    recent_low = float(df["low"].iloc[i - lookback:i].min())
    current_low = float(df["low"].iloc[i])
    if current_low <= recent_low:
        # Price new low but OI/volume dropping → LONG
        return _make_signal_pct("long", price, 0.4, 1.2)

    return None


def sig_funding_reversion_15m(df, i, extras=None):
    """STRATEGY 4: Funding rate reversion on 15m.
    After funding extreme (>0.05% or <-0.05%), funding reverts to neutral.
    Trade when funding crosses back through 0.02% from extreme.
    SL 0.4%, TP 1.2%.
    """
    if not extras:
        return None
    rate = extras.get("funding_rate", 0)
    prev_rate = extras.get("prev_funding_rate", 0)
    if rate == 0 or prev_rate == 0:
        return None
    price = float(df["close"].iloc[i])

    # Was extreme positive, now reverting down through 0.02%
    if prev_rate > 0.0005 and rate <= 0.0002 and rate < prev_rate:
        return _make_signal_pct("long", price, 0.4, 1.2)

    # Was extreme negative, now reverting up through -0.02%
    if prev_rate < -0.0005 and rate >= -0.0002 and rate > prev_rate:
        return _make_signal_pct("short", price, 0.4, 1.2)

    return None


def sig_vwap_reclaim_oi_15m(df, i, extras=None):
    """STRATEGY 5: VWAP reclaim with OI (volume) confirmation on 15m.
    Price reclaims VWAP from below + volume increasing → LONG.
    Price loses VWAP + volume increasing → SHORT.
    OI confirmation = volume > vol_ma (real buyers/sellers).
    SL 0.5%, TP 0.9%.
    """
    if i < 2:
        return None
    vwap = _safe_val(df["vwap"], i)
    vwap_p = _safe_val(df["vwap"], i - 1)
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(vwap) or _nan(vwap_p) or _nan(vol_ma) or vol_ma <= 0:
        return None

    price = float(df["close"].iloc[i])
    prev_close = float(df["close"].iloc[i - 1])
    curr_vol = float(df["volume"].iloc[i])

    # Volume must be above average (OI confirmation)
    if curr_vol < vol_ma:
        return None

    # Price reclaims VWAP from below → LONG
    if prev_close < vwap_p and price > vwap:
        return _make_signal_pct("long", price, 0.5, 0.9)

    # Price loses VWAP from above → SHORT
    if prev_close > vwap_p and price < vwap:
        return _make_signal_pct("short", price, 0.5, 0.9)

    return None


# --- COMBINED ---

def sig_rsi_vwap_confluence(df, i, extras=None):
    """RSI extreme + VWAP position."""
    rsi = _safe_val(df["rsi"], i)
    vwap = _safe_val(df["vwap"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(rsi) or _nan(vwap) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if rsi < 35 and price < vwap:
        return _make_signal("long", price, atr, 1.0, 2.5)
    if rsi > 65 and price > vwap:
        return _make_signal("short", price, atr, 1.0, 2.5)
    return None


def sig_ema_volume_confluence(df, i, extras=None):
    """EMA cross + volume confirmation."""
    if i < 2:
        return None
    e9 = _safe_val(df["ema_9"], i)
    e21 = _safe_val(df["ema_21"], i)
    e9p = _safe_val(df["ema_9"], i - 1)
    e21p = _safe_val(df["ema_21"], i - 1)
    atr = _safe_val(df["atr"], i)
    vol_ma = _safe_val(df["vol_ma"], i)
    if _nan(e9) or _nan(e21) or _nan(e9p) or _nan(e21p) or _nan(atr) or _nan(vol_ma):
        return None
    if atr <= 0 or vol_ma <= 0:
        return None
    vol = float(df["volume"].iloc[i])
    if vol < vol_ma * 1.5:
        return None
    price = float(df["close"].iloc[i])
    if e9p <= e21p and e9 > e21:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if e9p >= e21p and e9 < e21:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_trend_momentum(df, i, extras=None):
    """ADX + MACD + EMA aligned."""
    adx = _safe_val(df["adx"], i)
    macd_h = _safe_val(df["macd_hist"], i)
    e9 = _safe_val(df["ema_9"], i)
    e21 = _safe_val(df["ema_21"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(adx) or _nan(macd_h) or _nan(e9) or _nan(e21) or _nan(atr) or atr <= 0:
        return None
    if adx < 25:
        return None
    price = float(df["close"].iloc[i])
    if e9 > e21 and macd_h > 0 and price > e9:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if e9 < e21 and macd_h < 0 and price < e9:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_mtf_confluence(df, i, extras=None):
    """1h trend + 5m entry -- uses EMA200 for trend, EMA9/21 for entry."""
    if i < 2:
        return None
    e200 = _safe_val(df["ema_200"], i)
    e9 = _safe_val(df["ema_9"], i)
    e21 = _safe_val(df["ema_21"], i)
    e9p = _safe_val(df["ema_9"], i - 1)
    e21p = _safe_val(df["ema_21"], i - 1)
    atr = _safe_val(df["atr"], i)
    if _nan(e200) or _nan(e9) or _nan(e21) or _nan(e9p) or _nan(e21p) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if price > e200 and e9p <= e21p and e9 > e21:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < e200 and e9p >= e21p and e9 < e21:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_session_pattern(df, i, extras=None):
    """Pattern only during London/NY."""
    if "timestamp" not in df.columns:
        return None
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        h = ts.hour
    else:
        return None
    if not (7 <= h < 11 or 13 <= h < 17):
        return None
    # Use engulfing as the pattern
    return sig_engulfing(df, i, extras)


def sig_regime_rsi(df, i, extras=None):
    """RSI in RANGING, momentum in TRENDING."""
    adx = _safe_val(df["adx"], i)
    rsi = _safe_val(df["rsi"], i)
    atr = _safe_val(df["atr"], i)
    if _nan(adx) or _nan(rsi) or _nan(atr) or atr <= 0:
        return None
    price = float(df["close"].iloc[i])
    if adx < 20:  # Ranging
        if rsi < 30:
            return _make_signal("long", price, atr, 1.0, 2.0)
        if rsi > 70:
            return _make_signal("short", price, atr, 1.0, 2.0)
    elif adx > 30:  # Trending
        e9 = _safe_val(df["ema_9"], i)
        e21 = _safe_val(df["ema_21"], i)
        if not _nan(e9) and not _nan(e21):
            if e9 > e21 and rsi > 50:
                return _make_signal("long", price, atr, 1.5, 3.0)
            if e9 < e21 and rsi < 50:
                return _make_signal("short", price, atr, 1.5, 3.0)
    return None


def sig_volatility_breakout(df, i, extras=None):
    """ATR expansion + directional candle."""
    if i < 5:
        return None
    atr = _safe_val(df["atr"], i)
    if _nan(atr) or atr <= 0:
        return None
    # Check ATR expansion: current ATR > 1.5x ATR from 5 candles ago
    atr_prev = _safe_val(df["atr"], i - 5)
    if _nan(atr_prev) or atr_prev <= 0:
        return None
    if atr < atr_prev * 1.5:
        return None
    price = float(df["close"].iloc[i])
    open_p = float(df["open"].iloc[i])
    body = abs(price - open_p)
    full_range = float(df["high"].iloc[i]) - float(df["low"].iloc[i])
    if full_range == 0 or body / full_range < 0.6:
        return None
    if price > open_p:
        return _make_signal("long", price, atr, 1.5, 3.0)
    if price < open_p:
        return _make_signal("short", price, atr, 1.5, 3.0)
    return None


# ---------------------------------------------------------------------------
# ALL_STRATEGY_DEFS -- maps strategy name to config
# ---------------------------------------------------------------------------

ALL_STRATEGY_DEFS = {
    # MOMENTUM
    "ema_cross_9_21_5m":    {"label": "EMA 9/21 5m",        "emoji": "1",  "tf": "5m",  "signal_func": sig_ema_cross_9_21,    "max_hold": 240},
    "ema_cross_21_55_15m":  {"label": "EMA 21/55 15m",      "emoji": "2",  "tf": "15m", "signal_func": sig_ema_cross_21_55,   "max_hold": 480},
    "macd_cross_5m":        {"label": "MACD Cross 5m",      "emoji": "3",  "tf": "5m",  "signal_func": sig_macd_cross_5m,     "max_hold": 240},
    "macd_cross_15m":       {"label": "MACD Cross 15m",     "emoji": "4",  "tf": "15m", "signal_func": sig_macd_cross_15m,    "max_hold": 480},
    "adx_ema_5m":           {"label": "ADX+EMA 5m",         "emoji": "5",  "tf": "5m",  "signal_func": sig_adx_ema,           "max_hold": 240},
    "supertrend_5m":        {"label": "Supertrend 5m",      "emoji": "6",  "tf": "5m",  "signal_func": sig_supertrend_5m,     "max_hold": 240},
    "supertrend_15m":       {"label": "Supertrend 15m",     "emoji": "7",  "tf": "15m", "signal_func": sig_supertrend_15m,    "max_hold": 480},
    "parabolic_sar_5m":     {"label": "Parabolic SAR 5m",   "emoji": "8",  "tf": "5m",  "signal_func": sig_parabolic_sar,     "max_hold": 240},
    "ichimoku_breakout_1h": {"label": "Ichimoku Break 1h",  "emoji": "9",  "tf": "1h",  "signal_func": sig_ichimoku_breakout, "max_hold": 720},
    "hull_ma_cross_5m":     {"label": "Hull MA Cross 5m",   "emoji": "10", "tf": "5m",  "signal_func": sig_hull_ma_cross,     "max_hold": 240},
    # MEAN REVERSION
    "rsi_extreme_5m":       {"label": "RSI Extreme 5m",     "emoji": "11", "tf": "5m",  "signal_func": sig_rsi_extreme,       "max_hold": 120},
    "rsi_divergence_5m":    {"label": "RSI Divergence 5m",  "emoji": "12", "tf": "5m",  "signal_func": sig_rsi_divergence,    "max_hold": 180},
    "bb_touch_5m":          {"label": "BB Touch 5m",        "emoji": "13", "tf": "5m",  "signal_func": sig_bb_touch_reversal, "max_hold": 120},
    "bb_squeeze_5m":        {"label": "BB Squeeze 5m",      "emoji": "14", "tf": "5m",  "signal_func": sig_bb_squeeze_breakout,"max_hold": 240},
    "stoch_rsi_cross_5m":   {"label": "Stoch RSI Cross 5m", "emoji": "15", "tf": "5m",  "signal_func": sig_stoch_rsi_cross,   "max_hold": 120},
    "cci_extreme_5m":       {"label": "CCI Extreme 5m",     "emoji": "16", "tf": "5m",  "signal_func": sig_cci_extreme,       "max_hold": 120},
    "williams_r_5m":        {"label": "Williams %R 5m",     "emoji": "17", "tf": "5m",  "signal_func": sig_williams_r_extreme, "max_hold": 120},
    "mean_rev_ema_5m":      {"label": "Mean Rev EMA 5m",    "emoji": "18", "tf": "5m",  "signal_func": sig_mean_reversion_ema, "max_hold": 120},
    # VOLUME
    "vwap_bounce_1m":       {"label": "VWAP Bounce 1m",     "emoji": "19", "tf": "1m",  "signal_func": sig_vwap_bounce,       "max_hold": 60},
    "vwap_reclaim_5m":      {"label": "VWAP Reclaim 5m",    "emoji": "20", "tf": "5m",  "signal_func": sig_vwap_reclaim,      "max_hold": 180},
    "vol_spike_rev_5m":     {"label": "Vol Spike Rev 5m",   "emoji": "21", "tf": "5m",  "signal_func": sig_volume_spike_reversal,"max_hold": 120},
    "obv_divergence_15m":   {"label": "OBV Divergence 15m", "emoji": "22", "tf": "15m", "signal_func": sig_obv_divergence,    "max_hold": 240},
    "funding_fade_1h":      {"label": "Funding Fade 1h",    "emoji": "23", "tf": "1h",  "signal_func": sig_funding_rate_fade, "max_hold": 480},
    "liquidation_fade_5m":  {"label": "Liquidation Fade 5m","emoji": "24", "tf": "5m",  "signal_func": sig_liquidation_fade,  "max_hold": 120},
    "delta_div_5m":         {"label": "Delta Divergence 5m","emoji": "25", "tf": "5m",  "signal_func": sig_delta_divergence,  "max_hold": 120},
    # PRICE ACTION
    "order_block_5m":       {"label": "Order Block 5m",     "emoji": "26", "tf": "5m",  "signal_func": sig_order_block,       "max_hold": 240},
    "liq_sweep_5m":         {"label": "Liq Sweep 5m",       "emoji": "27", "tf": "5m",  "signal_func": sig_liquidity_sweep,   "max_hold": 240},
    "fvg_fill_5m":          {"label": "FVG Fill 5m",        "emoji": "28", "tf": "5m",  "signal_func": sig_fvg_fill,          "max_hold": 180},
    "bos_pullback_15m":     {"label": "BOS Pullback 15m",   "emoji": "29", "tf": "15m", "signal_func": sig_bos_pullback,      "max_hold": 480},
    "inside_bar_5m":        {"label": "Inside Bar 5m",      "emoji": "30", "tf": "5m",  "signal_func": sig_inside_bar,        "max_hold": 120},
    "engulfing_5m":         {"label": "Engulfing 5m",       "emoji": "31", "tf": "5m",  "signal_func": sig_engulfing,         "max_hold": 120},
    "pin_bar_5m":           {"label": "Pin Bar 5m",         "emoji": "32", "tf": "5m",  "signal_func": sig_pin_bar,           "max_hold": 120},
    "three_candle_rev_5m":  {"label": "3-Candle Rev 5m",    "emoji": "33", "tf": "5m",  "signal_func": sig_three_candle_reversal,"max_hold": 120},
    "hh_hl_entry_15m":      {"label": "HH/HL Entry 15m",   "emoji": "34", "tf": "15m", "signal_func": sig_hh_hl_entry,       "max_hold": 480},
    # SESSION
    "london_breakout_5m":   {"label": "London Break 5m",    "emoji": "35", "tf": "5m",  "signal_func": sig_london_breakout,   "max_hold": 240},
    "ny_breakout_5m":       {"label": "NY Break 5m",        "emoji": "36", "tf": "5m",  "signal_func": sig_ny_breakout,       "max_hold": 240},
    "asia_range_fade_5m":   {"label": "Asia Fade 5m",       "emoji": "37", "tf": "5m",  "signal_func": sig_asia_range_fade,   "max_hold": 180},
    "session_end_mr_5m":    {"label": "Session End MR 5m",  "emoji": "38", "tf": "5m",  "signal_func": sig_session_end_mr,    "max_hold": 60},
    "monday_range_5m":      {"label": "Monday Range 5m",    "emoji": "39", "tf": "5m",  "signal_func": sig_monday_range,      "max_hold": 480},
    # HL SPECIFIC
    "top_trader_bias_5m":   {"label": "Top Trader Bias 5m", "emoji": "40", "tf": "5m",  "signal_func": sig_top_trader_bias,   "max_hold": 240},
    "oi_spike_5m":          {"label": "OI Spike 5m",        "emoji": "41", "tf": "5m",  "signal_func": sig_oi_spike,          "max_hold": 120},
    "funding_extreme_1h":   {"label": "Funding Extreme 1h", "emoji": "42", "tf": "1h",  "signal_func": sig_funding_extreme,   "max_hold": 480},
    "perp_premium_5m":      {"label": "Perp Premium 5m",    "emoji": "43", "tf": "5m",  "signal_func": sig_perp_premium,      "max_hold": 240},
    # COMBINED
    "rsi_vwap_conf_5m":     {"label": "RSI+VWAP Conf 5m",   "emoji": "44", "tf": "5m",  "signal_func": sig_rsi_vwap_confluence,"max_hold": 120},
    "ema_vol_conf_5m":      {"label": "EMA+Vol Conf 5m",    "emoji": "45", "tf": "5m",  "signal_func": sig_ema_volume_confluence,"max_hold": 240},
    "trend_momentum_5m":    {"label": "Trend Momentum 5m",  "emoji": "46", "tf": "5m",  "signal_func": sig_trend_momentum,    "max_hold": 240},
    "mtf_confluence_5m":    {"label": "MTF Confluence 5m",   "emoji": "47", "tf": "5m",  "signal_func": sig_mtf_confluence,    "max_hold": 240},
    "session_pattern_5m":   {"label": "Session Pattern 5m",  "emoji": "48", "tf": "5m",  "signal_func": sig_session_pattern,   "max_hold": 120},
    "regime_rsi_5m":        {"label": "Regime RSI 5m",       "emoji": "49", "tf": "5m",  "signal_func": sig_regime_rsi,        "max_hold": 180},
    "volatility_break_5m":  {"label": "Vol Breakout 5m",     "emoji": "50", "tf": "5m",  "signal_func": sig_volatility_breakout,"max_hold": 240},
    # PERP-NATIVE (new)
    "funding_extreme_fade_15m": {"label": "Funding Fade 15m",     "emoji": "51", "tf": "15m", "signal_func": sig_funding_extreme_fade_15m, "max_hold": 480},
    "liq_cascade_fade_15m":     {"label": "Liq Cascade Fade 15m", "emoji": "52", "tf": "15m", "signal_func": sig_liquidation_cascade_fade_15m, "max_hold": 240},
    "oi_divergence_15m":        {"label": "OI Divergence 15m",    "emoji": "53", "tf": "15m", "signal_func": sig_oi_divergence_15m, "max_hold": 480},
    "funding_reversion_15m":    {"label": "Funding Revert 15m",   "emoji": "54", "tf": "15m", "signal_func": sig_funding_reversion_15m, "max_hold": 480},
    "vwap_reclaim_oi_15m":      {"label": "VWAP+OI 15m",         "emoji": "55", "tf": "15m", "signal_func": sig_vwap_reclaim_oi_15m, "max_hold": 240},
    # EXISTING retested on 15m
    "rsi_divergence_15m":       {"label": "RSI Divergence 15m",   "emoji": "56", "tf": "15m", "signal_func": sig_rsi_divergence, "max_hold": 480},
    "williams_r_15m":           {"label": "Williams %R 15m",      "emoji": "57", "tf": "15m", "signal_func": sig_williams_r_extreme, "max_hold": 240},
    "macd_cross_15m_v2":        {"label": "MACD Cross 15m v2",    "emoji": "58", "tf": "15m", "signal_func": sig_macd_cross_15m, "max_hold": 480},
}


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

    log.info("HTF BIAS UPDATE: bias=%s strength=%s (EMA_slope=%.4f%% RSI=%.1f)",
             state.htf_bias, state.htf_bias_strength, ema_slope_pct, rsi_val)


# ---------------------------------------------------------------------------
# Pre-Trade Checklist
# ---------------------------------------------------------------------------

_rejection_tg_count = 0
_rejection_tg_hour = 0
_rejection_tg_suppressed = 0

def _should_send_rejection_tg() -> bool:
    """Rate limit rejection Telegram notifications: max 3 per hour, then hourly summary."""
    global _rejection_tg_count, _rejection_tg_hour, _rejection_tg_suppressed
    current_hour = int(time.time()) // 3600
    if current_hour != _rejection_tg_hour:
        if _rejection_tg_suppressed > 0:
            asyncio.create_task(tg_send(f"Suppressed {_rejection_tg_suppressed} rejection notifications last hour"))
        _rejection_tg_hour = current_hour
        _rejection_tg_count = 0
        _rejection_tg_suppressed = 0
    _rejection_tg_count += 1
    if _rejection_tg_count <= 3:
        return True
    _rejection_tg_suppressed += 1
    return False


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

    # 1. No opposing position open
    if state.current_position is not None:
        pos_side = state.current_position["side"]
        if pos_side == side:
            return _block(strategy, side, f"Already have {side} position open (same direction)")
        else:
            return _block(strategy, side, f"Opposing {pos_side} position open -- close first")

    # 2. Regime check -- NO trading in VOLATILE
    if state.regime == Regime.VOLATILE:
        return _block(strategy, side, f"VOLATILE regime -- no trading")

    # 3. Daily loss check -- 3 losses = stop
    if state.losses_today >= MAX_LOSSES_PER_DAY:
        return _block(strategy, side, f"Daily loss limit ({state.losses_today}/{MAX_LOSSES_PER_DAY} losses)")

    if state.daily_loss_cap_hit:
        return _block(strategy, side, "Daily loss cap hit")

    # Equity drawdown killswitch
    if state.drawdown_killswitch_hit:
        return _block(strategy, side, f"Equity drawdown killswitch active (>={EQUITY_DRAWDOWN_KILLSWITCH*100:.0f}% drawdown from peak)")

    # 4. Max trades per day
    if state.trades_today >= MAX_TRADES_PER_DAY:
        return _block(strategy, side, f"Max trades per day ({state.trades_today}/{MAX_TRADES_PER_DAY})")

    # 5. Macro check -- CRITICAL = no new trades
    if _macro_intel_cache.get("text", ""):
        if "RISK_LEVEL: CRITICAL" in _macro_intel_cache["text"].upper():
            return _block(strategy, side, "CRITICAL macro event active")

    # 6. Global or strategy paused
    if state.paused:
        return _block(strategy, side, "Global trading paused")

    if state.strategy_paused.get(strategy, False):
        return _block(strategy, side, f"{strategy} paused (underperformance)")

    # 7. Strategy must be active
    if strategy not in state.active_strategies:
        return _block(strategy, side, f"{strategy} not in active strategies")

    # 8. Session filter: only London (07:00-11:00) and NY (13:00-17:00)
    hour = datetime.now(timezone.utc).hour
    in_session = (7 <= hour < 11) or (13 <= hour < 17)
    if not in_session:
        return _block(strategy, side, f"Outside trading sessions (hour={hour})")

    # 9. No weekends
    weekday = datetime.now(timezone.utc).weekday()
    if weekday >= 5:  # Saturday=5, Sunday=6
        return _block(strategy, side, "Weekend -- no trading")

    # 10. MANDATORY BIAS DIRECTION FILTER
    # Bullish bias (weak or strong) → longs only. Bearish → shorts only. Neutral → both.
    if state.htf_bias == "bullish" and side == "short":
        if _should_send_rejection_tg():
            asyncio.create_task(tg_send(f'\U0001f6ab BIAS FILTER: {strategy} {side.upper()} blocked \u2014 1H bias is {state.htf_bias.upper()}'))
        return _block(strategy, side, f"1H bias is BULLISH -- only LONG signals allowed")
    if state.htf_bias == "bearish" and side == "long":
        if _should_send_rejection_tg():
            asyncio.create_task(tg_send(f'\U0001f6ab BIAS FILTER: {strategy} {side.upper()} blocked \u2014 1H bias is {state.htf_bias.upper()}'))
        return _block(strategy, side, f"1H bias is BEARISH -- only SHORT signals allowed")

    # 11. PROPOSAL 5: Top-trader dominance bias filter
    # If top-3 traders show dominant LONG, block shorts; if dominant SHORT, block longs.
    # NEUTRAL allows both.
    intel = _intelligence_cache.get("report", {})
    patterns = intel.get("patterns", {})
    dominant = str(patterns.get("dominant_direction", "NEUTRAL")).upper()
    if dominant == "LONG" and side == "short":
        return _block(strategy, side, f"Top-trader dominance is LONG -- short blocked")
    if dominant == "SHORT" and side == "long":
        return _block(strategy, side, f"Top-trader dominance is SHORT -- long blocked")

    log.info("PASSED: %s %s -- all checks OK", strategy, side.upper())
    return True, "OK"


# ---------------------------------------------------------------------------
# Position Sizing & Order Execution
# ---------------------------------------------------------------------------

def calc_leverage_from_confidence(confidence: int) -> int:
    """Determine leverage from AI confidence score.
    GG standard: HARD CAPPED at 10x regardless of confidence. Higher leverage
    amplified losses during the $500→$85 drawdown, so scaling is disabled.
    """
    return min(BASE_LEVERAGE, MAX_LEVERAGE_HARD_CAP)


async def fetch_live_equity() -> float:
    """Fetch Vic's current Hyperliquid account equity in USD.
    Handles BOTH unified-account mode (balance in spot USDC)
    and legacy perp-only mode (balance in clearinghouse marginSummary).
    Falls back to state.live_equity or ACCOUNT_CAPITAL_FALLBACK if fetch fails.
    Also updates peak_equity and the drawdown killswitch.
    """
    global state
    try:
        loop = asyncio.get_running_loop()
        user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
        margin_summary = user_st.get("marginSummary", {}) if isinstance(user_st, dict) else {}
        perp_equity = float(margin_summary.get("accountValue", 0) or 0)

        # Unified account: balance lives in spot USDC, not perp marginSummary.
        # Use HL's POST /info with spotClearinghouseState.
        spot_equity = 0.0
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    "https://api.hyperliquid.xyz/info",
                    json={"type": "spotClearinghouseState", "user": HL_WALLET_ADDRESS},
                )
                if r.status_code == 200:
                    spot_data = r.json()
                    for bal in spot_data.get("balances", []) or []:
                        if bal.get("coin") == "USDC":
                            spot_equity = float(bal.get("total", 0) or 0)
                            break
        except Exception as spot_exc:
            log.warning(f"spotClearinghouseState fetch failed: {spot_exc}")

        # Under unified mode, perp_equity will be 0 and spot_equity holds the real balance.
        # Under legacy mode, perp_equity holds everything. Use whichever is larger.
        equity = max(perp_equity, spot_equity)
        log.info(f"fetch_live_equity -- perp=${perp_equity:.2f} spot=${spot_equity:.2f} using=${equity:.2f}")

        if equity <= 0:
            raise ValueError(f"Both perp (${perp_equity}) and spot (${spot_equity}) equity are zero")
        state.live_equity = equity
        if equity > state.peak_equity:
            state.peak_equity = equity
        if state.peak_equity > 0:
            drawdown = 1.0 - (equity / state.peak_equity)
            if drawdown >= EQUITY_DRAWDOWN_KILLSWITCH and not state.drawdown_killswitch_hit:
                state.drawdown_killswitch_hit = True
                state.paused = True
                save_state()
                await tg_send(
                    f"🛑 DRAWDOWN KILLSWITCH — Vic paused.\n"
                    f"Equity ${equity:.2f} vs peak ${state.peak_equity:.2f} "
                    f"({drawdown*100:.1f}% drawdown >= {EQUITY_DRAWDOWN_KILLSWITCH*100:.0f}% limit).\n"
                    f"Review trades before /resume."
                )
        return equity
    except Exception as e:
        log.warning(f"fetch_live_equity failed: {e} — using fallback")
        return state.live_equity if state.live_equity > 0 else ACCOUNT_CAPITAL_FALLBACK


def validate_tp(side: str, entry: float, tp: float, strategy: str) -> bool:
    """Validate take profit is in the correct direction. Called before EVERY order."""
    if side == "long" and tp <= entry:
        log.error(f"INVALID TP: {strategy} LONG entry={entry} tp={tp} — TP below entry. Trade cancelled.")
        asyncio.create_task(tg_send(f"\U0001f6ab TP ERROR — {strategy} LONG: entry ${entry:,.2f} but TP ${tp:,.2f} is below entry. Trade blocked."))
        return False
    if side == "short" and tp >= entry:
        log.error(f"INVALID TP: {strategy} SHORT entry={entry} tp={tp} — TP above entry. Trade cancelled.")
        asyncio.create_task(tg_send(f"\U0001f6ab TP ERROR — {strategy} SHORT: entry ${entry:,.2f} but TP ${tp:,.2f} is above entry. Trade blocked."))
        return False
    return True


async def execute_trade(strategy: str, side: str, entry: float,
                        sl_price: float, tp_price: float,
                        confidence: int = 7, max_leverage: Optional[int] = None):
    """Open a position (paper or live). Single position model.

    sl_price and tp_price are pre-calculated by the strategy.
    Leverage is determined by AI confidence score, capped by max_leverage if set.
    Position sized so that SL hit <= TARGET_RISK_PCT of account.
    """
    allowed, reason = can_open_trade(strategy, side)
    if not allowed:
        return

    # HARD VALIDATION: TP must be in the correct direction relative to signal entry
    if not validate_tp(side, entry, tp_price, strategy):
        return

    leverage = calc_leverage_from_confidence(confidence)
    if max_leverage is not None:
        leverage = min(leverage, max_leverage)

    # Enforce minimum SL distance (GG standard: 2% minimum)
    min_sl_dist = entry * (MIN_SL_PCT / 100.0)
    sl_distance = abs(entry - sl_price)
    if sl_distance < min_sl_dist:
        log.info("%s -- SL widened from %.0f to %.0f (%.1f%% minimum)", strategy, sl_distance, min_sl_dist, MIN_SL_PCT)
        sl_distance = min_sl_dist
        if side == "long":
            sl_price = round(entry - sl_distance)
        else:
            sl_price = round(entry + sl_distance)
    if sl_distance <= 0:
        log.warning("%s -- zero SL distance, cannot size position.", strategy)
        return

    # Fee-aware profit check: TP must clear at least 2× round-trip fees net
    tp_distance = abs(tp_price - entry)
    min_net_profit = FEE_PER_TRADE_USD * 2 * MIN_NET_PROFIT_MULT  # $12 min net
    # We'll size based on risk, so estimate gross TP $ at that size below. For now skip size-dependent check until after sizing.

    # Size: HARD 2% risk per trade (no confidence scaling — GG standard)
    account_equity = await fetch_live_equity()
    if state.drawdown_killswitch_hit:
        log.warning("%s -- drawdown killswitch active, cannot open trade", strategy)
        return
    risk_pct = TARGET_RISK_PCT
    risk_dollars = account_equity * risk_pct
    size = math.floor(risk_dollars / sl_distance * 100000) / 100000

    if size <= 0:
        log.warning("%s -- invalid size from sizing, skipping.", strategy)
        return

    # Check notional meets minimum ($10 on HL)
    notional = size * entry
    if notional < 10:
        log.warning("%s -- notional $%.2f below HL minimum, skipping.", strategy, notional)
        return

    # Fee-aware profit check: expected gross TP must clear 2× round-trip fees
    expected_tp_gross = size * tp_distance
    round_trip_fees = FEE_PER_TRADE_USD * 2
    min_required_profit = round_trip_fees * MIN_NET_PROFIT_MULT
    if expected_tp_gross < min_required_profit:
        log.warning(
            "%s -- TP gross $%.2f < min $%.2f (2× round-trip fees $%.2f × %.1f). Edge too thin after fees. Skipping.",
            strategy, expected_tp_gross, min_required_profit, round_trip_fees, MIN_NET_PROFIT_MULT
        )
        asyncio.create_task(tg_send(
            f"⚠️ {strategy} {side.upper()} rejected — TP too small vs fees "
            f"(gross ${expected_tp_gross:.2f} < min ${min_required_profit:.2f})"
        ))
        return

    # Cap to leverage limit
    max_size = math.floor(account_equity * leverage / entry * 100000) / 100000
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

            # Fetch actual fill price and recalculate TP/SL relative to real entry
            try:
                await asyncio.sleep(0.5)
                user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
                hl_positions = user_st.get("assetPositions", [])
                for p in hl_positions:
                    pd_pos = p.get("position", {})
                    if pd_pos.get("coin") == "BTC" and float(pd_pos.get("szi", 0)) != 0:
                        actual_entry = float(pd_pos.get("entryPx", 0))
                        if actual_entry > 0 and abs(actual_entry - entry) > 1:
                            log.info("%s -- Actual fill $%.2f vs signal entry $%.2f (diff $%.2f)",
                                     strategy, actual_entry, entry, actual_entry - entry)
                            # Preserve TP/SL distances, recalculate from actual fill price
                            old_tp_dist = abs(tp_price - entry)
                            old_sl_dist = sl_distance
                            entry = actual_entry
                            if side == "long":
                                tp_price = round(entry + old_tp_dist)
                                sl_price = round(entry - old_sl_dist)
                            else:
                                tp_price = round(entry - old_tp_dist)
                                sl_price = round(entry + old_sl_dist)
                            tp_distance = old_tp_dist
                            log.info("%s -- Recalculated from fill: SL $%.2f, TP $%.2f", strategy, sl_price, tp_price)
                        break
            except Exception as fill_exc:
                log.warning("%s -- Could not fetch actual fill price: %s (using signal entry)", strategy, fill_exc)

            # Re-validate TP against actual entry — absolute safety check
            if not validate_tp(side, entry, tp_price, strategy):
                log.error("%s -- TP INVALID after fill adjustment — emergency close", strategy)
                try:
                    await loop.run_in_executor(
                        None, lambda: hl_exchange.market_close("BTC", sz=size)
                    )
                    await tg_send(f"\U0001f6a8 {strategy} — TP invalid after fill. Position closed immediately.")
                except Exception as close_exc:
                    await tg_send(f"\U0001f6a8\U0001f6a8 CRITICAL: {strategy} TP invalid AND close failed: {close_exc}. CLOSE MANUALLY.")
                return

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
    strat_def = ALL_STRATEGY_DEFS.get(strategy, {})
    label = strat_def.get("label", strategy)
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

async def close_position(exit_price: float, reason: str, force: bool = False):
    """Close the current position and book PnL.
    force=True bypasses the 60-second minimum hold time (used by /closeall only).
    """
    pos = state.current_position
    if pos is None:
        return

    # Minimum 60-second hold time to prevent open/close loops
    if not force:
        open_time = datetime.fromisoformat(pos["open_time"])
        elapsed_sec = (datetime.now(timezone.utc) - open_time).total_seconds()
        if elapsed_sec < 60:
            log.info("Hold time guard: %.0fs < 60s minimum, ignoring close request (%s)", elapsed_sec, reason)
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
    state._ensure_strategy_metrics(strategy)
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

    strat_def = ALL_STRATEGY_DEFS.get(strategy, {})
    label = strat_def.get("label", strategy)
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
        save_state()
        log.info("DAILY CAP HIT AND SAVED — losses=%d cap_hit=%s", state.losses_today, state.daily_loss_cap_hit)
        await tg_send(
            f"\U0001f6d1 <b>{MAX_LOSSES_PER_DAY} losing trades today -- ALL trading stopped until midnight UTC</b>\n"
            f"Daily PnL: ${state.daily_pnl:+,.2f}"
        )

    # Check max daily loss (2% of LIVE account equity — tight cap)
    _daily_cap_equity = state.live_equity if state.live_equity > 0 else ACCOUNT_CAPITAL_FALLBACK
    if state.daily_pnl <= -(_daily_cap_equity * MAX_RISK_PCT * MAX_LOSSES_PER_DAY):
        state.daily_loss_cap_hit = True
        save_state()
        log.info("DAILY CAP HIT AND SAVED — pnl=$%.2f cap_hit=%s", state.daily_pnl, state.daily_loss_cap_hit)
        await tg_send(
            f"\U0001f6d1 <b>Daily loss limit hit (${state.daily_pnl:+,.2f})</b>\n"
            f"All trading stopped until tomorrow."
        )

    # Underperformance check
    await check_underperformance(strategy)


async def check_underperformance(strategy: str):
    """If a strategy drops below 40% win rate after 20+ trades, pause it."""
    m = state.metrics.get(strategy, {})
    if m.get("trade_count", 0) < UNDERPERFORMANCE_MIN_TRADES:
        return
    win_rate = m.get("wins", 0) / m["trade_count"]
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
# AI Market Brain -- Pre-trade analysis with confidence scoring
# ---------------------------------------------------------------------------

_macro_intel_cache = {"text": "", "fetched_at": 0}

# AI Brain rejection tracking -- per-strategy consecutive rejections
_ai_rejection_streak: dict[str, int] = {}                # strategy -> consecutive rejections
_ai_rejection_override_until: dict[str, float] = {}      # strategy -> timestamp when override expires
_tg_rejection_timestamps: list[float] = []               # timestamps of recent rejection messages


def _get_trading_session() -> str:
    """Return current trading session based on UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 7 <= hour < 16:    # London: 07:00-16:00 UTC
        return "london"
    elif 13 <= hour < 22:  # NY: 13:00-22:00 UTC (overlaps London)
        return "ny"
    elif 0 <= hour < 8:    # Asia: 00:00-08:00 UTC
        return "asia"
    return "off"


def _is_london_or_ny() -> bool:
    hour = datetime.now(timezone.utc).hour
    return 7 <= hour < 22  # London open through NY close


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
            f"IMPORTANT RULES:\n"
            f"- Macro/geopolitical news is INFORMATIONAL ONLY. You MUST NOT reject a trade solely because of macro news, geopolitical events, wars, tariffs, or general market fear.\n"
            f"- You may ONLY reject a trade for TECHNICAL reasons: wrong session, wrong regime for the strategy, signal conflicts, or technically invalid setup.\n"
            f"- If macro conditions are concerning, lower your confidence by 1-2 points but still APPROVE if the technical setup is valid.\n\n"
            f"Reply with EXACTLY this format:\n"
            f"DECISION: APPROVE or REJECT\n"
            f"CONFIDENCE: [1-10]\n"
            f"REASON: [1-2 sentences]\n\n"
            f"Score guide: 1-5 = may be rejected (technical reasons only), 6+ = trade will execute (AI brain informational only). 7-8 = solid trade (10x leverage), 9 = strong confluence (15x), 10 = exceptional setup (20x)."
        )

        payload = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
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
    """Run all active strategies. Strongest signal wins."""
    if state.daily_loss_cap_hit or state.losses_today >= MAX_LOSSES_PER_DAY:
        log.debug("run_all_strategies blocked: cap_hit=%s losses=%d", state.daily_loss_cap_hit, state.losses_today)
        return
    if state.paused:
        return
    if state.regime == Regime.VOLATILE:
        return
    if state.trades_today >= MAX_TRADES_PER_DAY:
        return

    # Session filter
    hour = datetime.now(timezone.utc).hour
    in_session = (7 <= hour < 11) or (13 <= hour < 17)
    if not in_session:
        return

    # Weekend filter
    if datetime.now(timezone.utc).weekday() >= 5:
        return

    # If a position is open, don't look for new signals
    if state.current_position is not None:
        return

    # Fetch data for different timeframes
    df_cache = {}
    for tf in ["1m", "5m", "15m", "1h"]:
        try:
            df_cache[tf] = await fetch_ohlcv(tf, 250)
            if not df_cache[tf].empty and len(df_cache[tf]) >= 55:
                df_cache[tf] = precompute_indicators(df_cache[tf])
        except Exception:
            df_cache[tf] = pd.DataFrame()

    # Build extras for strategies that need them
    extras = {
        "funding_rate": state.current_funding_rate,
        "prev_funding_rate": state.funding_history[-2]["rate"] if len(state.funding_history) >= 2 else 0,
        "funding_pctl": get_funding_percentile(state.current_funding_rate),
        "htf_bias": state.htf_bias,
    }
    intel = _intelligence_cache.get("report", {})
    patterns = intel.get("patterns", {})
    extras["top_trader_bias"] = patterns.get("dominant_direction", "NEUTRAL")

    # PROPOSAL 3: Session whitelist -- during Asia hours, restrict to high-WR range strategies
    session = _get_trading_session()
    ASIA_WHITELIST = {"liq_cascade_fade_15m", "inside_bar_5m", "ema_cross_9_21_5m"}

    # Gather signals from all active strategies
    signals = []
    for name in state.active_strategies:
        if state.strategy_paused.get(name, False):
            continue
        # PROPOSAL 3: filter to Asia whitelist when in Asia session
        if session == "asia" and name not in ASIA_WHITELIST:
            continue
        sdef = ALL_STRATEGY_DEFS.get(name)
        if not sdef:
            continue
        tf = sdef["tf"]
        df = df_cache.get(tf)
        if df is None or df.empty or len(df) < 55:
            continue
        try:
            state.strategy_evals += 1
            log.debug("EVAL: %s tf=%s bars=%d", name, tf, len(df))
            sig = sdef["signal_func"](df, len(df) - 1, extras)
            # PROPOSAL 4: obv_divergence_15m requires EMA(9)/EMA(21) trend alignment
            if sig and name == "obv_divergence_15m":
                e9 = _safe_val(df["ema_9"], len(df) - 1)
                e21 = _safe_val(df["ema_21"], len(df) - 1)
                if _nan(e9) or _nan(e21):
                    sig = None
                elif sig["side"] == "long" and not (e9 > e21):
                    log.debug("obv_divergence_15m LONG suppressed -- EMA9 not > EMA21")
                    sig = None
                elif sig["side"] == "short" and not (e9 < e21):
                    log.debug("obv_divergence_15m SHORT suppressed -- EMA9 not < EMA21")
                    sig = None
            if sig:
                sig["strategy"] = name
                sig["reason"] = f"{sdef['label']} signal"
                signals.append(sig)
        except Exception as exc:
            log.error("Strategy %s error: %s", name, exc)

    if not signals:
        return

    # Pick first signal
    signal = signals[0]

    # Pre-trade checklist
    allowed, reason = can_open_trade(signal["strategy"], signal["side"])
    if not allowed:
        return

    # AI Market Brain gate
    strat_name = signal["strategy"]
    approved, confidence, ai_reason = await ai_market_analysis(
        strat_name, signal["side"], signal["entry"],
        signal_reason=signal.get("reason", "")
    )

    # --- AI BRAIN RULES ---
    # Rule 1: Confidence 6+ = AI brain is INFORMATIONAL ONLY, cannot block the trade
    # Rule 2: Confidence 5 and below = AI can veto but ONLY for technical reasons
    # Rule 3: 3+ consecutive rejections from same strategy = informational only for 24h
    force_approve = False
    strat_def = ALL_STRATEGY_DEFS.get(strat_name, {})
    label = strat_def.get("label", strat_name)

    if confidence >= 6:
        force_approve = True
        if not approved:
            log.info("AI Brain confidence %d >= 6 -- informational only, trade proceeds for %s", confidence, strat_name)

    # Consecutive rejection override
    if not approved and not force_approve:
        _ai_rejection_streak[strat_name] = _ai_rejection_streak.get(strat_name, 0) + 1
        if _ai_rejection_streak[strat_name] >= 3 and strat_name not in _ai_rejection_override_until:
            _ai_rejection_override_until[strat_name] = time.time() + 86400
            log.info("AI Brain override activated for %s -- 3+ consecutive rejections, informational only for 24h", label)
    elif approved or force_approve:
        _ai_rejection_streak[strat_name] = 0

    if not approved and not force_approve:
        override_until = _ai_rejection_override_until.get(strat_name, 0)
        if time.time() < override_until:
            force_approve = True
            confidence = max(confidence, 7)
            log.info("AI Brain override active for %s -- treating as informational, forcing APPROVE", strat_name)

    if not approved and not force_approve:
        log.info("AI Market Brain REJECTED %s %s (confidence %d): %s",
                 strat_name, signal["side"], confidence, ai_reason)
        now = time.time()
        log.info("AI Brain REJECTED %s %s @ $%.2f (confidence %d): %s",
                 label, signal['side'], signal['entry'], confidence, ai_reason[:200])
        return
    else:
        if force_approve and not approved:
            approved = True
            log.info("AI Brain overridden for %s -- proceeding with trade (confidence %d)", strat_name, confidence)

    # Execute
    max_lev = signal.get("max_leverage")
    await execute_trade(
        signal["strategy"], signal["side"], signal["entry"],
        signal["sl"], signal["tp"],
        confidence=confidence,
        max_leverage=max_lev,
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

            # 3. Macro tighten
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
                    if side == "long" and sl < entry:
                        pos["sl"] = entry
                        await tg_send(f"\u26a0\ufe0f <b>CRITICAL macro</b> -- SL tightened to breakeven ${entry:,.0f}")
                    elif side == "short" and sl > entry:
                        pos["sl"] = entry
                        await tg_send(f"\u26a0\ufe0f <b>CRITICAL macro</b> -- SL tightened to breakeven ${entry:,.0f}")

            elif "RISK_LEVEL: HIGH" in macro_text:
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

            # 4. Intelligent early exit
            await _check_intelligent_exit(pos, price)

            # 5. Max hold time
            strategy = pos["strategy"]
            strat_def = ALL_STRATEGY_DEFS.get(strategy, {})
            max_hold_min = strat_def.get("max_hold", 480)
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

            # 1m boundary
            current_1m = int(current_ts // 60) * 60
            new_1m = current_1m > state.last_1m_candle_ts

            # 5m boundary
            current_5m = int(current_ts // 300) * 300
            new_5m = current_5m > state.last_5m_candle_ts

            # 15m boundary
            current_15m = int(current_ts // 900) * 900
            new_15m = current_15m > state.last_15m_candle_ts

            # 1h boundary
            current_1h = int(current_ts // 3600) * 3600
            new_1h = current_1h > state.last_1h_candle_ts

            if new_1m:
                state.last_1m_candle_ts = current_1m
            if new_5m:
                state.last_5m_candle_ts = current_5m
            if new_15m:
                state.last_15m_candle_ts = current_15m
            if new_1h:
                state.last_1h_candle_ts = current_1h

            # Run strategies on appropriate boundaries
            if new_1m or new_5m or new_15m or new_1h:
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

            try:
                from copy_engine import get_copy_status
                cs = get_copy_status()
                copy_text = f"Tracking {cs['tracked_traders']} traders | Copied: {cs['trades_executed']} | Skipped: {cs['trades_skipped']}"
            except Exception:
                copy_text = "copy engine status unavailable"

            log.info(
                "=== PERIODIC STATUS (v7 COPY ENGINE) ===\n"
                "  Regime: %s | HTF Bias: %s (%s) | BTC: $%.2f | Mode: %s\n"
                "  Copy engine: %s\n"
                "  Trades today: %d/%d | Losses: %d/%d | PnL: $%.2f\n"
                "  Position: %s\n"
                "  Equity: $%.2f | Paused: %s",
                state.regime.value, state.htf_bias, state.htf_bias_strength,
                state.last_btc_price, state.mode,
                copy_text,
                state.trades_today, MAX_TRADES_PER_DAY,
                state.losses_today, MAX_LOSSES_PER_DAY,
                state.daily_pnl,
                pos_text,
                state.live_equity,
                "YES" if state.paused else "NO",
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
                    log.info("Macro risk HIGH detected -- trading continues, AI Brain will factor in. %s", result[:200])
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
# Paginated Data Downloader
# ---------------------------------------------------------------------------

async def download_historical_data(timeframe: str, days: int = 90) -> pd.DataFrame:
    """Download N days of data, paginating in chunks. Log progress to Telegram."""
    if not hl_info:
        return pd.DataFrame()

    tf_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                  "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}
    interval_sec = tf_seconds.get(timeframe, 60)
    chunk_candles = 5000
    chunk_ms = chunk_candles * interval_sec * 1000

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 3600 * 1000)

    all_rows = []
    cursor = start_ms
    chunk_num = 0
    total_chunks = max(1, int((now_ms - start_ms) / chunk_ms) + 1)

    while cursor < now_ms:
        end = min(cursor + chunk_ms, now_ms)
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda s=cursor, e=end: hl_info.candles_snapshot("BTC", timeframe, s, e)
            )
            if raw:
                for c in raw:
                    all_rows.append({
                        "timestamp": pd.to_datetime(c["t"], unit="ms"),
                        "open": float(c["o"]),
                        "high": float(c["h"]),
                        "low": float(c["l"]),
                        "close": float(c["c"]),
                        "volume": float(c["v"]),
                    })
        except Exception as exc:
            log.error("Download chunk error (%s, chunk %d): %s", timeframe, chunk_num, exc)

        chunk_num += 1
        cursor = end
        if chunk_num % 5 == 0:
            pct = min(100, int(chunk_num / total_chunks * 100))
            log.info("Downloading %s: %d%% (%d candles so far)", timeframe, pct, len(all_rows))
        await asyncio.sleep(0.2)  # Rate limit

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    log.info("Downloaded %d %s candles (%d days)", len(df), timeframe, days)
    return df


# ---------------------------------------------------------------------------
# 90-Day Backtest Engine
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


def _calc_backtest_stats(strategy: str, trades: list) -> dict:
    """Calculate backtest statistics for a strategy."""
    if not trades:
        return {
            "strategy": strategy, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_r": 0.0, "max_drawdown": 0.0,
            "max_consec_losses": 0, "profit_factor": 0.0, "passed": False,
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    total_pnl = sum(t["pnl"] for t in trades)
    risk_per_trade = ACCOUNT_CAPITAL * TARGET_RISK_PCT  # $7.40

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

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in trades:
        if t["pnl"] <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Max drawdown as percentage of account
    max_dd_pct = max_dd / ACCOUNT_CAPITAL * 100

    # Profit factor = gross profit / gross loss
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # PASS criteria -- profit factor >= 1.3 AND positive PnL (+ min 20 trades, max DD < 30%)
    passed = (
        total_pnl > 0 and
        profit_factor >= 1.3 and
        len(trades) >= 20 and
        max_dd_pct < 30
    )

    return {
        "strategy": strategy, "total_trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 4), "total_pnl": round(total_pnl, 2),
        "avg_r": round(avg_r, 2), "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "max_consec_losses": max_consec,
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "passed": passed,
        "trades": trades[-10:],
    }


def _is_session_candle(ts) -> bool:
    """Check if a candle timestamp falls in London (07:00-11:00) or NY (13:00-17:00) sessions."""
    if hasattr(ts, "hour"):
        h = ts.hour
        return (7 <= h < 11) or (13 <= h < 17)
    return False


def _is_weekend_candle(ts) -> bool:
    """Check if a candle falls on Saturday or Sunday."""
    if hasattr(ts, "weekday"):
        return ts.weekday() >= 5
    return False


async def run_full_backtest():
    """Phase 1-3: Download 90 days, test all 50 strategies, report results."""
    log.info("Starting 90-day full backtest...")
    log.info("Research phase starting -- downloading 90 days of data...")

    # Download data for each timeframe
    data = {}
    for tf in ["5m", "15m", "1h"]:
        log.info("Downloading %s data...", tf)
        data[tf] = await download_historical_data(tf, 90)
        if not data[tf].empty:
            log.info("%s: %d candles downloaded", tf, len(data[tf]))
        else:
            log.warning("%s: No data received", tf)

    # Skip 1m download -- too much data, and most strategies use 5m+
    data["1m"] = data.get("5m", pd.DataFrame())  # Use 5m as proxy for 1m strategies

    # Download funding rate history for perp-native strategies
    log.info("Downloading funding rate history...")
    funding_df = await fetch_funding_rate_history(90)
    funding_rates_by_ts = {}
    if not funding_df.empty:
        log.info("Funding: %d rate entries downloaded", len(funding_df))
        for _, row in funding_df.iterrows():
            # Index by hour for lookup during backtest
            ts_key = row["timestamp"].floor("8h")  # Funding settles every 8h
            funding_rates_by_ts[ts_key] = row["funding_rate"]
    else:
        log.warning("Funding history: No data (will use 0)")

    # Precompute indicators
    log.info("Precomputing indicators...")
    for tf in data:
        if not data[tf].empty and len(data[tf]) >= 55:
            try:
                data[tf] = precompute_indicators(data[tf])
            except Exception as exc:
                log.error("Precompute error for %s: %s", tf, exc)

    # Precompute 1H bias for backtest (EMA50 slope + RSI)
    df_1h = data.get("1h", pd.DataFrame())
    bias_by_hour = {}
    if not df_1h.empty and len(df_1h) >= 55:
        ema50_bt = calc_ema(df_1h["close"], 50)
        rsi_bt = calc_rsi(df_1h["close"], 14)
        for idx in range(55, len(df_1h)):
            ts = df_1h["timestamp"].iloc[idx]
            ema_now = float(ema50_bt.iloc[idx])
            ema_prev = float(ema50_bt.iloc[max(0, idx - 5)])
            rsi_val = float(rsi_bt.iloc[idx])
            if np.isnan(ema_now) or np.isnan(ema_prev) or np.isnan(rsi_val):
                bias_by_hour[ts.floor("h")] = "neutral"
                continue
            slope_pct = (ema_now - ema_prev) / ema_prev * 100 if ema_prev > 0 else 0
            if slope_pct > 0 and rsi_val > 50:
                bias_by_hour[ts.floor("h")] = "bullish"
            elif slope_pct < 0 and rsi_val < 50:
                bias_by_hour[ts.floor("h")] = "bearish"
            else:
                bias_by_hour[ts.floor("h")] = "neutral"

    # Test each strategy
    results = {}
    passed_names = []
    failed_names = []

    log.info("Testing %d strategies...", len(ALL_STRATEGY_DEFS))

    for name, sdef in ALL_STRATEGY_DEFS.items():
        tf = sdef["tf"]
        df = data.get(tf, pd.DataFrame())
        if df.empty or len(df) < 100:
            results[name] = _calc_backtest_stats(name, [])
            failed_names.append(name)
            continue

        signal_func = sdef["signal_func"]
        max_bars = max(12, sdef["max_hold"] // 5)  # Convert hold time to bars

        # Walk through data
        trades = []
        daily_trade_count = {}
        prev_funding = 0.0
        i = 55

        while i < len(df) - max_bars:
            ts = df["timestamp"].iloc[i]

            # Skip weekends
            if _is_weekend_candle(ts):
                i += 1
                continue

            # Sessions only
            if not _is_session_candle(ts):
                i += 1
                continue

            # Max 4 trades per day
            day_key = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            if daily_trade_count.get(day_key, 0) >= MAX_TRADES_PER_DAY:
                i += 1
                continue

            # Look up funding rate and bias for this candle
            ts_8h = ts.floor("8h") if hasattr(ts, "floor") else ts
            ts_1h = ts.floor("h") if hasattr(ts, "floor") else ts
            current_funding = funding_rates_by_ts.get(ts_8h, 0.0)
            current_bias = bias_by_hour.get(ts_1h, "neutral")

            extras = {
                "funding_rate": current_funding,
                "prev_funding_rate": prev_funding,
                "funding_pctl": 50,  # Approximate in backtest
                "htf_bias": current_bias,
                "top_trader_bias": "NEUTRAL",
            }
            prev_funding = current_funding

            try:
                sig = signal_func(df, i, extras)
            except Exception:
                i += 1
                continue

            if sig:
                # MANDATORY BIAS FILTER in backtest
                if current_bias == "bullish" and sig["side"] == "short":
                    i += 1
                    continue
                if current_bias == "bearish" and sig["side"] == "long":
                    i += 1
                    continue

                # Validate TP direction
                if sig["side"] == "long" and sig["tp"] <= sig["entry"]:
                    i += 1
                    continue
                if sig["side"] == "short" and sig["tp"] >= sig["entry"]:
                    i += 1
                    continue

                # Enforce minimum SL 0.5%
                sl_dist = abs(sig["entry"] - sig["sl"])
                min_sl = sig["entry"] * 0.005
                if sl_dist < min_sl:
                    sl_dist = min_sl
                    if sig["side"] == "long":
                        sig["sl"] = round(sig["entry"] - sl_dist)
                    else:
                        sig["sl"] = round(sig["entry"] + sl_dist)

                size = (ACCOUNT_CAPITAL * TARGET_RISK_PCT) / sl_dist if sl_dist > 0 else 0
                if size <= 0:
                    i += 1
                    continue

                result = _simulate_trade_forward(df, i, sig["side"], sig["entry"],
                                                  sig["sl"], sig["tp"], size, max_bars)
                if result:
                    trades.append(result)
                    daily_trade_count[day_key] = daily_trade_count.get(day_key, 0) + 1
                    i += result.get("bars_held", 1) + 1
                    continue
            i += 1

        stats = _calc_backtest_stats(name, trades)
        results[name] = stats

        if stats["passed"]:
            passed_names.append(name)
        else:
            failed_names.append(name)

    # Save results
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        d = os.path.dirname(BACKTEST_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(BACKTEST_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        # Save failed strategies
        with open(FAILED_STRATEGIES_FILE, "w") as f:
            json.dump({"failed": failed_names, "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    except Exception as exc:
        log.error("Backtest save error: %s", exc)

    # Report results
    msg_lines = [f"\U0001f4ca <b>BACKTEST RESULTS (90 days)</b>\n"]
    msg_lines.append(f"<b>PASSED ({len(passed_names)}):</b>")
    for name in passed_names[:15]:
        s = results[name]
        msg_lines.append(f"  \u2705 {name}: WR {s['win_rate']*100:.0f}% | {s['total_trades']} trades | ${s['total_pnl']:+,.0f}")
    if len(passed_names) > 15:
        msg_lines.append(f"  ... and {len(passed_names) - 15} more")
    msg_lines.append(f"\n<b>FAILED ({len(failed_names)}):</b> {len(failed_names)} strategies did not meet criteria")
    log.info("Backtest results: %d passed, %d failed", len(passed_names), len(failed_names))

    return results, passed_names, failed_names


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
    for name in state.active_strategies:
        m = state.metrics.get(name, {})
        tc = m.get("trade_count", 0)
        wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
        metrics_text += f"  {name}: {tc} trades, WR {wr:.0f}%, PnL ${m.get('current_pnl', 0):+,.2f}\n"

    # Backtest results
    bt_text = ""
    for name in state.active_strategies:
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
                f"90-DAY BACKTEST RESULTS:\n{bt_text}\n"
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

    for name in state.active_strategies[:10]:
        m = state.metrics.get(name, {})
        tc = m.get("trade_count", 0)
        wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
        paused = state.strategy_paused.get(name, False)
        status = "PAUSED" if paused else "Active"
        summary_lines.append(
            f"  <b>{name}</b> [{status}]: WR {wr:.0f}% ({tc} trades) | PnL ${m.get('current_pnl', 0):+,.2f}"
        )

    if len(state.active_strategies) > 10:
        summary_lines.append(f"  ... and {len(state.active_strategies) - 10} more active strategies")

    summary_lines.append(f"\n<b>Today:</b> {len(today_trades)} trades | PnL ${state.daily_pnl:+,.2f}")
    summary_lines.append(f"\n<b>Review:</b>\n{sanitize_html(review_text[:1500])}")
    summary_lines.append(f"\n\u2753 <b>Use /approve to apply changes or /reject to discard.</b>")

    # Store review to file only -- send via /review command on demand
    log.info("Daily self-review complete. Use /review to see results.")


async def daily_reset_scheduler():
    """Reset daily PnL and counters at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (tomorrow - now).total_seconds()
        log.info("Daily reset in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)
        state.reset_daily()
        log.info("Daily reset complete. All strategies resumed.")


# ---------------------------------------------------------------------------
# Nightly Backtest Re-evaluation
# ---------------------------------------------------------------------------

async def backtest_scheduler():
    """Run backtest daily at midnight UTC and re-evaluate strategies."""
    while True:
        now = datetime.now(timezone.utc)
        target = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        if now.hour == 0 and now.minute < 5:
            target = now.replace(minute=5, second=0, microsecond=0)
        wait_seconds = (target - now).total_seconds()
        log.info("Next backtest in %.0f seconds.", wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            bt_results, passed_names, failed_names = await run_full_backtest()
            state.backtest_results = bt_results
            state.active_strategies = passed_names
            state.backtest_complete = True

            # Initialize metrics for new strategies
            for name in passed_names:
                state._ensure_strategy_metrics(name)
                state.strategy_paused[name] = False

            save_state()

            bt_msg = [f"\U0001f4ca <b>NIGHTLY BACKTEST (90 days)</b>\n"]
            bt_msg.append(f"<b>Passed:</b> {len(passed_names)} | <b>Failed:</b> {len(failed_names)}")
            for name in passed_names[:10]:
                bt = bt_results.get(name, {})
                wr = bt.get("win_rate", 0) * 100
                total = bt.get("total_trades", 0)
                bt_msg.append(f"  \u2705 {name}: {total} trades | WR {wr:.1f}%")
            if len(passed_names) > 10:
                bt_msg.append(f"  ... and {len(passed_names) - 10} more")
            bt_msg.append(f"\n<b>Active:</b> {len(passed_names)} strategies scanning for signals.")
            log.info("Nightly backtest complete: %d passed, %d failed", len(passed_names), len(failed_names))

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
        bt_results, passed_names, failed_names = await run_full_backtest()
        state.active_strategies = passed_names
        state.backtest_results = bt_results

        for name in passed_names:
            state._ensure_strategy_metrics(name)
            state.strategy_paused[name] = False

        save_state()

        lines.append(f"Passed: {len(passed_names)} | Failed: {len(failed_names)}")
        for name in passed_names[:10]:
            bt = bt_results.get(name, {})
            wr = bt.get("win_rate", 0) * 100
            total = bt.get("total_trades", 0)
            lines.append(f"  \u2705 {name}: {total} trades | WR {wr:.1f}%")
    except Exception as exc:
        lines.append(f"  Backtest error: {sanitize_html(str(exc))}")

    # Live performance
    lines.append("\n<b>=== LIVE PERFORMANCE ===</b>")
    for name in state.active_strategies[:10]:
        m = state.metrics.get(name, {})
        tc = m.get("trade_count", 0)
        wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
        lines.append(f"  <b>{name}</b>: {tc} trades | WR {wr:.0f}% | PnL ${m.get('current_pnl', 0):+,.2f}")

    total_pnl = sum(m.get("current_pnl", 0) for m in state.metrics.values())
    lines.append(f"\n<b>Total PnL:</b> ${total_pnl:+,.2f}")
    lines.append(f"<b>Lifetime trades:</b> {state.total_trade_count}")
    lines.append(f"<b>Mode:</b> {state.mode.upper()}")
    lines.append(f"<b>Active:</b> {len(state.active_strategies)} strategies")

    intel = _get_intelligence_summary()
    if intel:
        lines.append(f"\n<b>=== TOP TRADER INTEL ===</b>\n{sanitize_html(intel)}")

    log.info("Sunday report generated. Use /review to see results.")


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
                            "Hey! I'm Vic v7 COPY ENGINE.\n\n"
                            "Commands: /resume /pause /status /equity /copystatus /closeall"
                        )
                        continue

                    if cmd == "/resume":
                        from copy_engine import copy_state
                        state.paused = False
                        if state.drawdown_killswitch_hit:
                            state.drawdown_killswitch_hit = False
                            try:
                                eq = await fetch_live_equity()
                                state.peak_equity = eq
                            except Exception:
                                pass
                        save_state()
                        eq = state.live_equity or 0
                        await tg_reply(chat_id,
                            f"▶️ Vic v7 COPY ENGINE is now LIVE.\n"
                            f"Equity: ${eq:,.2f} | Tracking {len(copy_state.traders)} top traders.\n"
                            f"Drawdown killswitch: {EQUITY_DRAWDOWN_KILLSWITCH*100:.0f}%\n"
                            f"Next trade mirrors the next position change from a tracked trader."
                        )
                        continue

                    if cmd == "/pause":
                        state.paused = True
                        save_state()
                        await tg_reply(chat_id, "⏸ Vic PAUSED. No new trades. Send /resume to restart.")
                        continue

                    if cmd == "/equity":
                        try:
                            eq = await fetch_live_equity()
                            dd = (1 - eq / state.peak_equity) * 100 if state.peak_equity > 0 else 0
                            await tg_reply(chat_id,
                                f"💰 Equity: ${eq:,.2f}\n"
                                f"Peak: ${state.peak_equity:,.2f}\n"
                                f"Drawdown: {dd:.1f}%\n"
                                f"Killswitch: {'HIT' if state.drawdown_killswitch_hit else f'armed at {EQUITY_DRAWDOWN_KILLSWITCH*100:.0f}%'}"
                            )
                        except Exception as e:
                            await tg_reply(chat_id, f"Equity check error: {e}")
                        continue

                    if cmd == "/copystatus":
                        from copy_engine import get_copy_status
                        cs = get_copy_status()
                        lines = [f"📋 <b>Copy Engine Status</b>",
                                 f"Tracked traders: {cs['tracked_traders']}",
                                 f"Trades executed: {cs['trades_executed']}",
                                 f"Trades skipped: {cs['trades_skipped']}"]
                        for t in cs.get("top_traders", [])[:5]:
                            lines.append(f"  • {t['name']}({t['address']}) — {t['active_positions']} positions")
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/status":
                        eq = state.live_equity or 0
                        pos = state.current_position
                        pos_str = f"{pos['strategy']} {pos['side']} {pos.get('coin','BTC')} @ ${pos['entry']:,.2f}" if pos else "None"
                        await tg_reply(chat_id,
                            f"🤖 <b>Vic v7 Copy Engine</b>\n"
                            f"Mode: {state.mode} | {'PAUSED' if state.paused else 'LIVE'}\n"
                            f"Equity: ${eq:,.2f} | Position: {pos_str}\n"
                            f"Trades today: {state.trades_today}/{MAX_TRADES_PER_DAY}\n"
                            f"Losses: {state.losses_today}/{MAX_LOSSES_PER_DAY}\n"
                            f"Daily PnL: ${state.daily_pnl:+,.2f}"
                        )
                        continue

                    if cmd == "/strategies":
                        lines = [f"\U0001f4ca <b>Strategy Status</b> ({len(state.active_strategies)} active / {len(ALL_STRATEGY_DEFS)} total)\n"]
                        for name in state.active_strategies[:15]:
                            m = state.metrics.get(name, {})
                            tc = m.get("trade_count", 0)
                            wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
                            paused = state.strategy_paused.get(name, False)
                            status = "PAUSED" if paused else "Active"
                            bt = state.backtest_results.get(name, {})
                            bt_wr = bt.get("win_rate", 0) * 100
                            sdef = ALL_STRATEGY_DEFS.get(name, {})
                            label = sdef.get("label", name)
                            lines.append(
                                f"<b>{label}</b> [{status}]\n"
                                f"  Live: {tc} trades | WR {wr:.0f}% | PnL ${m.get('current_pnl', 0):+,.2f}\n"
                                f"  Backtest: WR {bt_wr:.0f}%"
                            )
                        if len(state.active_strategies) > 15:
                            lines.append(f"\n... and {len(state.active_strategies) - 15} more")
                        # Show inactive count
                        inactive = len(ALL_STRATEGY_DEFS) - len(state.active_strategies)
                        if inactive > 0:
                            lines.append(f"\n{inactive} strategies failed backtest criteria")
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/backtest":
                        try:
                            if os.path.exists(BACKTEST_FILE):
                                with open(BACKTEST_FILE, "r") as f:
                                    bt = json.load(f)
                                lines = [f"\U0001f4ca <b>Backtest Results</b>\nDate: {bt.get('timestamp', 'unknown')}\n"]
                                passed_count = 0
                                for name in ALL_STRATEGY_DEFS:
                                    s = bt.get(name, {})
                                    if not isinstance(s, dict):
                                        continue
                                    passed_icon = "\u2705" if s.get("passed") else "\u274c"
                                    lines.append(
                                        f"{passed_icon} <b>{name}</b>: {s.get('total_trades',0)} trades | "
                                        f"WR {s.get('win_rate',0)*100:.1f}% | "
                                        f"Avg R {s.get('avg_r',0):+.2f}"
                                    )
                                    if s.get("passed"):
                                        passed_count += 1
                                    if len(lines) > 30:
                                        lines.append("... (truncated)")
                                        break
                                lines.append(f"\n<b>Total passed: {passed_count}/{len(ALL_STRATEGY_DEFS)}</b>")
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
                        lines = [f"\U0001f4ca <b>Per-Strategy Metrics</b> ({len(state.active_strategies)} active)\n"]
                        for n in state.active_strategies[:15]:
                            m = state.metrics.get(n, {})
                            tc = m.get("trade_count", 0)
                            wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
                            exp = (m.get("total_r_achieved", 0) / tc) if tc > 0 else 0
                            paused = state.strategy_paused.get(n, False)
                            status = "PAUSED" if paused else "Active"
                            lines.append(
                                f"<b>{n}</b> [{status}]: {tc} trades | WR {wr:.0f}% | Exp {exp:+.2f}R\n"
                                f"  PnL ${m.get('current_pnl', 0):+,.2f} | MaxDD ${m.get('max_drawdown', 0):,.2f} | "
                                f"Streak {m.get('current_losing_streak', 0)}/{m.get('max_losing_streak', 0)}"
                            )
                        if len(state.active_strategies) > 15:
                            lines.append(f"\n... and {len(state.active_strategies) - 15} more")
                        await tg_reply(chat_id, "\n".join(lines))
                        continue

                    if cmd == "/regime":
                        msg = (
                            f"\U0001f50d <b>Current Regime: {state.regime.value}</b>\n\n"
                            f"1H Bias: {state.htf_bias} ({state.htf_bias_strength})\n"
                            f"BTC: ${state.last_btc_price:,.2f}\n"
                            f"Funding: {state.current_funding_rate:.6f}\n"
                            f"Active: {len(state.active_strategies)} strategies\n"
                            f"Research: {'complete' if state.research_complete else 'pending'}"
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
    for n in state.active_strategies[:10]:
        m = state.metrics.get(n, {})
        tc = m.get("trade_count", 0)
        wr = (m.get("wins", 0) / tc * 100) if tc > 0 else 0
        metrics_lines.append(f"  {n}: {tc} trades, WR {wr:.0f}%, PnL ${m.get('current_pnl', 0):+,.2f}")

    # Get copy engine status for context
    try:
        from copy_engine import get_copy_status, copy_state
        cs = get_copy_status()
        copy_info = (
            f"Tracked traders: {cs['tracked_traders']}\n"
            f"Copy trades executed: {cs['trades_executed']} | Skipped: {cs['trades_skipped']}\n"
            f"Top traders: {', '.join(t.get('display', t.get('address','?')[:10]) for t in cs.get('top_traders', [])[:5])}"
        )
    except Exception:
        copy_info = "Copy engine status unavailable"

    system_prompt = (
        f"You are Vic v7, an AI copy-trading agent on Hyperliquid. "
        f"You mirror positions from the top 15 Hyperliquid leaderboard traders by 90-day PnL. "
        f"You do NOT run your own TA strategies — the old 58 strategies were scrapped. "
        f"Your only job is to copy winning traders' entries with proper risk management.\n\n"
        f"Account: ~$500 | Max leverage: 10x | SL: 2% of equity per trade | "
        f"Drawdown killswitch: 20%\n\n"
        f"=== LIVE STATE ===\n"
        f"BTC: ${state.last_btc_price:,.2f} | Regime: {state.regime.value} | "
        f"Bias: {state.htf_bias} | Mode: {state.mode.upper()}\n"
        f"Equity: ${state.live_equity:,.2f} | Losses today: {state.losses_today}/{MAX_LOSSES_PER_DAY}\n"
        f"PnL today: ${state.daily_pnl:+,.2f} | Lifetime: {state.total_trade_count} trades\n"
        f"Paused: {'YES' if state.paused else 'NO'}\n\n"
        f"=== COPY ENGINE ===\n{copy_info}\n\n"
        f"=== POSITION ===\n{pos_text}\n\n"
        f"RULES: Never suggest opening manual trades. Never propose TA-based entries. "
        f"You copy traders, that's it. If GG asks why you aren't trading, explain the copy engine "
        f"is waiting for a tracked trader to open/change a position. "
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
    """Close all positions. Bypasses the 60-second minimum hold time."""
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
        "bot": "Vic v6.0",
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
        "active_strategies": len(state.active_strategies),
        "total_strategies": len(ALL_STRATEGY_DEFS),
        "has_position": state.current_position is not None,
        "funding_rate": state.current_funding_rate,
        "research_complete": state.research_complete,
    }


VIC_VERSION_SHA = os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")[:7]
VIC_VERSION_TAG = "v7-copy-engine-hl-leaderboard"


@app.get("/hl-dump")
async def hl_dump():
    """Dump raw Hyperliquid user_state for debugging equity parsing."""
    try:
        loop = asyncio.get_running_loop()
        user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
        return {"raw": user_st, "type": str(type(user_st))}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/version")
async def version_check():
    """Returns the exact git commit + feature tag of the deployed code.
    Use this to verify a Railway deploy landed the expected changes.
    """
    return {
        "commit_sha": VIC_VERSION_SHA,
        "feature_tag": VIC_VERSION_TAG,
        "deployed_at": state.startup_time,
    }


@app.get("/equity")
async def equity_check():
    """Live equity check — pulls actual Hyperliquid balance in real time.
    Use this to confirm Vic is correctly seeing his working capital.
    """
    try:
        eq = await fetch_live_equity()
        # Connection is OK if we got a real number from HL (not the silent fallback)
        connection = "ok" if state.live_equity > 0 else "check_hyperliquid"
        return {
            "live_equity_usd": round(eq, 2),
            "peak_equity_usd": round(state.peak_equity, 2),
            "drawdown_from_peak_pct": round((1 - eq / state.peak_equity) * 100, 2) if state.peak_equity > 0 else 0,
            "drawdown_killswitch_hit": state.drawdown_killswitch_hit,
            "drawdown_killswitch_threshold_pct": EQUITY_DRAWDOWN_KILLSWITCH * 100,
            "hyperliquid_wallet": HL_WALLET_ADDRESS[:10] + "..." if HL_WALLET_ADDRESS else "not_set",
            "connection": connection,
        }
    except Exception as e:
        return {"error": str(e), "connection": "failed"}


@app.post("/clear-phantom-position")
async def clear_phantom_position(token: str = Query("")):
    """Clear Vic's internal current_position if Hyperliquid shows no actual position.
    Used to fix state-drift bugs where Vic thinks it has a position that doesn't exist on HL.
    """
    if token != WEBHOOK_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    try:
        loop = asyncio.get_running_loop()
        user_st = await loop.run_in_executor(None, lambda: hl_info.user_state(HL_WALLET_ADDRESS))
        hl_positions = [p for p in user_st.get("assetPositions", []) or []
                        if float(p.get("position", {}).get("szi", 0)) != 0]
        before = state.current_position
        if not hl_positions:
            state.current_position = None
            save_state()
            await tg_send(
                f"🧹 Phantom position cleared.\n"
                f"Vic's internal state had a position that HL didn't recognise. Cleared."
            )
            return {"cleared": True, "was": before, "hl_positions": 0}
        return {"cleared": False, "reason": "HL shows real positions", "hl_positions": len(hl_positions)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/status")
async def full_status():
    strategies_status = {}
    for name in ALL_STRATEGY_DEFS:
        m = state.metrics.get(name, {})
        tc = m.get("trade_count", 0)
        win_rate = (m.get("wins", 0) / tc * 100) if tc > 0 else 0.0
        strategies_status[name] = {
            "active": name in state.active_strategies,
            "paused": state.strategy_paused.get(name, False),
            "metrics": {
                "win_rate": round(win_rate, 1),
                "total_trades": tc,
                "pnl": m.get("current_pnl", 0),
                "max_drawdown": m.get("max_drawdown", 0),
            },
        }

    return {
        "bot": "Vic v6",
        "mode": state.mode,
        "base_leverage": BASE_LEVERAGE,
        "max_leverage_hard_cap": MAX_LEVERAGE_HARD_CAP,
        "account_capital_fallback": ACCOUNT_CAPITAL_FALLBACK,
        "live_equity": round(state.live_equity, 2) if state.live_equity else None,
        "peak_equity": round(state.peak_equity, 2) if state.peak_equity else None,
        "drawdown_killswitch_hit": state.drawdown_killswitch_hit,
        "min_sl_pct": MIN_SL_PCT,
        "max_risk_pct": MAX_RISK_PCT,
        "fee_per_trade_usd": FEE_PER_TRADE_USD,
        "global_paused": state.paused,
        "research_complete": state.research_complete,
        "regime": state.regime.value,
        "htf_bias": f"{state.htf_bias} ({state.htf_bias_strength})",
        "btc_price": state.last_btc_price,
        "funding_rate": state.current_funding_rate,
        "strategies": strategies_status,
        "active_strategy_count": len(state.active_strategies),
        "total_strategy_count": len(ALL_STRATEGY_DEFS),
        "active_strategies": state.active_strategies,
        "current_position": state.current_position,
        "trades_today": state.trades_today,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
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
    # Clear drawdown killswitch on manual resume AND reset peak to current equity
    # (so the killswitch measures drawdown from this resume point, not the old peak)
    if state.drawdown_killswitch_hit:
        state.drawdown_killswitch_hit = False
        try:
            eq = await fetch_live_equity()
            state.peak_equity = eq
        except Exception:
            pass
    save_state()
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
    from copy_engine import copy_monitor_loop, backtest_copy_engine, get_copy_status, refresh_leaderboard

    _required = ["HL_WALLET_ADDRESS", "HL_PRIVATE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "CLAUDE_API_KEY"]
    _missing = [v for v in _required if not os.getenv(v)]
    if _missing:
        log.error(f"MISSING REQUIRED ENV VARS: {', '.join(_missing)} -- Vic cannot start properly")

    for filepath in [JOURNAL_FILE, STATE_FILE, BACKTEST_FILE, FAILED_STRATEGIES_FILE, INTELLIGENCE_FILE, REVIEW_FILE]:
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)

    state.startup_time = datetime.now(timezone.utc).isoformat()
    log.info("Vic v7 COPY ENGINE starting up -- mode: %s", state.mode)

    state.paused = True  # Always start paused — GG sends /resume when ready
    loaded = load_state()
    state.paused = True  # Override any persisted state — safety first

    now_ts = datetime.now(timezone.utc).timestamp()
    state.last_1m_candle_ts = int(now_ts // 60) * 60
    state.last_5m_candle_ts = int(now_ts // 300) * 300
    state.last_15m_candle_ts = int(now_ts // 900) * 900
    state.last_1h_candle_ts = int(now_ts // 3600) * 3600

    init_exchange()
    await recover_orphaned_positions()

    # Fetch live equity
    equity = await fetch_live_equity()

    # Run copy-engine backtest
    bt_msg = ""
    try:
        bt = await backtest_copy_engine(days=90, equity=500.0, fee_per_trade=FEE_PER_TRADE_USD)
        state.backtest_results = bt
        state.research_complete = True
        state.backtest_complete = True
        save_state()

        bt_msg = (
            f"\n\n<b>90-day backtest (top-trader copy):</b>\n"
            f"Trades: {bt['total_trades']} | Wins: {bt['wins']} | Losses: {bt['losses']}\n"
            f"Win rate: {bt['win_rate_pct']}%\n"
            f"PnL: ${bt['total_pnl']:+,.2f} (${bt['starting_equity']:.0f} → ${bt['final_equity']:,.2f})\n"
            f"Max drawdown: ${bt['max_drawdown_usd']:,.2f} ({bt['max_drawdown_pct']}%)\n"
            f"Fee/trade: ${bt['fee_per_trade']:.2f} | Traders analysed: {bt['traders_analysed']}"
        )
    except Exception as exc:
        log.error("Copy engine backtest failed: %s", exc)
        bt_msg = f"\n\n⚠️ Backtest failed: {exc}"

    # Startup announcement to GG
    startup_msg = (
        f"🤖 <b>Vic v7 COPY ENGINE deployed</b>\n"
        f"Commit: {VIC_VERSION_SHA} | Tag: {VIC_VERSION_TAG}\n"
        f"Mode: {state.mode.upper()} | Equity: ${equity:,.2f}\n\n"
        f"<b>What changed:</b>\n"
        f"• 58 TA strategies DISABLED\n"
        f"• Copy engine: mirrors top 15 Hyperliquid traders by month PnL\n"
        f"• Position sizing: SL = 2% of equity per trade\n"
        f"• Leverage hard cap: {MAX_LEVERAGE_HARD_CAP}x\n"
        f"• Drawdown killswitch: {EQUITY_DRAWDOWN_KILLSWITCH*100:.0f}%\n"
        f"• Fee filter: ${FEE_PER_TRADE_USD*2*2:.0f} min TP"
        f"{bt_msg}\n\n"
        f"⏸ <b>PAUSED — send /resume to go live</b>"
    )
    await tg_send(startup_msg)

    # Copy engine execution adapter
    async def _copy_execute(coin, side, size, entry, sl, tp, leverage, strategy_name):
        """Adapter: maps copy_engine calls to vic's execute_trade interface."""
        if coin != "BTC":
            # For non-BTC coins, use hl_exchange directly
            try:
                loop = asyncio.get_running_loop()
                is_buy = side == "long"
                result = await loop.run_in_executor(
                    None, lambda: hl_exchange.market_open(coin, is_buy=is_buy, sz=size)
                )
                if result.get("status") != "ok":
                    raise Exception(f"Order rejected: {result}")
                # Record position
                state.current_position = {
                    "strategy": strategy_name, "side": side, "entry": entry,
                    "sl": sl, "tp": tp, "size": size, "coin": coin,
                    "leverage": leverage, "opened_at": datetime.now(timezone.utc).isoformat(),
                }
                state.trades_today += 1
                save_state()
                return True
            except Exception as e:
                log.error(f"Copy execute {coin} failed: {e}")
                return False
        else:
            # BTC uses existing execute_trade with full safety
            await execute_trade(strategy_name, side, entry, sl, tp, confidence=7, max_leverage=leverage)
            return state.current_position is not None

    # Start background tasks (copy engine replaces strategy_monitor_loop)
    asyncio.create_task(copy_monitor_loop(
        equity_fn=fetch_live_equity,
        execute_fn=_copy_execute,
        tg_fn=tg_send,
        fee_filter=FEE_PER_TRADE_USD * 2 * 2,
        max_leverage=MAX_LEVERAGE_HARD_CAP,
        check_can_trade=can_open_trade,
        is_paused=lambda: state.paused,
    ))
    asyncio.create_task(regime_update_loop())
    asyncio.create_task(funding_rate_loop())
    asyncio.create_task(position_monitor_loop())
    asyncio.create_task(news_sentiment_monitor())
    asyncio.create_task(daily_summary_scheduler())
    asyncio.create_task(daily_reset_scheduler())
    asyncio.create_task(periodic_status_log())
    asyncio.create_task(telegram_polling_watchdog())
    asyncio.create_task(intelligence_loop())

    # Add copy status to /status endpoint
    @app.get("/copy-status")
    async def copy_status():
        return get_copy_status()

    log.info("Vic v7 COPY ENGINE — all background tasks started.")


@app.on_event("shutdown")
async def shutdown():
    log.info("Vic shutting down -- saving state.")
    save_state()
    close_exchange()
