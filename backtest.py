"""
Vic Backtest — BTC/USDC Perpetual Futures Strategy Backtester
Downloads 6 months of candle data from Hyperliquid public API,
runs the same 4 strategies as vic.py, and reports results.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "BTC"
LEVERAGE = 5
CAPITAL_PER_STRATEGY = 200.0
RISK_PER_TRADE = 20.0
TP_PER_TRADE = 60.0
MAX_DAILY_LOSS = 40.0
BB_SQUEEZE_THRESHOLD = 0.02
MONTHS_BACK = 6

DATA_DIR = Path(__file__).parent / "cache"
RESULTS_PATH = Path(__file__).parent / "backtest_results.json"

HL_CANDLES = "https://api.hyperliquid.xyz/info"


# ---------------------------------------------------------------------------
# Indicators (same as vic.py)
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


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP — resets daily (by UTC date)."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = typical_price * df["volume"]
    cum_tp_vol = df.groupby("date")["tp_vol"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()
    return cum_tp_vol / cum_vol


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

BAR_TO_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}


def download_candles(bar: str, months: int = MONTHS_BACK) -> pd.DataFrame:
    """Download candles from Hyperliquid public API with pagination and caching."""
    cache_file = DATA_DIR / f"btc_usdc_swap_{bar}_{months}m.csv"

    if cache_file.exists():
        print(f"  [cache] Loading {bar} data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        print(f"  [cache] {len(df)} candles loaded ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df

    DATA_DIR.mkdir(exist_ok=True)

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=months * 30)).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    interval = BAR_TO_INTERVAL.get(bar, bar)
    all_candles = []
    max_retries = 3
    page = 0

    print(f"  Downloading {bar} candles from Hyperliquid (going back {months} months)...")

    with httpx.Client(timeout=30) as client:
        cursor_end = end_ms
        while cursor_end > start_ms:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": SYMBOL,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": cursor_end,
                },
            }

            success = False
            for attempt in range(max_retries):
                try:
                    resp = client.post(HL_CANDLES, json=payload)
                    data = resp.json()
                    success = True
                    break
                except Exception as exc:
                    print(f"    API error (attempt {attempt+1}): {exc}")
                    time.sleep(2)

            if not success or not data:
                break

            rows = data
            if not rows:
                break

            all_candles.extend(rows)
            oldest_ts = rows[0]["t"]
            page += 1

            if page % 10 == 0:
                oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
                print(f"    Page {page}: {len(all_candles)} candles, oldest: {oldest_dt.strftime('%Y-%m-%d %H:%M')}")

            # Move cursor before the oldest candle we received
            cursor_end = oldest_ts - 1
            if len(rows) < 500:
                break

            time.sleep(0.12)

    if not all_candles:
        print(f"  WARNING: No candles downloaded for {bar}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Filter to our date range
    cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    df.to_csv(cache_file, index=False)
    print(f"  Downloaded {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
    print(f"  Cached to {cache_file}")
    return df


# ---------------------------------------------------------------------------
# Trade simulator
# ---------------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, name: str):
        self.name = name
        self.capital = CAPITAL_PER_STRATEGY
        self.equity_curve = [CAPITAL_PER_STRATEGY]
        self.trades = []
        self.daily_pnl = {}  # date -> pnl
        self.current_date = None
        self.daily_loss_today = 0.0
        self.paused_today = False
        self.position = None  # {side, entry, sl, tp, open_idx}
        self.peak_equity = CAPITAL_PER_STRATEGY

    def _update_day(self, dt):
        d = dt.date() if hasattr(dt, "date") else dt
        if d != self.current_date:
            self.current_date = d
            self.daily_loss_today = 0.0
            self.paused_today = False
            if d not in self.daily_pnl:
                self.daily_pnl[d] = 0.0

    def _calc_sl_tp(self, entry: float, side: str):
        notional = CAPITAL_PER_STRATEGY * LEVERAGE
        size_btc = notional / entry
        sl_dist = RISK_PER_TRADE / size_btc
        tp_dist = TP_PER_TRADE / size_btc
        if side == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return round(sl, 2), round(tp, 2)

    def open_trade(self, side: str, entry: float, timestamp, bar_idx: int):
        if self.position is not None or self.paused_today:
            return
        self._update_day(timestamp)
        if self.paused_today:
            return
        sl, tp = self._calc_sl_tp(entry, side)
        self.position = {
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "open_idx": bar_idx,
            "open_time": timestamp,
        }

    def check_exit(self, high: float, low: float, close: float, timestamp, bar_idx: int):
        """Check if SL or TP was hit during this bar. Returns True if closed."""
        if self.position is None:
            return False

        self._update_day(timestamp)
        pos = self.position
        notional = CAPITAL_PER_STRATEGY * LEVERAGE
        size_btc = notional / pos["entry"]

        exit_price = None
        reason = None

        if pos["side"] == "long":
            if low <= pos["sl"]:
                exit_price = pos["sl"]
                reason = "SL"
            elif high >= pos["tp"]:
                exit_price = pos["tp"]
                reason = "TP"
        else:  # short
            if high >= pos["sl"]:
                exit_price = pos["sl"]
                reason = "SL"
            elif low <= pos["tp"]:
                exit_price = pos["tp"]
                reason = "TP"

        if exit_price is not None:
            if pos["side"] == "long":
                pnl = (exit_price - pos["entry"]) * size_btc
            else:
                pnl = (pos["entry"] - exit_price) * size_btc
            pnl = round(pnl, 2)

            self.capital += pnl
            self.equity_curve.append(self.capital)
            self.peak_equity = max(self.peak_equity, self.capital)

            d = timestamp.date() if hasattr(timestamp, "date") else timestamp
            self.daily_pnl[d] = self.daily_pnl.get(d, 0.0) + pnl
            self.daily_loss_today += pnl

            self.trades.append({
                "side": pos["side"],
                "entry": pos["entry"],
                "exit": exit_price,
                "pnl": pnl,
                "reason": reason,
                "open_time": str(pos["open_time"]),
                "close_time": str(timestamp),
                "bars_held": bar_idx - pos["open_idx"],
            })

            self.position = None

            # Check daily loss limit
            if self.daily_loss_today <= -MAX_DAILY_LOSS:
                self.paused_today = True

            return True
        return False

    def force_close(self, close: float, timestamp, bar_idx: int):
        """Force close at end of data."""
        if self.position is None:
            return
        pos = self.position
        notional = CAPITAL_PER_STRATEGY * LEVERAGE
        size_btc = notional / pos["entry"]
        if pos["side"] == "long":
            pnl = (close - pos["entry"]) * size_btc
        else:
            pnl = (pos["entry"] - close) * size_btc
        pnl = round(pnl, 2)
        self.capital += pnl
        self.equity_curve.append(self.capital)
        d = timestamp.date() if hasattr(timestamp, "date") else timestamp
        self.daily_pnl[d] = self.daily_pnl.get(d, 0.0) + pnl
        self.trades.append({
            "side": pos["side"],
            "entry": pos["entry"],
            "exit": close,
            "pnl": pnl,
            "reason": "EOD",
            "open_time": str(pos["open_time"]),
            "close_time": str(timestamp),
            "bars_held": bar_idx - pos["open_idx"],
        })
        self.position = None

    def report(self) -> dict:
        if not self.trades:
            return {
                "strategy": self.name,
                "total_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "total_pnl": 0,
                "best_day": 0,
                "worst_day": 0,
                "sharpe_ratio": 0,
                "final_capital": CAPITAL_PER_STRATEGY,
            }

        wins = [t["pnl"] for t in self.trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trades if t["pnl"] <= 0]

        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = abs(float(dd.min()))

        total_pnl = round(self.capital - CAPITAL_PER_STRATEGY, 2)

        # Daily stats
        daily_vals = list(self.daily_pnl.values())
        best_day = max(daily_vals) if daily_vals else 0
        worst_day = min(daily_vals) if daily_vals else 0

        # Sharpe ratio (annualized, daily returns)
        if len(daily_vals) > 1:
            daily_arr = np.array(daily_vals)
            sharpe = (np.mean(daily_arr) / np.std(daily_arr)) * np.sqrt(252) if np.std(daily_arr) > 0 else 0
        else:
            sharpe = 0

        return {
            "strategy": self.name,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "profit_factor": round(float(profit_factor), 2),
            "max_drawdown": round(max_dd, 2),
            "total_pnl": total_pnl,
            "best_day": round(float(best_day), 2),
            "worst_day": round(float(worst_day), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "final_capital": round(self.capital, 2),
        }


# ---------------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------------

def backtest_ema_crossover(df_1m: pd.DataFrame) -> dict:
    """Strategy 2: EMA Crossover — 9/21 EMA cross on 1m, RSI filter."""
    print("\n[Strategy 2] EMA Crossover (9/21 EMA, 1m candles)...")
    engine = BacktestEngine("ema_crossover")

    df = df_1m.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi"] = rsi(df["close"], 14)

    total = len(df)
    for i in range(22, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        engine._update_day(ts)

        # Check exit first
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        prev_ema9 = df["ema9"].iloc[i - 1]
        prev_ema21 = df["ema21"].iloc[i - 1]
        curr_ema9 = df["ema9"].iloc[i]
        curr_ema21 = df["ema21"].iloc[i]
        rsi_val = df["rsi"].iloc[i]

        if np.isnan(rsi_val) or np.isnan(curr_ema9):
            continue

        # Long: 9 EMA crosses above 21 EMA, RSI < 70
        if prev_ema9 <= prev_ema21 and curr_ema9 > curr_ema21 and rsi_val < 70:
            engine.open_trade("long", close, ts, i)
        # Short: 9 EMA crosses below 21 EMA, RSI > 30
        elif prev_ema9 >= prev_ema21 and curr_ema9 < curr_ema21 and rsi_val > 30:
            engine.open_trade("short", close, ts, i)

        if i % 50000 == 0:
            print(f"    Processed {i}/{total} bars...")

    # Force close any remaining position
    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


def backtest_rsi_divergence(df_5m: pd.DataFrame) -> dict:
    """Strategy 3: RSI Divergence on 5m candles."""
    print("\n[Strategy 3] RSI Divergence (5m candles)...")
    engine = BacktestEngine("rsi_divergence")

    df = df_5m.copy()
    df["rsi"] = rsi(df["close"], 14)

    total = len(df)
    for i in range(30, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        # Look at last 20 bars (same as vic.py)
        start_idx = max(0, i - 19)
        close_vals = df["close"].values[start_idx:i + 1]
        rsi_vals = df["rsi"].values[start_idx:i + 1]

        if len(close_vals) < 5 or np.any(np.isnan(rsi_vals)):
            continue

        # Find local lows
        price_lows = []
        for j in range(1, len(close_vals) - 1):
            if close_vals[j] < close_vals[j - 1] and close_vals[j] < close_vals[j + 1]:
                price_lows.append((j, close_vals[j], rsi_vals[j]))

        # Find local highs
        price_highs = []
        for j in range(1, len(close_vals) - 1):
            if close_vals[j] > close_vals[j - 1] and close_vals[j] > close_vals[j + 1]:
                price_highs.append((j, close_vals[j], rsi_vals[j]))

        # Bullish divergence: price lower low, RSI higher low
        if len(price_lows) >= 2:
            prev_low = price_lows[-2]
            curr_low = price_lows[-1]
            if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
                engine.open_trade("long", close, ts, i)
                continue

        # Bearish divergence: price higher high, RSI lower high
        if len(price_highs) >= 2:
            prev_high = price_highs[-2]
            curr_high = price_highs[-1]
            if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
                engine.open_trade("short", close, ts, i)

        if i % 10000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


def backtest_vwap_bounce(df_1m: pd.DataFrame) -> dict:
    """Strategy 4: VWAP Bounce on 1m candles."""
    print("\n[Strategy 4] VWAP Bounce (1m candles)...")
    engine = BacktestEngine("vwap_bounce")

    df = df_1m.copy()
    df["vwap"] = calc_vwap(df)

    total = len(df)
    for i in range(2, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        open_ = df["open"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        vwap_val = df["vwap"].iloc[i]
        if np.isnan(vwap_val):
            continue

        price = close
        distance_pct = abs(price - vwap_val) / vwap_val
        if distance_pct > 0.001:
            continue

        is_bullish = close > open_
        is_bearish = close < open_

        prev_low = df["low"].iloc[i - 1]
        prev_high = df["high"].iloc[i - 1]

        # Long: prev bar low touched VWAP, current bullish candle above VWAP
        if prev_low <= vwap_val and is_bullish and price > vwap_val:
            engine.open_trade("long", price, ts, i)
        # Short: prev bar high touched VWAP, current bearish candle below VWAP
        elif prev_high >= vwap_val and is_bearish and price < vwap_val:
            engine.open_trade("short", price, ts, i)

        if i % 50000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


def backtest_bb_squeeze(df_5m: pd.DataFrame) -> dict:
    """Strategy 5: Bollinger Band Squeeze Breakout on 5m candles."""
    print("\n[Strategy 5] BB Squeeze (5m candles)...")
    engine = BacktestEngine("bb_squeeze")

    df = df_5m.copy()
    mid, upper, lower, bandwidth = bollinger_bands(df["close"], 20, 2.0)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_bw"] = bandwidth

    total = len(df)
    for i in range(25, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        prev_bw = df["bb_bw"].iloc[i - 1]
        curr_bw = df["bb_bw"].iloc[i]

        if np.isnan(prev_bw) or np.isnan(curr_bw):
            continue

        bb_upper = df["bb_upper"].iloc[i]
        bb_lower = df["bb_lower"].iloc[i]

        # Squeeze: prev bandwidth below threshold, current expanding
        if prev_bw < BB_SQUEEZE_THRESHOLD and curr_bw > prev_bw:
            if close > bb_upper:
                engine.open_trade("long", close, ts, i)
            elif close < bb_lower:
                engine.open_trade("short", close, ts, i)

        if i % 10000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: list):
    """Print a clean ASCII table."""
    print("\n" + "=" * 100)
    print("VIC BACKTEST RESULTS — BTC/USDT PERP | 5x Leverage | $200/strategy | $20 risk | 1:3 R/R")
    print(f"Period: ~{MONTHS_BACK} months")
    print("=" * 100)

    header = f"{'Strategy':<20} {'Trades':>7} {'Win%':>7} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>7} {'MaxDD':>8} {'PnL':>10} {'Sharpe':>7}"
    print(header)
    print("-" * 100)

    total_pnl = 0
    total_trades = 0
    for r in results:
        name = r["strategy"]
        total_pnl += r["total_pnl"]
        total_trades += r["total_trades"]
        print(
            f"{name:<20} {r['total_trades']:>7} {r['win_rate']:>6.1f}% "
            f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} "
            f"{r['profit_factor']:>6.2f}x ${r['max_drawdown']:>7.2f} "
            f"${r['total_pnl']:>+9.2f} {r['sharpe_ratio']:>7.2f}"
        )

    print("-" * 100)
    print(f"{'TOTAL':<20} {total_trades:>7} {'':>7} {'':>8} {'':>8} {'':>7} {'':>8} ${total_pnl:>+9.2f}")
    print("=" * 100)

    # Best/worst day per strategy
    print("\nDaily Extremes:")
    for r in results:
        print(f"  {r['strategy']:<20} Best day: ${r['best_day']:>+8.2f} | Worst day: ${r['worst_day']:>+8.2f} | Final capital: ${r['final_capital']:>8.2f}")
    print()


def generate_telegram_message(results: list) -> str:
    """Generate a Telegram-formatted summary."""
    total_pnl = sum(r["total_pnl"] for r in results)
    total_trades = sum(r["total_trades"] for r in results)

    lines = [
        f"<b>VIC BACKTEST RESULTS</b>",
        f"BTC/USDT Perp | 5x | {MONTHS_BACK}mo",
        f"$200/strategy | $20 risk | 1:3 R/R\n",
    ]

    strategy_emojis = {
        "ema_crossover": "2\ufe0f\u20e3",
        "rsi_divergence": "3\ufe0f\u20e3",
        "vwap_bounce": "4\ufe0f\u20e3",
        "bb_squeeze": "5\ufe0f\u20e3",
    }

    for r in results:
        emoji = strategy_emojis.get(r["strategy"], "")
        pnl_emoji = "\U0001f4b0" if r["total_pnl"] >= 0 else "\U0001f4b8"
        lines.append(
            f"{emoji} <b>{r['strategy']}</b>\n"
            f"   {r['total_trades']} trades | {r['win_rate']}% win | PF {r['profit_factor']}x\n"
            f"   {pnl_emoji} PnL: <b>${r['total_pnl']:+,.2f}</b> | DD: ${r['max_drawdown']:,.2f}"
        )

    pnl_icon = "\U0001f4b0" if total_pnl >= 0 else "\U0001f4b8"
    lines.append(f"\n<b>TOTAL:</b> {total_trades} trades | {pnl_icon} <b>${total_pnl:+,.2f}</b>")

    best = max(results, key=lambda r: r["total_pnl"])
    worst = min(results, key=lambda r: r["total_pnl"])
    lines.append(f"\n<b>Best:</b> {best['strategy']} (${best['total_pnl']:+,.2f})")
    lines.append(f"<b>Worst:</b> {worst['strategy']} (${worst['total_pnl']:+,.2f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=" * 60)
    print("VIC BACKTESTER")
    print(f"Symbol: BTC/USDT SWAP | Period: {MONTHS_BACK} months")
    print(f"Capital: ${CAPITAL_PER_STRATEGY}/strategy | Leverage: {LEVERAGE}x")
    print(f"Risk: ${RISK_PER_TRADE} SL / ${TP_PER_TRADE} TP (1:3 R/R)")
    print("=" * 60)

    # Download data
    print("\n--- Downloading Data ---")
    df_1m = download_candles("1m", MONTHS_BACK)
    df_5m = download_candles("5m", MONTHS_BACK)

    if df_1m.empty or df_5m.empty:
        print("ERROR: Could not download candle data. Exiting.")
        return

    print(f"\n1m candles: {len(df_1m):,}")
    print(f"5m candles: {len(df_5m):,}")

    # Run backtests
    print("\n--- Running Backtests ---")
    results = []

    results.append(backtest_ema_crossover(df_1m))
    results.append(backtest_rsi_divergence(df_5m))
    results.append(backtest_vwap_bounce(df_1m))
    results.append(backtest_bb_squeeze(df_5m))

    # Print summary
    print_summary_table(results)

    # Generate Telegram message
    tg_msg = generate_telegram_message(results)
    print("--- Telegram Message ---")
    print(tg_msg.replace("<b>", "").replace("</b>", ""))
    print()

    # Save results
    output = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbol": SYMBOL,
            "months": MONTHS_BACK,
            "leverage": LEVERAGE,
            "capital_per_strategy": CAPITAL_PER_STRATEGY,
            "risk_per_trade": RISK_PER_TRADE,
            "tp_per_trade": TP_PER_TRADE,
            "max_daily_loss": MAX_DAILY_LOSS,
        },
        "data": {
            "1m_candles": len(df_1m),
            "5m_candles": len(df_5m),
            "start_date": str(df_1m["timestamp"].min()),
            "end_date": str(df_1m["timestamp"].max()),
        },
        "strategies": results,
        "total_pnl": round(sum(r["total_pnl"] for r in results), 2),
        "total_trades": sum(r["total_trades"] for r in results),
        "telegram_message": tg_msg,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_PATH}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
