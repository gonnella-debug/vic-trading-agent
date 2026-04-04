"""
Vic Backtest — Alternative BTC/USDT Scalping Strategies
Tests 5 alternative strategies against the same cached 6-month data,
then compares all 9 strategies (4 original + 5 new) ranked by PnL.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

# ---------------------------------------------------------------------------
# Config (same as backtest.py)
# ---------------------------------------------------------------------------
SYMBOL = "BTC-USDT-SWAP"
LEVERAGE = 5
CAPITAL_PER_STRATEGY = 200.0
RISK_PER_TRADE = 20.0
TP_PER_TRADE = 60.0
MAX_DAILY_LOSS = 40.0
MONTHS_BACK = 6

DATA_DIR = Path(__file__).parent / "cache"

# Original strategy results (from previous backtest run)
ORIGINAL_RESULTS = [
    {
        "strategy": "rsi_divergence",
        "total_trades": 0,  # placeholder
        "win_rate": 31.3,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 1.37,
        "max_drawdown": 0,
        "total_pnl": 420.0,
        "best_day": 0,
        "worst_day": 0,
        "sharpe_ratio": 0,
        "final_capital": 620.0,
    },
    {
        "strategy": "bb_squeeze",
        "total_trades": 0,
        "win_rate": 29.6,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 1.23,
        "max_drawdown": 0,
        "total_pnl": 320.0,
        "best_day": 0,
        "worst_day": 0,
        "sharpe_ratio": 0,
        "final_capital": 520.0,
    },
    {
        "strategy": "vwap_bounce",
        "total_trades": 0,
        "win_rate": 28.8,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 1.16,
        "max_drawdown": 0,
        "total_pnl": 162.0,
        "best_day": 0,
        "worst_day": 0,
        "sharpe_ratio": 0,
        "final_capital": 362.0,
    },
    {
        "strategy": "ema_crossover",
        "total_trades": 0,
        "win_rate": 22.6,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 0.86,
        "max_drawdown": 0,
        "total_pnl": -203.0,
        "best_day": 0,
        "worst_day": 0,
        "sharpe_ratio": 0,
        "final_capital": -3.0,
    },
]


# ---------------------------------------------------------------------------
# Trade simulator (copied from backtest.py)
# ---------------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, name: str):
        self.name = name
        self.capital = CAPITAL_PER_STRATEGY
        self.equity_curve = [CAPITAL_PER_STRATEGY]
        self.trades = []
        self.daily_pnl = {}
        self.current_date = None
        self.daily_loss_today = 0.0
        self.paused_today = False
        self.position = None
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
        else:
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

            if self.daily_loss_today <= -MAX_DAILY_LOSS:
                self.paused_today = True

            return True
        return False

    def force_close(self, close: float, timestamp, bar_idx: int):
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

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = abs(float(dd.min()))

        total_pnl = round(self.capital - CAPITAL_PER_STRATEGY, 2)

        daily_vals = list(self.daily_pnl.values())
        best_day = max(daily_vals) if daily_vals else 0
        worst_day = min(daily_vals) if daily_vals else 0

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
# Strategy 1: MACD Histogram Reversal (5m)
# ---------------------------------------------------------------------------

def backtest_macd_histogram(df_5m: pd.DataFrame) -> dict:
    """MACD(12,26,9) histogram flips = trade signal."""
    print("\n[Alt 1] MACD Histogram Reversal (5m candles)...")
    engine = BacktestEngine("macd_histogram")

    df = df_5m.copy()
    macd_ind = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_hist"] = macd_ind.macd_diff()

    # Threshold: median of absolute histogram changes (non-zero) * 0.5
    hist_changes = df["macd_hist"].diff().abs().dropna()
    hist_changes = hist_changes[hist_changes > 0]
    flip_threshold = hist_changes.median() * 0.5
    print(f"    Histogram flip threshold: {flip_threshold:.2f}")

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

        prev_hist = df["macd_hist"].iloc[i - 1]
        curr_hist = df["macd_hist"].iloc[i]

        if np.isnan(prev_hist) or np.isnan(curr_hist):
            continue

        change = curr_hist - prev_hist

        # Long: histogram flips from negative to positive with strong change
        if prev_hist < 0 and curr_hist > 0 and abs(change) > flip_threshold:
            engine.open_trade("long", close, ts, i)
        # Short: histogram flips from positive to negative with strong change
        elif prev_hist > 0 and curr_hist < 0 and abs(change) > flip_threshold:
            engine.open_trade("short", close, ts, i)

        if i % 10000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


# ---------------------------------------------------------------------------
# Strategy 2: Stochastic RSI Crossover (5m)
# ---------------------------------------------------------------------------

def backtest_stoch_rsi(df_5m: pd.DataFrame) -> dict:
    """StochRSI(14,14,3,3) %K/%D crossover in overbought/oversold zones."""
    print("\n[Alt 2] Stochastic RSI Crossover (5m candles)...")
    engine = BacktestEngine("stoch_rsi")

    df = df_5m.copy()
    stoch_rsi = ta.momentum.StochRSIIndicator(
        df["close"], window=14, smooth1=3, smooth2=3
    )
    df["stoch_k"] = stoch_rsi.stochrsi_k() * 100  # scale to 0-100
    df["stoch_d"] = stoch_rsi.stochrsi_d() * 100

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

        prev_k = df["stoch_k"].iloc[i - 1]
        prev_d = df["stoch_d"].iloc[i - 1]
        curr_k = df["stoch_k"].iloc[i]
        curr_d = df["stoch_d"].iloc[i]

        if any(np.isnan(v) for v in [prev_k, prev_d, curr_k, curr_d]):
            continue

        # Long: %K crosses above %D in oversold zone (below 20)
        if prev_k <= prev_d and curr_k > curr_d and curr_k < 20:
            engine.open_trade("long", close, ts, i)
        # Short: %K crosses below %D in overbought zone (above 80)
        elif prev_k >= prev_d and curr_k < curr_d and curr_k > 80:
            engine.open_trade("short", close, ts, i)

        if i % 10000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


# ---------------------------------------------------------------------------
# Strategy 3: Volume Spike (1m)
# ---------------------------------------------------------------------------

def backtest_volume_spike(df_1m: pd.DataFrame) -> dict:
    """Volume > 2x 20-period average with directional candle = trade signal."""
    print("\n[Alt 3] Volume Spike (1m candles)...")
    engine = BacktestEngine("volume_spike")

    df = df_1m.copy()
    df["vol_sma20"] = df["volume"].rolling(window=20).mean()

    total = len(df)
    for i in range(25, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        open_ = df["open"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_avg = df["vol_sma20"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        if np.isnan(vol_avg) or vol_avg == 0:
            continue

        # Volume spike: current volume > 2x the 20-period average
        if vol > 2.0 * vol_avg:
            body = close - open_
            candle_range = high - low
            if candle_range == 0:
                continue
            # Require meaningful body (at least 40% of candle range)
            if abs(body) / candle_range < 0.4:
                continue

            if body > 0:  # Bullish candle
                engine.open_trade("long", close, ts, i)
            else:  # Bearish candle
                engine.open_trade("short", close, ts, i)

        if i % 50000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


# ---------------------------------------------------------------------------
# Strategy 4: Supertrend (5m)
# ---------------------------------------------------------------------------

def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Calculate Supertrend indicator."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)  # 1 = up (green), -1 = down (red)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            supertrend.iloc[i] = supertrend.iloc[i - 1]
            direction.iloc[i] = direction.iloc[i - 1]
            continue

        # Lower band logic
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            pass  # keep current lower_band
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Upper band logic
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            pass
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Direction
        if direction.iloc[i - 1] == -1:  # was red (downtrend)
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1  # flip to green
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
        else:  # was green (uptrend)
            if df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1  # flip to red
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]

    return supertrend, direction


def backtest_supertrend(df_5m: pd.DataFrame) -> dict:
    """Supertrend(10,3) direction flip = trade signal."""
    print("\n[Alt 4] Supertrend (5m candles)...")
    engine = BacktestEngine("supertrend")

    df = df_5m.copy()
    df["supertrend"], df["st_direction"] = calc_supertrend(df, period=10, multiplier=3.0)

    total = len(df)
    for i in range(15, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        prev_dir = df["st_direction"].iloc[i - 1]
        curr_dir = df["st_direction"].iloc[i]

        if np.isnan(prev_dir) or np.isnan(curr_dir):
            continue

        # Long: flips from red (-1) to green (1)
        if prev_dir == -1 and curr_dir == 1:
            engine.open_trade("long", close, ts, i)
        # Short: flips from green (1) to red (-1)
        elif prev_dir == 1 and curr_dir == -1:
            engine.open_trade("short", close, ts, i)

        if i % 10000 == 0:
            print(f"    Processed {i}/{total} bars...")

    if engine.position is not None:
        engine.force_close(df["close"].iloc[-1], df["timestamp"].iloc[-1], len(df) - 1)

    result = engine.report()
    print(f"    Done: {result['total_trades']} trades, PnL: ${result['total_pnl']:+,.2f}")
    return result


# ---------------------------------------------------------------------------
# Strategy 5: Ichimoku Cloud Breakout (5m)
# ---------------------------------------------------------------------------

def backtest_ichimoku(df_5m: pd.DataFrame) -> dict:
    """Ichimoku Cloud breakout with Tenkan/Kijun filter."""
    print("\n[Alt 5] Ichimoku Cloud Breakout (5m candles)...")
    engine = BacktestEngine("ichimoku_cloud")

    df = df_5m.copy()
    ichi = ta.trend.IchimokuIndicator(
        df["high"], df["low"],
        window1=9, window2=26, window3=52
    )
    df["tenkan"] = ichi.ichimoku_conversion_line()
    df["kijun"] = ichi.ichimoku_base_line()
    df["senkou_a"] = ichi.ichimoku_a()
    df["senkou_b"] = ichi.ichimoku_b()

    total = len(df)
    for i in range(55, total):
        ts = df["timestamp"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        engine._update_day(ts)
        engine.check_exit(high, low, close, ts, i)

        if engine.position is not None or engine.paused_today:
            continue

        tenkan = df["tenkan"].iloc[i]
        kijun = df["kijun"].iloc[i]
        span_a = df["senkou_a"].iloc[i]
        span_b = df["senkou_b"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        if any(np.isnan(v) for v in [tenkan, kijun, span_a, span_b]):
            continue

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        # Long: price breaks above cloud + Tenkan > Kijun
        if prev_close <= cloud_top and close > cloud_top and tenkan > kijun:
            engine.open_trade("long", close, ts, i)
        # Short: price breaks below cloud + Tenkan < Kijun
        elif prev_close >= cloud_bottom and close < cloud_bottom and tenkan < kijun:
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

def print_comparison_table(all_results: list):
    """Print all 9 strategies ranked by PnL."""
    sorted_results = sorted(all_results, key=lambda r: r["total_pnl"], reverse=True)

    print("\n" + "=" * 110)
    print("ALL 9 STRATEGIES RANKED BY PnL — BTC/USDT PERP | 5x Leverage | $200/strategy | $20 SL / $60 TP | 1:3 R/R")
    print(f"Period: ~{MONTHS_BACK} months of cached data")
    print("=" * 110)

    header = (
        f"{'Rank':<5} {'Strategy':<22} {'Trades':>7} {'Win%':>7} {'AvgWin':>8} "
        f"{'AvgLoss':>8} {'PF':>7} {'MaxDD':>8} {'PnL':>10} {'Sharpe':>7} {'Source':>10}"
    )
    print(header)
    print("-" * 110)

    for rank, r in enumerate(sorted_results, 1):
        source = "ORIGINAL" if r["strategy"] in ["rsi_divergence", "bb_squeeze", "vwap_bounce", "ema_crossover"] else "NEW"
        marker = " ***" if r["total_pnl"] < 0 else ""
        print(
            f"{rank:<5} {r['strategy']:<22} {r['total_trades']:>7} {r['win_rate']:>6.1f}% "
            f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} "
            f"{r['profit_factor']:>6.2f}x ${r['max_drawdown']:>7.2f} "
            f"${r['total_pnl']:>+9.2f} {r['sharpe_ratio']:>7.2f} {source:>10}{marker}"
        )

    print("-" * 110)
    total_new = sum(r["total_pnl"] for r in all_results if r["strategy"] not in ["rsi_divergence", "bb_squeeze", "vwap_bounce", "ema_crossover"])
    total_orig = sum(r["total_pnl"] for r in all_results if r["strategy"] in ["rsi_divergence", "bb_squeeze", "vwap_bounce", "ema_crossover"])
    print(f"      Original 4 strategies total PnL: ${total_orig:>+9.2f}")
    print(f"      New 5 alternative strategies PnL: ${total_new:>+9.2f}")
    print("=" * 110)

    # Recommendation
    print("\n" + "=" * 110)
    print("RECOMMENDATION: Best 4 strategies to use (Slot 1 = TradingView webhook, Slots 2-5 = auto)")
    print("=" * 110)

    profitable = [r for r in sorted_results if r["total_pnl"] > 0]
    top4 = profitable[:4] if len(profitable) >= 4 else sorted_results[:4]

    for slot, r in enumerate(top4, 2):
        source = "ORIGINAL" if r["strategy"] in ["rsi_divergence", "bb_squeeze", "vwap_bounce", "ema_crossover"] else "NEW"
        print(f"  Slot {slot}: {r['strategy']:<22} PnL: ${r['total_pnl']:>+9.2f} | Win: {r['win_rate']}% | PF: {r['profit_factor']}x [{source}]")

    losers = [r for r in sorted_results if r["total_pnl"] <= 0]
    if losers:
        print(f"\n  DROP these losers:")
        for r in losers:
            source = "ORIGINAL" if r["strategy"] in ["rsi_divergence", "bb_squeeze", "vwap_bounce", "ema_crossover"] else "NEW"
            print(f"    - {r['strategy']:<22} PnL: ${r['total_pnl']:>+9.2f} | PF: {r['profit_factor']}x [{source}]")

    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=" * 70)
    print("VIC ALTERNATIVE STRATEGY BACKTESTER")
    print(f"Symbol: BTC/USDT SWAP | Period: {MONTHS_BACK} months (cached)")
    print(f"Capital: ${CAPITAL_PER_STRATEGY}/strategy | Leverage: {LEVERAGE}x")
    print(f"Risk: ${RISK_PER_TRADE} SL / ${TP_PER_TRADE} TP (1:3 R/R)")
    print(f"Max daily loss: ${MAX_DAILY_LOSS}/strategy")
    print("=" * 70)

    # Load cached data
    print("\n--- Loading Cached Data ---")
    cache_1m = DATA_DIR / "btc_usdt_swap_1m_6m.csv"
    cache_5m = DATA_DIR / "btc_usdt_swap_5m_6m.csv"

    if not cache_1m.exists() or not cache_5m.exists():
        print("ERROR: Cached data not found. Run backtest.py first.")
        return

    df_1m = pd.read_csv(cache_1m, parse_dates=["timestamp"])
    df_5m = pd.read_csv(cache_5m, parse_dates=["timestamp"])

    print(f"  1m candles: {len(df_1m):,} ({df_1m['timestamp'].min()} to {df_1m['timestamp'].max()})")
    print(f"  5m candles: {len(df_5m):,} ({df_5m['timestamp'].min()} to {df_5m['timestamp'].max()})")

    # Run alternative backtests
    print("\n--- Running Alternative Strategy Backtests ---")
    alt_results = []

    alt_results.append(backtest_macd_histogram(df_5m))
    alt_results.append(backtest_stoch_rsi(df_5m))
    alt_results.append(backtest_volume_spike(df_1m))
    alt_results.append(backtest_supertrend(df_5m))
    alt_results.append(backtest_ichimoku(df_5m))

    # Print individual alternative results
    print("\n" + "=" * 100)
    print("ALTERNATIVE STRATEGY RESULTS")
    print("=" * 100)
    header = f"{'Strategy':<22} {'Trades':>7} {'Win%':>7} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>7} {'MaxDD':>8} {'PnL':>10} {'Sharpe':>7}"
    print(header)
    print("-" * 100)
    for r in sorted(alt_results, key=lambda x: x["total_pnl"], reverse=True):
        print(
            f"{r['strategy']:<22} {r['total_trades']:>7} {r['win_rate']:>6.1f}% "
            f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} "
            f"{r['profit_factor']:>6.2f}x ${r['max_drawdown']:>7.2f} "
            f"${r['total_pnl']:>+9.2f} {r['sharpe_ratio']:>7.2f}"
        )
    print("-" * 100)
    alt_total = sum(r["total_pnl"] for r in alt_results)
    print(f"{'ALT TOTAL':<22} {sum(r['total_trades'] for r in alt_results):>7} {'':>7} {'':>8} {'':>8} {'':>7} {'':>8} ${alt_total:>+9.2f}")
    print("=" * 100)

    # Daily extremes for alternatives
    print("\nDaily Extremes (Alternative Strategies):")
    for r in alt_results:
        print(f"  {r['strategy']:<22} Best day: ${r['best_day']:>+8.2f} | Worst day: ${r['worst_day']:>+8.2f} | Final capital: ${r['final_capital']:>8.2f}")

    # Combined comparison of all 9
    all_results = ORIGINAL_RESULTS + alt_results
    print_comparison_table(all_results)

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
        "alternative_strategies": alt_results,
        "all_strategies_ranked": sorted(all_results, key=lambda r: r["total_pnl"], reverse=True),
    }

    results_path = Path(__file__).parent / "backtest_alternatives_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
