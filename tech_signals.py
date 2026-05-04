"""
RoboTrader S4 — tech_signals.py
Indicadores técnicos determinísticos.
Mismas funciones que walk.py usa internamente.
"""
import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    up   = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs   = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(s: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(s, fast)
    ema_slow = _ema(s, slow)
    macd     = ema_fast - ema_slow
    sig      = _ema(macd, signal)
    return macd, sig, macd - sig

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def add_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Añade EMA10/30, RSI14, MACD, ATR al DataFrame. In-place sobre copia."""
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ema_10"]                        = _ema(df["close"], 10)
    df["ema_30"]                        = _ema(df["close"], 30)
    df["rsi_14"]                        = _rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_diff"] = _macd(df["close"])
    df["atr"]                           = _atr(df, 14)
    return df
