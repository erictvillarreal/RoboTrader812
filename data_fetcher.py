"""
RoboTrader S4 — data_fetcher.py
Descarga velas de Binance Futures (/fapi/v1/klines).
Paginación automática, dedup, sort por open_time.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config import FUTURES_BASE, TESTNET_BASE, USE_TESTNET, INTERVAL

# Binance bloquea algunas regiones (451). Fallback a endpoints alternativos.
_FUTURES_ENDPOINTS = [
    FUTURES_BASE,           # fapi.binance.com (principal)
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]
_BASE = TESTNET_BASE if USE_TESTNET else FUTURES_BASE

_INTERVAL_MAP = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"}

def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _interval_ms(interval: str) -> int:
    n = int(interval[:-1])
    unit = interval[-1]
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]

def get_klines(symbol: str, interval: str, start_ms: Optional[int] = None,
               end_ms: Optional[int] = None, limit: int = 1000) -> list:
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"Intervalo no soportado: {interval}")
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    if end_ms:   params["endTime"]   = end_ms

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RoboTraderS4/1.0)",
        "Accept":     "application/json",
    }
    endpoints = [TESTNET_BASE] if USE_TESTNET else _FUTURES_ENDPOINTS
    last_err = None
    for base in endpoints:
        try:
            url = base + "/fapi/v1/klines"
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code in (451, 403):
                last_err = f"{r.status_code} from {base}"
                continue
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                last_err = f"respuesta vacía de {base}"
                continue
            return r.json()
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Todos los endpoints fallaron. Último error: {last_err}")

def klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open","high","low","close","volume","quote_vol","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["open_time","open","high","low","close","volume",
             "close_time","trades","taker_base","taker_quote"]].copy()
    return df.sort_values("open_time").reset_index(drop=True)

def get_historical_data(symbol: str = "BTCUSDT", interval: str = INTERVAL,
                        limit: int = 4000, start: Optional[datetime] = None,
                        end: Optional[datetime] = None,
                        sleep_sec: float = 0.25) -> pd.DataFrame:
    """
    Descarga las ultimas `limit` velas. Para el loop de trader.py,
    limit=300 — se pide en una sola llamada sin startTime/endTime
    para maxima compatibilidad con Binance en cualquier region.
    """
    frames = []
    remaining = limit

    while remaining > 0:
        batch = min(1000, remaining)
        # Sin startTime ni endTime — Binance devuelve las N mas recientes
        kl = get_klines(symbol, interval, limit=batch)
        if not kl:
            break
        df = klines_to_df(kl)
        if df.empty:
            break
        frames.append(df)
        remaining -= len(df)
        # Para limit <= 1000 una sola llamada es suficiente
        if limit <= 1000:
            break
        # Para limit > 1000 paginar hacia atras
        first_open_ms = int(df["open_time"].iloc[0].timestamp() * 1000)
        end_ms = first_open_ms - 1
        if start and first_open_ms <= _to_ms(start):
            break
        time.sleep(sleep_sec)
        # Segunda vuelta: pedir con endTime
        batch2 = min(1000, remaining)
        if batch2 <= 0:
            break
        kl2 = get_klines(symbol, interval, end_ms=end_ms, limit=batch2)
        if not kl2:
            break
        df2 = klines_to_df(kl2)
        if df2.empty:
            break
        frames.append(df2)
        remaining -= len(df2)
        break  # maximo 2 paginas para el loop de trading

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return out.iloc[-limit:].reset_index(drop=True)

if __name__ == "__main__":
    df = get_historical_data("BTCUSDT", "1h", limit=10)
    print(df[["open_time","open","high","low","close","volume"]].tail())